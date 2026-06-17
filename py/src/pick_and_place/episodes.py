# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Sample and prepare random pick-and-carry episodes under live physics.

Shared by ``view_trajectory`` (sim-only viewer), ``real_analytic_pick_and_place``
(the hardware path) and ``record_episodes`` (batch dataset generation). All need
to draw random source/target cube poses
and near-neutral start/end arm poses, build the scene with a dynamic cube, and
search ``pick_and_carry_candidates`` (vetting each with a collision preflight)
for a trajectory that runs clean — resampling the poses until one is found.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from pick_and_place import build_scene
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import So101Kinematics, derive_kinematics
from pick_and_place.trajectory import (
    GRIPPER_OPEN,
    NEUTRAL_ARM_JOINTS,
    GraspChoice,
    Trajectory,
    trajectory_candidates,
)
from pick_and_place.workspace_overlays import (
    AZIMUTH_MAX,
    AZIMUTH_MIN,
    PAN_AXIS,
    WORKSPACE_OVERLAYS,
)

_CLEARANCE_OVERLAY = next(o for o in WORKSPACE_OVERLAYS if o.name == "workspace_clearance_pregrasp")

# ±radians of random joint perturbation applied to the neutral start/end pose.
_NEAR_NEUTRAL_JOINT_SCALE = 0.4
# Per-joint overrides of the perturbation scale. ``shoulder_lift``, ``elbow_flex``
# and ``wrist_flex`` are held tighter than the rest because they are the levers
# that tilt the gripper down toward the floor: a full ±0.4 swing on them is what
# drives the near-neutral start/end pose down toward the ground. Tightening them to
# ±0.2 keeps almost every sampled pose above the clearance gate below (≈99% pass),
# so the gate rarely has to resample while still guaranteeing the floor margin.
_JOINT_SCALE_OVERRIDES: dict[str, float] = {
    "shoulder_lift": 0.2,
    "elbow_flex": 0.2,
    "wrist_flex": 0.2,
}
# Minimum height (m) the lowest gripper-jaw corner must clear the floor by for a
# sampled start pose to be accepted, so the arm begins well up in the air rather
# than skimming (or buried in) the ground.
MIN_START_CLEARANCE = 0.10


class EpisodeSamplingError(RuntimeError):
    """Raised when no collision-free trajectory is found within the attempt budget."""


def sample_cube(rng: np.random.Generator) -> CubePose:
    """Sample a cube pose uniformly inside the clearance-pregrasp annular sector."""
    r_inner, r_outer = _CLEARANCE_OVERLAY.inner_radius, _CLEARANCE_OVERLAY.outer_radius
    # Uniform area sampling: draw r² uniformly so density is flat in 2-D.
    r = math.sqrt(rng.uniform(r_inner**2, r_outer**2))
    theta = rng.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
    yaw = rng.uniform(0.0, 2 * math.pi)
    return CubePose(
        x=PAN_AXIS[0] + r * math.cos(theta),
        y=PAN_AXIS[1] + r * math.sin(theta),
        z=CUBE_HALF_SIZE,
        yaw=yaw,
    )


def sample_near_neutral(rng: np.random.Generator) -> tuple[dict[str, float], float]:
    """Return arm joints and gripper perturbed slightly from the neutral pose.

    Each joint is perturbed by ±its scale (``_JOINT_SCALE_OVERRIDES`` for the
    tightened joints, else ``_NEAR_NEUTRAL_JOINT_SCALE``).
    """
    joints = {}
    for name, value in NEUTRAL_ARM_JOINTS.items():
        scale = _JOINT_SCALE_OVERRIDES.get(name, _NEAR_NEUTRAL_JOINT_SCALE)
        joints[name] = value + rng.uniform(-scale, scale)
    gripper = float(rng.uniform(0.0, GRIPPER_OPEN))
    return joints, gripper


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value


def build_geom_sets(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    """Return (robot_geom_ids, env_geom_ids).

    Robot geoms: all geoms on bodies other than the worldbody and the pick_cube.
    Environment geoms: floor and pick_cube — the things we check the robot against.
    """
    world_body_id = 0
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    robot_geom_ids = {
        gid
        for gid in range(model.ngeom)
        if model.geom_bodyid[gid] not in (world_body_id, cube_body_id)
    }
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return robot_geom_ids, {cube_geom_id, floor_geom_id}


def scan_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> list[tuple[str, str]]:
    """Return (name1, name2) for robot↔environment and robot↔robot contacts."""
    hits = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        g1_robot = g1 in robot_geom_ids
        g2_robot = g2 in robot_geom_ids
        if (g1_robot and g2 in env_geom_ids) or (g2_robot and g1 in env_geom_ids) or (g1_robot and g2_robot):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
            hits.append((n1, n2))
    return hits


def _preflight(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    actuator_id: dict[str, int],
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> list[tuple[float, str, str]]:
    """Simulate the full trajectory in a shadow MjData and return collision events.

    The shadow starts at the trajectory's own start pose (``start_joints`` /
    ``start_gripper``) — the same pose the recorded run begins from — so the
    approach swing off the start pose and the retreat onto the end pose are both
    part of what gets vetted, not just the cube-handling middle.
    """
    shadow = mujoco.MjData(model)
    for name, value in trajectory.start_joints.items():
        set_joint(model, shadow, name, value)
        shadow.ctrl[actuator_id[name]] = value
    set_joint(model, shadow, "gripper", trajectory.start_gripper)
    shadow.ctrl[actuator_id["gripper"]] = trajectory.start_gripper

    mujoco.mj_forward(model, shadow)

    events: list[tuple[float, str, str]] = []
    while shadow.time < trajectory.duration:
        frame = trajectory.evaluate(shadow.time)
        for name, value in frame.joints.items():
            shadow.ctrl[actuator_id[name]] = value
        shadow.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, shadow)
        for n1, n2 in scan_contacts(model, shadow, robot_geom_ids, env_geom_ids):
            events.append((shadow.time, n1, n2))
    return events


_JAW_PREFIXES = ("fixed_jaw_col", "moving_jaw_col")


def _is_jaw(n: str) -> bool:
    return n.startswith(_JAW_PREFIXES)


def is_unexpected(n1: str, n2: str) -> bool:
    """False only for jaw↔cube contacts, which are the intentional grasp."""
    return not ((_is_jaw(n1) and n2 == "pick_cube") or (_is_jaw(n2) and n1 == "pick_cube"))


def jaw_geom_ids(model: mujoco.MjModel) -> list[int]:
    """Geom ids of the gripper-jaw collision boxes — the lowest-reaching robot
    parts near neutral, hence what the start-pose floor clearance is measured on."""
    return [
        gid
        for gid in range(model.ngeom)
        if _is_jaw(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
    ]


def jaw_floor_clearance(
    model: mujoco.MjModel, data: mujoco.MjData, jaw_ids: list[int]
) -> float:
    """Height of the lowest gripper-jaw corner above the floor for the current
    (already forward-kinematics'd) configuration. The jaws are oriented boxes, so
    each box's lowest corner is its center minus the world-z extent of its half
    sizes; the minimum over all jaw geoms is the clearance."""
    clearance = math.inf
    for gid in jaw_ids:
        half_sizes = model.geom_size[gid]
        world_z_row = data.geom_xmat[gid].reshape(3, 3)[2]
        half_height = float(np.abs(world_z_row) @ half_sizes)
        clearance = min(clearance, float(data.geom_xpos[gid][2]) - half_height)
    return clearance


@dataclass
class Episode:
    """A prepared, collision-free pick-and-carry ready to run under physics.

    ``model``/``data`` are compiled with the cube as a dynamic free body and the
    arm initialised at the start pose (both ``qpos`` and ``ctrl`` set). The
    ``trajectory`` already carries the sampled start/end poses.
    """

    source: CubePose
    target: CubePose
    start_joints: dict[str, float]
    start_gripper: float
    end_joints: dict[str, float]
    end_gripper: float
    model: mujoco.MjModel
    data: mujoco.MjData
    kinematics: So101Kinematics
    actuator_id: dict[str, int]
    robot_geom_ids: set[int] = field(repr=False)
    env_geom_ids: set[int] = field(repr=False)
    trajectory: Trajectory = field(repr=False)
    attempts: int = 1

    @property
    def grasp(self) -> GraspChoice:
        return self.trajectory.grasp


def _build_model(
    source: CubePose,
    include_environment: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_scene(include_environment=include_environment)
    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    half_yaw = source.yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()  # make the cube a real dynamic body
    model = spec.compile()
    return model, mujoco.MjData(model)


def prepare_episode(
    rng: np.random.Generator,
    source: CubePose | None = None,
    target: CubePose | None = None,
    *,
    max_attempts: int | None = None,
    verbose: bool = False,
    include_environment: bool = False,
) -> Episode:
    """Sample poses and return the first collision-free pick-and-carry.

    ``source``/``target`` pin those cube poses (otherwise they are resampled
    each attempt); the start/end arm poses are always sampled fresh. With both
    poses pinned and no feasible trajectory, raises immediately. Otherwise keeps
    resampling until success or, if ``max_attempts`` is set, that many attempts
    fail — then raises :class:`EpisodeSamplingError`.
    """
    fixed_source = source is not None
    fixed_target = target is not None

    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1

        ep_source = source if fixed_source else sample_cube(rng)
        ep_target = target if fixed_target else sample_cube(rng)
        start_joints, start_gripper = sample_near_neutral(rng)
        end_joints, end_gripper = sample_near_neutral(rng)

        model, data = _build_model(ep_source, include_environment=include_environment)
        kinematics = derive_kinematics(model)
        actuator_id = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
        }
        for name, value in start_joints.items():
            set_joint(model, data, name, value)
            data.ctrl[actuator_id[name]] = value
        # Initialise the gripper qpos too, not just its target: otherwise it
        # starts at the model default (closed) and the servo snaps it open on the
        # first step, polluting the viewer and the recorded state with a fake jerk.
        set_joint(model, data, "gripper", start_gripper)
        data.ctrl[actuator_id["gripper"]] = start_gripper
        mujoco.mj_forward(model, data)

        # Reject start poses that sit on or too near the floor before spending a
        # trajectory search on them — the arm must begin in the air. (The start
        # arm pose is resampled every attempt, so retrying is always meaningful.)
        clearance = jaw_floor_clearance(model, data, jaw_geom_ids(model))
        if clearance < MIN_START_CLEARANCE:
            if verbose:
                print(f"attempt {attempt}: start jaw clearance {clearance:.3f}m too low, resampling...")
            continue

        robot_geom_ids, env_geom_ids = build_geom_sets(model)

        trajectory = None
        for traj in trajectory_candidates(kinematics, ep_source, ep_target, start_joints, start_gripper, end_joints, end_gripper):
            grasp = traj.grasp
            events = _preflight(model, traj, actuator_id, robot_geom_ids, env_geom_ids)
            unexpected = [(t, n1, n2) for t, n1, n2 in events if is_unexpected(n1, n2)]
            if not unexpected:
                trajectory = traj
                if verbose:
                    print(
                        f"source: x={ep_source.x:.4f}  y={ep_source.y:.4f}"
                        f"  yaw={math.degrees(ep_source.yaw):.1f}°"
                        f"  target: x={ep_target.x:.4f}  y={ep_target.y:.4f}"
                        f"  yaw={math.degrees(ep_target.yaw):.1f}°"
                    )
                    print(
                        f"grasp: face={grasp.face}  elbow={grasp.elbow}"
                        f"  carry={traj.carry.mode}  (attempt {attempt})"
                    )
                break
            if verbose:
                seen_pairs: set[tuple[str, str]] = set()
                for t, n1, n2 in unexpected:
                    key = (min(n1, n2), max(n1, n2))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        print(f"skip {grasp.face}/{grasp.elbow}: collision t={t:.3f}s  {n1} ↔ {n2}")

        if trajectory is not None:
            return Episode(
                source=ep_source,
                target=ep_target,
                start_joints=start_joints,
                start_gripper=start_gripper,
                end_joints=end_joints,
                end_gripper=end_gripper,
                model=model,
                data=data,
                kinematics=kinematics,
                actuator_id=actuator_id,
                robot_geom_ids=robot_geom_ids,
                env_geom_ids=env_geom_ids,
                trajectory=trajectory,
                attempts=attempt,
            )

        if fixed_source and fixed_target:
            raise EpisodeSamplingError(
                "no collision-free pick-and-carry found for this source/target"
            )
        if verbose:
            print(f"attempt {attempt}: no trajectory found, resampling...")

    raise EpisodeSamplingError(f"no collision-free trajectory within {max_attempts} attempts")
