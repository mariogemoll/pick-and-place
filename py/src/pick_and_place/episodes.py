# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Sample and prepare random pick-and-carry episodes under live physics.

Shared by ``pick_and_place/sim.py`` (sim-only viewer), ``pick_and_place/real.py``
(the hardware path) and ``record_episodes`` (batch dataset generation). All need
to draw random source/target cube poses
and near-neutral start/end arm poses, build the scene with a dynamic cube, and
search ``pick_and_carry_candidates`` (vetting each with a collision preflight)
for a trajectory that runs clean — resampling the poses until one is found.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place import build_scene
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import ARM_JOINT_NAMES, So101Kinematics, derive_kinematics
from pick_and_place.paper_detection import add_paper_target_marker
from pick_and_place.robot_dynamics import set_actuator_activation
from pick_and_place.trajectory import (
    GRIPPER_OPEN,
    NEUTRAL_ARM_JOINTS,
    GraspChoice,
    Trajectory,
    free_grasp_candidates,
    grasp_candidates,
    trajectory_candidates_for_grasp,
)
from pick_and_place.workspace_overlays import (
    AZIMUTH_MAX,
    AZIMUTH_MIN,
    CANONICAL_PICKUP_OVERLAY,
    PAN_AXIS,
    CUBE_PLACEMENT_OVERLAY,
    is_cube_drop_allowed,
    is_cube_placement_allowed,
    is_cube_recovery_target_allowed,
    is_target_plate_position_allowed,
)

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
# Shoulder-pan half-range for look-around search poses. Near the ±1.92 rad pan
# limit (with a small margin) so the search sweeps the arm across nearly its full
# lateral travel to clear the overhead view, rather than the tight start jitter.
_HUNT_PAN_SCALE = 1.7
# Minimum height (m) the lowest gripper-jaw corner must clear the floor by for a
# sampled start pose to be accepted, so the arm begins well up in the air rather
# than skimming (or buried in) the ground.
MIN_START_CLEARANCE = 0.10
PICKUP_YAW_DEVIATION = math.pi / 4.0


def pickup_yaw_from_azimuth(azimuth: float, deviation: float = 0.0) -> float:
    """Return cube yaw relative to the local pickup azimuth frame."""
    return azimuth + deviation


class EpisodeSamplingError(RuntimeError):
    """Raised when no collision-free trajectory is found within the attempt budget."""


@dataclass(frozen=True)
class PreflightCollision:
    """One unexpected contact observed while simulating a candidate trajectory."""

    time: float
    phase: str
    phase_time: float
    geom1: str
    geom2: str
    body1: str
    body2: str
    dist: float
    position: tuple[float, float, float]


@dataclass(frozen=True)
class PlacementError:
    """Final cube placement error relative to an episode target."""

    cube_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    dx: float
    dy: float
    dz: float
    xy: float

    def summary(self) -> str:
        return (
            f"placement error: xy={self.xy * 1000:.1f} mm "
            f"(dx={self.dx * 1000:+.1f}, dy={self.dy * 1000:+.1f}), "
            f"z={self.dz * 1000:+.1f} mm"
        )


def sample_cube(rng: np.random.Generator) -> CubePose:
    """Sample a cube pose in the canonical pick-lift sector."""
    r_inner = CANONICAL_PICKUP_OVERLAY.inner_radius
    r_outer = CANONICAL_PICKUP_OVERLAY.outer_radius
    while True:
        # Uniform radial sampling to prevent points bunching up at the outer edge.
        r = rng.uniform(r_inner, r_outer)
        theta = rng.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
        x = PAN_AXIS[0] + r * math.cos(theta)
        y = PAN_AXIS[1] + r * math.sin(theta)
        if is_cube_placement_allowed(x, y):
            break
    yaw = pickup_yaw_from_azimuth(
        theta,
        rng.uniform(-PICKUP_YAW_DEVIATION, PICKUP_YAW_DEVIATION),
    )
    return CubePose(
        x=x,
        y=y,
        z=CUBE_HALF_SIZE,
        yaw=yaw,
    )


def sample_recovery_cube(rng: np.random.Generator) -> CubePose:
    """Sample a conservative pickup-zone target for unrecorded cube recovery."""
    while True:
        pose = sample_cube(rng)
        if is_cube_recovery_target_allowed(pose.x, pose.y):
            return pose


def sample_target(rng: np.random.Generator) -> CubePose:
    """Sample a target in the broader drop sector with room for the plate."""
    r_inner = CUBE_PLACEMENT_OVERLAY.inner_radius
    r_outer = CUBE_PLACEMENT_OVERLAY.outer_radius
    while True:
        r = rng.uniform(r_inner, r_outer)
        theta = rng.uniform(AZIMUTH_MIN, AZIMUTH_MAX)
        x = PAN_AXIS[0] + r * math.cos(theta)
        y = PAN_AXIS[1] + r * math.sin(theta)
        if is_cube_drop_allowed(x, y) and is_target_plate_position_allowed(x, y):
            return CubePose(x=x, y=y, z=CUBE_HALF_SIZE)


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


def sample_hunt_pose(rng: np.random.Generator) -> tuple[dict[str, float], float]:
    """Return a search pose: a wide shoulder-pan swing, the rest near neutral.

    The arm itself can sit between the fixed overhead camera and the cube or
    drop-zone square, so the look-around search swings the pan far wider than the
    near-neutral start jitter to clear the view from a range of angles. The tilt
    joints stay near neutral, keeping the gripper well above the floor."""
    joints, gripper = sample_near_neutral(rng)
    joints["shoulder_pan"] = NEUTRAL_ARM_JOINTS["shoulder_pan"] + rng.uniform(
        -_HUNT_PAN_SCALE, _HUNT_PAN_SCALE
    )
    return joints, gripper


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value


def get_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[jid]])


def cube_quat_from_pose(pose: CubePose) -> tuple[float, float, float, float]:
    """MuJoCo ``w, x, y, z`` quaternion for ``pose``'s intrinsic ZYX rotation."""
    half_roll = pose.roll / 2.0
    half_pitch = pose.pitch / 2.0
    half_yaw = pose.yaw / 2.0
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def set_cube_pose(model: mujoco.MjModel, data: mujoco.MjData, source: CubePose) -> None:
    """Move the freejoint ``pick_cube`` to ``source`` in an existing model's data.

    Lets a single persistent model be reused across episodes (so a live viewer can
    stay bound to it) instead of recompiling one per cube pose."""
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    jnt_adr = model.body_jntadr[cube_body_id]
    qpos_adr = model.jnt_qposadr[jnt_adr]
    qvel_adr = model.jnt_dofadr[jnt_adr]
    data.qpos[qpos_adr:qpos_adr + 3] = (source.x, source.y, source.z)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = cube_quat_from_pose(source)
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def placement_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: CubePose,
) -> PlacementError:
    """Measure the current cube-center offset from the target center."""
    mujoco.mj_forward(model, data)
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    cube_xyz = tuple(float(v) for v in data.xpos[cube_body_id])
    target_xyz = (float(target.x), float(target.y), float(CUBE_HALF_SIZE))
    dx = cube_xyz[0] - target_xyz[0]
    dy = cube_xyz[1] - target_xyz[1]
    dz = cube_xyz[2] - target_xyz[2]
    return PlacementError(
        cube_xyz=cube_xyz,
        target_xyz=target_xyz,
        dx=dx,
        dy=dy,
        dz=dz,
        xy=math.hypot(dx, dy),
    )


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


def _trajectory_phase_at(trajectory: Trajectory, elapsed: float) -> tuple[str, float]:
    t = elapsed
    for phase in trajectory.phases:
        if t < phase.duration:
            return phase.name, t
        t -= phase.duration
    if not trajectory.phases:
        return "none", elapsed
    last = trajectory.phases[-1]
    return last.name, min(max(t, 0.0), last.duration)


def _preflight(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    actuator_id: dict[str, int],
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
    *,
    detailed: bool = False,
) -> list[tuple[float, str, str]] | list[PreflightCollision]:
    """Simulate the full trajectory in a shadow MjData and return collision events.

    The shadow starts at the trajectory's own start pose (``start_joints`` /
    ``start_gripper``) — the same pose the recorded run begins from — so the
    approach swing off the start pose and the retreat onto the end pose are both
    part of what gets vetted, not just the cube-handling middle.
    """
    shadow = mujoco.MjData(model)
    if trajectory.source is not None:
        set_cube_pose(model, shadow, trajectory.source)
    for name, value in trajectory.start_joints.items():
        set_joint(model, shadow, name, value)
        shadow.ctrl[actuator_id[name]] = value
        set_actuator_activation(model, shadow, actuator_id[name], value)
    set_joint(model, shadow, "gripper", trajectory.start_gripper)
    shadow.ctrl[actuator_id["gripper"]] = trajectory.start_gripper
    set_actuator_activation(model, shadow, actuator_id["gripper"], trajectory.start_gripper)

    mujoco.mj_forward(model, shadow)

    events: list[tuple[float, str, str]] | list[PreflightCollision] = []
    while shadow.time < trajectory.duration:
        frame = trajectory.evaluate(shadow.time)
        for name, value in frame.joints.items():
            shadow.ctrl[actuator_id[name]] = value
        shadow.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, shadow)
        for i in range(shadow.ncon):
            contact = shadow.contact[i]
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            g1_robot = g1 in robot_geom_ids
            g2_robot = g2 in robot_geom_ids
            if not (
                (g1_robot and g2 in env_geom_ids)
                or (g2_robot and g1 in env_geom_ids)
                or (g1_robot and g2_robot)
            ):
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
            if not detailed:
                events.append((shadow.time, n1, n2))
                continue
            phase, phase_time = _trajectory_phase_at(trajectory, shadow.time)
            b1_id = int(model.geom_bodyid[g1])
            b2_id = int(model.geom_bodyid[g2])
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1_id) or str(b1_id)
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2_id) or str(b2_id)
            events.append(
                PreflightCollision(
                    time=float(shadow.time),
                    phase=phase,
                    phase_time=float(phase_time),
                    geom1=n1,
                    geom2=n2,
                    body1=b1,
                    body2=b2,
                    dist=float(contact.dist),
                    position=tuple(float(x) for x in contact.pos),
                )
            )
    return events




_JAW_PREFIXES = ("fixed_jaw_col", "moving_jaw_col")


def _is_jaw(n: str) -> bool:
    return n.startswith(_JAW_PREFIXES)


def is_unexpected(n1: str, n2: str) -> bool:
    """False only for jaw↔cube contacts, which are the intentional grasp."""
    return not ((_is_jaw(n1) and n2 == "pick_cube") or (_is_jaw(n2) and n1 == "pick_cube"))


def make_carry_collision_checker(
    model: mujoco.MjModel,
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
) -> Callable[[dict[str, float]], bool]:
    """Build a cheap per-configuration collision check for screening carry
    candidates during planning, before committing to the much more expensive
    full-trajectory preflight (``_preflight``, which steps real dynamics).

    Uses pure kinematics + collision detection -- no integration, no contact
    dynamics -- so it's fast enough to run on every candidate. The cube isn't
    positioned here (its pose isn't tracked by this cheap check), so it's
    excluded from the environment set entirely; the full preflight remains the
    authoritative check for anything cube-related.
    """
    shadow = mujoco.MjData(model)
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    env_geom_ids_no_cube = {gid for gid in env_geom_ids if model.geom_bodyid[gid] != cube_body_id}
    qpos_adr = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ARM_JOINT_NAMES
    }

    def check(joints: dict[str, float]) -> bool:
        for name, value in joints.items():
            shadow.qpos[qpos_adr[name]] = value
        mujoco.mj_kinematics(model, shadow)
        mujoco.mj_collision(model, shadow)
        for i in range(shadow.ncon):
            contact = shadow.contact[i]
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            g1_robot = g1 in robot_geom_ids
            g2_robot = g2 in robot_geom_ids
            if not (
                (g1_robot and g2 in env_geom_ids_no_cube)
                or (g2_robot and g1 in env_geom_ids_no_cube)
                or (g1_robot and g2_robot)
            ):
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
            if is_unexpected(n1, n2):
                return False
        return True

    return check


def _preflight_collision_is_unexpected(event: PreflightCollision) -> bool:
    return is_unexpected(event.geom1, event.geom2)


def _print_preflight_debug(
    attempt: int,
    traj: Trajectory,
    events: list[PreflightCollision],
    *,
    limit: int,
) -> None:
    grasp = traj.grasp
    if grasp is None:
        label = "unknown grasp"
    else:
        label = f"{grasp.face}/{grasp.elbow}"
    print(f"preflight {attempt=} {label}: {len(events)} unexpected contacts")

    phase_counts = Counter(event.phase for event in events)
    pair_counts = Counter((min(event.geom1, event.geom2), max(event.geom1, event.geom2)) for event in events)
    if phase_counts:
        phase_summary = ", ".join(f"{phase}={count}" for phase, count in phase_counts.most_common())
        print(f"  by phase: {phase_summary}")
    for (g1, g2), count in pair_counts.most_common(6):
        print(f"  pair {count:4d}x: {g1} <-> {g2}")

    for event in events[:limit]:
        penetration_mm = max(0.0, -event.dist) * 1000.0
        print(
            f"  t={event.time:.3f}s phase={event.phase}:{event.phase_time:.3f}s "
            f"penetration={penetration_mm:.2f}mm "
            f"{event.geom1}({event.body1}) <-> {event.geom2}({event.body2}) "
            f"pos=({event.position[0]:.3f}, {event.position[1]:.3f}, {event.position[2]:.3f})"
        )
    if len(events) > limit:
        print(f"  ... {len(events) - limit} more contacts omitted")


def _cube_start_array(source: CubePose) -> np.ndarray:
    return np.array((source.x, source.y, source.z, source.yaw), dtype=float)


def _trajectory_pose_array(pose: CubePose) -> np.ndarray:
    return np.array((pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw), dtype=float)


def _save_failed_preflight_trajectory(
    path: Path,
    model: mujoco.MjModel,
    trajectory: Trajectory,
    actuator_id: dict[str, int],
    events: list[PreflightCollision],
) -> None:
    """Save a rejected candidate as a replayable qpos rollout plus metadata."""
    shadow = mujoco.MjData(model)
    for name, value in trajectory.start_joints.items():
        set_joint(model, shadow, name, value)
        shadow.ctrl[actuator_id[name]] = value
        set_actuator_activation(model, shadow, actuator_id[name], value)
    set_joint(model, shadow, "gripper", trajectory.start_gripper)
    shadow.ctrl[actuator_id["gripper"]] = trajectory.start_gripper
    set_actuator_activation(model, shadow, actuator_id["gripper"], trajectory.start_gripper)
    mujoco.mj_forward(model, shadow)

    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    ctrl: list[np.ndarray] = []
    t_values: list[float] = []
    phase_names: list[str] = []
    phase_times: list[float] = []

    while shadow.time < trajectory.duration:
        frame = trajectory.evaluate(shadow.time)
        for name, value in frame.joints.items():
            shadow.ctrl[actuator_id[name]] = value
        shadow.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, shadow)
        phase, phase_time = _trajectory_phase_at(trajectory, shadow.time)
        qpos.append(shadow.qpos.copy())
        qvel.append(shadow.qvel.copy())
        ctrl.append(shadow.ctrl.copy())
        t_values.append(float(shadow.time))
        phase_names.append(phase)
        phase_times.append(float(phase_time))

    grasp = trajectory.grasp
    carry = trajectory.carry
    event_rows = np.array(
        [
            (
                event.time,
                event.phase,
                event.phase_time,
                event.geom1,
                event.geom2,
                event.body1,
                event.body2,
                event.dist,
                *event.position,
            )
            for event in events
        ],
        dtype=[
            ("time", "f8"),
            ("phase", "U32"),
            ("phase_time", "f8"),
            ("geom1", "U96"),
            ("geom2", "U96"),
            ("body1", "U96"),
            ("body2", "U96"),
            ("dist", "f8"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        ctrl=np.asarray(ctrl),
        t=np.asarray(t_values),
        phase=np.asarray(phase_names),
        phase_t=np.asarray(phase_times),
        control_hz=np.array(1.0 / model.opt.timestep),
        cube_start=_cube_start_array(trajectory.source),
        source=_trajectory_pose_array(trajectory.source),
        target=_trajectory_pose_array(trajectory.target),
        grasp_face=np.array("" if grasp is None else grasp.face),
        grasp_elbow=np.array("" if grasp is None else grasp.elbow),
        carry_mode=np.array("" if carry is None else carry.mode),
        duration=np.array(trajectory.duration),
        collision_events=event_rows,
    )


def _write_failed_trajectory_note(
    failed_trajectory_dir: Path | None,
    reason: str,
    *,
    source: CubePose | None,
    target: CubePose | None,
) -> None:
    if failed_trajectory_dir is None:
        return
    failed_trajectory_dir.mkdir(parents=True, exist_ok=True)
    note = {
        "reason": reason,
        "source": None if source is None else _trajectory_pose_array(source).tolist(),
        "target": None if target is None else _trajectory_pose_array(target).tolist(),
    }
    (failed_trajectory_dir / "planning_failed.json").write_text(
        json.dumps(note, indent=2) + "\n"
    )


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
    offwidth: int = 1280,
    offheight: int = 720,
    paper_target_marker: bool = False,
    background_panorama: Path | str | None = None,
    table_texture: Path | str | None = None,
    robot_dynamics: bool | str | Path = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_scene(
        include_environment=include_environment,
        background_panorama=background_panorama,
        table_texture=table_texture,
        robot_dynamics=robot_dynamics,
    )
    if paper_target_marker:
        add_paper_target_marker(spec)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, offwidth)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, offheight)
    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    cube.quat = cube_quat_from_pose(source)
    cube.add_freejoint()  # make the cube a real dynamic body
    model = spec.compile()
    return model, mujoco.MjData(model)


def prepare_episode(
    rng: np.random.Generator,
    source: CubePose | None = None,
    target: CubePose | None = None,
    *,
    start_joints: dict[str, float] | None = None,
    start_gripper: float | None = None,
    end_joints: dict[str, float] | None = None,
    end_gripper: float | None = None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    max_attempts: int | None = None,
    verbose: bool = False,
    include_environment: bool = False,
    offwidth: int = 1280,
    offheight: int = 720,
    preflight_debug: bool = False,
    preflight_debug_limit: int = 12,
    failed_trajectory_dir: Path | None = None,
    failed_trajectory_limit: int = 8,
    free_grasp: bool = False,
    target_sampler: Callable[[np.random.Generator], CubePose] | None = None,
) -> Episode:
    """Sample poses and return the first collision-free pick-and-carry.

    ``source``/``target`` pin those cube poses (otherwise they are resampled
    each attempt). ``target_sampler`` can override target sampling when no
    target is pinned. The start and end arm poses are sampled fresh each attempt
    unless their joint/gripper pairs are pinned (e.g. to continue a replay from
    a previous episode's final pose). Pass ``model``/``data`` to reuse a single
    persistent scene (its ``pick_cube`` freejoint is moved to ``source``) instead
    of compiling a fresh model each attempt — required so a live viewer can stay
    bound across episodes. With the cube and start poses all pinned and no
    feasible trajectory, raises immediately. Otherwise keeps resampling until
    success or, if ``max_attempts`` is set, that many attempts fail — then raises
    :class:`EpisodeSamplingError`.

    ``free_grasp`` keeps the low recovery drop timing, but pickup itself uses
    the same canonical full-range grasp as normal episodes.
    """
    fixed_source = source is not None
    fixed_target = target is not None
    fixed_start = start_joints is not None
    fixed_end = end_joints is not None
    reuse_model = model is not None

    if (start_joints is None) != (start_gripper is None):
        raise ValueError("start_joints and start_gripper must be provided together")
    if (end_joints is None) != (end_gripper is None):
        raise ValueError("end_joints and end_gripper must be provided together")
    if fixed_target and target_sampler is not None:
        raise ValueError("target and target_sampler are mutually exclusive")

    if failed_trajectory_dir is not None:
        failed_trajectory_dir.mkdir(parents=True, exist_ok=True)

    source_allowed = is_cube_drop_allowed if free_grasp else is_cube_placement_allowed
    if fixed_source and not source_allowed(source.x, source.y):
        zone = "allowed drop zone" if free_grasp else "allowed pickup zone"
        reason = f"source ({source.x:.4f}, {source.y:.4f}) is outside the {zone}"
        _write_failed_trajectory_note(failed_trajectory_dir, reason, source=source, target=target)
        raise EpisodeSamplingError(reason)
    if fixed_target and not is_cube_drop_allowed(target.x, target.y):
        reason = f"target ({target.x:.4f}, {target.y:.4f}) is outside the allowed drop zone"
        _write_failed_trajectory_note(failed_trajectory_dir, reason, source=source, target=target)
        raise EpisodeSamplingError(reason)

    attempt = 0
    saved_failed_trajectories = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1

        ep_source = source if fixed_source else sample_cube(rng)
        ep_target = target if fixed_target else (target_sampler or sample_target)(rng)
        if fixed_start:
            ep_start_joints, ep_start_gripper = dict(start_joints), float(start_gripper)
        else:
            ep_start_joints, ep_start_gripper = sample_near_neutral(rng)
        if fixed_end:
            ep_end_joints, ep_end_gripper = dict(end_joints), float(end_gripper)
        else:
            ep_end_joints, ep_end_gripper = sample_near_neutral(rng)

        if reuse_model:
            ep_model, ep_data = model, data
            set_cube_pose(ep_model, ep_data, ep_source)
        else:
            ep_model, ep_data = _build_model(
                ep_source,
                include_environment=include_environment,
                offwidth=offwidth,
                offheight=offheight,
            )
        kinematics = derive_kinematics(ep_model)
        actuator_id = {
            mujoco.mj_id2name(ep_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(ep_model.nu)
        }
        for name, value in ep_start_joints.items():
            set_joint(ep_model, ep_data, name, value)
            ep_data.ctrl[actuator_id[name]] = value
            set_actuator_activation(ep_model, ep_data, actuator_id[name], value)
        # Initialise the gripper qpos too, not just its target: otherwise it
        # starts at the model default (closed) and the servo snaps it open on the
        # first step, polluting the viewer and the recorded state with a fake jerk.
        set_joint(ep_model, ep_data, "gripper", ep_start_gripper)
        ep_data.ctrl[actuator_id["gripper"]] = ep_start_gripper
        set_actuator_activation(ep_model, ep_data, actuator_id["gripper"], ep_start_gripper)
        # Clear any residual velocity left in a reused scene from a prior episode.
        ep_data.qvel[:] = 0.0
        mujoco.mj_forward(ep_model, ep_data)

        # Reject start poses that sit on or too near the floor before spending a
        # trajectory search on them — the arm must begin in the air. When the start
        # pose is pinned (the real arm's current pose) it can't be resampled, so we
        # only warn and carry on.
        clearance = jaw_floor_clearance(ep_model, ep_data, jaw_geom_ids(ep_model))
        if clearance < MIN_START_CLEARANCE:
            if fixed_start:
                if verbose:
                    print(f"attempt {attempt}: pinned start jaw clearance {clearance:.3f}m below {MIN_START_CLEARANCE}m, proceeding")
            else:
                if verbose:
                    print(f"attempt {attempt}: start jaw clearance {clearance:.3f}m too low, resampling...")
                continue

        robot_geom_ids, env_geom_ids = build_geom_sets(ep_model)
        carry_ok = make_carry_collision_checker(ep_model, robot_geom_ids, env_geom_ids)

        trajectory = None
        grasp_iter = (
            free_grasp_candidates(kinematics, ep_source)
            if free_grasp
            else grasp_candidates(kinematics, ep_source)
        )
        for candidate_grasp in grasp_iter:
            candidate_trajectories = trajectory_candidates_for_grasp(
                kinematics,
                ep_source,
                ep_target,
                ep_start_joints,
                ep_start_gripper,
                ep_end_joints,
                ep_end_gripper,
                candidate_grasp,
                free_grasp=free_grasp,
                carry_ok=carry_ok,
            )
            for traj in candidate_trajectories:
                grasp = traj.grasp
                collect_preflight_detail = preflight_debug or failed_trajectory_dir is not None
                if collect_preflight_detail:
                    detail_events = _preflight(
                        ep_model,
                        traj,
                        actuator_id,
                        robot_geom_ids,
                        env_geom_ids,
                        detailed=True,
                    )
                    unexpected_detail = [
                        event for event in detail_events if _preflight_collision_is_unexpected(event)
                    ]
                    unexpected = [(event.time, event.geom1, event.geom2) for event in unexpected_detail]
                else:
                    events = _preflight(ep_model, traj, actuator_id, robot_geom_ids, env_geom_ids)
                    unexpected = [(t, n1, n2) for t, n1, n2 in events if is_unexpected(n1, n2)]
                if not unexpected:
                    trajectory = traj
                    if verbose:
                        print(
                            f"source: x={ep_source.x:.4f}  y={ep_source.y:.4f}"
                            f"  yaw={math.degrees(ep_source.yaw):.1f}°"
                            f"  target: x={ep_target.x:.4f}  y={ep_target.y:.4f}"
                        )
                        print(
                            f"grasp: face={grasp.face}  elbow={grasp.elbow}"
                            f"  carry={traj.carry.mode}  (attempt {attempt})"
                        )
                    break
                if preflight_debug:
                    _print_preflight_debug(
                        attempt,
                        traj,
                        unexpected_detail,
                        limit=preflight_debug_limit,
                    )
                if (
                    failed_trajectory_dir is not None
                    and saved_failed_trajectories < failed_trajectory_limit
                ):
                    grasp_label = (
                        "unknown"
                        if traj.grasp is None
                        else f"{traj.grasp.face}_{traj.grasp.elbow}"
                    )
                    path = (
                        failed_trajectory_dir
                        / f"attempt_{attempt:03d}_candidate_{saved_failed_trajectories + 1:03d}_{grasp_label}.npz"
                    )
                    _save_failed_preflight_trajectory(
                        path,
                        ep_model,
                        traj,
                        actuator_id,
                        unexpected_detail,
                    )
                    saved_failed_trajectories += 1
                    if verbose:
                        print(f"saved rejected trajectory: {path}")
                if verbose:
                    seen_pairs: set[tuple[str, str]] = set()
                    for t, n1, n2 in unexpected:
                        key = (min(n1, n2), max(n1, n2))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            print(f"skip {grasp.face}/{grasp.elbow}: collision t={t:.3f}s  {n1} ↔ {n2}")
            if trajectory is not None:
                break

        if trajectory is not None:
            return Episode(
                source=ep_source,
                target=ep_target,
                start_joints=ep_start_joints,
                start_gripper=ep_start_gripper,
                end_joints=ep_end_joints,
                end_gripper=ep_end_gripper,
                model=ep_model,
                data=ep_data,
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
