# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Sample a random pick-and-carry episode that runs clean.

Shared by ``pick_and_place/sim.py`` (sim-only viewer), ``pick_and_place/real.py``
(the hardware path) and ``record_episodes`` (batch dataset generation). All need
the same thing: poses drawn from
:mod:`pick_and_place.scripted.episode_sampling`, a trajectory planned for them,
and :mod:`pick_and_place.runtime.preflight` run over each candidate until one
comes back without an unexpected contact — resampling when none does.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.core.grasp_perturbation import GraspPerturbation
from pick_and_place.core.miscalibration import MiscalibrationDraw
from pick_and_place.core.robot_dynamics import set_actuator_activation
from pick_and_place.core.workspace_bounds import is_cube_drop_allowed, is_cube_pickup_allowed
from pick_and_place.scripted.episode_sampling import sample_cube, sample_near_neutral, sample_target
from pick_and_place.scripted.grasp import GraspChoice, free_grasp_candidates, grasp_candidates
from pick_and_place.scripted.trajectory import Trajectory, trajectory_candidates_for_grasp
from pick_and_place.runtime.preflight import (
    PreflightDebug,
    preflight,
    preflight_collision_is_unexpected,
    print_preflight_debug,
    save_failed_preflight_trajectory,
    write_failed_trajectory_note,
)
from pick_and_place.sim.collisions import (
    build_geom_sets,
    is_unexpected,
    jaw_floor_clearance,
    jaw_geom_ids,
    make_carry_collision_checker,
)
from pick_and_place.sim.derive_kinematics import derive_kinematics
from pick_and_place.sim.model import build_model, set_cube_pose, set_joint


# Minimum height (m) the lowest gripper-jaw corner must clear the floor by for a
# sampled start pose to be accepted, so the arm begins well up in the air rather
# than skimming (or buried in) the ground.
MIN_START_CLEARANCE = 0.10


class EpisodeSamplingError(RuntimeError):
    """Raised when no collision-free trajectory is found within the attempt budget."""


@dataclass
class Episode:
    """A prepared, collision-free pick-and-carry ready to run under physics.

    ``model``/``data`` are compiled with the cube as a dynamic free body and the
    arm initialised at the start pose (both ``qpos`` and ``ctrl`` set). The
    ``trajectory`` already carries the sampled start/end poses.

    ``source``/``target`` are the *true* cube poses physics and scoring use.
    ``believed_source``/``believed_target`` are what the planner acted on; they
    differ from the true poses only when a ``miscalibration`` draw was injected
    (the ``trajectory`` is planned entirely in the believed frame).

    ``grasp_perturbation``, when set, is a *deliberate* extra error folded into
    ``believed_source`` only, so the first grasp misses and the recovery can be
    recorded. Kept separate from ``miscalibration`` because that field means
    "error the real rig exhibits" and is read as such downstream.
    """

    source: CubePose
    target: CubePose
    believed_source: CubePose
    believed_target: CubePose
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
    miscalibration: MiscalibrationDraw | None = None
    grasp_perturbation: GraspPerturbation | None = None

    @property
    def grasp(self) -> GraspChoice:
        return self.trajectory.grasp


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
    debug: PreflightDebug = PreflightDebug(),
    free_grasp: bool = False,
    target_sampler: Callable[[np.random.Generator], CubePose] | None = None,
    miscalibration: MiscalibrationDraw | None = None,
    grasp_perturbation: GraspPerturbation | None = None,
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

    ``miscalibration`` splits the episode into true and believed state: the
    trajectory is planned and preflighted against the *believed* cube/target
    (true pose plus the draw's belief error) — everything the real planner
    would know — while the returned ``data`` keeps the cube at the true pose
    and holds the arm at the physically-true start joints (planned start plus
    the draw's joint-zero offsets), so running the plan open-loop misses the
    way real reaching misses.
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

    if debug.trajectory_dir is not None:
        debug.trajectory_dir.mkdir(parents=True, exist_ok=True)

    source_allowed = is_cube_drop_allowed if free_grasp else is_cube_pickup_allowed
    if fixed_source and not source_allowed(source.x, source.y):
        zone = "allowed drop zone" if free_grasp else "allowed pickup zone"
        reason = f"source ({source.x:.4f}, {source.y:.4f}) is outside the {zone}"
        write_failed_trajectory_note(debug.trajectory_dir, reason, source=source, target=target)
        raise EpisodeSamplingError(reason)
    if fixed_target and not is_cube_drop_allowed(target.x, target.y):
        reason = f"target ({target.x:.4f}, {target.y:.4f}) is outside the allowed drop zone"
        write_failed_trajectory_note(debug.trajectory_dir, reason, source=source, target=target)
        raise EpisodeSamplingError(reason)

    attempt = 0
    saved_failed_trajectories = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1

        ep_source = source if fixed_source else sample_cube(rng)
        ep_target = target if fixed_target else (target_sampler or sample_target)(rng)
        if miscalibration is not None:
            believed_source = miscalibration.believe_cube(ep_source)
            believed_target = miscalibration.believe_target(ep_target)
        else:
            believed_source, believed_target = ep_source, ep_target
        # Layered on top of the belief rather than folded into it: the draw above
        # models error the real rig exhibits, this is a deliberate fumble, and
        # keeping them separate is what lets a dataset be split by perturbation
        # type afterwards. Only the *source* is perturbed -- the fumble under
        # study is the grasp, and a wrong drop target would fail the placement
        # instead, which is a different episode.
        if grasp_perturbation is not None:
            believed_source = grasp_perturbation.apply(believed_source)
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
            ep_model, ep_data = build_model(
                ep_source,
                include_environment=include_environment,
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
            free_grasp_candidates(kinematics, believed_source)
            if free_grasp
            else grasp_candidates(kinematics, believed_source)
        )
        for candidate_grasp in grasp_iter:
            candidate_trajectories = trajectory_candidates_for_grasp(
                kinematics,
                believed_source,
                believed_target,
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
                if debug.detailed:
                    detail_events = preflight(
                        ep_model,
                        traj,
                        actuator_id,
                        robot_geom_ids,
                        env_geom_ids,
                        detailed=True,
                    )
                    unexpected_detail = [
                        event for event in detail_events if preflight_collision_is_unexpected(event)
                    ]
                    unexpected = [(event.time, event.geom1, event.geom2) for event in unexpected_detail]
                else:
                    events = preflight(ep_model, traj, actuator_id, robot_geom_ids, env_geom_ids)
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
                if debug.print_contacts:
                    print_preflight_debug(
                        attempt,
                        traj,
                        unexpected_detail,
                        limit=debug.contact_limit,
                    )
                if (
                    debug.trajectory_dir is not None
                    and saved_failed_trajectories < debug.trajectory_limit
                ):
                    grasp_label = (
                        "unknown"
                        if traj.grasp is None
                        else f"{traj.grasp.face}_{traj.grasp.elbow}"
                    )
                    path = (
                        debug.trajectory_dir
                        / f"attempt_{attempt:03d}_candidate_{saved_failed_trajectories + 1:03d}_{grasp_label}.npz"
                    )
                    save_failed_preflight_trajectory(
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
            if miscalibration is not None:
                # The plan is vetted; now put physics into the *true* start
                # state: the arm physically sits at the planned start plus the
                # drawn joint-zero offsets (a real servo commanded ``theta``
                # rests at model angle ``theta + offset``). The cube already
                # sits at the true pose — only the plan used the believed one.
                offsets = miscalibration.offsets_rad(0.0)
                for name, value in ep_start_joints.items():
                    true_value = value + offsets.get(name, 0.0)
                    set_joint(ep_model, ep_data, name, true_value)
                    ep_data.ctrl[actuator_id[name]] = true_value
                    set_actuator_activation(
                        ep_model, ep_data, actuator_id[name], true_value
                    )
                ep_data.qvel[:] = 0.0
                mujoco.mj_forward(ep_model, ep_data)
            return Episode(
                source=ep_source,
                target=ep_target,
                believed_source=believed_source,
                believed_target=believed_target,
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
                miscalibration=miscalibration,
                grasp_perturbation=grasp_perturbation,
            )

        if fixed_source and fixed_target:
            raise EpisodeSamplingError(
                "no collision-free pick-and-carry found for this source/target"
            )
        if verbose:
            print(f"attempt {attempt}: no trajectory found, resampling...")

    raise EpisodeSamplingError(f"no collision-free trajectory within {max_attempts} attempts")
