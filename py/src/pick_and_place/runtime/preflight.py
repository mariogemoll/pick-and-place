# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Vet a planned trajectory by running it under live physics.

The cheap kinematic screen in :mod:`pick_and_place.sim.collisions` rejects
configurations that are already in collision; this is the authoritative check
that a whole trajectory runs clean. It steps a shadow ``MjData`` from the
trajectory's own start pose through to its end, so the approach swing and the
retreat are vetted too, and reports every unexpected contact along the way.

A rejected candidate can be saved as a replayable rollout, which is how a
preflight failure gets looked at rather than merely counted.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.robot_dynamics import set_actuator_activation
from pick_and_place.scripted.trajectory import Trajectory
from pick_and_place.sim.collisions import is_unexpected
from pick_and_place.sim.model import set_cube_pose, set_joint


@dataclass(frozen=True)
class PreflightDebug:
    """What to report about the trajectory candidates preflight rejects.

    Silent by default. Both kinds of report need the same per-contact detail,
    which costs a second, slower physics pass over every candidate, so a run that
    plans thousands of episodes must not pay for it unasked — :attr:`detailed`
    is what decides.
    """

    print_contacts: bool = False
    contact_limit: int = 12
    trajectory_dir: Path | None = None
    trajectory_limit: int = 8

    @property
    def detailed(self) -> bool:
        """Whether anything here needs the per-contact detail."""
        return self.print_contacts or self.trajectory_dir is not None


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


def preflight(
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


def preflight_collision_is_unexpected(event: PreflightCollision) -> bool:
    return is_unexpected(event.geom1, event.geom2)


def print_preflight_debug(
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


def save_failed_preflight_trajectory(
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


def write_failed_trajectory_note(
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
