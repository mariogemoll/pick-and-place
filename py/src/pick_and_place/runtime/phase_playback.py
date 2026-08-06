# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Play one trajectory phase, tick by tick, against the arm and the simulator.

This is the inner loop of the hardware path. Once per control tick it evaluates
the phase at the current trajectory time, writes the resulting joint set points
into the simulator, steps physics forward by exactly one tick's worth of
substeps, clamps the same set points into the real frame and sends them to the
servos, then reads the motors back.

**The simulator is the plant, not a preview.** Trajectory time comes from
``data.time``, so the phase advances at the rate physics does; the sleep at the
bottom of the loop is what keeps that rate near wall time. Scaling by ``speed``
slows every phase uniformly without touching the planner.

The descent is the one phase with feedback, and it is what most of the length
here is: it consumes wrist-camera estimates, re-solves its grasp toward them,
and decides for itself when it has settled — or that it never will, in which
case it backs up to pregrasp and comes in again before giving up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import joints_to_action
from pick_and_place.planning.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
)
from pick_and_place.runtime.descent import follow_estimate
from pick_and_place.runtime.wrist_servo import WristServo, show_frame
from pick_and_place.sim.collisions import unexpected_contact_pairs
from pick_and_place.sim.model import set_cube_pose
from pick_and_place.spec.robot import CONTROL_HZ


@dataclass(frozen=True)
class Plant:
    """What a phase is played against: the simulator, the arm, and the clock.

    Named as one thing because every unit below needs most of it, and threading
    a dozen arguments through the tick loop is what made this code resist being
    split in the first place.

    ``to_real`` maps a phase's sim-frame output to a clamped hardware command,
    ``readback`` reads the motors, and ``on_tick`` receives each
    ``(commanded, actual)`` pair for logging and recording.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    actuator_id: dict[str, int]
    robot_geom_ids: Any
    env_geom_ids: Any
    kinematics: Any
    follower: Any
    viewer: Any
    substeps_per_tick: int
    speed: float
    to_real: Callable[[dict[str, float], float], np.ndarray]
    readback: Callable[[np.ndarray], np.ndarray]
    on_tick: Callable[..., None]


@dataclass(frozen=True)
class WristView:
    """The wrist camera, if there is one, and whether its frames are being shown."""

    servo: WristServo | None
    camera_id: int
    renderer: Any = None
    show: bool = False


@dataclass(frozen=True)
class PhaseResult:
    """Where a phase left things.

    ``phase`` comes back because the descent re-solves its own grasp as the cube
    estimate moves, so what finished is not always what started.
    """

    outcome: str  # "completed" | "restart"
    phase: Any
    tracked_source: CubePose
    commanded: np.ndarray
    contacts: set[tuple[str, str]]


def _report_new_contacts(plant: Plant, previous: set[tuple[str, str]], phase_t: float) -> set:
    """Print unexpected contacts that were not there last tick; return the current set."""
    current = unexpected_contact_pairs(
        plant.model, plant.data, plant.robot_geom_ids, plant.env_geom_ids
    )
    for pair in current - previous:
        print(f"collision phase_t={phase_t:.3f}s  {pair[0]} ↔ {pair[1]}")
    return current


def play_phase(
    plant: Plant,
    wrist: WristView,
    phase,
    *,
    tracked_source: CubePose,
    contacts: set[tuple[str, str]],
    commanded: np.ndarray,
) -> PhaseResult:
    """Run ``phase`` to its end, or until the viewer closes or the descent gives up.

    Returns with ``outcome="restart"`` only from the descent, which is the one
    phase that can decide the episode is not worth continuing. Every other phase
    simply plays out. A closed viewer ends the loop without a verdict — the
    caller checks for that, because it means the whole episode is over, not just
    this phase.
    """
    from pick_and_place.planning.trajectory import DescentPhase

    servo = wrist.servo
    data = plant.data
    control_period = 1.0 / CONTROL_HZ

    is_descent = isinstance(phase, DescentPhase)
    playback_start = data.time
    next_tick = time.monotonic()
    convergence = DescentServoConvergence() if is_descent else None
    retry = DescentServoRetryState() if is_descent else None
    saw_detection = False
    deadline = max(phase.duration, DESCENT_SERVO_MAX_DURATION) if is_descent else phase.duration
    last_estimate_id, last_preview_id = (
        servo.begin_phase(is_descent) if servo is not None else (-1, -1)
    )

    def result(outcome: str) -> PhaseResult:
        """Snapshot the loop's current state. Reads the latest bindings when called."""
        return PhaseResult(outcome, phase, tracked_source, commanded, contacts)

    while plant.viewer.is_running():
        raw_phase_t = (data.time - playback_start) * plant.speed
        phase_t = (
            retry.command_phase_t(raw_phase_t, phase.duration)
            if retry is not None
            else raw_phase_t
        )

        preview_bgr = None
        if servo is not None and (wrist.show or is_descent):
            if is_descent:
                estimate, preview = servo.sample(
                    data.cam_xpos[wrist.camera_id].copy(),
                    data.cam_xmat[wrist.camera_id].reshape(3, 3).copy(),
                )
                if estimate is not None and estimate.frame_id != last_estimate_id:
                    last_estimate_id = estimate.frame_id
                    correction = follow_estimate(
                        phase, tracked_source, estimate.source, plant.kinematics
                    )
                    phase = correction.phase
                    tracked_source = correction.tracked
                    saw_detection = True
                    if convergence is not None:
                        convergence.observe(tracked_source)
                    set_cube_pose(plant.model, data, correction.measured)
                if wrist.show and preview is not None and preview.frame_id != last_preview_id:
                    last_preview_id = preview.frame_id
                    preview_bgr = preview.bgr.copy()
            elif wrist.show:
                snapshot = servo.reader.latest()
                if snapshot is not None:
                    preview_bgr = snapshot.bgr.copy()

            if preview_bgr is not None and wrist.show:
                show_frame(
                    preview_bgr,
                    renderer=wrist.renderer,
                    model=plant.model,
                    data=data,
                    undistort_map=servo.undistort_map,
                    rectify=not is_descent,
                )

        frame = phase.evaluate(phase_t)
        for name, value in frame.joints.items():
            data.ctrl[plant.actuator_id[name]] = value
        data.ctrl[plant.actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(plant.model, data, nstep=plant.substeps_per_tick)
        contacts = _report_new_contacts(plant, contacts, phase_t)

        commanded = plant.to_real(frame.joints, frame.gripper)
        plant.follower.send_action(joints_to_action(commanded))
        plant.on_tick(
            commanded,
            plant.readback(commanded),
            servo_active=is_descent,
            servo_source=(
                np.array(
                    [tracked_source.x, tracked_source.y, tracked_source.z, tracked_source.yaw]
                )
                if is_descent
                else None
            ),
        )
        plant.viewer.sync()

        if is_descent:
            assert retry is not None and convergence is not None
            # The jaws can hide every tag on the way down. Back off to pregrasp
            # and come in again rather than close blind on a cube not being seen.
            if (
                servo is not None
                and not saw_detection
                and not retry.is_backing_up()
                and raw_phase_t >= phase.duration
                and retry.can_retry()
            ):
                retry.start_backup(raw_phase_t)
                print(
                    "warning: descent saw no cube tags; backing up to pregrasp and "
                    f"retrying ({retry.retries_started}/{retry.max_retries})"
                )
            if retry.is_backing_up():
                if retry.backup_complete(raw_phase_t):
                    retry.finish_backup()
                    convergence = DescentServoConvergence()
                    saw_detection = False
                    playback_start = data.time
                    next_tick = time.monotonic()
                continue
            if phase_t >= deadline:
                if saw_detection:
                    print(
                        f"warning: descent visual servo hit {deadline:.1f}s cap before "
                        f"settling ({convergence.stable_frames}/"
                        f"{DESCENT_SERVO_STABLE_FRAMES} stable frames)"
                    )
                    return result("restart")
                elif servo is not None:
                    print(
                        f"warning: descent visual servo hit {deadline:.1f}s cap without "
                        "a cube detection"
                    )
                    return result("restart")
                break
            if servo is None:
                if phase_t >= phase.duration:
                    break
            elif phase_t >= phase.duration and convergence.is_stable():
                break
        elif phase_t >= phase.duration:
            break

        next_tick += control_period
        remaining = next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -control_period:
            # Do not issue a burst of catch-up commands after a long vision, I/O,
            # or scheduling stall.
            next_tick = time.monotonic()

    return result("completed")
