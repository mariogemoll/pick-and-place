# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Play one trajectory phase against the arm, tick by tick.

This is the inner loop of the hardware path. Once per control tick it evaluates
the phase at the current trajectory time and hands the resulting joint set points
to the plant, which writes them into the believed shadow, steps physics by
exactly one tick's worth of substeps, sends the clamped real-frame command to the
servos and paces the loop to the control rate.

**The simulator is the plant, not a preview.** Trajectory time comes from the
plant's clock, so the phase advances at the rate physics does. Scaling by
``speed`` slows every phase uniformly without touching the planner.

The descent is the one phase with feedback, and it is what most of the length
here is: it consumes wrist-camera estimates, re-solves its grasp toward them,
and decides for itself when it has settled — or that it never will, in which
case it backs up to pregrasp and comes in again before giving up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.plant.real import RealPlant
from pick_and_place.planning.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
)
from pick_and_place.runtime.descent import follow_estimate
from pick_and_place.runtime.wrist_servo import show_frame


@dataclass(frozen=True)
class RealRun:
    """What surrounds the plant for one hardware run: the window and the logging.

    ``on_tick`` receives each ``(commanded, actual)`` pair for the tracking log
    and the dataset. None of it is part of commanding or observing, which is why
    it sits here rather than on the plant.
    """

    viewer: Any
    on_tick: Callable[..., None]


@dataclass(frozen=True)
class WristView:
    """The wrist camera preview, if one is being shown."""

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


def play_phase(
    plant: RealPlant,
    run: RealRun,
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

    servo = plant.servo

    is_descent = isinstance(phase, DescentPhase)
    playback_start = plant.time
    convergence = DescentServoConvergence() if is_descent else None
    retry = DescentServoRetryState() if is_descent else None
    saw_detection = False
    deadline = max(phase.duration, DESCENT_SERVO_MAX_DURATION) if is_descent else phase.duration
    last_preview_id = plant.begin_phase(is_descent)

    def result(outcome: str) -> PhaseResult:
        """Snapshot the loop's current state. Reads the latest bindings when called."""
        return PhaseResult(outcome, phase, tracked_source, commanded, contacts)

    while run.viewer.is_running():
        raw_phase_t = (plant.time - playback_start) * plant.speed
        phase_t = (
            retry.command_phase_t(raw_phase_t, phase.duration)
            if retry is not None
            else raw_phase_t
        )

        if is_descent:
            sighting = plant.sighting(tracked_source)
            if sighting.usable:
                correction = follow_estimate(
                    phase, tracked_source, sighting.pose, plant.kinematics
                )
                phase = correction.phase
                tracked_source = correction.tracked
                saw_detection = True
                if convergence is not None:
                    convergence.observe(tracked_source)
                plant.set_believed_cube(correction.measured)

        if wrist.show and servo is not None:
            last_preview_id = _show_preview(plant, wrist, is_descent, last_preview_id)

        frame = phase.evaluate(phase_t)
        commanded = plant.step(frame.joints, frame.gripper)
        current = plant.new_contacts()
        for pair in current - contacts:
            print(f"collision phase_t={phase_t:.3f}s  {pair[0]} ↔ {pair[1]}")
        contacts = current

        run.on_tick(
            commanded,
            plant.readback(),
            servo_active=is_descent,
            servo_source=(
                np.array(
                    [tracked_source.x, tracked_source.y, tracked_source.z, tracked_source.yaw]
                )
                if is_descent
                else None
            ),
        )
        run.viewer.sync()

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
                    playback_start = plant.time
                    plant.resync_clock()
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

    return result("completed")


def _show_preview(
    plant: RealPlant, wrist: WristView, is_descent: bool, last_preview_id: int
) -> int:
    """Put the newest camera frame on screen; return the preview id now shown.

    Nothing here feeds control, so it is deliberately best-effort: during the
    descent the window waits for a frame the detector has actually annotated,
    and outside it takes whatever the camera thread last read.
    """
    servo = plant.servo
    if is_descent:
        preview = plant.last_preview
        if preview is None or preview.frame_id == last_preview_id:
            return last_preview_id
        bgr, shown = preview.bgr, preview.frame_id
    else:
        snapshot = servo.reader.latest()
        if snapshot is None:
            return last_preview_id
        bgr, shown = snapshot.bgr, last_preview_id
    show_frame(
        bgr.copy(),
        renderer=wrist.renderer,
        model=plant.model,
        data=plant.data,
        undistort_map=servo.undistort_map,
        rectify=not is_descent,
    )
    return shown
