# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Play one trajectory phase in the simulator, recording every control tick.

The sim twin of :mod:`pick_and_place.runtime.phase_playback`, and shaped like it
on purpose. Both now command and observe through
:class:`~pick_and_place.plant.interface.Plant`, so what is left here is the part
that is genuinely about *recording* rather than about which world is being
driven:

**A tick is captured before it is commanded.** The row for time t holds the state
and images as they are *before* the set point is written, which is the ordering a
real recording has, where the motors are read before they have tracked the new
command. Everything else in the loop follows from putting the capture first.

Every tick goes into the trajectory artifact, which holds the true world and the
believed one side by side and no pixels at all; a dataset row, with its two
rendered cameras, is written on top of that only when something is recording.

The descent is again the one phase with feedback: it consumes a wrist-camera
sighting each tick, re-solves its grasp toward it, and decides for itself when it
has settled — or backs up to pregrasp and comes in again, or gives up and asks
for the episode to be restarted rather than close the jaws blind.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.data.trajectory_artifact import TrajectoryWriter
from pick_and_place.plant.sim import SimPlant
from pick_and_place.planning.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
)
from pick_and_place.runtime.believed_frame import BelievedFrame
from pick_and_place.runtime.descent import follow_estimate
from pick_and_place.runtime.sim_tick_recorder import SimTickRecorder
from pick_and_place.runtime.wrist_mixed_view import blend_mixed, show_mixed
from pick_and_place.sim.model import get_cube_qpos


@dataclasses.dataclass(frozen=True)
class SimRun:
    """What surrounds the plant for one recorded run: the window and the operator.

    Kept apart from the plant because none of it is part of commanding or
    observing — a viewer that is closed or an operator skipping ahead ends the
    run without saying anything about the world it was driving.
    """

    viewer: Any = None
    should_stop: Callable[[], bool] | None = None
    verbose: bool = True


@dataclasses.dataclass(frozen=True)
class SimPhaseResult:
    """Where a phase left things.

    ``phase`` comes back because the descent re-solves its own grasp as the cube
    estimate moves, so what finished is not always what started.
    """

    outcome: str  # "completed" | "restart" | "stopped"
    phase: Any
    tracked_source: CubePose
    contacts: set[tuple[str, str]]


def play_phase(
    plant: SimPlant,
    phase,
    *,
    run: SimRun,
    belief: BelievedFrame,
    recorder: SimTickRecorder | None = None,
    artifact: TrajectoryWriter,
    tracked_source: CubePose,
    contacts: set[tuple[str, str]] = frozenset(),
    show_wrist_mixed: bool = False,
) -> SimPhaseResult:
    """Run ``phase`` to its end, or until the caller stops it or the descent gives up.

    ``outcome="restart"`` comes only from the descent, which is the one phase that
    can decide the episode is not worth recording. ``"stopped"`` is the caller's
    own ``should_stop`` — a closed viewer, or the operator skipping ahead.
    """
    from pick_and_place.planning.trajectory import DescentPhase

    is_descent = plant.servo is not None and isinstance(phase, DescentPhase)
    playback_start = plant.time
    convergence = DescentServoConvergence() if is_descent else None
    retry = DescentServoRetryState() if is_descent else None
    saw_detection = False
    deadline = max(phase.duration, DESCENT_SERVO_MAX_DURATION) if is_descent else phase.duration

    def result(outcome: str) -> SimPhaseResult:
        """Snapshot the loop's current state. Reads the latest bindings when called."""
        return SimPhaseResult(outcome, phase, tracked_source, contacts)

    while True:
        if run.should_stop is not None and run.should_stop():
            return result("stopped")
        raw_phase_t = (plant.time - playback_start) * plant.speed
        phase_t = (
            retry.command_phase_t(raw_phase_t, phase.duration)
            if retry is not None
            else raw_phase_t
        )

        sighting = None
        if is_descent or show_wrist_mixed:
            sighting = plant.sighting(tracked_source)
        if is_descent and sighting.usable:
            correction = follow_estimate(phase, tracked_source, sighting.pose, plant.kinematics)
            phase = correction.phase
            tracked_source = correction.tracked
            saw_detection = True
            convergence.observe(tracked_source)

        if show_wrist_mixed:
            # Rendered after the servo update, so the underlay shows the believed
            # cube as this tick's estimate left it.
            true_rgb, camera_position, camera_rotation = plant.last_look
            show_mixed(
                blend_mixed(
                    true_rgb,
                    plant.servo.render_believed(tracked_source),
                    sighting,
                    plant.servo.camera_matrix,
                    camera_position,
                    camera_rotation,
                )
            )

        frame = phase.evaluate(min(phase_t, phase.duration))
        # Read the tick's ground truth once, in both frames, and hand the same
        # values to the artifact and to the dataset row. Two readers would drift:
        # the pan jitter advances with the clock, so a second read is a second
        # draw of the offsets it separates the frames by.
        true_state, believed_state = belief.state_pair()
        true_cube_pose = get_cube_qpos(plant.model, plant.data)
        action = sim_frame_to_real(frame.joints, frame.gripper)
        artifact.record(
            phase_name=phase.name,
            true_state=true_state,
            believed_state=believed_state,
            action=action,
            true_cube_pose=true_cube_pose,
            believed_cube_pose=tracked_source,
            wrist_sighting=None if sighting is None else sighting.pose,
        )
        if recorder is not None:
            recorder.record(
                believed_state=believed_state,
                action=action,
                true_cube_pose=true_cube_pose,
            )

        if is_descent:
            if retry.is_backing_up():
                if retry.backup_complete(raw_phase_t):
                    retry.finish_backup()
                    convergence = DescentServoConvergence()
                    saw_detection = False
                    playback_start = plant.time
            elif not saw_detection and raw_phase_t >= phase.duration and retry.can_retry():
                # The jaws can hide every tag on the way down. Back off to
                # pregrasp and come in again rather than close blind.
                retry.start_backup(raw_phase_t)
                if run.verbose:
                    print(
                        "warning: descent saw no cube tags; backing up to "
                        "pregrasp and retrying "
                        f"({retry.retries_started}/{retry.max_retries})"
                    )
            elif phase_t >= deadline:
                if run.verbose:
                    if saw_detection:
                        print(
                            "warning: descent visual servo hit "
                            f"{deadline:.1f}s cap before settling "
                            f"({convergence.stable_frames}/"
                            f"{DESCENT_SERVO_STABLE_FRAMES} stable frames)"
                        )
                    else:
                        print(
                            "warning: descent visual servo hit "
                            f"{deadline:.1f}s cap without a cube detection"
                        )
                return result("restart")
            elif phase_t >= phase.duration and convergence.is_stable():
                break
        elif phase_t >= phase.duration:
            break

        plant.step(frame.joints, frame.gripper)

        current = plant.new_contacts()
        if run.verbose:
            for pair in current - contacts:
                print(f"collision t={raw_phase_t:.3f}s  {pair[0]} ↔ {pair[1]}")
        contacts = current

        if run.viewer is not None:
            run.viewer.sync()

    return result("completed")
