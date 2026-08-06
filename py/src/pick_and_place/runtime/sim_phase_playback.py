# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Play one trajectory phase in the simulator, recording every control tick.

The sim twin of :mod:`pick_and_place.runtime.phase_playback`, and shaped like it
on purpose — evaluate the phase, command the actuators, step physics by exactly
one tick's worth of substeps — with the two differences that matter:

**A tick is captured before it is commanded.** The row for time t holds the state
and images as they are *before* the set point is written, which is the ordering a
real recording has, where the motors are read before they have tracked the new
command. Everything else in the loop follows from putting the capture first.

**There is no arm.** The position-servo actuators are the plant; what a real run
would send to the servos is instead added to ``data.ctrl``, with the drawn
joint-zero offsets folded in so physics runs the true joints while the plan and
the recorded rows stay in the believed frame.

The descent is again the one phase with feedback: it consumes a wrist-camera
sighting each tick, re-solves its grasp toward it, and decides for itself when it
has settled — or backs up to pregrasp and comes in again, or gives up and asks
for the episode to be restarted rather than close the jaws blind.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable

import mujoco

from pick_and_place.core.geometry import CubePose
from pick_and_place.planning.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
)
from pick_and_place.runtime.believed_frame import BelievedFrame
from pick_and_place.runtime.descent import follow_estimate
from pick_and_place.runtime.sim_tick_recorder import SimTickRecorder
from pick_and_place.runtime.sim_wrist_servo import CubeSighting, SimWristServo
from pick_and_place.runtime.wrist_mixed_view import blend_mixed, show_mixed
from pick_and_place.sim.collisions import unexpected_contact_pairs
from pick_and_place.spec.robot import CONTROL_HZ

#: What a tick sees when nothing looked: no tags, no pose, no solve.
NOTHING_SEEN = CubeSighting(detections=[], pose=None, estimate=None)


@dataclasses.dataclass(frozen=True)
class SimPlant:
    """What a phase is played against: the simulator, the clock, and the window.

    Named as one thing because every unit below needs most of it. ``speed``
    scales trajectory time against simulated time; ``realtime`` sleeps out the
    rest of each tick so a viewer runs at the control rate, which a recording
    does not want.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    actuator_id: dict[str, int]
    robot_geom_ids: Any
    env_geom_ids: Any
    kinematics: Any
    substeps_per_tick: int
    speed: float = 1.0
    realtime: bool = False
    verbose: bool = True
    viewer: Any = None
    should_stop: Callable[[], bool] | None = None


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
    belief: BelievedFrame,
    servo: SimWristServo | None = None,
    recorder: SimTickRecorder | None = None,
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

    data = plant.data
    control_period = 1.0 / CONTROL_HZ

    is_descent = servo is not None and isinstance(phase, DescentPhase)
    playback_start = data.time
    convergence = DescentServoConvergence() if is_descent else None
    retry = DescentServoRetryState() if is_descent else None
    saw_detection = False
    deadline = max(phase.duration, DESCENT_SERVO_MAX_DURATION) if is_descent else phase.duration

    def result(outcome: str) -> SimPhaseResult:
        """Snapshot the loop's current state. Reads the latest bindings when called."""
        return SimPhaseResult(outcome, phase, tracked_source, contacts)

    while True:
        if plant.should_stop is not None and plant.should_stop():
            return result("stopped")
        tick_start = time.monotonic()
        raw_phase_t = (data.time - playback_start) * plant.speed
        phase_t = (
            retry.command_phase_t(raw_phase_t, phase.duration)
            if retry is not None
            else raw_phase_t
        )

        sighting = NOTHING_SEEN
        true_rgb = camera_position = camera_rotation = None
        if is_descent or show_wrist_mixed:
            true_rgb, camera_position, camera_rotation = servo.look(tracked_source)
        if is_descent:
            sighting = servo.solve(true_rgb, camera_position, camera_rotation)
            if sighting.pose is not None:
                correction = follow_estimate(
                    phase, tracked_source, sighting.pose, plant.kinematics
                )
                phase = correction.phase
                tracked_source = correction.tracked
                saw_detection = True
                convergence.observe(tracked_source)

        if show_wrist_mixed:
            # Rendered after the servo update, so the underlay shows the believed
            # cube as this tick's estimate left it.
            show_mixed(
                blend_mixed(
                    true_rgb,
                    servo.render_believed(tracked_source),
                    sighting,
                    servo.camera_matrix,
                    camera_position,
                    camera_rotation,
                )
            )

        frame = phase.evaluate(min(phase_t, phase.duration))
        if recorder is not None:
            recorder.record(frame, phase.name, belief.offsets_deg())

        if is_descent:
            if retry.is_backing_up():
                if retry.backup_complete(raw_phase_t):
                    retry.finish_backup()
                    convergence = DescentServoConvergence()
                    saw_detection = False
                    playback_start = data.time
            elif not saw_detection and raw_phase_t >= phase.duration and retry.can_retry():
                # The jaws can hide every tag on the way down. Back off to
                # pregrasp and come in again rather than close blind.
                retry.start_backup(raw_phase_t)
                if plant.verbose:
                    print(
                        "warning: descent saw no cube tags; backing up to "
                        "pregrasp and retrying "
                        f"({retry.retries_started}/{retry.max_retries})"
                    )
            elif phase_t >= deadline:
                if plant.verbose:
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

        for name, value in frame.joints.items():
            data.ctrl[plant.actuator_id[name]] = value
        for name, offset in belief.offsets_rad().items():
            if name in frame.joints:
                data.ctrl[plant.actuator_id[name]] += offset
        data.ctrl[plant.actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(plant.model, data, nstep=plant.substeps_per_tick)

        current = unexpected_contact_pairs(
            plant.model, data, plant.robot_geom_ids, plant.env_geom_ids
        )
        if plant.verbose:
            for pair in current - contacts:
                print(f"collision t={raw_phase_t:.3f}s  {pair[0]} ↔ {pair[1]}")
        contacts = current

        if plant.viewer is not None:
            plant.viewer.sync()

        if plant.realtime:
            remaining = control_period - (time.monotonic() - tick_start)
            if remaining > 0:
                time.sleep(remaining)

    return result("completed")
