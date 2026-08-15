# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Play one trajectory phase, tick by tick, against whatever world is behind the plant.

There used to be two of these — one for the rig, one for the simulator — and they
were the same loop written twice. What actually differs between them is where the
image comes from, what receives the commands, whether the detector runs on a
thread or inline, and what drives the clock, and all four now live behind
:mod:`pick_and_place.plant`. What is left is the loop, and it is the same loop.

**A tick is observed before it is commanded.** The row for time t holds the state
as it was *before* the set point went out, which is the dataset's central
invariant: a policy is trained on what it will actually have when it has to
decide. It is also why a phase's last tick is recorded but never commanded — the
phase ends between the two.

The descent is the one phase with feedback, and it is most of the length here: it
consumes a wrist-camera sighting each tick, re-solves its grasp toward it, and
decides for itself when it has settled — or backs up to pregrasp and comes in
again, or gives up and asks for the episode to be abandoned rather than close the
jaws blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pick_and_place.core.geometry import CubePose
from pick_and_place.data.trajectory_artifact import TrajectoryWriter
from pick_and_place.plant.interface import NOTHING_SEEN, Observation, Plant, Sighting
from pick_and_place.scripted.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
)
from pick_and_place.scripted.descent import follow_estimate


def _noop(*args: Any, **kwargs: Any) -> None:
    pass


@dataclass(frozen=True)
class Tick:
    """One control tick, as it happened: what was seen, and what was sent."""

    observation: Observation
    action: Any
    phase_name: str
    is_descent: bool
    sighting: Sighting
    tracked_source: CubePose


@dataclass(frozen=True)
class Run:
    """What surrounds the plant for one run: the window, the recording, the operator.

    None of it is part of commanding or observing, which is why it sits here
    rather than on the plant. Every hook has a default, so a bare ``Run()`` plays
    an episode and reports nothing.

    ``on_tick`` receives each :class:`Tick` and is where a dataset row is
    written. ``on_look`` receives the same tick's sighting for a preview window.
    ``should_stop`` is the operator: a closed viewer, or a skip.
    """

    on_tick: Callable[..., None] = _noop
    on_look: Callable[..., None] = _noop
    sync: Callable[[], None] = _noop
    should_stop: Callable[[], bool] = lambda: False
    verbose: bool = True


@dataclass(frozen=True)
class PhaseResult:
    """Where a phase left things.

    ``phase`` comes back because the descent re-solves its own grasp as the cube
    estimate moves, so what finished is not always what started.
    """

    outcome: str  # "completed" | "restart" | "stopped"
    phase: Any
    tracked_source: CubePose
    action: Any = None
    contacts: set[tuple[str, str]] = field(default_factory=set)


def play_phase(
    plant: Plant,
    phase,
    *,
    run: Run,
    artifact: TrajectoryWriter,
    tracked_source: CubePose,
    contacts: set[tuple[str, str]] = frozenset(),
    action: Any = None,
    watch: bool = False,
) -> PhaseResult:
    """Run ``phase`` to its end, or until the operator stops it or the descent gives up.

    ``outcome="restart"`` comes only from the descent, which is the one phase that
    can decide the episode is not worth keeping. ``"stopped"`` is the operator's.

    ``watch`` asks for a sighting on every tick rather than only during the
    descent, which a preview window wants and control does not.
    """
    from pick_and_place.scripted.trajectory import DescentPhase

    # Without a wrist camera the descent is not a servo at all: it plays out
    # open loop, which is what a vetted feedforward plan wants.
    is_descent = plant.has_wrist_camera and isinstance(phase, DescentPhase)
    playback_start = plant.time
    convergence = DescentServoConvergence() if is_descent else None
    retry = DescentServoRetryState() if is_descent else None
    saw_detection = False
    deadline = max(phase.duration, DESCENT_SERVO_MAX_DURATION) if is_descent else phase.duration
    plant.begin_phase(is_descent)

    def result(outcome: str) -> PhaseResult:
        """Snapshot the loop's current state. Reads the latest bindings when called."""
        return PhaseResult(outcome, phase, tracked_source, action, contacts)

    while True:
        if run.should_stop():
            return result("stopped")
        raw_phase_t = (plant.time - playback_start) * plant.speed
        phase_t = (
            retry.command_phase_t(raw_phase_t, phase.duration)
            if retry is not None
            else raw_phase_t
        )

        sighting: Sighting = NOTHING_SEEN
        if is_descent or watch:
            sighting = plant.sighting(tracked_source)
        if is_descent and sighting.usable:
            correction = follow_estimate(phase, tracked_source, sighting.pose, plant.kinematics)
            phase = correction.phase
            tracked_source = correction.tracked
            saw_detection = True
            convergence.observe(tracked_source)
            plant.set_believed_cube(correction.measured)
        # After the servo update, so a preview shows the belief as this tick left it.
        run.on_look(plant, sighting, tracked_source, is_descent)

        frame = phase.evaluate(min(phase_t, phase.duration))
        observation: Observation = plant.observe()
        action = plant.to_real(frame.joints, frame.gripper)
        artifact.record(
            phase_name=phase.name,
            true_state=observation.true_state,
            believed_state=observation.state,
            action=action,
            true_cube_pose=observation.true_cube_pose,
            believed_cube_pose=tracked_source,
            wrist_sighting=sighting.pose,
        )
        run.on_tick(
            Tick(
                observation=observation,
                action=action,
                phase_name=phase.name,
                is_descent=is_descent,
                sighting=sighting,
                tracked_source=tracked_source,
            )
        )

        if is_descent:
            if retry.is_backing_up():
                if retry.backup_complete(raw_phase_t):
                    retry.finish_backup()
                    convergence = DescentServoConvergence()
                    saw_detection = False
                    playback_start = plant.time
                    plant.resync_clock()
            elif not saw_detection and raw_phase_t >= phase.duration and retry.can_retry():
                # The jaws can hide every tag on the way down. Back off to
                # pregrasp and come in again rather than close blind.
                retry.start_backup(raw_phase_t)
                if run.verbose:
                    print(
                        "warning: descent saw no cube tags; backing up to pregrasp and "
                        f"retrying ({retry.retries_started}/{retry.max_retries})"
                    )
            elif phase_t >= deadline:
                if run.verbose:
                    if saw_detection:
                        print(
                            f"warning: descent visual servo hit {deadline:.1f}s cap before "
                            f"settling ({convergence.stable_frames}/"
                            f"{DESCENT_SERVO_STABLE_FRAMES} stable frames)"
                        )
                    else:
                        print(
                            f"warning: descent visual servo hit {deadline:.1f}s cap without "
                            "a cube detection"
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
        run.sync()

    return result("completed")
