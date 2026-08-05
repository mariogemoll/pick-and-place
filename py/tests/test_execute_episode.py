# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Driving a whole episode through the executor, with no cameras and no recording.

The hardware execution path was taken to be untestable because it opens four USB
cameras and streams to a real arm. Neither is reachable from here, but neither is
required: with ``recording=None`` and ``wrist_camera=None`` the executor needs
only a follower that accepts actions and a viewer that says it is running. That
leaves phase playback, the per-phase epilogues, checkpoint replanning and
teardown under test, which is the part a decomposition would break silently.

The wrist-servo and recording branches still need camera doubles and are not
covered here.
"""

import contextlib
import io

import numpy as np
import pytest

from pick_and_place.core.joint_frames import (
    action_to_joints,
    follower_clamp_limits,
    joints_to_action,
    sim_frame_to_real,
)
from pick_and_place.runtime.episodes import prepare_episode
from pick_and_place.runtime.executor import execute_episode
from pick_and_place.spec.robot import HARDWARE_SIMULATION_HZ

# Compresses a ~9 s trajectory into ~1.6 s of ticks. The commanded end pose is
# the same at 4x; only how finely the trajectory is sampled changes.
SPEED = 8.0
PLANNED_PHASES = (
    "approach",
    "descent",
    "grasp",
    "lift",
    "carry",
    "drop_descent",
    "release",
    "retreat",
)


class FakeFollower:
    """Accepts every command and reports it straight back as the measured pose.

    Perfect tracking, so any command the executor sees on readback is one it sent.
    """

    def __init__(self, start: np.ndarray) -> None:
        self.pose = np.asarray(start, dtype=float)
        self.commands: list[np.ndarray] = []

    def get_observation(self) -> dict[str, float]:
        return joints_to_action(self.pose)

    def send_action(self, action: dict[str, float]) -> None:
        self.pose = action_to_joints(action, self.pose)
        self.commands.append(self.pose.copy())


class StubViewer:
    """Always running, never draws."""

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


class ClosingViewer:
    """Reports itself closed from the ``at`` th check onward, as a window close would."""

    def __init__(self, at: int) -> None:
        self.at = at
        self.checks = 0

    def is_running(self) -> bool:
        self.checks += 1
        return self.checks < self.at

    def sync(self) -> None:
        pass


def _episode(seed: int = 0, free_grasp: bool = False):
    episode = prepare_episode(
        np.random.default_rng(seed), max_attempts=40, free_grasp=free_grasp
    )
    # The hardware path steps the sim in whole control ticks, so the model has to
    # run at the hardware substep rate. Every caller sets this after preparing.
    episode.model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    return episode


def _run(episode, viewer=None, **kwargs):
    """Run an episode, returning the status, the follower, and the phases it entered.

    The printed phase log is the only external evidence of the control flow, and
    it is what distinguishes a straight playback from one that replanned.
    """
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = execute_episode(
            episode,
            follower=follower,
            viewer=viewer if viewer is not None else StubViewer(),
            speed=SPEED,
            **kwargs,
        )
    executed = [
        line.split(": ", 1)[1]
        for line in out.getvalue().splitlines()
        if line.startswith("Executing phase: ")
    ]
    return status, follower, executed


@pytest.fixture(scope="module")
def completed_run():
    """One full episode, shared by the assertions that only read its result."""
    episode = _episode()
    status, follower, executed = _run(episode)
    return episode, status, np.asarray(follower.commands), executed


def test_a_prepared_episode_runs_to_completion(completed_run) -> None:
    _, status, commands, _ = completed_run

    assert status == "success"
    assert len(commands) > 0


def test_every_phase_of_the_plan_is_executed(completed_run) -> None:
    """Each planned phase is entered, in order. A checkpoint replan may re-enter
    one, so the executed log is a supersequence rather than an exact match."""
    episode, _, _, executed = completed_run

    assert tuple(phase.name for phase in episode.trajectory.phases) == PLANNED_PHASES
    remaining = list(executed)
    for name in PLANNED_PHASES:
        assert name in remaining, f"phase {name} never executed: {executed}"
        remaining = remaining[remaining.index(name) + 1 :]


def test_the_arm_is_driven_the_whole_way(completed_run) -> None:
    """A run that ramped on and then stalled would still return success, so assert
    the commands actually move and end somewhere other than they started."""
    _, _, commands, _ = completed_run

    assert not np.allclose(commands[0], commands[-1])


def test_no_command_leaves_the_follower_limits(completed_run) -> None:
    """Every command goes through ``clamp_and_warn``; nothing may reach the servos
    outside their range even when the planner asks for it."""
    episode, _, commands, _ = completed_run
    low, high = follower_clamp_limits(episode.kinematics)

    assert np.all(commands >= low - 1e-9)
    assert np.all(commands <= high + 1e-9)


def test_the_run_is_reproducible() -> None:
    """The oracle a refactor is checked against: same seed, same command stream,
    bit for bit. Wall-clock pacing must not reach the commanded values."""
    first_status, first, _ = _run(_episode())
    second_status, second, _ = _run(_episode())

    assert first_status == second_status
    np.testing.assert_array_equal(np.asarray(first.commands), np.asarray(second.commands))


def test_closing_the_viewer_mid_episode_asks_the_caller_to_restart() -> None:
    """A closed window aborts the trajectory, and an aborted trajectory is not a
    success — the caller has to re-home rather than record it."""
    status, follower, _ = _run(_episode(), viewer=ClosingViewer(at=40))

    assert status == "restart"
    assert len(follower.commands) > 0


def test_speed_must_be_positive() -> None:
    episode = _episode()
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))

    with pytest.raises(ValueError, match="speed must be positive"):
        execute_episode(episode, follower=follower, viewer=StubViewer(), speed=0.0)


def test_recording_rest_to_rest_needs_a_recording() -> None:
    """The flag only widens what gets written, so asking for it without a dataset
    is a caller mistake rather than a no-op."""
    episode = _episode()
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))

    with pytest.raises(ValueError, match="record_rest_to_rest requires recording"):
        execute_episode(
            episode,
            follower=follower,
            viewer=StubViewer(),
            speed=SPEED,
            record_rest_to_rest=True,
        )


def test_a_model_that_cannot_hit_the_control_rate_is_rejected() -> None:
    """``prepare_episode`` leaves the model at the sim timestep, which does not
    divide the control period. Running anyway would silently drift the sim clock
    away from the tick clock, so the executor refuses."""
    episode = prepare_episode(np.random.default_rng(0), max_attempts=40)
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))

    with pytest.raises(ValueError, match="cannot produce"):
        execute_episode(episode, follower=follower, viewer=StubViewer(), speed=SPEED)
