# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Ramping the follower onto a pose, driven by a fake device."""

import numpy as np
import pytest

from pick_and_place.core.joint_frames import action_to_joints, joints_to_action
from pick_and_place.runtime.ramp import RAMP_DURATION, ramp_duration, ramp_follower
from pick_and_place.spec.robot import CONTROL_HZ, JOINT_NAMES

LOW = np.array([-90.0, -90.0, -90.0, -90.0, -90.0, 0.0])
HIGH = np.array([90.0, 90.0, 90.0, 90.0, 90.0, 100.0])


class FakeFollower:
    """Records every commanded pose and reports the last one as measured."""

    def __init__(self, start: np.ndarray) -> None:
        self.pose = np.asarray(start, dtype=float)
        self.commands: list[np.ndarray] = []

    def get_observation(self) -> dict[str, float]:
        return joints_to_action(self.pose)

    def send_action(self, action: dict[str, float]) -> None:
        self.pose = action_to_joints(action, self.pose)
        self.commands.append(self.pose.copy())


class ClosingViewer:
    """A viewer that reports itself closed from the ``at`` th check onward."""

    def __init__(self, at: int) -> None:
        self.at = at
        self.checks = 0

    def is_running(self) -> bool:
        self.checks += 1
        return self.checks < self.at


def _target(value: float = 30.0) -> np.ndarray:
    return np.full(len(JOINT_NAMES), value)


def test_duration_is_the_floor_when_the_move_is_small() -> None:
    current = np.zeros(len(JOINT_NAMES))
    assert ramp_duration(current, _target(5.0), 60.0) == RAMP_DURATION


def test_duration_stretches_to_hold_the_velocity_cap() -> None:
    current = np.zeros(len(JOINT_NAMES))
    assert ramp_duration(current, _target(90.0), 10.0) == pytest.approx(9.0)


def test_duration_ignores_the_gripper() -> None:
    current = np.zeros(len(JOINT_NAMES))
    target = np.zeros(len(JOINT_NAMES))
    target[-1] = 100.0
    assert ramp_duration(current, target, 1.0) == RAMP_DURATION


def test_no_cap_means_the_fixed_duration() -> None:
    current = np.zeros(len(JOINT_NAMES))
    assert ramp_duration(current, _target(180.0)) == RAMP_DURATION
    assert ramp_duration(current, _target(180.0), 0.0) == RAMP_DURATION


def test_ramp_ends_exactly_on_the_target() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    assert ramp_follower(follower, _target(), LOW, HIGH, set(), duration=0.2)
    np.testing.assert_allclose(follower.commands[-1], _target())


def test_ramp_starts_and_ends_at_rest() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    ramp_follower(follower, _target(), LOW, HIGH, set(), duration=1.0)
    steps = np.diff(np.array(follower.commands)[:, 0])
    # Smoothstep: the first and last increments are the smallest of the run.
    assert steps[0] < steps.max() / 2.0
    assert steps[-1] < steps.max() / 2.0


def test_ramp_is_monotonic_towards_the_target() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    ramp_follower(follower, _target(), LOW, HIGH, set(), duration=1.0)
    column = np.array(follower.commands)[:, 0]
    assert np.all(np.diff(column) >= 0.0)


def test_every_command_is_clamped_and_warned_once() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    warned: set[str] = set()
    ramp_follower(follower, _target(200.0), LOW, HIGH, warned, duration=0.2)
    commands = np.array(follower.commands)
    assert commands.max() <= HIGH.max()
    np.testing.assert_allclose(commands[-1][:5], HIGH[:5])
    # One warning per arm joint, and the gripper's 200 -> 100 clip too.
    assert warned == set(JOINT_NAMES)


def test_step_count_follows_the_duration_at_the_control_rate() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    ramp_follower(follower, _target(), LOW, HIGH, set(), duration=0.5)
    assert len(follower.commands) == round(0.5 * CONTROL_HZ)


def test_a_zero_duration_still_sends_the_target_once() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    ramp_follower(follower, _target(), LOW, HIGH, set(), duration=0.0)
    assert len(follower.commands) == 1
    np.testing.assert_allclose(follower.commands[-1], _target())


def test_on_tick_sees_the_eased_fraction_and_the_sent_command() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    seen: list[tuple[float, float]] = []
    ramp_follower(
        follower,
        _target(),
        LOW,
        HIGH,
        set(),
        duration=0.2,
        on_tick=lambda alpha, command: seen.append((alpha, float(command[0]))),
    )
    assert [alpha for alpha, _ in seen] == sorted(alpha for alpha, _ in seen)
    assert seen[-1] == (1.0, 30.0)
    assert len(seen) == len(follower.commands)


def test_a_closing_viewer_stops_the_ramp_before_the_next_command() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    viewer = ClosingViewer(at=3)
    assert not ramp_follower(
        follower, _target(), LOW, HIGH, set(), duration=1.0, viewer=viewer
    )
    assert len(follower.commands) == 2


def test_a_live_viewer_runs_the_ramp_to_the_end() -> None:
    follower = FakeFollower(np.zeros(len(JOINT_NAMES)))
    viewer = ClosingViewer(at=10_000)
    assert ramp_follower(follower, _target(), LOW, HIGH, set(), duration=0.2, viewer=viewer)
    np.testing.assert_allclose(follower.commands[-1], _target())
