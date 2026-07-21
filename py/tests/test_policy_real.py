# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.follower import JOINT_NAMES
from pick_and_place.policy_controllers import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.policy_controllers import ControllerFailure
from pick_and_place.policy_real import (
    PhysicalEpisodeOutcome,
    calibrated_state,
    classify_physical_outcome,
    raw_command,
    run_physical_policy_episode,
)


class StubFollower:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=float)
        self.commands = []

    def get_observation(self):
        return {f"{name}.pos": self.state[i] for i, name in enumerate(JOINT_NAMES)}

    def send_action(self, action):
        self.commands.append(action)


class StubController:
    def __init__(self, action, terminal_after=1):
        self.action = np.asarray(action, dtype=float)
        self.terminal_after = terminal_after
        self.reset_count = 0
        self.observations = []
        self.failure = None

    def reset(self):
        self.reset_count += 1
        self.observations.clear()

    def act(self, observation):
        self.observations.append(observation)
        return self.action

    @property
    def terminal(self):
        return len(self.observations) >= self.terminal_after

    @property
    def succeeded(self):
        return self.terminal


def test_physical_adapter_resets_controller_and_passes_only_policy_observation():
    state = np.arange(len(JOINT_NAMES), dtype=float)
    follower = StubFollower(state)
    controller = StubController(state + 1.0)
    overhead = np.zeros((4, 5, 3), dtype=np.uint8)
    wrist = np.ones((3, 2, 3), dtype=np.uint8)
    observed = []
    ticks = []

    result = run_physical_policy_episode(
        controller,
        follower=follower,
        overhead_rgb=lambda: overhead,
        wrist_rgb=lambda: wrist,
        clamp_low=np.full(len(JOINT_NAMES), -100.0),
        clamp_high=np.full(len(JOINT_NAMES), 100.0),
        control_hz=30.0,
        max_steps=10,
        observation_callback=observed.append,
        tick_callback=ticks.append,
        clock=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert result.succeeded
    assert result.control_steps == 1
    assert controller.reset_count == 1
    assert set(controller.observations[0]) == {
        STATE_FEATURE,
        OVERHEAD_FEATURE,
        WRIST_FEATURE,
    }
    np.testing.assert_array_equal(controller.observations[0][STATE_FEATURE], state)
    assert observed[0] is controller.observations[0]
    assert ticks[0].observation is controller.observations[0]
    np.testing.assert_array_equal(ticks[0].command, state + 1.0)
    assert follower.commands[0] == {
        f"{name}.pos": state[i] + 1.0 for i, name in enumerate(JOINT_NAMES)
    }


def test_physical_adapter_clamps_actions_and_reports_timeout():
    follower = StubFollower(np.zeros(len(JOINT_NAMES)))
    controller = StubController(np.full(len(JOINT_NAMES), 20.0), terminal_after=20)

    with pytest.warns(UserWarning, match="clamped"):
        result = run_physical_policy_episode(
            controller,
            follower=follower,
            overhead_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
            wrist_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
            clamp_low=np.full(len(JOINT_NAMES), -5.0),
            clamp_high=np.full(len(JOINT_NAMES), 5.0),
            control_hz=30.0,
            max_steps=2,
            clock=lambda: 0.0,
            sleep=lambda _: None,
        )

    assert result.timed_out
    assert result.control_steps == 2
    assert all(value == 5.0 for value in follower.commands[-1].values())


def test_joint_offsets_are_applied_to_observations_and_removed_from_commands():
    raw = np.arange(len(JOINT_NAMES), dtype=float)
    offsets = {name: index + 0.25 for index, name in enumerate(JOINT_NAMES[:5])}
    calibrated = calibrated_state(raw, offsets)

    np.testing.assert_allclose(raw_command(calibrated, offsets), raw)
    assert calibrated[-1] == raw[-1]


def test_physical_adapter_slew_limits_and_keeps_30_hz_cadence():
    follower = StubFollower(np.zeros(len(JOINT_NAMES)))
    controller = StubController(np.full(len(JOINT_NAMES), 30.0), terminal_after=2)
    now = [0.0]
    sleeps = []

    def sleep(duration):
        sleeps.append(duration)
        now[0] += duration

    result = run_physical_policy_episode(
        controller,
        follower=follower,
        overhead_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        wrist_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        clamp_low=np.full(len(JOINT_NAMES), -100.0),
        clamp_high=np.full(len(JOINT_NAMES), 100.0),
        control_hz=30.0,
        max_steps=3,
        max_slew_per_second=30.0,
        clock=lambda: now[0],
        sleep=sleep,
    )

    assert result.succeeded
    assert sleeps == pytest.approx([1.0 / 30.0])
    assert set(follower.commands[0].values()) == {1.0}
    assert set(follower.commands[1].values()) == {2.0}


class StubRecording:
    def __init__(self):
        self.ticks = []
        self.commits = 0
        self.discards = 0

    def record_tick(self, tick):
        self.ticks.append(tick)

    def commit(self):
        self.commits += 1

    def discard(self):
        self.discards += 1


@pytest.mark.parametrize(
    ("pickup", "placement", "outcome"),
    [
        (False, True, PhysicalEpisodeOutcome.PICKUP_FAILURE),
        (True, False, PhysicalEpisodeOutcome.PLACEMENT_FAILURE),
        (True, True, PhysicalEpisodeOutcome.SUCCESS),
    ],
)
def test_physical_verification_controls_recording_commit(pickup, placement, outcome):
    follower = StubFollower(np.zeros(len(JOINT_NAMES)))
    controller = StubController(np.zeros(len(JOINT_NAMES)))
    recording = StubRecording()

    result = run_physical_policy_episode(
        controller,
        follower=follower,
        overhead_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        wrist_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        clamp_low=np.full(len(JOINT_NAMES), -100.0),
        clamp_high=np.full(len(JOINT_NAMES), 100.0),
        control_hz=30.0,
        max_steps=2,
        pickup_verifier=lambda _: pickup,
        placement_verifier=lambda: placement,
        recording=recording,
        clock=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert result.outcome is outcome
    assert recording.commits == int(outcome is PhysicalEpisodeOutcome.SUCCESS)
    assert recording.discards == int(outcome is not PhysicalEpisodeOutcome.SUCCESS)


def test_operator_abort_is_classified_and_recording_is_discarded():
    follower = StubFollower(np.zeros(len(JOINT_NAMES)))
    controller = StubController(np.zeros(len(JOINT_NAMES)), terminal_after=10)
    controller.act = lambda observation: (_ for _ in ()).throw(KeyboardInterrupt())
    recording = StubRecording()

    result = run_physical_policy_episode(
        controller,
        follower=follower,
        overhead_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        wrist_rgb=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
        clamp_low=np.full(len(JOINT_NAMES), -100.0),
        clamp_high=np.full(len(JOINT_NAMES), 100.0),
        control_hz=30.0,
        max_steps=2,
        recording=recording,
        clock=lambda: 0.0,
        sleep=lambda _: None,
    )

    assert result.outcome is PhysicalEpisodeOutcome.OPERATOR_ABORT
    assert recording.discards == 1


def test_outcome_classifier_preserves_controller_failure():
    failure = ControllerFailure("planning_error", "no safe trajectory")
    outcome = classify_physical_outcome(
        controller_succeeded=False,
        controller_failure=failure,
        timed_out=False,
        pickup_verified=None,
        placement_verified=None,
    )
    assert outcome is PhysicalEpisodeOutcome.CONTROLLER_FAILURE
