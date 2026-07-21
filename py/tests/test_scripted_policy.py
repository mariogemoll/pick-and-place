# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from types import SimpleNamespace

import numpy as np

from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.policy_controllers import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.scripted_policy import ScriptedPolicy, ScriptedPolicyState


class StubLocalizer:
    def __init__(self, cubes=(), targets=()):
        self.cubes = iter(cubes)
        self.targets = iter(targets)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def localize_cube(self, image):
        del image
        return next(self.cubes, None)

    def localize_drop_target(self, image, **kwargs):
        del image
        self.target_kwargs = kwargs
        return next(self.targets, None)


def _observation(state=None):
    return {
        STATE_FEATURE: np.arange(6, dtype=np.float32) if state is None else state,
        OVERHEAD_FEATURE: np.zeros((8, 8, 3), dtype=np.uint8),
        WRIST_FEATURE: np.ones((8, 8, 3), dtype=np.uint8),
        "privileged.cube_position": np.array([9.0, 9.0, 9.0]),
    }


def _policy(localizer, *, max_steps=2):
    return ScriptedPolicy(localizer, np.ones((4, 3)), max_localization_steps=max_steps)


def test_scripted_policy_localizes_from_images_and_holds_reported_state():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    localizer = StubLocalizer(cubes=[cube], targets=[target])
    policy = _policy(localizer)

    action = policy.act(_observation())

    np.testing.assert_array_equal(action, np.arange(6, dtype=np.float32))
    assert policy.state is ScriptedPolicyState.READY
    assert policy.cube_pose is cube
    assert policy.drop_target is target
    assert localizer.target_kwargs["target_color"] == "black"
    np.testing.assert_array_equal(
        localizer.target_kwargs["workspace_corners_world"], np.ones((4, 3))
    )


def test_scripted_policy_times_out_explicitly_and_keeps_holding():
    policy = _policy(StubLocalizer(), max_steps=2)

    policy.act(_observation())
    second_action = policy.act(_observation())
    failed_action = policy.act(_observation(state=np.full(6, 3.0, dtype=np.float32)))

    np.testing.assert_array_equal(second_action, np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(failed_action, np.full(6, 3.0, dtype=np.float32))
    assert policy.terminal
    assert policy.failure is not None
    assert policy.failure.code == "localization_timeout"
    assert "cube and drop target" in policy.failure.message


def test_scripted_policy_reset_clears_episode_state_and_resets_localizer():
    cube = CubePose(0.1, 0.2, CUBE_HALF_SIZE)
    target = SimpleNamespace(xy=(0.2, -0.1))
    localizer = StubLocalizer(cubes=[cube], targets=[target])
    policy = _policy(localizer)
    policy.act(_observation())

    policy.reset()

    assert policy.state is ScriptedPolicyState.LOCALIZING
    assert policy.cube_pose is None
    assert policy.drop_target is None
    assert policy.failure is None
    assert localizer.reset_count == 2


def test_scripted_policy_turns_bad_image_observation_into_terminal_failure():
    policy = _policy(StubLocalizer())
    observation = _observation()
    del observation[WRIST_FEATURE]

    action = policy.act(observation)

    np.testing.assert_array_equal(action, observation[STATE_FEATURE])
    assert policy.terminal
    assert policy.failure is not None
    assert policy.failure.code == "localization_error"
    assert WRIST_FEATURE in policy.failure.message
