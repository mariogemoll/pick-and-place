# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    LeRobotPolicyController,
    NoOpPolicyController,
    ScriptedPolicyController,
)


def _observation():
    return {
        STATE_FEATURE: np.arange(6, dtype=np.float32),
        OVERHEAD_FEATURE: np.zeros((8, 8, 3), dtype=np.uint8),
        WRIST_FEATURE: np.ones((8, 8, 3), dtype=np.uint8),
        "privileged.cube_position": np.array([0.1, 0.2, 0.3]),
    }


def test_no_op_controller_holds_observed_state():
    action = NoOpPolicyController().act(_observation())
    np.testing.assert_array_equal(action, np.arange(6, dtype=np.float32))


def test_scripted_controller_restarts_and_holds_final_action():
    first = np.zeros(6, dtype=np.float32)
    second = np.ones(6, dtype=np.float32)
    controller = ScriptedPolicyController([first, second])

    np.testing.assert_array_equal(controller.act({}), first)
    np.testing.assert_array_equal(controller.act({}), second)
    np.testing.assert_array_equal(controller.act({}), second)
    controller.reset()
    np.testing.assert_array_equal(controller.act({}), first)


def test_lerobot_controller_maps_canonical_camera_names_and_resets_policy():
    class Policy:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    policy = Policy()
    captured = {}

    def predict(observation, *args, **kwargs):
        captured.update(observation)
        return np.arange(6, dtype=np.float32)

    controller = LeRobotPolicyController(
        policy=policy,
        preprocessor=None,
        postprocessor=None,
        device="cpu",
        image_keys=("observation.images.camera1", "observation.images.camera2"),
        predict_action_fn=predict,
    )
    controller.reset()
    action = controller.act(_observation())

    assert policy.reset_count == 1
    assert set(captured) == {
        STATE_FEATURE,
        "observation.images.camera1",
        "observation.images.camera2",
    }
    np.testing.assert_array_equal(captured["observation.images.camera1"], _observation()[OVERHEAD_FEATURE])
    np.testing.assert_array_equal(action, np.arange(6, dtype=np.float32))
