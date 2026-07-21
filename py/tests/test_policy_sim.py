# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from dataclasses import replace
from pathlib import Path

import numpy as np

from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    NoOpPolicyController,
    ControllerFailure,
)
from pick_and_place.policy_evaluation import ScenarioManifest
from pick_and_place.policy_sim import (
    PolicySimEnv,
    evaluate_policy_episode,
    joint_qpos_addresses,
    real_action_to_sim_ctrl,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class DummyRenderer:
    def __init__(self, model, *, height, width):
        self.model = model
        self.height = height
        self.width = width
        self.camera = ""
        self.closed = False

    def update_scene(self, data, *, camera):
        del data
        self.camera = camera

    def render(self):
        value = 10 if self.camera == "wrist_camera" else 20
        return np.full((self.height, self.width, 3), value, dtype=np.uint8)

    def close(self):
        self.closed = True


def _scenario(max_steps=2):
    manifest = ScenarioManifest.load(REPOSITORY_ROOT / "config/evaluation/smoke_v1.json")
    return replace(manifest.scenarios[0], max_steps=max_steps)


def test_visual_env_exposes_only_deployable_observation_and_privileged_info():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
    )
    try:
        observation, info = env.reset(options={"scenario": _scenario()})

        assert set(observation) == {STATE_FEATURE, OVERHEAD_FEATURE, WRIST_FEATURE}
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(observation[WRIST_FEATURE], 10)
        np.testing.assert_array_equal(observation[OVERHEAD_FEATURE], 20)
        assert "cube_position_m" in info["task_state"]
        assert "task_state" not in observation
    finally:
        env.close()


def test_reset_reproduces_explicit_scenario_state():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
    )
    scenario = _scenario()
    try:
        first_observation, first_info = env.reset(options={"scenario": scenario})
        env.step(first_observation[STATE_FEATURE])
        second_observation, second_info = env.reset(options={"scenario": scenario})

        np.testing.assert_array_equal(first_observation[STATE_FEATURE], second_observation[STATE_FEATURE])
        assert first_info["task_state"] == second_info["task_state"]
    finally:
        env.close()


def test_joint_miscalibration_lives_in_environment_and_is_hidden_from_observation():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
    )
    scenario = replace(
        _scenario(),
        miscalibration_sample={"joint_offsets_deg": {"shoulder_pan": 7.5}},
    )
    try:
        observation, info = env.reset(options={"scenario": scenario})

        np.testing.assert_allclose(
            observation[STATE_FEATURE],
            scenario.initial_robot_state_real,
            atol=1e-5,
        )
        initial_ctrl = real_action_to_sim_ctrl(scenario.initial_robot_state_real)
        true_qpos = env.data.qpos[joint_qpos_addresses(env.model)]
        np.testing.assert_allclose(true_qpos[0], initial_ctrl[0] + np.deg2rad(7.5))
        assert set(observation) == {STATE_FEATURE, OVERHEAD_FEATURE, WRIST_FEATURE}
        assert "miscalibration" not in info
    finally:
        env.close()


def test_no_op_controller_times_out_without_false_success():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
    )
    try:
        result = evaluate_policy_episode(env, NoOpPolicyController(), _scenario())

        assert result.control_steps == 2
        assert not result.success
        assert result.failures.missed_pickup
        assert result.failures.timeout
    finally:
        env.close()


def test_controller_failure_stops_episode_and_is_reported():
    class FailingController(NoOpPolicyController):
        failure = None

        def act(self, observation):
            self.failure = ControllerFailure("localization_error", "camera frame is invalid")
            return super().act(observation)

        def reset(self):
            self.failure = None

    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
    )
    try:
        result = evaluate_policy_episode(env, FailingController(), _scenario(max_steps=10))

        assert result.control_steps == 1
        assert result.controller_failure == {
            "code": "localization_error",
            "message": "camera frame is invalid",
        }
        assert not result.failures.timeout
    finally:
        env.close()
