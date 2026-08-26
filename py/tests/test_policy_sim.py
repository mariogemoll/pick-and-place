# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pick_and_place.spec.controller import ControllerFailure, GOAL_FEATURE, OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.policies.policy_controllers import NoOpPolicyController
from pick_and_place.policies.policy_evaluation import ScenarioManifest
from pick_and_place.runtime.policy_sim import (
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


def _perturbed_scenario():
    manifest = ScenarioManifest.load(
        REPOSITORY_ROOT / "config/evaluation/scripted_perturbation_smoke_v1.json"
    )
    return manifest.scenarios[0]


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
        assert "cube_orientation_wxyz" in info["task_state"]
        assert "task_state" not in observation
    finally:
        env.close()


def test_state_only_env_does_not_create_a_renderer():
    def fail_renderer(*args, **kwargs):
        del args, kwargs
        raise AssertionError("state-only observations must not render cameras")

    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=fail_renderer,
        include_images=False,
    )
    try:
        observation, _ = env.reset(options={"scenario": _scenario()})

        assert set(observation) == {STATE_FEATURE}
        assert env.observation_space.contains(observation)
        env.step(observation[STATE_FEATURE])
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


def test_task_state_reads_current_cube_orientation() -> None:
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
        include_images=False,
    )
    try:
        env.reset(options={"scenario": _scenario()})
        orientation_wxyz = np.array([0.5, 0.5, 0.5, 0.5])
        env.data.qpos[env._cube_qpos_adr + 3 : env._cube_qpos_adr + 7] = orientation_wxyz

        np.testing.assert_allclose(env._task_state().cube_orientation_wxyz, orientation_wxyz)
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
        miscalibration_sample={
            "joint_offsets_deg": {"shoulder_pan": 7.5},
            "pan_jitter": None,
            "cube_belief_error": [0.0, 0.0, 0.0, 0.0],
            "target_belief_error": [0.0, 0.0],
        },
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


def test_full_miscalibration_is_reproducible_and_separates_belief_from_truth():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
        include_images=False,
    )
    scenario = replace(
        _scenario(),
        miscalibration_sample={
            "joint_offsets_deg": {},
            "pan_jitter": {"sigma_deg": 2.0, "tau_s": 10.0, "seed": 42},
            "cube_belief_error": [0.01, -0.02, 0.003, 0.1],
            "target_belief_error": [-0.04, 0.05],
        },
    )
    try:
        first_observation, first_info = env.reset(options={"scenario": scenario})
        first_pan_offset = env._joint_offsets_rad()["shoulder_pan"]
        env.step(first_observation[STATE_FEATURE])
        assert env._joint_offsets_rad()["shoulder_pan"] != pytest.approx(first_pan_offset)
        _, second_info = env.reset(options={"scenario": scenario})

        truth = second_info["task_state"]
        belief = second_info["believed_task_state"]
        assert belief["cube_position_m"] == pytest.approx(
            np.asarray(truth["cube_position_m"]) + [0.01, -0.02, 0.003]
        )
        assert belief["target_xy_m"] == pytest.approx(
            np.asarray(truth["target_xy_m"]) + [-0.04, 0.05]
        )
        assert belief["cube_orientation_wxyz"] != pytest.approx(
            truth["cube_orientation_wxyz"]
        )
        assert env._joint_offsets_rad()["shoulder_pan"] == pytest.approx(first_pan_offset)
        assert first_info["believed_task_state"] == second_info["believed_task_state"]
    finally:
        env.close()


def test_physics_draw_is_applied_and_nominal_reset_restores_the_model():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
        include_images=False,
    )
    nominal = _scenario()
    perturbed = replace(
        nominal,
        physics_sample={
            "joint_gain_scale": {"shoulder_pan": 0.8},
            "joint_time_constant_scale": {"shoulder_pan": 1.2},
            "extra_joint_friction": {"shoulder_pan": 0.01},
            "tracking_bias_scale": 0.5,
            "mass_scale": 1.1,
            "friction_scale": 1.3,
            "damping_scale": 0.9,
        },
    )
    try:
        env.reset(options={"scenario": nominal})
        nominal_mass = env.model.body_mass.copy()
        nominal_friction = env.model.geom_friction.copy()

        env.reset(options={"scenario": perturbed})
        np.testing.assert_allclose(env.model.body_mass, nominal_mass * 1.1)
        np.testing.assert_allclose(env.model.geom_friction, nominal_friction * 1.3)
        assert np.any(env._tracking_bias)

        env.reset(options={"scenario": nominal})
        np.testing.assert_array_equal(env.model.body_mass, nominal_mass)
        np.testing.assert_array_equal(env.model.geom_friction, nominal_friction)
        np.testing.assert_array_equal(env._tracking_bias, 0.0)
    finally:
        env.close()


def test_frozen_camera_perturbations_are_environment_only_and_resettable():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
    )
    camera_ids = np.array([
        env.model.camera(name).id for name in ("overhead_camera", "wrist_camera")
    ])
    try:
        env.reset(options={"scenario": _scenario()})
        nominal_positions = env.model.cam_pos[camera_ids].copy()
        nominal_quaternions = env.model.cam_quat[camera_ids].copy()

        observation, info = env.reset(options={"scenario": _perturbed_scenario()})

        assert not np.allclose(env.model.cam_pos[camera_ids], nominal_positions)
        assert not np.allclose(env.model.cam_quat[camera_ids], nominal_quaternions)
        assert set(observation) == {STATE_FEATURE, OVERHEAD_FEATURE, WRIST_FEATURE}
        assert "domain_randomization_sample" not in info
        assert "miscalibration_sample" not in info

        env.reset(options={"scenario": _scenario()})
        np.testing.assert_allclose(env.model.cam_pos[camera_ids], nominal_positions)
        np.testing.assert_allclose(env.model.cam_quat[camera_ids], nominal_quaternions)
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


def test_the_goal_is_absent_unless_the_env_was_built_with_one():
    """An unconditioned policy must see exactly the observation it was trained on."""
    env = PolicySimEnv(image_hw=(16, 16), render_hw=(32, 32), renderer_factory=DummyRenderer)
    try:
        observation, _ = env.reset(options={"scenario": _scenario()})
        assert GOAL_FEATURE not in observation
    finally:
        env.close()


def test_a_goal_env_hands_over_the_scenario_target_rather_than_a_reading_of_it():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
        include_goal=True,
    )
    scenario = _scenario()
    try:
        observation, _ = env.reset(options={"scenario": scenario})

        assert set(observation) == {GOAL_FEATURE, STATE_FEATURE, OVERHEAD_FEATURE, WRIST_FEATURE}
        assert env.observation_space.contains(observation)
        np.testing.assert_allclose(observation[GOAL_FEATURE], scenario.target_position_m[:2])
        # Every tick of the episode carries it, as every frame of the export does.
        stepped, *_ = env.step(observation[STATE_FEATURE])
        np.testing.assert_allclose(stepped[GOAL_FEATURE], scenario.target_position_m[:2])
    finally:
        env.close()


def test_a_goal_observation_does_not_alias_the_env_between_episodes():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
        include_goal=True,
    )
    first = _scenario()
    second = replace(first, scenario_id="moved", target_position_m=(0.05, -0.05, 0.0125))
    try:
        observation, _ = env.reset(options={"scenario": first})
        held = observation[GOAL_FEATURE]
        env.reset(options={"scenario": second})

        np.testing.assert_allclose(held, first.target_position_m[:2])
    finally:
        env.close()


def test_a_state_only_env_still_carries_the_goal():
    env = PolicySimEnv(
        image_hw=(16, 16),
        render_hw=(32, 32),
        renderer_factory=DummyRenderer,
        include_images=False,
        include_goal=True,
    )
    try:
        observation, _ = env.reset(options={"scenario": _scenario()})
        assert set(observation) == {STATE_FEATURE, GOAL_FEATURE}
        assert env.observation_space.contains(observation)
    finally:
        env.close()
