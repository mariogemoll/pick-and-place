# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.data.diffusion_policy_dataset import normalize_min_max
from pick_and_place.dppo_rl.env import (
    DppoTaskEnv,
    EnvConfig,
    normalize_state,
    unnormalize_action,
)
from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
from pick_and_place.spec.action_encoding import ACTION_ENCODING_KEY, ActionEncoding
from pick_and_place.spec.controller import STATE_FEATURE

OVERHEAD_VALUE = 20
WRIST_VALUE = 10


class DummyRenderer:
    """Renders a flat, per-camera constant so channel order stays checkable."""

    def __init__(self, model, *, height, width):
        self.model = model
        self.height = height
        self.width = width
        self.camera = ""

    def update_scene(self, data, *, camera):
        del data
        self.camera = camera

    def render(self):
        value = WRIST_VALUE if self.camera == "wrist_camera" else OVERHEAD_VALUE
        return np.full((self.height, self.width, 3), value, dtype=np.uint8)

    def close(self):
        pass


def _normalization(tmp_path, action_encoding=ActionEncoding.ABSOLUTE):
    path = tmp_path / f"normalization-{action_encoding.value}.npz"
    np.savez(
        path,
        obs_min=np.full(6, -100.0, dtype=np.float32),
        obs_max=np.full(6, 100.0, dtype=np.float32),
        action_min=np.full(6, -100.0, dtype=np.float32),
        action_max=np.full(6, 100.0, dtype=np.float32),
        **{ACTION_ENCODING_KEY: action_encoding.value},
    )
    return path


def _config(tmp_path, **overrides):
    defaults = {
        "normalization_path": _normalization(tmp_path),
        "image_hw": (96, 96),
        "render_hw": (120, 160),
        "cond_steps": 2,
        "act_steps": 2,
        "max_steps": 3,
        "renderer_factory": DummyRenderer,
    }
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _env(tmp_path, **overrides):
    return DppoTaskEnv(_config(tmp_path, **overrides))


def test_normalization_inverts_the_exporters_min_max_map():
    raw = np.array([[-1.0, 2.0, 30.0], [5.0, -2.0, 10.0], [0.0, 0.0, 20.0]], dtype=np.float32)
    normalized, minimum, maximum = normalize_min_max(raw)
    assert normalize_state(raw, minimum, maximum) == pytest.approx(normalized, abs=1e-5)
    assert unnormalize_action(normalized, minimum, maximum) == pytest.approx(raw, abs=1e-3)


def test_observation_matches_the_checkpoints_input_contract(tmp_path):
    env = _env(tmp_path)
    try:
        observation = env.reset()
        assert set(observation) == {"state", "rgb"}
        assert observation["state"].shape == (2, 6)
        assert observation["state"].dtype == np.float32
        assert observation["rgb"].shape == (2, 6, 96, 96)
        assert observation["rgb"].dtype == np.uint8
        # Overhead occupies the first three channels, wrist the last three, the
        # order the export wrote and the policy server feeds.
        assert np.all(observation["rgb"][:, :3] == OVERHEAD_VALUE)
        assert np.all(observation["rgb"][:, 3:] == WRIST_VALUE)
        # A one-observation history is padded with its oldest entry, as the
        # closed-loop controller does on its first tick.
        assert np.array_equal(observation["state"][0], observation["state"][1])
    finally:
        env.close()


def test_state_is_normalized_into_the_policys_range(tmp_path):
    env = _env(tmp_path)
    try:
        observation = env.reset()
        # The neutral start is well inside the +/-100 bounds this test declares.
        assert np.all(np.abs(observation["state"]) <= 1.0)
    finally:
        env.close()


def test_action_chunk_advances_one_control_tick_per_row(tmp_path):
    env = _env(tmp_path, act_steps=2, max_steps=10)
    try:
        env.reset()
        _, _, _, _, info = env.step(np.zeros((2, 6), dtype=np.float32))
        assert info["episode"].control_steps == 2
        _, _, _, _, info = env.step(np.zeros((2, 6), dtype=np.float32))
        assert info["episode"].control_steps == 4
    finally:
        env.close()


def test_an_absolute_chunk_is_commanded_as_it_is(tmp_path):
    env = _env(tmp_path, act_steps=2, max_steps=10)
    commanded, _ = _record_commands(env)
    try:
        env.reset()
        env.step(np.full((2, 6), 0.5, dtype=np.float32))
    finally:
        env.close()

    for command in commanded:
        np.testing.assert_allclose(command, np.full(6, 50.0), atol=1e-4)


def test_a_delta_chunk_is_integrated_onto_each_ticks_measured_joints(tmp_path):
    env = _env(
        tmp_path,
        act_steps=2,
        max_steps=10,
        normalization_path=_normalization(tmp_path, ActionEncoding.DELTA),
    )
    commanded, measured = _record_commands(env)
    try:
        env.reset()
        start = env._measured_joints.copy()
        # +1.0 normalized is +100 raw against the bounds this test declares.
        env.step(np.ones((2, 6), dtype=np.float32))
    finally:
        env.close()

    np.testing.assert_allclose(commanded[0], start + 100.0, atol=1e-4)
    # The arm moves under the first command, and the second row of the chunk is
    # measured from where it ended up -- not from where the chunk was predicted.
    assert np.any(np.abs(measured[0] - start) > 1e-3)
    np.testing.assert_allclose(commanded[1], measured[0] + 100.0, atol=1e-4)


def test_the_dense_reward_pays_for_every_tick_the_cube_stays_on_target(tmp_path):
    env = _env(tmp_path, act_steps=3, max_steps=9, dense_success_reward=True)
    settled = [False, True, True]
    try:
        env.reset()
        # The oracle is driven by physics, so stub the one fact the reward reads.
        inner_step = env._env.step

        def settling_step(action):
            observation, reward, terminated, truncated, info = inner_step(action)
            info = {**info, "settled_on_target": settled.pop(0) if settled else True}
            return observation, reward, terminated, truncated, info

        env._env.step = settling_step
        _, reward, _, _, _ = env.step(np.zeros((3, 6), dtype=np.float32))
    finally:
        env.close()

    # Two of the chunk's three ticks had the cube on target.
    assert reward == pytest.approx(2.0)


def test_the_sparse_reward_still_pays_once_and_ends_the_episode(tmp_path):
    env = _env(tmp_path, act_steps=2, max_steps=10)
    try:
        # The default keeps the underlying environment terminating on success,
        # which is what every scored evaluation depends on.
        assert env._env.terminate_on_success
        assert not env.config.dense_success_reward
    finally:
        env.close()

    dense = _env(tmp_path, act_steps=2, max_steps=10, dense_success_reward=True)
    try:
        assert not dense._env.terminate_on_success
    finally:
        dense.close()


def _record_commands(env):
    """Capture what reaches the simulator, and what it reports back."""
    commanded: list[np.ndarray] = []
    measured: list[np.ndarray] = []
    inner_step = env._env.step

    def recording_step(action):
        commanded.append(np.asarray(action, dtype=np.float64).copy())
        result = inner_step(action)
        measured.append(np.asarray(result[0][STATE_FEATURE], dtype=np.float64).copy())
        return result

    env._env.step = recording_step
    return commanded, measured


def test_reward_is_sparse_and_the_episode_restarts_on_truncation(tmp_path):
    env = _env(tmp_path, act_steps=2, max_steps=2)
    try:
        first = env.reset()
        first_scenario = env._summary.scenario_id
        observation, reward, terminated, truncated, info = env.step(
            np.zeros((2, 6), dtype=np.float32)
        )
        assert reward == 0.0
        # The step limit is reported as terminal so DPPO does not bootstrap a
        # timed-out episode from the next episode's fresh starting observation.
        assert terminated
        assert truncated
        assert not info["episode"].success
        assert info["episode"].scenario_id == first_scenario
        # The next episode has already started, so its history is padded again.
        assert np.array_equal(observation["state"][0], observation["state"][1])
        assert env._summary.scenario_id != first_scenario
        assert first["rgb"].shape == observation["rgb"].shape
    finally:
        env.close()


def test_vector_env_batches_workers_over_disjoint_scenes(tmp_path):
    config = _config(tmp_path, act_steps=2, max_steps=4)
    # Fork keeps the stub renderer importable in the worker; production spawns,
    # because an OpenGL context does not survive a fork.
    venv = DppoVectorEnv(config, n_envs=2, start_method="fork")
    try:
        observation = venv.reset_arg()
        assert observation["state"].shape == (2, 2, 6)
        assert observation["rgb"].shape == (2, 2, 6, 96, 96)
        observation, rewards, terminated, truncated, infos = venv.step(
            np.zeros((2, 2, 6), dtype=np.float32)
        )
        assert rewards.shape == (2,)
        assert terminated.shape == truncated.shape == (2,)
        assert not terminated.any()
        scenario_ids = {info["episode"].scenario_id for info in infos}
        assert scenario_ids == {"dppo-train-000000", "dppo-train-000001"}
    finally:
        venv.close()


def test_vector_env_rejects_a_mismatched_action_batch(tmp_path):
    config = _config(tmp_path, act_steps=2, max_steps=4)
    venv = DppoVectorEnv(config, n_envs=2, start_method="fork")
    try:
        with pytest.raises(ValueError):
            venv.step(np.zeros((3, 2, 6), dtype=np.float32))
    finally:
        venv.close()
