# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.diffusion_policy_dataset import normalize_min_max
from pick_and_place.dppo_rl.env import (
    DppoTaskEnv,
    EnvConfig,
    normalize_state,
    unnormalize_action,
)
from pick_and_place.dppo_rl.vector_env import DppoVectorEnv

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


def _normalization(tmp_path):
    path = tmp_path / "normalization.npz"
    np.savez(
        path,
        obs_min=np.full(6, -100.0, dtype=np.float32),
        obs_max=np.full(6, 100.0, dtype=np.float32),
        action_min=np.full(6, -100.0, dtype=np.float32),
        action_max=np.full(6, 100.0, dtype=np.float32),
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
