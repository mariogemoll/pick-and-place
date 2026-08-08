# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The training loop end to end, against stubs.

Everything expensive in a DSRL run -- MuJoCo, the ViT, the U-Net -- is replaced
here, leaving the plumbing: that features are encoded once and reused as the
next step's current features, that warmup draws standard normal noise instead of
the actor's, that the latent is reshaped into the chunk the environment expects,
and that transitions land in the buffer with the right widths. Those are the
joins a real run would only fail at after provisioning a pod.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pick_and_place.dsrl.sac import SacConfig  # noqa: E402
from pick_and_place.dsrl.trainer import DsrlTrainer, TrainConfig  # noqa: E402

N_ENVS = 3
HORIZON_STEPS = 4
ACTION_DIM = 2
ACT_STEPS = 2
FEATURE_DIM = 6
PRIVILEGED_DIM = 5
COND_STEPS = 2
EPISODE_STEPS = 5


class _StubModel:
    """Stands in for the frozen diffusion policy."""

    horizon_steps = HORIZON_STEPS
    action_dim = ACTION_DIM

    def __init__(self) -> None:
        self.seen_noise: list[np.ndarray] = []


class _StubVectorEnv:
    """A DppoVectorEnv-shaped environment with no simulator behind it."""

    def __init__(self) -> None:
        self.step_count = 0
        self.chunks: list[np.ndarray] = []

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "state": np.full((N_ENVS, COND_STEPS, 6), self.step_count, dtype=np.float32),
            "rgb": np.zeros((N_ENVS, COND_STEPS, 6, 8, 8), dtype=np.uint8),
            "privileged": np.full(
                (N_ENVS, COND_STEPS, PRIVILEGED_DIM), self.step_count, dtype=np.float32
            ),
        }

    def reset_arg(self, options_list=None):
        del options_list
        return self._observation()

    def step(self, chunk):
        self.chunks.append(np.asarray(chunk))
        self.step_count += 1
        done = np.zeros(N_ENVS, dtype=bool)
        if self.step_count % EPISODE_STEPS == 0:
            done[:] = True
        reward = np.full(N_ENVS, 0.5, dtype=np.float32)
        infos = [
            {"episode": SimpleNamespace(success=bool(index == 0), scenario_id=f"s{index}")}
            for index in range(N_ENVS)
        ]
        return self._observation(), reward, done, np.zeros(N_ENVS, dtype=bool), infos

    def close(self):
        pass


def _trainer(tmp_path, monkeypatch, model, venv, train_config):
    """Wire the trainer with the two frozen-policy calls stubbed out."""
    import pick_and_place.dsrl.trainer as trainer_module

    def visual_features(_model, cond):
        # A deterministic function of the observation, as the real encoder is.
        state = cond["state"].reshape(cond["state"].shape[0], -1)
        return state[:, :1].repeat(1, FEATURE_DIM).float()

    def denoise(target, _cond, noise):
        target.seen_noise.append(noise.detach().cpu().numpy())
        return noise.clone()

    monkeypatch.setattr(trainer_module, "visual_features", visual_features)
    monkeypatch.setattr(trainer_module, "denoise", denoise)

    def sac_config(actor_feature_dim: int, critic_feature_dim: int) -> SacConfig:
        return SacConfig(
            latent_dim=HORIZON_STEPS * ACTION_DIM,
            actor_feature_dim=actor_feature_dim,
            critic_feature_dim=critic_feature_dim,
            hidden_dim=8,
            n_layers=1,
            device="cpu",
        )

    return DsrlTrainer(
        model=model,
        venv=venv,
        act_steps=ACT_STEPS,
        sac_config_factory=sac_config,
        train_config=train_config,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )


def test_the_loop_runs_and_writes_a_checkpoint(tmp_path, monkeypatch):
    model, venv = _StubModel(), _StubVectorEnv()
    config = TrainConfig(
        total_iterations=12,
        warmup_iterations=4,
        gradient_steps_per_iteration=1,
        batch_size=4,
        buffer_capacity=64,
        save_freq=0,
        log_freq=100,
    )
    trainer = _trainer(tmp_path, monkeypatch, model, venv, config)
    result = trainer.run()

    assert len(venv.chunks) == 12
    # Only the executed prefix is handed to the environment, not the horizon.
    assert venv.chunks[0].shape == (N_ENVS, ACT_STEPS, ACTION_DIM)
    # The noise is reshaped back into a chunk before denoising.
    assert model.seen_noise[0].shape == (N_ENVS, HORIZON_STEPS, ACTION_DIM)
    assert (tmp_path / "state_12.pt").exists()
    # Episodes end every EPISODE_STEPS across all envs, and each one is recorded.
    assert result["final"]["episodes"] == (12 // EPISODE_STEPS) * N_ENVS
    assert result["final"]["success_rate"] == pytest.approx(1 / N_ENVS)


def test_warmup_uses_the_base_policys_own_noise(tmp_path, monkeypatch):
    """Warmup must fill the buffer with N(0, I), not with an untrained actor.

    The actor is bounded by action_magnitude and starts near zero, so seeding
    the buffer from it would describe a policy nobody intends to deploy.
    """
    model, venv = _StubModel(), _StubVectorEnv()
    config = TrainConfig(
        total_iterations=200,
        warmup_iterations=200,
        gradient_steps_per_iteration=1,
        batch_size=4,
        buffer_capacity=1024,
        save_freq=0,
        log_freq=1000,
    )
    trainer = _trainer(tmp_path, monkeypatch, model, venv, config)
    trainer.run()

    drawn = np.concatenate([noise.reshape(-1) for noise in model.seen_noise])
    assert drawn.std() == pytest.approx(1.0, abs=0.1)
    # A standard normal routinely leaves the actor's box; an actor's draw could not.
    assert np.abs(drawn).max() > 1.5


def test_transitions_land_in_the_buffer_with_the_configured_widths(tmp_path, monkeypatch):
    model, venv = _StubModel(), _StubVectorEnv()
    config = TrainConfig(
        total_iterations=6,
        warmup_iterations=6,
        gradient_steps_per_iteration=1,
        batch_size=2,
        buffer_capacity=64,
        save_freq=0,
        log_freq=1000,
    )
    trainer = _trainer(tmp_path, monkeypatch, model, venv, config)
    trainer.run()

    assert trainer.buffer is not None
    assert len(trainer.buffer) == 6 * N_ENVS
    batch = trainer.buffer.sample(5, torch.device("cpu"), np.random.default_rng(0))
    assert batch["actor_features"].shape == (5, FEATURE_DIM)
    assert batch["critic_features"].shape == (5, COND_STEPS * PRIVILEGED_DIM)
    assert batch["action"].shape == (5, HORIZON_STEPS * ACTION_DIM)


def test_an_observable_critic_reads_the_actors_features(tmp_path, monkeypatch):
    """With privileged_critic off, both networks see exactly the same input."""
    model, venv = _StubModel(), _StubVectorEnv()
    config = TrainConfig(
        total_iterations=3,
        warmup_iterations=3,
        gradient_steps_per_iteration=1,
        batch_size=2,
        buffer_capacity=64,
        save_freq=0,
        log_freq=1000,
        privileged_critic=False,
    )
    trainer = _trainer(tmp_path, monkeypatch, model, venv, config)
    trainer.run()

    assert trainer.buffer is not None
    batch = trainer.buffer.sample(4, torch.device("cpu"), np.random.default_rng(0))
    assert torch.equal(batch["actor_features"], batch["critic_features"])
