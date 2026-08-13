# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The DSRL training loop.

Per parallel environment step: encode the observation once with the frozen
diffusion policy's own encoder, ask the latent actor for a noise vector, denoise
it into an action chunk, step the environment, and store the transition as
features. Then take a fixed number of gradient steps on the replay buffer.

The cost profile is worth stating, because it is what decides how much of this
loop is worth optimizing. A parallel environment step costs one batched pass of
the ViT plus ten U-Net denoising steps, plus ``act_steps`` control ticks of
MuJoCo physics and rendering in every worker -- the same bill the DPPO strand's
rollouts paid, and on the measured RTX 5090 configuration it dominates. A
gradient step costs three small MLP passes at batch 256, so the update-to-data
ratio can be raised well past the paper's default before it shows up in the wall
clock. That asymmetry is the reverse of PPO's here, where the update was the
GPU-bound half.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pick_and_place.dsrl.noise_policy import denoise, latent_shape, visual_features
from pick_and_place.dsrl.replay import ReplayBuffer, ReplaySpec
from pick_and_place.dsrl.sac import LatentSac, SacConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """How one DSRL run is shaped."""

    total_iterations: int = 4000
    # Parallel environment steps whose noise is drawn from N(0, I) instead of
    # from the actor, filling the buffer with the base policy's own behavior
    # before anything is learned from it. The paper's "initial rollouts".
    warmup_iterations: int = 200
    gradient_steps_per_iteration: int = 20
    batch_size: int = 256
    buffer_capacity: int = 400_000
    # No in-training evaluation knob on purpose. Every deterministic eval on the
    # DPPO strand swung +/-6% and "resolved nothing, as always" -- the paired
    # oracle in check_dppo_rl_env.py is the measurement. What is logged here is
    # the rolling success rate of the rollouts already being collected, which is
    # free and only ever read as a liveness signal.
    save_freq: int = 100
    log_freq: int = 10
    seed: int = 42
    # Whether the critic reads the environment's privileged observation. The
    # actor never does, so the policy that deploys is unchanged either way.
    privileged_critic: bool = True


@dataclass
class RunStats:
    """Episode outcomes seen during training, as a rolling window."""

    successes: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    window: int = 200

    def add(self, success: bool, episode_return: float) -> None:
        self.successes.append(float(success))
        self.returns.append(episode_return)
        del self.successes[: -self.window]
        del self.returns[: -self.window]

    def summary(self) -> dict[str, float]:
        if not self.successes:
            return {"episodes": 0}
        return {
            "episodes": len(self.successes),
            "success_rate": float(np.mean(self.successes)),
            "mean_return": float(np.mean(self.returns)),
        }


def _critic_features(
    observation: dict[str, np.ndarray],
    actor_features: torch.Tensor,
    *,
    privileged: bool,
) -> torch.Tensor:
    """What the critic reads, flattened over the observation history."""
    if not privileged:
        return actor_features
    values = observation["privileged"]
    return torch.from_numpy(values.reshape(len(values), -1)).to(
        actor_features.device, dtype=torch.float32
    )


def critic_feature_dim(
    observation: dict[str, np.ndarray], actor_feature_dim: int, *, privileged: bool
) -> int:
    if not privileged:
        return actor_feature_dim
    return int(np.prod(observation["privileged"].shape[1:]))


class DsrlTrainer:
    """Owns the environment, the frozen policy and the latent learner."""

    def __init__(
        self,
        *,
        model: Any,
        venv: Any,
        act_steps: int,
        # Called once, with (actor_feature_dim, critic_feature_dim) measured off
        # the first observation rather than predicted from the config.
        sac_config_factory: Callable[[int, int], SacConfig],
        train_config: TrainConfig,
        output_dir: Path,
        device: torch.device,
        on_log: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        self.model = model
        self.venv = venv
        self.act_steps = act_steps
        self.config = train_config
        self.output_dir = output_dir
        self.device = device
        self.on_log = on_log
        self._sac_config = sac_config_factory
        self.horizon_steps, self.action_dim = latent_shape(model)
        self.latent_dim = self.horizon_steps * self.action_dim
        self.rng = np.random.default_rng(train_config.seed)
        self.stats = RunStats()
        self.agent: LatentSac | None = None
        self.buffer: ReplayBuffer | None = None
        self._episode_returns: np.ndarray | None = None

    # -- observation plumbing --------------------------------------------

    def _cond(self, observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(observation[key]).float().to(self.device)
            for key in ("state", "rgb")
        }

    def _features(
        self, observation: dict[str, np.ndarray]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        cond = self._cond(observation)
        actor = visual_features(self.model, cond)
        critic = _critic_features(
            observation, actor, privileged=self.config.privileged_critic
        )
        return cond, actor, critic

    # -- acting -----------------------------------------------------------

    def _latent(self, actor_features: torch.Tensor, *, warmup: bool) -> torch.Tensor:
        """The latent-noise action to play this step.

        During warmup this is the standard normal draw the diffusion policy
        would have made for itself, so the buffer starts out describing the base
        policy rather than an untrained actor's preferences.
        """
        if warmup:
            return torch.randn(
                (actor_features.shape[0], self.latent_dim), device=self.device
            )
        assert self.agent is not None
        return self.agent.actor.act(actor_features)

    def _chunk(
        self, cond: dict[str, torch.Tensor], latent: torch.Tensor
    ) -> np.ndarray:
        noise = latent.view(-1, self.horizon_steps, self.action_dim)
        actions = denoise(self.model, cond, noise)
        return actions[:, : self.act_steps].cpu().numpy()

    # -- the loop ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        observation = self.venv.reset_arg()
        cond, actor_features, critic_features = self._features(observation)

        self.agent = LatentSac(
            self._sac_config(actor_features.shape[1], critic_features.shape[1])
        )
        self.buffer = ReplayBuffer(
            ReplaySpec(
                actor_feature_dim=actor_features.shape[1],
                critic_feature_dim=critic_features.shape[1],
                latent_dim=self.latent_dim,
                capacity=self.config.buffer_capacity,
            )
        )
        n_envs = actor_features.shape[0]
        self._episode_returns = np.zeros(n_envs, dtype=np.float64)
        log.info(
            "replay: %d transitions x %d B = %.1f GB at capacity",
            self.buffer.capacity,
            self.buffer.bytes_per_transition(),
            self.buffer.capacity * self.buffer.bytes_per_transition() / 1e9,
        )

        history: list[dict[str, float]] = []
        started = time.perf_counter()
        for iteration in range(self.config.total_iterations):
            warmup = iteration < self.config.warmup_iterations
            latent = self._latent(actor_features, warmup=warmup)
            chunk = self._chunk(cond, latent)
            observation, reward, terminated, truncated, infos = self.venv.step(chunk)
            done = terminated | truncated

            next_cond, next_actor, next_critic = self._features(observation)
            self.buffer.add(
                actor_features=actor_features.cpu().numpy(),
                critic_features=critic_features.cpu().numpy(),
                action=latent.cpu().numpy(),
                reward=reward,
                done=done.astype(np.float32),
                next_actor_features=next_actor.cpu().numpy(),
                next_critic_features=next_critic.cpu().numpy(),
            )
            self._record_episodes(reward, done, infos)
            cond, actor_features, critic_features = next_cond, next_actor, next_critic

            diagnostics: dict[str, float] = {}
            if not warmup and len(self.buffer) >= self.config.batch_size:
                diagnostics = self._update()

            if iteration % self.config.log_freq == 0:
                entry = {
                    "iteration": iteration,
                    "warmup": float(warmup),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                    "buffer": len(self.buffer),
                    **self.stats.summary(),
                    **diagnostics,
                }
                history.append(entry)
                log.info("%s", json.dumps(entry))
                if self.on_log is not None:
                    self.on_log(entry)
            if self.config.save_freq and iteration % self.config.save_freq == 0 and iteration:
                self.save(iteration)

        self.save(self.config.total_iterations)
        return {"history": history, "final": self.stats.summary()}

    def _update(self) -> dict[str, float]:
        assert self.agent is not None and self.buffer is not None
        diagnostics: dict[str, float] = {}
        for _ in range(self.config.gradient_steps_per_iteration):
            batch = self.buffer.sample(self.config.batch_size, self.device, self.rng)
            diagnostics = self.agent.update(batch)
        return diagnostics

    def _record_episodes(
        self, reward: np.ndarray, done: np.ndarray, infos: Any
    ) -> None:
        assert self._episode_returns is not None
        self._episode_returns += np.asarray(reward, dtype=np.float64)
        for index, finished in enumerate(done):
            if not finished:
                continue
            summary = infos[index]["episode"]
            self.stats.add(bool(summary.success), float(self._episode_returns[index]))
            self._episode_returns[index] = 0.0

    def save(self, iteration: int) -> Path:
        assert self.agent is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"state_{iteration}.pt"
        torch.save({"itr": iteration, **self.agent.state_dict()}, path)
        log.info("wrote %s", path)
        return path
