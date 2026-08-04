# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Bind the pick-and-place environment into DPPO's image PPO fine-tuner.

``TrainAgent.__init__`` builds its vectorized environment through the vendored
``env.gym_utils.make_async``, which only knows robomimic, d4rl, D3IL and
furniture. Rather than reimplement DPPO's PPO loop -- the whole point of using
the official implementation is that the algorithm is theirs -- this substitutes
the builder for the duration of that constructor and hands back
:class:`DppoVectorEnv`.

Hydra instantiates this class in place of
``agent.finetune.train_ppo_diffusion_img_agent.TrainPPOImgDiffusionAgent``; every
other config key keeps its upstream meaning.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from agent.finetune import train_agent as dppo_train_agent
from agent.finetune.train_ppo_diffusion_img_agent import TrainPPOImgDiffusionAgent

from pick_and_place.dppo_rl.env import EnvConfig
from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
from pick_and_place.sim.scene_appearance import parse_appearance

log = logging.getLogger(__name__)


class TimedVectorEnv:
    """A :class:`DppoVectorEnv` that reports how the wall clock splits.

    On a rented GPU the question that decides the machine shape is whether the
    iteration is spent collecting rollouts (CPU physics and rendering, GPU
    largely idle) or updating (GPU-bound). DPPO logs only the total, so this
    prints the split once per rollout batch.
    """

    def __init__(self, venv: DppoVectorEnv, steps_per_iteration: int, act_steps: int) -> None:
        self._venv = venv
        self._steps_per_iteration = steps_per_iteration
        self._act_steps = act_steps
        self._rollout_seconds = 0.0
        self._inference_seconds = 0.0
        self._steps = 0
        self._batch_ended_at: float | None = None
        self._step_ended_at: float | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._venv, name)

    def reset_arg(self, options_list=None):
        started = time.perf_counter()
        result = self._venv.reset_arg(options_list)
        self._rollout_seconds += time.perf_counter() - started
        return result

    def step(self, action_venv):
        started = time.perf_counter()
        # The gap since the previous step returned is the policy's sampling for
        # this step: within a batch, nothing else runs between two steps.
        if self._step_ended_at is not None:
            self._inference_seconds += started - self._step_ended_at
        result = self._venv.step(action_venv)
        self._step_ended_at = time.perf_counter()
        self._rollout_seconds += self._step_ended_at - started
        self._steps += 1
        if self._steps % self._steps_per_iteration == 0:
            ticks = self._steps_per_iteration * self._venv.n_envs * self._act_steps
            # Everything since the last batch ended that was neither stepping the
            # environments nor sampling actions is the PPO update.
            update_seconds = (
                self._step_ended_at
                - self._batch_ended_at
                - self._rollout_seconds
                - self._inference_seconds
                if self._batch_ended_at is not None
                else float("nan")
            )
            log.info(
                f"env {self._rollout_seconds:.1f}s for {ticks} control ticks "
                f"({ticks / max(self._rollout_seconds, 1e-9):.0f} ticks/s); "
                f"sampling {self._inference_seconds:.1f}s; "
                f"previous update {update_seconds:.1f}s"
            )
            self._rollout_seconds = 0.0
            self._inference_seconds = 0.0
            self._batch_ended_at = self._step_ended_at
            self._step_ended_at = None
        return result


class PickAndPlacePPOImgAgent(TrainPPOImgDiffusionAgent):
    """DPPO's image PPO agent over the simulated pick-and-place task."""

    def __init__(self, cfg):
        env_config = EnvConfig(
            normalization_path=Path(cfg.normalization_path),
            image_hw=(int(cfg.shape_meta.obs.rgb.shape[1]), int(cfg.shape_meta.obs.rgb.shape[2])),
            render_hw=tuple(int(value) for value in cfg.env.render_hw),
            cond_steps=int(cfg.cond_steps),
            act_steps=int(cfg.act_steps),
            control_hz=float(cfg.env.control_hz),
            max_steps=int(cfg.env.max_episode_steps),
            seed_base=int(cfg.env.scene_seed_base),
            scene_appearance=parse_appearance(str(cfg.env.scene_appearance))[1],
            shaping_weight=float(cfg.env.get("shaping_weight", 0.0)),
            gamma=float(cfg.train.gamma),
            privileged_obs=bool(cfg.env.get("privileged_obs", False)),
            debug_action_reward=bool(cfg.env.get("debug_action_reward", False)),
        )
        if env_config.cond_steps != int(cfg.img_cond_steps):
            raise ValueError(
                "this environment stacks one camera history per state step, so "
                "cond_steps and img_cond_steps must match"
            )
        mujoco_gl = cfg.env.get("mujoco_gl", "egl")

        def make_env(*args, **kwargs):
            venv = DppoVectorEnv(env_config, int(cfg.env.n_envs), mujoco_gl=mujoco_gl)
            return TimedVectorEnv(venv, int(cfg.train.n_steps), env_config.act_steps)

        with _patched(dppo_train_agent, "make_async", make_env):
            super().__init__(cfg)

        # The 5090 has no reason to run these in full fp32.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        log.info(
            f"Fine-tuning {cfg.base_policy_path} over {cfg.env.n_envs} environments "
            f"at {env_config.control_hz:g} Hz, {cfg.act_steps} of {cfg.horizon_steps} "
            f"actions executed per query, scene seed base {env_config.seed_base}, "
            f"scene appearance {cfg.env.scene_appearance}."
        )

    def run(self):
        try:
            super().run()
        finally:
            with contextlib.suppress(Exception):
                self.venv.close()


@contextlib.contextmanager
def _patched(module, name: str, replacement):
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def episode_summaries(infos) -> list[dict[str, Any]]:
    """Flatten the per-step ``info`` payloads into plain episode records."""
    summaries = []
    for info in infos:
        episode = info.get("episode")
        if episode is None:
            continue
        summaries.append({
            "scenario_id": episode.scenario_id,
            "control_steps": episode.control_steps,
            "success": episode.success,
            "final_xy_error_m": (
                None if np.isnan(episode.final_xy_error_m) else episode.final_xy_error_m
            ),
            **{f"milestone_{name}": value for name, value in episode.milestones.items()},
        })
    return summaries
