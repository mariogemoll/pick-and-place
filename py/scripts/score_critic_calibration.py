#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure whether the DPPO critic predicts returns at all.

PPO improves a policy only insofar as its advantages carry signal. The advantage
is ``return - value``, so a critic whose predictions are uncorrelated with actual
returns turns every update into a biased random walk -- which degrades a strong
behavior-cloned policy monotonically, at a rate set by the learning rate and
insensitive to everything else. That is precisely the pattern every fine-tuning
run so far shows, and none could have detected it, because DPPO
logs value *loss* (which falls happily when the critic learns to predict a
constant) but never explained variance.

This rolls out the policy, records the critic's value at each step alongside the
reward actually received, and reports:

- **explained variance** of the critic against realized discounted returns.
  1.0 is perfect, 0.0 means "no better than predicting the mean", and negative
  means actively worse than the mean. Below ~0.2 the advantages are mostly noise.
- the correlation and the spread of both series, so a degenerate critic that
  outputs a near-constant is visible directly.

Example:

    PYTHONPATH=third_party/dppo MUJOCO_GL=egl python py/scripts/score_critic_calibration.py \\
      --config config/diffusion_policy/ft_ppo_so101_unet_img.yaml \\
      --checkpoint output/dp_blue_cube_1000/checkpoint/state_1500.pt \\
      --finetuned-checkpoint output/dppo_ft_checkpoints/shaped_state_5.pt \\
      --normalization output/dp_blue_cube_1000/artifact/normalization.npz \\
      --episodes 12 --n-envs 4 --shaping-weight 0.5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument("--render-hw", type=int, nargs=2, default=None)
    parser.add_argument("--shaping-weight", type=float, default=None)
    parser.add_argument("--sampling-std", type=float, default=0.003)
    parser.add_argument("--scene-seed-base", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def explained_variance(predicted: np.ndarray, actual: np.ndarray) -> float:
    """1 - Var(actual - predicted) / Var(actual); 0 means "no better than the mean"."""
    variance = float(np.var(actual))
    if variance == 0.0:
        return float("nan")
    return float(1.0 - np.var(actual - predicted) / variance)


def main() -> None:
    args = _parse_args()

    import hydra
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now().strftime(pattern), replace=True
    )

    from pick_and_place.dppo_rl.env import EnvConfig
    from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
    from pick_and_place.scene_appearance import parse_appearance

    for name in ("DPPO_LOG_DIR", "DPPO_DATA_DIR", "DPPO_BASE_POLICY"):
        os.environ.setdefault(name, "unused-by-this-check")

    config = OmegaConf.load(args.config)
    config.base_policy_path = str(args.checkpoint)
    config.normalization_path = str(args.normalization)
    config.device = args.device
    config.model.min_sampling_denoising_std = args.sampling_std
    OmegaConf.resolve(config)

    gamma = float(config.train.gamma)
    shaping = (
        args.shaping_weight
        if args.shaping_weight is not None
        else float(config.env.get("shaping_weight", 0.0))
    )
    env_config = EnvConfig(
        normalization_path=args.normalization,
        image_hw=(int(config.shape_meta.obs.rgb.shape[1]), int(config.shape_meta.obs.rgb.shape[2])),
        render_hw=tuple(args.render_hw or [int(v) for v in config.env.render_hw]),
        cond_steps=int(config.cond_steps),
        act_steps=int(config.act_steps),
        control_hz=float(config.env.control_hz),
        max_steps=int(config.env.max_episode_steps),
        seed_base=int(args.scene_seed_base or config.env.scene_seed_base),
        scene_appearance=parse_appearance(str(config.env.scene_appearance))[1],
        shaping_weight=shaping,
        gamma=gamma,
        # The asymmetric critic reads this key; without it the critic has no input.
        privileged_obs=bool(config.env.get("privileged_obs", False)),
    )

    torch.manual_seed(args.seed)
    model = hydra.utils.instantiate(config.model)
    state = torch.load(args.finetuned_checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    device = torch.device(args.device)
    print(f"Loaded critic from {args.finetuned_checkpoint} (iteration {state.get('itr')}).")

    venv = DppoVectorEnv(env_config, args.n_envs, mujoco_gl=args.mujoco_gl)
    # Per environment: the running list of (value, reward) for the current episode.
    pending: list[list[tuple[float, float]]] = [[] for _ in range(args.n_envs)]
    values: list[float] = []
    returns: list[float] = []
    finished = 0
    try:
        observation = venv.reset_arg()
        while finished < args.episodes:
            with torch.no_grad():
                # Every key the env emits: the actor reads state/rgb, the
                # asymmetric critic reads the privileged one.
                cond = {
                    key: torch.from_numpy(value).float().to(device)
                    for key, value in observation.items()
                }
                value = model.critic(cond, no_augment=True).cpu().numpy().flatten()
                samples = model(cond=cond, deterministic=False)
            actions = samples.trajectories.cpu().numpy()[:, : env_config.act_steps]
            observation, rewards, terminated, truncated, _ = venv.step(actions)
            for index in range(args.n_envs):
                pending[index].append((float(value[index]), float(rewards[index])))
                if terminated[index] or truncated[index]:
                    # Discounted return from each step to the end of the episode.
                    running = 0.0
                    episode_values, episode_returns = [], []
                    for step_value, step_reward in reversed(pending[index]):
                        running = step_reward + gamma * running
                        episode_values.append(step_value)
                        episode_returns.append(running)
                    values.extend(episode_values)
                    returns.extend(episode_returns)
                    pending[index] = []
                    finished += 1
    finally:
        venv.close()

    value_array = np.array(values)
    return_array = np.array(returns)
    summary = {
        "episodes": finished,
        "steps": int(value_array.size),
        "explained_variance": explained_variance(value_array, return_array),
        "correlation": float(np.corrcoef(value_array, return_array)[0, 1]),
        "value_mean": float(value_array.mean()),
        "value_std": float(value_array.std()),
        "return_mean": float(return_array.mean()),
        "return_std": float(return_array.std()),
        "shaping_weight": shaping,
        "gamma": gamma,
        "finetuned_checkpoint": str(args.finetuned_checkpoint),
    }
    print(json.dumps(summary, indent=2))
    verdict = summary["explained_variance"]
    if verdict < 0.2:
        print(
            "\nVERDICT: the critic explains almost none of the return variance, so "
            "advantages are dominated by noise and PPO cannot be expected to improve "
            "the policy."
        )
    else:
        print("\nVERDICT: the critic carries real signal; look elsewhere for the defect.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
