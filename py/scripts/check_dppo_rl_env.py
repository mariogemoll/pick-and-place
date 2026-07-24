#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run the pretrained policy in the DPPO fine-tuning environment, deterministically.

This is the gate before spending GPU hours on RL: it answers whether the policy
behaves in the training environment the way it did in the closed-loop harness
that measured its baseline. Fine-tuning a policy that has been silently broken
by an observation-pipeline mismatch -- channel order, state normalization, image
geometry, control rate -- looks exactly like fine-tuning a policy with nothing
to learn from, and costs the same.

It drives the same ``PPODiffusion`` model, sampler and vectorized environment the
fine-tuner uses, with ``deterministic=True`` and no gradient updates, then reports
the milestone rates the evaluation oracle tracks (cube moved, cube lifted, and
placed within 6 cm). Small-sample noise is large, so read this as "does the
policy still act on the cube", not as a success-rate measurement.

Example (on a rendering-capable machine, from the repository root):

    PYTHONPATH=third_party/dppo MUJOCO_GL=egl python py/scripts/check_dppo_rl_env.py \\
      --config config/diffusion_policy/ft_ppo_so101_unet_img.yaml \\
      --checkpoint output/dp_blue_cube_1000/checkpoint/state_1500.pt \\
      --normalization output/dp_blue_cube_1000/artifact/normalization.npz \\
      --episodes 12 --n-envs 4 --output outputs/dppo-rl-env-check/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the fine-tuning YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretrained state_*.pt")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument(
        "--render-hw",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="override the config's render resolution (a coarser one is faster but "
        "no longer the geometry the policy was trained on)",
    )
    parser.add_argument(
        "--scene-seed-base",
        type=int,
        default=None,
        help="defaults to the config's training stream, so this checks the scenes "
        "fine-tuning will actually visit",
    )
    parser.add_argument(
        "--scene-appearance",
        default=None,
        help="override the config's appearance preset (e.g. as-recorded), to check "
        "what a differently trained checkpoint sees",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=None,
        help="a DPPO fine-tuning checkpoint to load over the pretrained one. Its "
        "state dict holds actor/actor_ft/critic rather than pretraining EMA "
        "weights, so it is loaded after the model is built. Point --checkpoint at "
        "the pretrained policy either way: it defines the architecture.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample the way rollout collection does instead of deterministically. "
        "This is the policy PPO actually learns from, so its success rate is what "
        "decides whether there is a reward signal to learn from at all.",
    )
    parser.add_argument(
        "--sampling-std",
        type=float,
        default=None,
        help="override the exploration noise floor (min_sampling_denoising_std). "
        "Its units are normalized actions, so on this task's absolute joint "
        "commands 0.1 is roughly 9 degrees per joint. The separate likelihood "
        "floor is not touched: it exists for numerical stability, not "
        "exploration, and nothing here computes log-probabilities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds torch's sampling. ``deterministic=True`` suppresses the "
        "per-step denoising noise but the chain still starts from a random "
        "latent, so two runs of one checkpoint on identical scenes differ "
        "slightly. Fixing this makes a paired comparison differ only by weights.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import hydra
    import torch
    from omegaconf import OmegaConf

    # ``eval`` is the launcher's resolver; ``now`` is normally Hydra's, and the
    # config's run directory interpolates it even though nothing here writes one.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now(UTC).strftime(pattern), replace=True
    )

    from pick_and_place.dppo_rl.env import EnvConfig
    from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
    from pick_and_place.scene_appearance import parse_appearance

    config = OmegaConf.load(args.config)
    config.base_policy_path = str(args.checkpoint)
    config.normalization_path = str(args.normalization)
    config.device = args.device
    if args.sampling_std is not None:
        config.model.min_sampling_denoising_std = args.sampling_std
    # The checkpoint and normalization come from the command line here, and
    # nothing is written to a run directory, so the launcher's environment
    # variables only need to exist for interpolation to succeed.
    for name in ("DPPO_LOG_DIR", "DPPO_DATA_DIR", "DPPO_BASE_POLICY"):
        os.environ.setdefault(name, "unused-by-this-check")
    OmegaConf.resolve(config)

    env_config = EnvConfig(
        normalization_path=args.normalization,
        image_hw=(int(config.shape_meta.obs.rgb.shape[1]), int(config.shape_meta.obs.rgb.shape[2])),
        render_hw=tuple(args.render_hw or [int(value) for value in config.env.render_hw]),
        cond_steps=int(config.cond_steps),
        act_steps=int(config.act_steps),
        control_hz=float(config.env.control_hz),
        max_steps=int(config.env.max_episode_steps),
        seed_base=int(
            args.scene_seed_base
            if args.scene_seed_base is not None
            else config.env.scene_seed_base
        ),
        scene_appearance=parse_appearance(
            args.scene_appearance or str(config.env.scene_appearance)
        )[1],
    )

    torch.manual_seed(args.seed)
    model = hydra.utils.instantiate(config.model)
    if args.finetuned_checkpoint is not None:
        state = torch.load(
            args.finetuned_checkpoint, map_location=args.device, weights_only=True
        )
        model.load_state_dict(state["model"])
        print(
            f"Loaded fine-tuned weights from {args.finetuned_checkpoint} "
            f"(iteration {state.get('itr')})."
        )
    model.eval()
    device = torch.device(args.device)

    venv = DppoVectorEnv(env_config, args.n_envs, mujoco_gl=args.mujoco_gl)
    episodes: list[dict] = []
    started = time.perf_counter()
    try:
        observation = venv.reset_arg()
        while len(episodes) < args.episodes:
            with torch.no_grad():
                cond = {
                    key: torch.from_numpy(observation[key]).float().to(device)
                    for key in ("state", "rgb")
                }
                samples = model(cond=cond, deterministic=not args.stochastic)
            actions = samples.trajectories.cpu().numpy()[:, : env_config.act_steps]
            observation, _, terminated, truncated, infos = venv.step(actions)
            for index, done in enumerate(terminated | truncated):
                if not done:
                    continue
                summary = infos[index]["episode"]
                episodes.append({
                    "scenario_id": summary.scenario_id,
                    "success": summary.success,
                    "control_steps": summary.control_steps,
                    "final_xy_error_m": summary.final_xy_error_m,
                    "unexpected_collision": summary.unexpected_collision,
                    "out_of_bounds": summary.out_of_bounds,
                    **summary.milestones,
                })
    finally:
        venv.close()

    episodes = episodes[: args.episodes]
    total = len(episodes)
    errors = [
        episode["final_xy_error_m"]
        for episode in episodes
        if not math.isnan(episode["final_xy_error_m"])
    ]
    milestone_names = [key for key in episodes[0] if isinstance(episodes[0][key], bool)]
    summary = {
        "episodes": total,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "rates": {
            name: sum(episode[name] for episode in episodes) / total
            for name in milestone_names
        },
        "placed_within_6cm": sum(error <= 0.06 for error in errors) / total,
        "median_final_xy_error_m": float(np.median(errors)) if errors else None,
        "config": {
            "checkpoint": str(args.checkpoint),
            "finetuned_checkpoint": (
                None if args.finetuned_checkpoint is None else str(args.finetuned_checkpoint)
            ),
            "render_hw": list(env_config.render_hw),
            "act_steps": env_config.act_steps,
            "control_hz": env_config.control_hz,
            "max_steps": env_config.max_steps,
            "scene_seed_base": env_config.seed_base,
            "scene_appearance": env_config.scene_appearance.name,
            "stochastic": args.stochastic,
            "sampling_std": (
                args.sampling_std
                if args.sampling_std is not None
                else float(config.model.min_sampling_denoising_std)
            ),
            "seed": args.seed,
            "ddim_steps": int(config.model.ddim_steps),
        },
    }

    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "episodes": episodes}, indent=2) + "\n"
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
