#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure per-scene success by replaying each scene many times.

A single rollout per scene cannot separate the two reasons an episode fails.
Either the scene is one this policy essentially cannot do -- the outcome was
settled at reset -- or the scene is winnable and this particular rollout was
sloppy. The distinction decides whether policy-gradient fine-tuning has anything
to grip: PPO's advantage is ``R - V(s)``, so an outcome determined by ``s``
contributes either zero signal (a critic that sees it) or pure noise (one that
does not). Only variation *within* a scene points at the action.

So this replays a fixed set of scenes ``--repeats`` times each, sampling the way
rollout collection does, and records every episode. The resulting per-scene
success rate is the measurement: a distribution piled up at 0 and 1 means scene
difficulty dominates; one spread across the middle means action noise does.

Scenes are materialized from ``seed_base + index`` alone, so the analysis side
can reconstruct each scene's geometry from its ``scenario_id`` without storing it.

Example (on a rendering-capable machine, from the repository root):

    PYTHONPATH=third_party/dppo MUJOCO_GL=egl python py/scripts/scene_difficulty_sweep.py \\
      --config config/diffusion_policy/ft_ppo_so101_unet_img.yaml \\
      --checkpoint output/dp_blue_cube_1000/checkpoint/state_1500.pt \\
      --normalization output/dp_blue_cube_1000/artifact/normalization.npz \\
      --scenes 128 --repeats 8 --n-envs 64 --output outputs/scene-difficulty/sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the fine-tuning YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretrained state_*.pt")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument(
        "--scenes",
        type=int,
        default=128,
        help="how many distinct scenes to measure. Kept a multiple of --n-envs so "
        "every pass visits exactly the same set.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="rollouts per scene. The per-scene success rate is a binomial "
        "estimate on this many samples, so 8 resolves 0/8 from 4/8 comfortably "
        "and is what the bimodality question needs.",
    )
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument(
        "--scene-seed-base",
        type=int,
        default=6_000_000,
        help="defaults to the held-out stream used by the paired A/B, which no "
        "fine-tuning run trained on.",
    )
    parser.add_argument("--scene-appearance", default=None)
    parser.add_argument(
        "--act-steps",
        type=int,
        default=None,
        help="how many of the predicted actions to execute per query, overriding "
        "the config. This is an inference-time choice -- the network still "
        "predicts horizon_steps actions -- so the same weights can be measured at "
        "several chunk lengths. Shorter chunks re-plan more often: at 8 and 10 Hz "
        "the policy commits to 0.8 s of open-loop motion, which spans the entire "
        "gripper close.",
    )
    parser.add_argument("--finetuned-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="replay without sampling noise. Every pass is then identical apart "
        "from the initial latent, which measures how little of the outcome the "
        "policy's own randomness accounts for.",
    )
    parser.add_argument(
        "--sampling-std",
        type=float,
        default=None,
        help="exploration noise floor. Defaults to the config's "
        "min_sampling_denoising_std, i.e. the distribution PPO actually samples.",
    )
    parser.add_argument("--seed", type=int, default=0, help="pass p uses seed + p")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.scenes % args.n_envs:
        raise SystemExit(
            f"--scenes ({args.scenes}) must be a multiple of --n-envs ({args.n_envs}) "
            "so every pass visits the same scene set"
        )

    import hydra
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now().strftime(pattern), replace=True
    )

    from pick_and_place.dppo_rl.env import EnvConfig
    from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
    from pick_and_place.sim.scene_appearance import parse_appearance

    config = OmegaConf.load(args.config)
    config.base_policy_path = str(args.checkpoint)
    config.normalization_path = str(args.normalization)
    config.device = args.device
    if args.sampling_std is not None:
        config.model.min_sampling_denoising_std = args.sampling_std
    for name in ("DPPO_LOG_DIR", "DPPO_DATA_DIR", "DPPO_BASE_POLICY"):
        os.environ.setdefault(name, "unused-by-this-check")
    OmegaConf.resolve(config)

    env_config = EnvConfig(
        normalization_path=args.normalization,
        image_hw=(
            int(config.shape_meta.obs.rgb.shape[1]),
            int(config.shape_meta.obs.rgb.shape[2]),
        ),
        render_hw=tuple(int(value) for value in config.env.render_hw),
        cond_steps=int(config.cond_steps),
        act_steps=int(args.act_steps if args.act_steps is not None else config.act_steps),
        control_hz=float(config.env.control_hz),
        max_steps=int(config.env.max_episode_steps),
        seed_base=int(args.scene_seed_base),
        scene_appearance=parse_appearance(
            args.scene_appearance or str(config.env.scene_appearance)
        )[1],
    )

    model = hydra.utils.instantiate(config.model)
    if args.finetuned_checkpoint is not None:
        state = torch.load(
            args.finetuned_checkpoint, map_location=args.device, weights_only=True
        )
        model.load_state_dict(state["model"])
    model.eval()
    device = torch.device(args.device)

    episodes: list[dict] = []
    started = time.perf_counter()
    for repeat in range(args.repeats):
        # A fresh vector env restarts every worker's scene stream at its offset,
        # so pass p visits scene indices 0..scenes-1 exactly like pass 0. The
        # torch seed is what differs, which is precisely the action noise.
        torch.manual_seed(args.seed + repeat)
        venv = DppoVectorEnv(env_config, args.n_envs, mujoco_gl=args.mujoco_gl)
        # Worker i draws scene indices i, i + n_envs, i + 2 * n_envs, ... so a
        # per-worker quota pins the scene set exactly. Stopping on a global count
        # instead would let a pass whose episodes ran short pull in later scenes,
        # and the passes would no longer be comparable.
        quota = args.scenes // args.n_envs
        done_per_worker = [0] * args.n_envs
        try:
            observation = venv.reset_arg()
            while min(done_per_worker) < quota:
                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(observation[key]).float().to(device)
                        for key in ("state", "rgb")
                    }
                    samples = model(cond=cond, deterministic=args.deterministic)
                actions = samples.trajectories.cpu().numpy()[:, : env_config.act_steps]
                observation, _, terminated, truncated, infos = venv.step(actions)
                for index, done in enumerate(terminated | truncated):
                    if not done or done_per_worker[index] >= quota:
                        continue
                    summary = infos[index]["episode"]
                    episodes.append({
                        "repeat": repeat,
                        "scenario_id": summary.scenario_id,
                        "success": summary.success,
                        "control_steps": summary.control_steps,
                        "final_xy_error_m": summary.final_xy_error_m,
                        "unexpected_collision": summary.unexpected_collision,
                        "out_of_bounds": summary.out_of_bounds,
                        **summary.milestones,
                    })
                    done_per_worker[index] += 1
        finally:
            venv.close()
        rate = sum(e["success"] for e in episodes if e["repeat"] == repeat) / args.scenes
        print(
            f"pass {repeat + 1}/{args.repeats}: {rate:.3f} over {args.scenes} scenes "
            f"({time.perf_counter() - started:.0f}s elapsed)",
            flush=True,
        )

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_scene[episode["scenario_id"]].append(episode)
    # Scenes a pass finished out of order still get their full complement, but a
    # truncated final wave can leave one short; only complete scenes are scored.
    complete = {
        scene: runs for scene, runs in by_scene.items() if len(runs) == args.repeats
    }
    per_scene_rate = {
        scene: sum(run["success"] for run in runs) / len(runs)
        for scene, runs in complete.items()
    }
    rates = np.array(sorted(per_scene_rate.values()))
    errors = [
        episode["final_xy_error_m"]
        for episode in episodes
        if not math.isnan(episode["final_xy_error_m"])
    ]

    summary = {
        "scenes_requested": args.scenes,
        "scenes_complete": len(complete),
        "repeats": args.repeats,
        "episodes": len(episodes),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "overall_success": float(np.mean([e["success"] for e in episodes])),
        # The headline: how much of the outcome is the scene rather than the roll.
        "scenes_never_solved": float(np.mean(rates == 0.0)) if len(rates) else None,
        "scenes_always_solved": float(np.mean(rates == 1.0)) if len(rates) else None,
        "scenes_mixed": float(np.mean((rates > 0.0) & (rates < 1.0))) if len(rates) else None,
        "per_scene_rate_std": float(np.std(rates)) if len(rates) else None,
        "median_final_xy_error_m": float(np.median(errors)) if errors else None,
        "config": {
            "checkpoint": str(args.checkpoint),
            "finetuned_checkpoint": (
                None if args.finetuned_checkpoint is None else str(args.finetuned_checkpoint)
            ),
            "scene_seed_base": env_config.seed_base,
            "scene_appearance": env_config.scene_appearance.name,
            "deterministic": args.deterministic,
            "sampling_std": float(config.model.min_sampling_denoising_std),
            "act_steps": env_config.act_steps,
            "horizon_steps": int(config.horizon_steps),
            "max_steps": env_config.max_steps,
            "seed": args.seed,
            "n_envs": args.n_envs,
        },
    }
    print(json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"summary": summary, "per_scene_rate": per_scene_rate, "episodes": episodes},
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
