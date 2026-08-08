#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Does the diffusion policy's input noise move its actions, and its outcomes?

The gate DSRL has to clear before a training run is worth paying for. Steering
can only re-weight modes the behavior-cloned policy already has, and every
demonstration in this project comes from a deterministic analytic planner -- so
the policy may simply ignore its noise, in which case there is nothing to steer.

Two modes:

    # Action spread: roll out the base policy, and at each step denoise K extra
    # noise draws from the same state to see how far apart the chunks land.
    python scripts/measure_dsrl_steerability.py spread \
        --config ../config/diffusion_policy/dsrl_so101.yaml \
        --checkpoint <state_500.pt> --normalization <artifact>/normalization.npz

    # Outcome spread: pair R independent scorings of the same scene stream, each
    # produced by `check_dppo_rl_env.py --seed <r>`, and report how often
    # repeats disagree about a scene. That disagreement is the headroom.
    python scripts/measure_dsrl_steerability.py outcomes \
        run-seed1.json run-seed2.json run-seed3.json run-seed4.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pick_and_place.dsrl.steerability import (
    combine_action_spreads,
    load_episode_records,
    measure_action_spread,
    summarize_outcome_spread,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    spread = subparsers.add_parser("spread", help="action spread under resampled noise")
    spread.add_argument("--config", type=Path, required=True)
    spread.add_argument("--checkpoint", type=Path, required=True)
    spread.add_argument("--normalization", type=Path, required=True)
    spread.add_argument("--device", default="cuda:0")
    spread.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    spread.add_argument("--n-envs", type=int, default=8)
    spread.add_argument("--steps", type=int, default=60, help="parallel steps to sample")
    spread.add_argument("--draws", type=int, default=16, help="noise draws per state")
    spread.add_argument("--seed", type=int, default=0)
    spread.add_argument("--output", type=Path, default=None)

    outcomes = subparsers.add_parser("outcomes", help="outcome spread across repeats")
    outcomes.add_argument("scores", type=Path, nargs="+", help="score JSONs to pair")
    outcomes.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run_spread(args: argparse.Namespace) -> dict:
    import hydra
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now(UTC).strftime(pattern), replace=True
    )

    from pick_and_place.dppo_rl.env import EnvConfig
    from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
    from pick_and_place.dsrl.noise_policy import denoise, latent_shape
    from pick_and_place.sim.scene_appearance import parse_appearance

    config = OmegaConf.load(args.config)
    config.base_policy_path = str(args.checkpoint)
    config.normalization_path = str(args.normalization)
    config.device = args.device
    for name in ("DPPO_LOG_DIR", "DPPO_DATA_DIR", "DPPO_BASE_POLICY"):
        os.environ.setdefault(name, "unused-by-this-script")
    OmegaConf.resolve(config)

    env_config = EnvConfig(
        normalization_path=args.normalization,
        image_hw=(
            int(config.shape_meta.obs.rgb.shape[1]),
            int(config.shape_meta.obs.rgb.shape[2]),
        ),
        render_hw=tuple(int(value) for value in config.env.render_hw),
        cond_steps=int(config.cond_steps),
        act_steps=int(config.act_steps),
        control_hz=float(config.env.control_hz),
        max_steps=int(config.env.max_episode_steps),
        seed_base=int(config.env.scene_seed_base),
        scene_appearance=parse_appearance(str(config.env.scene_appearance))[1],
    )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = hydra.utils.instantiate(config.model)
    model.eval()
    horizon_steps, action_dim = latent_shape(model)

    venv = DppoVectorEnv(env_config, args.n_envs, mujoco_gl=args.mujoco_gl)
    spreads = []
    try:
        observation = venv.reset_arg()
        for _ in range(args.steps):
            cond = {
                key: torch.from_numpy(observation[key]).float().to(device)
                for key in ("state", "rgb")
            }
            # One extra denoise per draw at the same state. The first draw is
            # the one actually played, so the rollout stays an ordinary rollout
            # of the base policy and the states are on its own distribution.
            chunks = []
            for _ in range(args.draws):
                noise = torch.randn(
                    (args.n_envs, horizon_steps, action_dim), device=device
                )
                chunks.append(denoise(model, cond, noise).cpu().numpy())
            stacked = np.stack(chunks)
            spreads.append(measure_action_spread(stacked))
            observation, *_ = venv.step(stacked[0][:, : env_config.act_steps])
    finally:
        venv.close()

    combined = combine_action_spreads(spreads)
    return {
        "mode": "spread",
        "draws": args.draws,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "checkpoint": str(args.checkpoint),
        "action_spread": combined.as_dict(),
    }


def main() -> None:
    args = _parse_args()
    if args.mode == "spread":
        payload = _run_spread(args)
    else:
        runs = [load_episode_records(path) for path in args.scores]
        payload = {
            "mode": "outcomes",
            "scores": [str(path) for path in args.scores],
            "outcome_spread": summarize_outcome_spread(runs).as_dict(),
        }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
