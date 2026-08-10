#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a flow policy in its PPO fine-tuning environment, and score it.

Two jobs, deliberately in one script, because they need the identical stack:

- the gate before spending GPU time on RL. It drives the same :class:`FlowPPO`
  model, sampler and vectorized environment the fine-tuner uses, without
  gradients, and reports the oracle's milestone rates. A pipeline mismatch and
  "RL had nothing to learn from" look identical once training starts, and cost
  the same to discover.
- the measurement afterwards. Every episode is recorded with its scenario id, so
  two runs over the same seed stream pair scene by scene and the difference
  between a base and a fine-tuned checkpoint can be tested rather than eyeballed.

With ``--stochastic`` it samples the way rollout collection does. That success
rate is the one that decides whether the exploration noise leaves a reward
signal to learn from at all: too much and every episode fails, too little and
the scene draw is the only source of variance.

Example (from the repository root):

    PYTHONPATH=third_party/dppo MUJOCO_GL=egl python py/scripts/check_flow_rl_env.py \\
      --config config/flow_policy/ft_ppo_so101_flow.yaml \\
      --checkpoint $PAP_DATA_ROOT/outputs/<run>/checkpoint.pt \\
      --export $PAP_DATA_ROOT/outputs/<export> \\
      --episodes 20 --n-envs 4 --device cpu \\
      --output outputs/flow-rl-env-check/summary.json
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
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="the behavior-cloned flow checkpoint"
    )
    parser.add_argument(
        "--export", type=Path, required=True, help="the export directory it was trained against"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument(
        "--scene-seed-base",
        type=int,
        default=None,
        help="defaults to the config's training stream, so this checks the scenes "
        "fine-tuning will actually visit. Selection uses 6,000,000; 7,000,000 is "
        "the one-time validation stream and nothing may be tuned against it.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=None,
        help="a fine-tuning checkpoint to load over the behavior-cloned one. Its "
        "state dict holds actor/actor_ft/critic rather than a bare velocity "
        "field, so it is loaded after the model is built. Point --checkpoint at "
        "the behavior-cloned policy either way: it defines the architecture.",
    )
    parser.add_argument(
        "--act-steps",
        type=int,
        default=None,
        help="executed actions per query, overriding the config's. The chunk "
        "schedule is an operating point rather than a property of the checkpoint.",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=None,
        help="Euler integration steps, overriding the config's. Changing it "
        "changes the sampler the fine-tuned chain is defined over, so a "
        "fine-tuned checkpoint must be scored at the value it was trained with.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample through the SDE the way rollout collection does, instead of "
        "integrating the ODE. This is the policy PPO actually learns from.",
    )
    parser.add_argument(
        "--sampling-noise-scale",
        type=float,
        default=None,
        help="override the SDE noise scale. Its units are normalized actions, so "
        "on this task's absolute joint commands 0.1 is roughly 9 degrees on "
        "joint 1 at the start of the chain. Only meaningful with --stochastic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds torch's sampling. The chain starts from a random latent even "
        "when integration is deterministic, so fixing this is what makes a "
        "paired comparison differ only by weights.",
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
    from pick_and_place.dppo_rl.observations import FlowStateObservation
    from pick_and_place.dppo_rl.vector_env import DppoVectorEnv

    config = OmegaConf.load(args.config)
    config.base_policy_path = str(args.checkpoint)
    config.flow_export_path = str(args.export)
    config.device = args.device
    if args.act_steps is not None:
        config.act_steps = args.act_steps
    if args.flow_steps is not None:
        config.flow_steps = args.flow_steps
    if args.sampling_noise_scale is not None:
        config.model.sampling_noise_scale = args.sampling_noise_scale
    # The checkpoint and export come from the command line here, and nothing is
    # written to a run directory, so the launcher's environment variables only
    # need to exist for interpolation to succeed.
    for name in ("DPPO_LOG_DIR", "DPPO_DATA_DIR", "DPPO_BASE_POLICY"):
        os.environ.setdefault(name, "unused-by-this-check")
    OmegaConf.resolve(config)

    env_config = EnvConfig(
        observation=FlowStateObservation(export_dir=args.export),
        cond_steps=int(config.cond_steps),
        act_steps=int(config.act_steps),
        control_hz=float(config.env.control_hz),
        max_steps=int(config.env.max_episode_steps),
        seed_base=int(
            args.scene_seed_base
            if args.scene_seed_base is not None
            else config.env.scene_seed_base
        ),
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
                cond = {"state": torch.from_numpy(observation["state"]).float().to(device)}
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
    # Time to placement, over the episodes that placed. Under the dense reward an
    # episode's return is the ticks it spent settled, so this is what the
    # objective actually pays for -- and unlike the success rate it is nowhere
    # near saturated. It only means "time to placement" because this check runs
    # with the terminating reward, which ends the episode at the placement.
    settled_steps = [
        episode["control_steps"] for episode in episodes if episode["success"]
    ]
    summary = {
        "episodes": total,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "rates": {
            name: sum(episode[name] for episode in episodes) / total
            for name in milestone_names
        },
        "control_steps_to_settle": {
            "median": float(np.median(settled_steps)) if settled_steps else None,
            "p10": float(np.percentile(settled_steps, 10)) if settled_steps else None,
            "p90": float(np.percentile(settled_steps, 90)) if settled_steps else None,
            "fastest": min(settled_steps) if settled_steps else None,
            "counted": len(settled_steps),
        },
        "placed_within_6cm": sum(error <= 0.06 for error in errors) / total,
        "median_final_xy_error_m": float(np.median(errors)) if errors else None,
        "config": {
            "checkpoint": str(args.checkpoint),
            "export": str(args.export),
            "finetuned_checkpoint": (
                None if args.finetuned_checkpoint is None else str(args.finetuned_checkpoint)
            ),
            "act_steps": env_config.act_steps,
            "control_hz": env_config.control_hz,
            "max_steps": env_config.max_steps,
            "scene_seed_base": env_config.seed_base,
            "flow_steps": int(config.model.flow_steps),
            "ft_denoising_steps": int(config.model.ft_denoising_steps),
            "stochastic": args.stochastic,
            "sampling_noise_scale": float(config.model.sampling_noise_scale),
            "seed": args.seed,
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
