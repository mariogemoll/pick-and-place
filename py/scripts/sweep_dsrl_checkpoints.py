#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Score many DSRL checkpoints against their base on one fixed scene set.

`check_dppo_rl_env.py` scores one policy per invocation, which means rebuilding
32 MuJoCo workers and reloading the diffusion policy every time. Sweeping a
run's checkpoints that way spends most of its wall clock on startup. This loads
the base policy once, builds the vectorized environment once, and then walks the
checkpoints through it -- including the base itself, so every comparison is
paired scene for scene by construction rather than by matching seeds afterwards.

Why a sweep at all: the DPPO strand established that checkpoint cadence matters
as much as seed count here. Across six seeds, four had a significantly-positive
checkpoint but at seed-specific iterations, and a sweep scoring only the
pre-registered one would have called three of them failures.

    python scripts/sweep_dsrl_checkpoints.py \\
        --config ../config/diffusion_policy/dsrl_so101.yaml \\
        --checkpoint <base state_500.pt> --normalization <artifact>/normalization.npz \\
        --actors <run>/dsrl --episodes 256 --scene-seed-base 6000000 \\
        --output sweep.json

`--noise-scale` scores the *base* policy with its latent draw scaled, with no
actor at all. That measures how much of the reachable behaviour lies outside the
standard normal shell the policy was trained under, which is the ceiling any
particular `action_magnitude` is competing for.
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

from pick_and_place.spec.action_encoding import read_action_encoding


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="the base policy")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument(
        "--actors",
        type=Path,
        default=None,
        help="directory of DSRL state_*.pt files, or a single file",
    )
    parser.add_argument(
        "--itrs",
        default=None,
        help="comma-separated iterations to score; default is every checkpoint found",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        nargs="*",
        default=None,
        help="score the base with its latent draw multiplied by each of these",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=0,
        help=(
            "instead of playing the actor's mode, sample this many latent actions "
            "and play the one the trained critic rates highest. The steerability "
            "gate measured 31 percent of scenes flipping outcome on the noise draw "
            "alone, with a per-scene hindsight ceiling of 0.827 against a 0.705 "
            "base; a single-shot actor cannot reach that, but a critic that can "
            "rank candidates at the state might. Only valid for an actor trained "
            "with --observable-critic: a privileged critic reads simulator state "
            "that does not exist at deployment, so ranking with it would not be a "
            "policy."
        ),
    )
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--scene-seed-base", type=int, default=6_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="do not score the base policy first (it is the pairing reference)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mcnemar(base: dict[str, bool], other: dict[str, bool]) -> dict[str, float]:
    """Exact two-sided McNemar over the scenarios both policies covered."""
    shared = sorted(set(base) & set(other))
    fixed = sum(1 for s in shared if other[s] and not base[s])
    broke = sum(1 for s in shared if base[s] and not other[s])
    total = fixed + broke
    if total == 0:
        p = 1.0
    else:
        # Two-sided exact binomial against p = 0.5.
        tail = sum(math.comb(total, k) for k in range(min(fixed, broke) + 1))
        p = min(1.0, 2.0 * tail / (2.0**total))
    return {
        "n": len(shared),
        "base_success": sum(base[s] for s in shared) / len(shared) if shared else float("nan"),
        "success": sum(other[s] for s in shared) / len(shared) if shared else float("nan"),
        "fixed": fixed,
        "broke": broke,
        "mcnemar_p": p,
    }


def main() -> None:
    args = _parse_args()

    import hydra
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now(UTC).strftime(pattern), replace=True
    )

    from pick_and_place.dppo_rl.env import EnvConfig
    from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
    from pick_and_place.dsrl.noise_policy import denoise, latent_shape, visual_features
    from pick_and_place.dsrl.sac import LatentActor, SacConfig, TwinCritic
    from pick_and_place.variants.appearance import parse_appearance

    with np.load(args.normalization) as bounds:
        action_encoding = read_action_encoding(bounds)

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
        render_hw=tuple(int(v) for v in config.env.render_hw),
        cond_steps=int(config.cond_steps),
        act_steps=int(config.act_steps),
        control_hz=float(config.env.control_hz),
        max_steps=int(config.env.max_episode_steps),
        seed_base=args.scene_seed_base,
        scene_appearance=parse_appearance(str(config.env.scene_appearance))[1],
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = hydra.utils.instantiate(config.model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    horizon_steps, action_dim = latent_shape(model)

    # What to score, in order. Each entry is (label, callable or None).
    if args.actors is None:
        paths: list[Path] = []
    elif args.actors.is_dir():
        paths = sorted(
            args.actors.glob("state_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
    else:
        paths = [args.actors]
    if args.itrs:
        wanted = {int(value) for value in args.itrs.split(",")}
        paths = [p for p in paths if int(p.stem.split("_")[1]) in wanted]

    venv = DppoVectorEnv(env_config, args.n_envs, mujoco_gl=args.mujoco_gl)
    results: dict[str, dict] = {}

    def run(label: str, action_of) -> dict[str, bool]:
        """Roll `--episodes` episodes, returning scenario id -> success."""
        started = time.perf_counter()
        outcomes: dict[str, bool] = {}
        episodes: list[dict] = []
        torch.manual_seed(args.seed)
        # The scene stream is endless and only advances, so without this each
        # policy after the first would be scored on different scenes and the
        # comparison would not be paired at all.
        venv.rewind()
        observation = venv.reset_arg()
        while len(episodes) < args.episodes:
            with torch.no_grad():
                cond = {
                    key: torch.from_numpy(observation[key]).float().to(device)
                    for key in ("state", "rgb")
                }
                actions = action_of(cond)[:, : env_config.act_steps].cpu().numpy()
            observation, _, terminated, truncated, infos = venv.step(actions)
            for index, done in enumerate(terminated | truncated):
                if not done:
                    continue
                summary = infos[index]["episode"]
                episodes.append({
                    "scenario_id": summary.scenario_id,
                    "success": bool(summary.success),
                    "control_steps": summary.control_steps,
                    **summary.milestones,
                })
        episodes = episodes[: args.episodes]
        for episode in episodes:
            outcomes.setdefault(episode["scenario_id"], episode["success"])
        rate = sum(e["success"] for e in episodes) / len(episodes)
        elapsed = time.perf_counter() - started
        results[label] = {
            "episodes": len(episodes),
            "success_rate": rate,
            "elapsed_s": round(elapsed, 1),
            "outcomes": outcomes,
        }
        print(f"{label}: {rate:.3f} over {len(episodes)} episodes ({elapsed:.0f}s)", flush=True)
        return outcomes

    base_outcomes: dict[str, bool] = {}
    try:
        if not args.skip_base:
            base_outcomes = run(
                "base", lambda cond: model(cond=cond, deterministic=True).trajectories
            )

        for scale in args.noise_scale or []:
            def scaled(cond, scale=scale):
                noise = torch.randn(
                    (cond["state"].shape[0], horizon_steps, action_dim), device=device
                )
                return denoise(model, cond, noise * scale)

            run(f"base-noise-{scale}", scaled)

        for path in paths:
            state = torch.load(path, map_location=args.device, weights_only=True)
            actor_config = SacConfig(**{**state["config"], "device": args.device})
            actor = LatentActor(actor_config).to(device)
            actor.load_state_dict(state["actor"])
            actor.eval()
            itr = int(path.stem.split("_")[1])

            if args.best_of_n < 2:
                def steered(cond, actor=actor):
                    latent = actor.act(visual_features(model, cond), deterministic=True)
                    return denoise(model, cond, latent.view(-1, horizon_steps, action_dim))

                run(f"itr-{itr}", steered)
                continue

            if actor_config.critic_feature_dim != actor_config.actor_feature_dim:
                raise SystemExit(
                    f"{path} was trained with a privileged critic "
                    f"(critic dim {actor_config.critic_feature_dim} != actor dim "
                    f"{actor_config.actor_feature_dim}). Ranking candidates with it "
                    "would use simulator state that does not exist at deployment, so "
                    "the result would not be a policy. Retrain with "
                    "--observable-critic."
                )
            critic = TwinCritic(actor_config).to(device)
            critic.load_state_dict(state["critic"])
            critic.eval()

            def ranked(cond, actor=actor, critic=critic):
                """Sample n candidate noises, play the one the critic likes best."""
                features = visual_features(model, cond)
                batch = features.shape[0]
                repeated = features.repeat_interleave(args.best_of_n, dim=0)
                latents, _ = actor(repeated)
                # Same features for actor and critic, which is what makes this a
                # deployable ranking rather than an oracle.
                values = critic(repeated, latents).min(dim=0).values
                best = values.view(batch, args.best_of_n).argmax(dim=1)
                chosen = latents.view(batch, args.best_of_n, -1)[
                    torch.arange(batch, device=device), best
                ]
                return denoise(model, cond, chosen.view(-1, horizon_steps, action_dim))

            run(f"itr-{itr}-best{args.best_of_n}", ranked)
    finally:
        venv.close()

    comparisons = {}
    if base_outcomes:
        for label, entry in results.items():
            if label == "base":
                continue
            comparisons[label] = _mcnemar(base_outcomes, entry["outcomes"])

    payload = {
        "config": {
            "base_checkpoint": str(args.checkpoint),
            "actors": None if args.actors is None else str(args.actors),
            "episodes": args.episodes,
            "scene_seed_base": args.scene_seed_base,
            "seed": args.seed,
            "action_encoding": action_encoding.value,
            "n_envs": args.n_envs,
        },
        "results": {
            label: {k: v for k, v in entry.items() if k != "outcomes"}
            for label, entry in results.items()
        },
        "outcomes": {label: entry["outcomes"] for label, entry in results.items()},
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n=== paired against the base ===", flush=True)
    for label, comparison in sorted(
        comparisons.items(), key=lambda kv: -kv[1]["success"]
    ):
        delta = comparison["success"] - comparison["base_success"]
        print(
            f"{label:16} {comparison['success']:.3f} "
            f"({delta:+.3f})  fixed {comparison['fixed']:3d} / broke {comparison['broke']:3d}  "
            f"p = {comparison['mcnemar_p']:.4g}",
            flush=True,
        )
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
