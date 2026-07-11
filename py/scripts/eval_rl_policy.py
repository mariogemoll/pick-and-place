#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a saved PPO policy under a reset stage / reward profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_and_place.rl import CURRICULUM_PHASES, REWARD_PROFILES, ReverseCurriculumEnv


def _make_env(
    pool: Path,
    stage: int,
    phase_fraction: float,
    phase_end_fraction: float | None,
    reward_profile: str,
    seed: int,
):
    def factory():
        env = ReverseCurriculumEnv(
            pool,
            stage=stage,
            phase_fraction=phase_fraction,
            phase_end_fraction=phase_end_fraction,
            reward_profile=reward_profile,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "episode_pools" / "eval",
        help=(
            "directory of recorded episode_*.npz to reset from (default: the "
            "held-out pool py/out/episode_pools/eval; see split_episode_pool.py)"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "rl" / "stage0",
        help="directory containing latest.zip and vecnormalize.pkl",
    )
    parser.add_argument("--stage", type=int, default=0, help="curriculum stage to evaluate")
    parser.add_argument(
        "--reward-profile",
        choices=REWARD_PROFILES,
        default="carry-drop",
        help="reward profile / skill objective (default: carry-drop)",
    )
    parser.add_argument(
        "--phase-fraction",
        type=float,
        default=0.0,
        help="start this fraction into the selected phase (default 0)",
    )
    parser.add_argument(
        "--phase-end-fraction",
        type=float,
        help=(
            "end reset sampling at this fraction through the selected phase instead "
            "of sampling through trajectory end"
        ),
    )
    parser.add_argument("--episodes", type=int, default=100, help="evaluation episodes")
    parser.add_argument("--seed", type=int, default=100_000, help="evaluation seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample from the policy instead of deterministic mean actions",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="ignore the checkpoint and sample random actions as a baseline",
    )
    args = parser.parse_args()

    if not 0 <= args.stage < len(CURRICULUM_PHASES):
        parser.error(f"--stage must be in 0..{len(CURRICULUM_PHASES) - 1}")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.phase_end_fraction is not None:
        if not 0.0 <= args.phase_end_fraction <= 1.0:
            parser.error("--phase-end-fraction must be in [0, 1]")
        if args.phase_end_fraction < args.phase_fraction:
            parser.error("--phase-end-fraction must be >= --phase-fraction")

    model_path = args.checkpoint_dir / "latest.zip"
    vecnormalize_path = args.checkpoint_dir / "vecnormalize.pkl"
    if not args.random_actions:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not vecnormalize_path.exists():
            raise FileNotFoundError(vecnormalize_path)

    env = DummyVecEnv([
        _make_env(
            args.pool,
            args.stage,
            args.phase_fraction,
            args.phase_end_fraction,
            args.reward_profile,
            args.seed,
        )
    ])
    if args.random_actions:
        model = None
    else:
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(model_path), env=env)

    successes = 0
    collisions = 0
    out_of_bounds = 0
    truncations = 0
    lengths: list[int] = []
    returns: list[float] = []

    for episode in range(args.episodes):
        obs = env.reset()
        done = np.array([False])
        total = 0.0
        steps = 0
        last_info = {}
        while not done[0]:
            if model is None:
                action = np.array([env.action_space.sample()])
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, done, infos = env.step(action)
            total += float(reward[0])
            steps += 1
            last_info = infos[0]

        successes += bool(last_info.get("success", False))
        collisions += bool(last_info.get("collision", False))
        out_of_bounds += bool(last_info.get("out_of_bounds", False))
        truncations += bool(last_info.get("TimeLimit.truncated", False))
        lengths.append(steps)
        returns.append(total)
        print(
            f"[{episode:03d}] steps={steps:3d} return={total:.1f} "
            f"{'SUCCESS' if last_info.get('success') else 'miss'}"
        )

    env.close()
    print(
        f"\nstage {args.stage} ({CURRICULUM_PHASES[args.stage]!r}) "
        f"reward {args.reward_profile!r} "
        f"{'random-action' if args.random_actions else 'stochastic' if args.stochastic else 'deterministic'} eval"
    )
    print(f"successes:     {successes}/{args.episodes} ({successes / args.episodes:.1%})")
    print(f"collisions:    {collisions}/{args.episodes}")
    print(f"out_of_bounds: {out_of_bounds}/{args.episodes}")
    print(f"truncations:   {truncations}/{args.episodes}")
    print(f"mean return:   {float(np.mean(returns)):.3f}")
    print(f"mean length:   {float(np.mean(lengths)):.1f} steps")


if __name__ == "__main__":
    main()
