#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a trained RL hover policy and report success rate / error stats.

DEPRECATED — reference / smoke test only. Drives the throwaway hover milestone
env (``rl.hover_env``), which is **not** on the frozen 31-dim RL contract and
**not** part of the curriculum. See ``docs/rl-curriculum-roadmap.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_and_place.rl.hover_env import ApproachToHoverEnv

_DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "out" / "rl_hover" / "best_model.zip"
_DEFAULT_NORM = Path(__file__).resolve().parents[1] / "out" / "rl_hover" / "vec_normalize_final.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--vec-normalize", type=Path, default=_DEFAULT_NORM)
    parser.add_argument("-n", "--num-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5000)
    args = parser.parse_args()

    env = DummyVecEnv([lambda: ApproachToHoverEnv()])
    env = VecNormalize.load(str(args.vec_normalize), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.checkpoint, env=env, device="auto")

    tip_dists, height_violations, successes = [], [], []
    obs = env.reset()
    ep = 0
    while ep < args.num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if done[0]:
            ep += 1
            ep_info = info[0]
            tip_dists.append(ep_info.get("tip_dist", float("nan")))
            height_violations.append(ep_info.get("height_violation", 0.0))
            successes.append(ep_info.get("success", False))
            if ep % 10 == 0:
                print(f"  {ep}/{args.num_episodes} episodes done...")

    n = len(successes)
    print(f"\n{'='*48}")
    print(f"episodes:            {n}")
    print(f"success rate:        {sum(successes)}/{n}  ({100*sum(successes)/n:.1f}%)")
    print(f"tip dist  median:    {np.nanmedian(tip_dists)*1000:.1f} mm   "
          f"max: {np.nanmax(tip_dists)*1000:.1f} mm")
    print(f"height violations:   {sum(v > 0 for v in height_violations)}/{n}")
    print(f"{'='*48}")

    env.close()


if __name__ == "__main__":
    main()
