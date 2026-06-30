#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Smoke-test the reverse-curriculum env: API conformance + scripted-replay reward.

Two checks:

1. Gymnasium's ``check_env`` validates the obs/action spaces, reset/step
   signatures, and dtypes.

2. A scripted-replay rollout: reset at a stage, then feed the *recorded* commanded
   set points from the reset frame onward back in as actions. A faithful env must
   reproduce the demonstrated motion and fire the success oracle — this validates
   the whole obs -> action -> reward -> reset loop against ground truth before any
   policy is trained on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from pick_and_place.rl import CURRICULUM_PHASES, EpisodePool, ReverseCurriculumEnv


def _replay_rollout(env: ReverseCurriculumEnv, pool_dir: Path, seed: int) -> dict:
    """Reset, then replay the source episode's commanded stream as actions."""
    _, info = env.reset(seed=seed)
    record = np.load(pool_dir / info["source"], allow_pickle=True)
    commanded = record["commanded"]
    start = info["reset_frame"]

    reward_total = 0.0
    success = False
    steps = 0
    # Feed the demonstrator's next set points; clamp to the trajectory's end.
    for frame in range(start + 1, len(commanded)):
        _, reward, terminated, truncated, step_info = env.step(commanded[frame])
        reward_total += reward
        steps += 1
        if terminated or truncated:
            success = bool(step_info["success"])
            break
    return {
        "source": info["source"],
        "phase": info["phase"],
        "reset_frame": start,
        "steps": steps,
        "reward": reward_total,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "episodes",
        help="directory of recorded episode_*.npz (default: py/out/episodes)",
    )
    parser.add_argument("--stage", type=int, default=0, help="curriculum stage to test")
    parser.add_argument(
        "--rollouts", type=int, default=10, help="scripted-replay rollouts to run"
    )
    args = parser.parse_args()

    pool = EpisodePool(args.pool)
    print(
        f"pool: {len(pool)} successful episodes "
        f"({pool.skipped_failures} skipped), phases {pool.phase_names}"
    )

    env = ReverseCurriculumEnv(pool, stage=args.stage)
    print("running gymnasium check_env ...")
    check_env(env, skip_render_check=True)
    print("check_env passed")

    print(
        f"\nscripted-replay rollouts at stage {args.stage} "
        f"(phase {CURRICULUM_PHASES[args.stage]!r}):"
    )
    successes = 0
    for i in range(args.rollouts):
        result = _replay_rollout(env, args.pool, seed=i)
        successes += result["success"]
        print(
            f"  [{i}] {result['source']} frame {result['reset_frame']:>4} "
            f"-> {result['steps']:>4} steps, reward {result['reward']:.0f}, "
            f"{'SUCCESS' if result['success'] else 'miss'}"
        )
    print(f"\n{successes}/{args.rollouts} scripted replays reached the success oracle")
    env.close()


if __name__ == "__main__":
    main()
