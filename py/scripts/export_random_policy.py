#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export a randomly-initialized (untrained) PPO checkpoint on PickPlaceEnv.

Just for fun / a baseline to compare against: builds the same
``PPO("MlpPolicy", ...)`` architecture the curriculum trains (matching
``curricula/pick_place.yaml``'s ``net_arch``), but never calls ``.learn()`` — the
saved weights are whatever torch's default initialization produced. Pair with
``view_rl_policy.py`` to watch a random policy flail at the cube.

Writes ``model_final.zip`` + ``vec_normalize_final.pkl`` into ``--out``, named so
``view_rl_policy.py`` resolves the VecNormalize stats automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_and_place.rl import curriculum as cur
from pick_and_place.rl.pick_place_env import PickPlaceEnv

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "out" / "rl_random"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0, help="seeds the random weight init")
    args = parser.parse_args()

    env = VecNormalize(
        DummyVecEnv([lambda: PickPlaceEnv()]), norm_obs=True, norm_reward=True, clip_obs=10.0
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out / cur.FINAL_MODEL_STEM))
    env.save(str(args.out / cur.FINAL_VECNORMALIZE_NAME))

    checkpoint = cur.final_model_path(args.out)
    print(f"Untrained checkpoint written to {args.out}")
    print(f"View it with:\n  .venv/bin/python scripts/view_rl_policy.py --checkpoint {checkpoint}")


if __name__ == "__main__":
    main()
