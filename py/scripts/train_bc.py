#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train the rung-1 behavior-cloning MLP on recorded analytic demonstrations.

Reads every ``episode_*.npz`` under ``--episodes-dir`` into a flat
``(observation, action)`` table and fits a small MLP by MSE regression, writing a
single checkpoint that :mod:`scripts.eval_policy` can roll out in sim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.il.bc import resolve_device, train_bc
from pick_and_place.il.dataset import load_dataset

_DEFAULT_EPISODES = Path(__file__).resolve().parents[1] / "out" / "episodes"
_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "out" / "bc_policy.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=_DEFAULT_EPISODES)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[256, 256],
        help="hidden layer widths (default: 256 256)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="train on every episode, not just successful demos",
    )
    args = parser.parse_args()

    obs, act = load_dataset(args.episodes_dir, successful_only=not args.all_episodes)
    policy = train_bc(
        obs,
        act,
        hidden=tuple(args.hidden),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=resolve_device(args.device),
    )
    policy.save(args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
