#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Split a recorded episode directory into disjoint train/eval pools.

RL training resets from episode snapshots, so a policy evaluated on the same
episodes it trained from could pass by memorizing those scenes. This script
shuffles the successful episodes of one recorded directory with a fixed seed
and divides them into two pools under one parent directory: ``train/`` (the
reset distribution for training) and ``eval/`` (held-out episodes that all
evaluation resets are drawn from, never seen during training).

Successful episodes beyond the two requested pool sizes are discarded, so the
pools can be given round sizes. Files are renumbered sequentially per pool, and
each pool gets a ``manifest.json`` mapping every file back to its source
recording.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _successful_episodes(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz found in {directory}")
    episodes = []
    for path in paths:
        with np.load(path, allow_pickle=True) as record:
            if bool(record["success"]):
                episodes.append(path)
    return episodes


def _write_pool(episodes: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True)
    manifest = []
    for i, source in enumerate(episodes):
        name = f"episode_{i:05d}.npz"
        shutil.copy2(source, out_dir / name)
        manifest.append({"file": name, "source": str(source)})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "out" / "episodes",
        help="recorded episode directory to split (default: py/out/episodes)",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=400,
        help="episodes held out for the eval pool (default 400)",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        help="episodes for the train pool (default: all not held out for eval)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "episode_pools",
        help="parent directory for train/ and eval/ (default: py/out/episode_pools)",
    )
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed (default 0)")
    args = parser.parse_args()

    train_dir = args.out / "train"
    eval_dir = args.out / "eval"
    for directory in (train_dir, eval_dir):
        if directory.exists():
            parser.error(f"{directory} already exists — remove it or choose another --out")

    episodes = _successful_episodes(args.source)
    train_count = (
        args.train_count if args.train_count is not None else len(episodes) - args.eval_count
    )
    if args.eval_count < 1 or train_count < 1 or args.eval_count + train_count > len(episodes):
        parser.error(
            f"need eval-count >= 1, train-count >= 1, and their sum <= "
            f"{len(episodes)} (successful episodes in {args.source})"
        )
    order = np.random.default_rng(args.seed).permutation(len(episodes))
    eval_episodes = [episodes[i] for i in sorted(order[: args.eval_count])]
    train_episodes = [
        episodes[i] for i in sorted(order[args.eval_count : args.eval_count + train_count])
    ]

    _write_pool(train_episodes, train_dir)
    _write_pool(eval_episodes, eval_dir)
    print(f"train: {len(train_episodes)} episodes -> {train_dir}")
    print(f"eval:  {len(eval_episodes)} episodes -> {eval_dir}")


if __name__ == "__main__":
    main()
