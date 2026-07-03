#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Split a LeRobotDataset into train/val subsets by episode, for BC training.

Episodes (not frames) are the split unit, so no near-duplicate frames leak
between train and val. The split is a deterministic seeded shuffle (see
``--seed``) of the dataset's episode indices, so re-running with the same
``--seed`` always reproduces the same assignment.

Uses ``pick_and_place.dataset_subset.write_subset_dataset`` to reindex data and
metadata without re-encoding any video -- see that module's docstring for why
re-encoding would otherwise be unavoidable and lossy for a scattered episode
subset like a val split.

Dry run by default (prints the split sizes); pass ``--write`` to actually
create the two subsets. The source dataset is never modified.

Example:

    python py/scripts/split_train_val_episodes.py \
        --src path/to/dataset-success --val-fraction 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from pick_and_place.dataset_subset import load_all_episodes, write_subset_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source LeRobotDataset root")
    parser.add_argument(
        "--train-dst",
        type=Path,
        default=None,
        help="train output root (default: <src>-train alongside the source)",
    )
    parser.add_argument(
        "--val-dst",
        type=Path,
        default=None,
        help="val output root (default: <src>-val alongside the source)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="fraction of episodes to hold out for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the deterministic shuffle that assigns episodes to train/val (default: 0)",
    )
    parser.add_argument("--write", action="store_true", help="perform the split")
    args = parser.parse_args()

    train_dst = args.train_dst if args.train_dst is not None else args.src.with_name(f"{args.src.name}-train")
    val_dst = args.val_dst if args.val_dst is not None else args.src.with_name(f"{args.src.name}-val")
    if args.write:
        for dst in (train_dst, val_dst):
            if dst.exists():
                raise SystemExit(f"output {dst} already exists; remove it or pick another destination")

    episodes = load_all_episodes(args.src)
    episode_indices = episodes["episode_index"].tolist()

    shuffled = episode_indices.copy()
    random.Random(args.seed).shuffle(shuffled)
    num_val = round(len(shuffled) * args.val_fraction)
    val_indices = sorted(shuffled[:num_val])
    train_indices = sorted(shuffled[num_val:])

    print(
        f"{args.src}: {len(episode_indices)} episode(s) -> "
        f"{len(train_indices)} train, {len(val_indices)} val (seed={args.seed})"
    )

    if not args.write:
        print(f"\nDry run: would write train to {train_dst}, val to {val_dst}.")
        print("Pass --write to perform the split.")
        return

    write_subset_dataset(args.src, train_dst, f"{args.src.name}-train", train_indices, episodes=episodes)
    print(f"Wrote {len(train_indices)} episode(s) to {train_dst}")

    write_subset_dataset(args.src, val_dst, f"{args.src.name}-val", val_indices, episodes=episodes)
    print(f"Wrote {len(val_indices)} episode(s) to {val_dst}")


if __name__ == "__main__":
    main()
