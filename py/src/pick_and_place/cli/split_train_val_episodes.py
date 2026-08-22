# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Split a LeRobotDataset into train/val subsets by episode, for BC training.

Episodes (not frames) are the split unit, so no near-duplicate frames leak
between train and val. The split is a deterministic seeded shuffle (see
``--seed``) of the dataset's episode indices, so re-running with the same
``--seed`` always reproduces the same assignment.

Uses ``pick_and_place.data.dataset_subset.write_subset_dataset`` to reindex data and
metadata without re-encoding any video -- see that module's docstring for why
re-encoding would otherwise be unavoidable and lossy for a scattered episode
subset like a val split.

Dry run by default (prints the split sizes); pass ``--write`` to actually
create the two subsets. The source dataset is never modified.

Example:

    pap split-train-val-episodes \
        --src path/to/dataset-success --val-fraction 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from pick_and_place.cli.common import add_seed_argument
from pick_and_place.cli.dataset import (
    add_source_dataset_argument,
    add_val_fraction_argument,
    add_write_argument,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.data.dataset_subset import load_all_episodes, write_subset_dataset


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the train/validation split."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_source_dataset_argument(parser)
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
    add_val_fraction_argument(parser)
    add_seed_argument(
        parser,
        default=0,
        help="seed for the deterministic shuffle that assigns episodes to train/val (default: 0)",
    )
    add_write_argument(parser, help="perform the split")
    return parser


def run(args: argparse.Namespace) -> None:
    """Write the train and validation datasets, or report the split."""
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


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
