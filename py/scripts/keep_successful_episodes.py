#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drop failed episodes from a LeRobotDataset, keeping only ``success == True``.

``consolidate_datasets.py`` and ``combine_datasets.py`` deliberately keep every
episode, successful or not, so consumers can define "success" themselves. This
script is that consumer: it reads the ``success`` column already recorded on
each episode and writes a new dataset containing only the successful ones,
using ``pick_and_place.dataset_subset.write_subset_dataset`` to reindex data
and metadata without re-encoding any video (see that module's docstring for
why re-encoding would otherwise be unavoidable and lossy).

Dry run by default (prints how many episodes would be dropped); pass
``--write`` to actually create the filtered copy. The source dataset is never
modified.

Example:

    python py/scripts/keep_successful_episodes.py \
        --src path/to/dataset --dst path/to/dataset-success
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.dataset_subset import load_all_episodes, write_subset_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source LeRobotDataset root")
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="output dataset root (default: <src>-success alongside the source)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="repo id for the output dataset (default: <src dir name>-success)",
    )
    parser.add_argument("--write", action="store_true", help="perform the filtering")
    args = parser.parse_args()

    dst_root = args.dst if args.dst is not None else args.src.with_name(f"{args.src.name}-success")
    if args.write and dst_root.exists():
        raise SystemExit(f"output {dst_root} already exists; remove it or pick another --dst")

    episodes = load_all_episodes(args.src)
    success = episodes.set_index("episode_index")["success"]
    failed_indices = success.index[~success].tolist()

    print(
        f"{args.src}: {len(success)} episode(s), {len(success) - len(failed_indices)} successful, "
        f"{len(failed_indices)} failed"
    )

    if not failed_indices:
        raise SystemExit("Every episode already succeeded; nothing to filter.")

    kept_indices = success.index[success].tolist()

    if not args.write:
        print(f"\nDry run: would write {len(kept_indices)} episode(s) to {dst_root}.")
        print("Pass --write to perform the filtering.")
        return

    repo_id = args.repo_id or f"{args.src.name}-success"
    write_subset_dataset(args.src, dst_root, repo_id, kept_indices, episodes=episodes)

    print(f"\nWrote {len(kept_indices)} episode(s) to {dst_root}")


if __name__ == "__main__":
    main()
