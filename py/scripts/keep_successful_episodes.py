#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drop failed episodes from a LeRobotDataset, keeping only the successful ones.

``consolidate_datasets.py`` and ``combine_datasets.py`` deliberately keep every
episode, successful or not, so consumers can define "success" themselves. This
script is that consumer: it derives success from the recorded placement points
(``placement_detected`` and ``cube_end``/``target`` within ``--xy-tolerance``,
via ``successful_episode_mask``) and writes a new dataset containing only the
successful ones, using ``pick_and_place.dataset_subset.write_subset_dataset`` to
reindex data and metadata without re-encoding any video (see that module's
docstring for why re-encoding would otherwise be unavoidable and lossy).

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

from pick_and_place.dataset_subset import (
    SUCCESS_XY_TOLERANCE_M,
    load_all_episodes,
    successful_episode_mask,
    write_subset_dataset,
)


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
    parser.add_argument(
        "--xy-tolerance",
        type=float,
        default=SUCCESS_XY_TOLERANCE_M,
        help=f"success placement-XY tolerance in metres (default: {SUCCESS_XY_TOLERANCE_M})",
    )
    parser.add_argument("--write", action="store_true", help="perform the filtering")
    args = parser.parse_args()

    dst_root = args.dst if args.dst is not None else args.src.with_name(f"{args.src.name}-success")
    if args.write and dst_root.exists():
        raise SystemExit(f"output {dst_root} already exists; remove it or pick another --dst")

    episodes = load_all_episodes(args.src)
    success = successful_episode_mask(episodes, args.xy_tolerance).to_numpy()
    kept_indices = episodes["episode_index"].to_numpy()[success].tolist()
    n_failed = len(episodes) - len(kept_indices)

    print(
        f"{args.src}: {len(episodes)} episode(s), {len(kept_indices)} successful, {n_failed} failed"
    )

    if n_failed == 0:
        raise SystemExit("Every episode already succeeded; nothing to filter.")

    if not args.write:
        print(f"\nDry run: would write {len(kept_indices)} episode(s) to {dst_root}.")
        print("Pass --write to perform the filtering.")
        return

    repo_id = args.repo_id or f"{args.src.name}-success"
    write_subset_dataset(args.src, dst_root, repo_id, kept_indices, episodes=episodes)

    print(f"\nWrote {len(kept_indices)} episode(s) to {dst_root}")


if __name__ == "__main__":
    main()
