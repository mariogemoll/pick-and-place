#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Finalize exactly N successful staged simulation episodes into one dataset.

``record_sim.py`` writes one independently finalized dataset per attempted
episode and can be run repeatedly against the same destination to top up that
staging area. This command is the separate commit phase: it derives placement
success from the recorded metadata, selects the first requested number in
global-index order, and aggregates them without re-encoding video.

The command is a dry run unless ``--write`` is passed. On a successful write,
the selected episode directories are removed by default; failed and excess
episodes remain staged for inspection or a different final dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.dataset_subset import SUCCESS_XY_TOLERANCE_M
from pick_and_place.sim_dataset_staging import (
    episode_staging_root,
    find_episode_datasets,
    merge_episodes,
    staged_episode_dirs,
    successful_episode_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="final dataset destination; staged episodes live at <root>_episodes",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="exact number of successful episodes to merge",
    )
    parser.add_argument(
        "--repo-id",
        default="local/pick-and-place-so101-sim",
        help="repository id stored in the finalized dataset",
    )
    parser.add_argument(
        "--xy-tolerance",
        type=float,
        default=SUCCESS_XY_TOLERANCE_M,
        help=f"maximum successful placement XY error in metres (default: {SUCCESS_XY_TOLERANCE_M})",
    )
    parser.add_argument(
        "--keep-episodes",
        action="store_true",
        help="retain selected per-episode datasets after a successful merge",
    )
    parser.add_argument("--write", action="store_true", help="perform the final merge")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.xy_tolerance <= 0.0:
        parser.error("--xy-tolerance must be positive")

    episodes_root = episode_staging_root(args.dataset_root)
    staged = staged_episode_dirs(episodes_root)
    complete = find_episode_datasets(episodes_root)
    successful = successful_episode_datasets(complete, args.xy_tolerance)
    selected = successful[: args.episodes]

    print(f"Staging root: {episodes_root}")
    print(
        f"Found {len(staged)} attempted, {len(complete)} complete, "
        f"and {len(successful)} successful episode(s)."
    )
    if len(successful) < args.episodes:
        needed = args.episodes - len(successful)
        raise SystemExit(
            f"Need {needed} more successful episode(s); run record_sim.py again "
            "against the same --dataset-root, then retry finalization."
        )

    excess = len(successful) - len(selected)
    print(
        f"Ready to merge exactly {len(selected)} successful episode(s) into "
        f"{args.dataset_root}; {excess} successful episode(s) will remain staged."
    )
    if not args.write:
        print("Dry run: pass --write to perform the merge.")
        return
    if args.dataset_root.exists():
        raise SystemExit(f"output already exists: {args.dataset_root}")

    merge_episodes(
        selected,
        output_root=args.dataset_root,
        output_repo_id=args.repo_id,
        keep_episodes=args.keep_episodes,
    )


if __name__ == "__main__":
    main()
