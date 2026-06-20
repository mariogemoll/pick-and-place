#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Push a local LeRobotDataset to the Hugging Face Hub.

Example:

    cd py
    python scripts/pick_and_place/push_dataset_to_hub.py \
        --root /path/to/pick-and-place-private-1-512 \
        --repo-id YOUR_HF_USERNAME/pick-and-place-private-1-512
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="local dataset root (contains meta/, data/, videos/)"
    )
    parser.add_argument("--repo-id", required=True, help="destination repo id, e.g. username/dataset-name")
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="create/push as a private repo (default: true)",
    )
    parser.add_argument(
        "--push-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the videos/ directory in the upload",
    )
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    print(f"Pushing {args.root} -> {args.repo_id} (private={args.private})")
    dataset.push_to_hub(private=args.private, push_videos=args.push_videos)
    print("Done.")


if __name__ == "__main__":
    main()
