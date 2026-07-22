#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export a LeRobot dataset for visual Diffusion Policy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pick_and_place.diffusion_policy_dataset import export_diffusion_policy_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source LeRobot dataset root")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new Diffusion Policy dataset directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="square image size; must be a multiple of 8 (default: 96)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="export only the first N episodes for a smoke run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel video decoding processes (default: 2)",
    )
    args = parser.parse_args()

    manifest = export_diffusion_policy_dataset(
        args.src,
        args.output,
        image_size=args.image_size,
        max_episodes=args.max_episodes,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
