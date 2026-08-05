# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags for writing a LeRobot dataset.

The sim recorder and the teleop recorder write the same schema, so a policy can
be trained on both together; these are the knobs that decide where it lands and
how the video is encoded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_TASK = "Pick up the cube and place it at the target."


def add_dataset_arguments(
    parser: argparse.ArgumentParser, *, repo_id: str, vcodec: str, vcodec_help: str
) -> None:
    """Add where the dataset is written and how its video is encoded.

    ``repo_id`` and ``vcodec`` differ by recorder: the sim recorder pins a
    software codec because MuJoCo already owns the GPU, while the teleop recorder
    is free to probe for a hardware encoder.
    """
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="output directory for the LeRobotDataset (default: datasets/<timestamp>)",
    )
    parser.add_argument("--repo-id", default=repo_id, help="dataset repo id stored in metadata")
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="natural-language task instruction saved with every frame",
    )
    parser.add_argument("--vcodec", default=vcodec, help=vcodec_help)
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode video during capture; --no-streaming-encoding falls back to "
        "PNG-then-encode",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="background image-writer threads for PNG-then-encode mode",
    )
