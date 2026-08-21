# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags for the datasets and staged episodes the commands read and write.

The sim recorder and the teleop recorder write the same schema, so a policy can
be trained on both together; these are the knobs that decide where it lands and
how the video is encoded. The offline tools that read a dataset back --
filtering it, splitting it, re-rendering it, measuring it -- name their inputs
and outputs through the same declarations, so a path that means "a LeRobot
dataset root" says so identically everywhere.
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


def add_episodes_root_argument(parser: argparse.ArgumentParser, *, help: str) -> None:
    """Add ``--episodes-root``: the sim recorder's staging area, not a dataset.

    Staged episodes are one directory per episode holding per-frame ground truth,
    which is what the visibility measurements read; a finalized LeRobotDataset has
    thrown that away.
    """
    parser.add_argument("--episodes-root", type=Path, required=True, help=help)


def add_max_episodes_argument(parser: argparse.ArgumentParser, *, help: str) -> None:
    """Add ``--max-episodes``, the smoke-run cap on how much of a set is read."""
    parser.add_argument("--max-episodes", type=int, default=None, help=help)


def add_image_size_argument(parser: argparse.ArgumentParser, *, default: int = 96) -> None:
    """Add ``--image-size``, the square input the image policy's export is built at.

    Shared because a visibility measurement is only about the export it describes:
    measuring at 96 what was exported at 128 answers a question nobody asked.
    """
    parser.add_argument(
        "--image-size",
        type=int,
        default=default,
        help=f"square image size; must be a multiple of 8 (default: {default})",
    )


def add_source_dataset_argument(
    parser: argparse.ArgumentParser, *, help: str = "source LeRobotDataset root"
) -> None:
    """Add ``--src``, the dataset an offline tool reads."""
    parser.add_argument("--src", type=Path, required=True, help=help)
