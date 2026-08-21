# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags that belong to no one subsystem: where a command writes, and its seed.

``--output`` was declared by hand in fifteen scripts and ``--seed`` in seven,
which is how six of them ended up with no help text at all and one with a
``--seed`` that was never given a type. Neither flag has a subsystem to live in,
so they live here.

Every function takes its help text rather than supplying one: what a command
writes is the one thing about ``--output`` that is not shared.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_output_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str,
    required: bool = False,
    default: Path | None = None,
    short: bool = False,
) -> None:
    """Add ``--output``, the path a command writes its result to.

    ``help`` is mandatory: a path flag that does not say what lands there is the
    one thing a reader cannot recover from the source.
    """
    names = ["-o", "--output"] if short else ["--output"]
    parser.add_argument(*names, type=Path, default=default, required=required, help=help)


def add_out_dir_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str,
    required: bool = False,
    default: Path | None = None,
) -> None:
    """Add ``--out-dir``, for the commands that write a directory of files."""
    parser.add_argument("--out-dir", type=Path, default=default, required=required, help=help)


def add_seed_argument(
    parser: argparse.ArgumentParser, *, default: int | None, help: str, flag: str = "--seed"
) -> None:
    """Add the RNG seed that makes a command's draws repeatable.

    ``default`` is deliberately not shared: ``None`` means "draw one" for the
    commands that sample a scene, while the offline ones pin a fixed seed so two
    runs of the same command agree.
    """
    parser.add_argument(flag, type=int, default=default, help=help)


def add_output_size_arguments(
    parser: argparse.ArgumentParser, *, width: int, height: int, noun: str
) -> None:
    """Add ``--width`` and ``--height``: the pixel size of what a command produces.

    Not to be confused with :func:`pick_and_place.cli.rig.add_capture_size_arguments`,
    which asks a camera for a resolution, or with
    :func:`pick_and_place.cli.scene.add_render_size_arguments`, which sets the
    offscreen MuJoCo render that a frame is then reduced from. All three spell
    themselves differently on purpose.
    """
    parser.add_argument(
        "--width", type=int, default=width, help=f"{noun} width in pixels (default: {width})"
    )
    parser.add_argument(
        "--height", type=int, default=height, help=f"{noun} height in pixels (default: {height})"
    )
