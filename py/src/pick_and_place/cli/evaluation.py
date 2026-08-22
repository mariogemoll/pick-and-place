# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags for the frozen scenario suites: which one, how it is sharded, how it is named.

A suite is the whole reason two policies' numbers can be compared, so the
commands that generate one, rewrite one, and score against one have to mean the
same thing by ``--manifest``, ``--limit`` and ``--offset``. Sharding especially:
``--offset`` is applied before ``--limit`` here and would be a silently
different suite if some other command applied them the other way round.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.core.paths import REPO_ROOT

EVALUATION_DIR = REPO_ROOT / "config" / "evaluation"


def add_manifest_argument(parser: argparse.ArgumentParser, *, default: Path) -> None:
    """Add ``--manifest``, the frozen scenario suite a command reads."""
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default,
        help=f"frozen scenario manifest (default: {_repository_relative(default)})",
    )


def _repository_relative(path: Path) -> str:
    """Name a committed default by its path in the repository, not on this machine.

    Help text is read far more often than it is generated, and an absolute
    checkout path in it is noise everywhere but the box that printed it.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def add_shard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--limit`` and ``--offset``, which cut one suite into concurrent slices.

    Scenarios are independent and each carries its own seed, so the slices stay
    comparable and their union is the result a serial run would have produced --
    which is what ``merge_evaluation_shards.py`` reassembles. It also means a
    crashed shard costs only its own slice.
    """
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N scenarios, for a non-headline wiring check",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip the first N scenarios, applied before --limit. Together the two shard one "
        "suite across concurrent workers; the shards stay comparable because each scenario is "
        "independent and carries its own seed, and merge_evaluation_shards.py reassembles them",
    )


def add_suite_name_argument(parser: argparse.ArgumentParser, *, help: str) -> None:
    """Add ``--suite``, the name a written manifest carries in its own header.

    The name reaches stored results, so what a suite is called is part of what a
    number means; ``help`` says where each command's default comes from.
    """
    parser.add_argument("--suite", default=None, help=help)


def add_scenarios_per_file_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--scenarios-per-file``, which spreads one suite over several files."""
    parser.add_argument(
        "--scenarios-per-file",
        type=int,
        default=None,
        help="write a sharded manifest with at most this many scenarios per compressed file",
    )
