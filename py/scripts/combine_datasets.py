#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Merge same-schema LeRobotDatasets into one combined dataset for training.

This script merges several same-schema dataset roots into a single
training-ready dataset, e.g. multiple outputs of
``convert_dataset_resolution.py`` into one combined dataset.

Uses ``lerobot.datasets.aggregate.aggregate_datasets``, same as
``consolidate_datasets.py``: episode/frame reindexing, video concatenation
(stream-copy, no re-encoding -- no video quality loss), task-string
unification, and stats aggregation all come from there. Every episode from
every source is kept.

Dry run by default (prints the discovered sources only); pass ``--write`` to
actually merge. Source dataset roots are never modified or deleted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def source_episode_count(root: Path) -> int:
    total = 0
    for parquet_path in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        total += len(pd.read_parquet(parquet_path, columns=["episode_index"]))
    return total


def discover_dataset_roots(parent: Path) -> list[Path]:
    """Every immediate subdirectory of ``parent`` that looks like a dataset root."""
    return sorted(p.parent.parent for p in parent.glob("*/meta/info.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "dataset roots to combine (default: every immediate subdirectory of "
            "--parent that has a meta/info.json)"
        ),
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets-512",
        help="where to look for sources when none are given explicitly (default: datasets-512/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dataset root (default: <--parent>/combined)",
    )
    parser.add_argument("--write", action="store_true", help="perform the merge")
    args = parser.parse_args()

    sources = args.sources if args.sources else discover_dataset_roots(args.parent)
    if not sources:
        raise SystemExit(f"No dataset roots found under {args.parent}")

    out_dir = args.out_dir if args.out_dir is not None else args.parent / "combined"
    if out_dir in sources:
        raise SystemExit(f"Output {out_dir} can't also be one of the sources being merged")

    print(f"Combining {len(sources)} dataset(s) -> {out_dir}:")
    source_total = 0
    for root in sources:
        count = source_episode_count(root)
        source_total += count
        print(f"  {root}: {count} episode(s)")
    print(f"  total: {source_total} episode(s)")

    if not args.write:
        print("\nDry run: pass --write to merge.")
        return

    if out_dir.exists():
        raise SystemExit(f"Output already exists, refusing to overwrite: {out_dir}")

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[f"local/pick-and-place-{i:03d}" for i in range(len(sources))],
        aggr_repo_id="local/pick-and-place-combined",
        roots=sources,
        aggr_root=out_dir,
    )

    merged_total = source_episode_count(out_dir)
    if merged_total != source_total:
        raise SystemExit(
            f"Episode count mismatch after merging: {source_total} source vs "
            f"{merged_total} merged. Merged output left at {out_dir} for inspection."
        )
    print(f"\nWrote {merged_total} episode(s) to {out_dir}")


if __name__ == "__main__":
    main()
