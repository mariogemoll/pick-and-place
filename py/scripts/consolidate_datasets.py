#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Merge per-run LeRobotDatasets into one dataset per recording day.

Each invocation of ``pick_and_place/real.py`` writes its own LeRobotDataset
under a ``<YYYYMMDD>_<HHMMSS>`` directory. This script discovers every such
run directory under one or more source roots, groups them by calendar day
(parsed from the directory name), and merges each day's runs into a single
LeRobotDataset using ``lerobot.datasets.aggregate.aggregate_datasets`` — which
already handles episode/frame reindexing, video concatenation, task-string
unification, and stats aggregation across dataset roots with an identical
schema.

Every episode is kept, successful or not, with its raw fields untouched
(``placement_detected``, ``cube_end_*``, ``target_*``, ``pickup_gripper_delta``,
etc.) so any consumer can define and threshold "success" themselves rather than
trusting a bolted-on derived column.

Dry run by default (prints the discovered grouping only); pass ``--write`` to
actually merge. Source run directories are never modified or deleted.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd


def discover_runs(source_root: Path) -> list[tuple[datetime.datetime, Path]]:
    """Return ``(timestamp, run_root)`` for every LeRobotDataset directly under
    ``source_root``, skipping anything whose name isn't a recording timestamp."""
    runs = []
    for info_path in sorted(source_root.glob("*/meta/info.json")):
        run_root = info_path.parent.parent
        try:
            ts = datetime.datetime.strptime(run_root.name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        runs.append((ts, run_root))
    return runs


def group_by_day(
    runs: list[tuple[datetime.datetime, Path]],
) -> dict[datetime.date, list[Path]]:
    by_day: dict[datetime.date, list[tuple[datetime.datetime, Path]]] = {}
    for ts, root in runs:
        by_day.setdefault(ts.date(), []).append((ts, root))
    return {day: [root for _, root in sorted(entries)] for day, entries in sorted(by_day.items())}


def source_episode_count(run_root: Path) -> int:
    total = 0
    for parquet_path in sorted(run_root.glob("meta/episodes/chunk-*/file-*.parquet")):
        total += len(pd.read_parquet(parquet_path, columns=["episode_index"]))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_roots",
        nargs="*",
        type=Path,
        default=[Path(__file__).resolve().parents[1] / "datasets"],
        help="parent directories to scan for run directories (default: py/datasets)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets",
        help="output root; each day is written to <out-dir>/<YYYYMMDD> "
        "(default: repo root datasets/)",
    )
    parser.add_argument("--write", action="store_true", help="perform the merge")
    args = parser.parse_args()

    runs = [run for root in args.source_roots for run in discover_runs(root)]
    if not runs:
        raise SystemExit(f"No run directories found under {args.source_roots}")

    by_day = group_by_day(runs)

    print(f"Discovered {len(runs)} run(s) across {len(by_day)} day(s):")
    for day, day_roots in by_day.items():
        episode_count = sum(source_episode_count(root) for root in day_roots)
        out_path = args.out_dir / f"{day:%Y%m%d}"
        print(
            f"  {day:%Y-%m-%d}: {len(day_roots)} run(s), {episode_count} episode(s) -> {out_path}"
        )

    if not args.write:
        print("\nDry run: pass --write to merge.")
        return

    from lerobot.datasets.aggregate import aggregate_datasets

    for day, day_roots in by_day.items():
        out_path = args.out_dir / f"{day:%Y%m%d}"
        if out_path.exists():
            raise SystemExit(f"Output already exists, refusing to overwrite: {out_path}")

        source_total = sum(source_episode_count(root) for root in day_roots)
        print(f"\nMerging {day:%Y-%m-%d} ({len(day_roots)} run(s), {source_total} episode(s))...")
        aggregate_datasets(
            repo_ids=[f"local/pick-and-place-{i:03d}" for i in range(len(day_roots))],
            aggr_repo_id=f"local/pick-and-place-{day:%Y%m%d}",
            roots=day_roots,
            aggr_root=out_path,
        )

        merged_total = source_episode_count(out_path)
        if merged_total != source_total:
            raise SystemExit(
                f"Episode count mismatch after merging {day:%Y-%m-%d}: "
                f"{source_total} source vs {merged_total} merged. "
                f"Merged output left at {out_path} for inspection."
            )
        print(f"  Wrote {merged_total} episode(s) to {out_path}")


if __name__ == "__main__":
    main()
