#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Re-materialize an older recording's episode metadata to the clean schema, in place.

Recordings made before the schema settled carry accumulated cruft: derived
columns that are a pure function of stored ones (``success``,
``placement_dx/dy/dz/placement_xy``), all-constant pose components
(``*_z``/``*_roll``/``*_pitch``, ``cube_target_yaw``), a duplicate target point
(``placement_target_*`` is identical to ``cube_target_*``), a mostly-empty
free-text ``placement_check_error``, and older column names.

This rewrites each ``meta/episodes/*.parquet`` of ``--src`` in place to the
current clean schema, keeping only the columns that carry information:

    cube_start_x, cube_start_y, cube_start_yaw   planar pick pose
    cube_end_x,   cube_end_y                      measured final cube (overhead)
    placement_detected                            cube seen at check time
    target_x,     target_y                        black-square marker centre
    driver                                        analytic | teleop | ...
    pickup_*  (6 columns)                          live gripper readback

Older names are renamed (``placement_cube_* -> cube_end_*``,
``cube_target_* -> target_*``); the cruft columns are dropped; a ``driver``
column is added as ``--driver`` (default ``analytic``, correct for the legacy
analytic-only sessions) if the dataset has none yet; and any missing
``pickup_*`` column is added as NaN (the earliest sessions predate gripper
readback -- NaN = unknown, not reconstructible). Data rows, video, stats and
LeRobot bookkeeping columns are left untouched, and nothing outside
``meta/episodes`` is read or written.

Dry run by default (prints the resulting column set); pass ``--write`` to apply.
Idempotent: re-running on an already-clean dataset is a no-op.

Example:

    python py/scripts/migrate_dataset_schema.py --src datasets/20260701 --write
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pick_and_place.dataset_subset import BOOKKEEPING_COLUMNS, BOOKKEEPING_PREFIXES

# Clean project column <- source column it is read/renamed from.
RENAMES = {
    "cube_start_x": "cube_start_x",
    "cube_start_y": "cube_start_y",
    "cube_start_yaw": "cube_start_yaw",
    "cube_end_x": "placement_cube_x",
    "cube_end_y": "placement_cube_y",
    "placement_detected": "placement_detected",
    "target_x": "cube_target_x",
    "target_y": "cube_target_y",
}

# Live-gripper readback columns kept in the schema, with the fill used for
# sessions recorded before pickup tracking existed (NaN / "" = unknown).
PICKUP_COLUMNS: dict[str, object] = {
    "pickup_check_phase": "",
    "pickup_confidence": np.nan,
    "pickup_empty_gripper_position": np.nan,
    "pickup_gripper_delta": np.nan,
    "pickup_gripper_margin": np.nan,
    "pickup_gripper_position": np.nan,
}


def is_bookkeeping(column: str) -> bool:
    return column in BOOKKEEPING_COLUMNS or column.startswith(BOOKKEEPING_PREFIXES)


def clean_frame(df: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Return ``df`` rewritten to the clean schema (bookkeeping columns preserved)."""
    out = df[[c for c in df.columns if is_bookkeeping(c)]].copy()

    for dest, src in RENAMES.items():
        if src in df.columns:
            out[dest] = df[src]
        elif dest in df.columns:  # already migrated
            out[dest] = df[dest]
        else:
            raise SystemExit(f"cannot build {dest!r}: neither {src!r} nor {dest!r} present")

    out["driver"] = df["driver"] if "driver" in df.columns else driver

    for col, fill in PICKUP_COLUMNS.items():
        out[col] = df[col] if col in df.columns else fill

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="dataset root to migrate in place")
    parser.add_argument(
        "--driver",
        default="analytic",
        help="value for the driver column when the dataset has none yet (default: analytic)",
    )
    parser.add_argument("--write", action="store_true", help="apply the migration in place")
    args = parser.parse_args()

    parquet_paths = sorted(args.src.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"no meta/episodes/*.parquet under {args.src}")

    total_episodes = 0
    before_cols: list[str] = []
    after_cols: list[str] = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        cleaned = clean_frame(df, args.driver)
        total_episodes += len(df)
        before_cols = [c for c in df.columns if not is_bookkeeping(c)]
        after_cols = [c for c in cleaned.columns if not is_bookkeeping(c)]
        if args.write:
            cleaned.to_parquet(path, index=False)

    dropped = sorted(set(before_cols) - set(after_cols))
    added = sorted(set(after_cols) - set(before_cols))
    verb = "Migrated" if args.write else "Would migrate"
    print(f"{args.src}: {total_episodes} episode(s) across {len(parquet_paths)} metadata file(s)")
    print(f"  {verb} to project columns: {after_cols}")
    print(f"  dropped: {dropped or '(none)'}")
    print(f"  added:   {added or '(none)'}")
    if not args.write:
        print("\nDry run: pass --write to apply in place.")


if __name__ == "__main__":
    main()
