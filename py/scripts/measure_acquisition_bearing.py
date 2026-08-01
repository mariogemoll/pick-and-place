#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure where a closed-loop rollout aims during acquisition.

Answers docs/DPPO_CLOSED_LOOP_STALL_HANDOFF.md's remaining open question about
training duration: was the ~22 degree acquisition miss already present at an
early checkpoint, or did the extra epochs sharpen it?

The base joint j1 swings the whole arm left and right, so a policy reaching
correctly for an object has j1 equal to that object's bearing regardless of
what the rest of the arm is doing. This robot's j1 runs opposite to ``atan2``,
so the convention (confirmed in the handoff doc to 2.5 degrees) is::

    j1 = -atan2(y, x)

Consumes the ``--trajectory-json`` written by ``run_policy_sim.py`` and reports,
per segment and pooled, the mean absolute j1 error against the cube's bearing
and against the drop plate's bearing. Comparing the two separates "aiming at
the cube" from "aiming at the plate": a policy that has learned the plate
shortcut correlates with the plate and not the cube.

Acquisition is taken as the opening ``--acquisition-ticks`` of each segment,
since a rollout that never grasps has no phase spans to read. That is the
phase the handoff doc identifies as failing.

Example:

    python py/scripts/measure_acquisition_bearing.py \\
      --trajectory outputs/bearing/state_200.json \\
      --trajectory outputs/bearing/state_1100.json \\
      --output outputs/bearing/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# j1 is the first joint in the recorded real-unit state vector, in degrees.
J1_INDEX = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help="a trajectory JSON from run_policy_sim.py --trajectory-json (repeatable)",
    )
    parser.add_argument(
        "--acquisition-ticks",
        type=int,
        default=25,
        help="ticks from each segment's start treated as acquisition (default: 25)",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _bearing_deg(x: float, y: float) -> float:
    """Expected j1 for an object at ``(x, y)``, in degrees."""
    return -math.degrees(math.atan2(y, x))


def _wrap_deg(value: float) -> float:
    """Wrap an angle difference into [-180, 180]."""
    return (value + 180.0) % 360.0 - 180.0


def _summarize(errors: list[float]) -> dict:
    if not errors:
        return {"count": 0}
    arr = np.asarray(errors, dtype=float)
    return {
        "mean_abs_deg": round(float(np.mean(np.abs(arr))), 2),
        "median_abs_deg": round(float(np.median(np.abs(arr))), 2),
        "p95_abs_deg": round(float(np.percentile(np.abs(arr), 95)), 2),
        "count": int(arr.size),
    }


def _correlation(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if va.std() < 1e-9 or vb.std() < 1e-9:
        return None
    return round(float(np.corrcoef(va, vb)[0, 1]), 3)


def analyze(payload: dict, acquisition_ticks: int) -> dict:
    """Per-segment and pooled acquisition bearing error for one rollout."""
    by_segment: dict[int, list[dict]] = {}
    for record in payload["ticks"]:
        by_segment.setdefault(int(record["segment"]), []).append(record)

    cube_errors: list[float] = []
    plate_errors: list[float] = []
    j1_values: list[float] = []
    cube_bearings: list[float] = []
    plate_bearings: list[float] = []
    segments: list[dict] = []

    for segment_id in sorted(by_segment):
        records = sorted(by_segment[segment_id], key=lambda r: r["tick"])[:acquisition_ticks]
        if not records:
            continue
        seg_cube: list[float] = []
        seg_plate: list[float] = []
        for record in records:
            j1 = float(record["state_real"][J1_INDEX])
            cube_x, cube_y = record["cube_xyz"][0], record["cube_xyz"][1]
            plate_x, plate_y = record["target_xy"]
            cube_bearing = _bearing_deg(cube_x, cube_y)
            plate_bearing = _bearing_deg(plate_x, plate_y)

            seg_cube.append(_wrap_deg(j1 - cube_bearing))
            seg_plate.append(_wrap_deg(j1 - plate_bearing))
            j1_values.append(j1)
            cube_bearings.append(cube_bearing)
            plate_bearings.append(plate_bearing)

        cube_errors.extend(seg_cube)
        plate_errors.extend(seg_plate)
        first = records[0]
        segments.append(
            {
                "segment": segment_id,
                "cube_xy": [round(first["cube_xyz"][0], 4), round(first["cube_xyz"][1], 4)],
                "plate_xy": [round(v, 4) for v in first["target_xy"]],
                "vs_cube": _summarize(seg_cube),
                "vs_plate": _summarize(seg_plate),
            }
        )

    # Averaging across the whole acquisition window understates how well the
    # policy aims: every segment restarts from the neutral pose, where j1 is 0
    # regardless of where the cube is, so the opening ticks contribute no
    # correlation by construction. Breaking the same data down by tick offset
    # shows whether the arm converges onto the cube's bearing as it reaches.
    by_offset: dict[str, dict] = {}
    for offset in range(0, acquisition_ticks + 1, 5):
        window = [
            record
            for records in by_segment.values()
            for record in records
            if offset <= int(record["tick"]) % max(payload.get("resample_every") or 10**9, 1)
            < offset + 5
        ]
        if not window:
            continue
        cube_window: list[float] = []
        j1_window: list[float] = []
        cube_bearing_window: list[float] = []
        for record in window:
            j1 = float(record["state_real"][J1_INDEX])
            cube_bearing = _bearing_deg(record["cube_xyz"][0], record["cube_xyz"][1])
            cube_window.append(_wrap_deg(j1 - cube_bearing))
            j1_window.append(j1)
            cube_bearing_window.append(cube_bearing)
        by_offset[f"ticks_{offset}_{offset + 4}"] = {
            "vs_cube": _summarize(cube_window),
            "j1_correlation_with_cube": _correlation(j1_window, cube_bearing_window),
        }

    return {
        "checkpoint": payload.get("checkpoint"),
        "scene_appearance": payload.get("scene_appearance"),
        "acquisition_ticks": acquisition_ticks,
        "segments_analyzed": len(segments),
        "pooled": {
            "vs_cube": _summarize(cube_errors),
            "vs_plate": _summarize(plate_errors),
            "j1_correlation_with_cube": _correlation(j1_values, cube_bearings),
            "j1_correlation_with_plate": _correlation(j1_values, plate_bearings),
        },
        "by_tick_offset": by_offset,
        "per_segment": segments,
    }


def main() -> None:
    args = _parse_args()
    report = {}
    for path in args.trajectory:
        payload = json.loads(path.read_text())
        report[path.stem] = analyze(payload, args.acquisition_ticks)

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
