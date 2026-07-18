#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fit the shoulder-pan zero offset from a hand-eye offset measurement.

The dominant hand-eye error is a rotation about the robot base's vertical
axis: the tangential component of the world-frame cube offset grows linearly
with the cube's radius from the base with near-zero intercept. This script
loads a measurement JSON written by ``measure_hand_eye_offset`` (whose pair
frames must carry world-frame deltas), computes the implied rotation per frame
(tangential offset / radius), and reports the robust per-episode and overall
pan offset in degrees.

The resulting value is the shoulder_pan entry for the exporter's
``--joint-offsets-deg`` and,
sign-flipped appropriately, the correction for the real scripted pipeline's
kinematics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", type=Path, help="measure_hand_eye_offset --output JSON")
    args = parser.parse_args()

    with args.measurement.open() as f:
        summary = json.load(f)

    all_thetas: list[float] = []
    print("per-episode implied pan offset:")
    for episode in summary["episodes"]:
        episode_dir = Path(episode["name"])
        with (episode_dir / "pairs.json").open() as f:
            index = json.load(f)
        cube_by_frame = {fr["frame"]: fr["cube"] for fr in index["frames"]}
        thetas = []
        for frame in episode["frames"]:
            if "delta_world_mm" not in frame:
                continue
            cube = cube_by_frame[frame["frame"]]
            radius = math.hypot(cube["x"], cube["y"])
            u_t = (-cube["y"] / radius, cube["x"] / radius)
            delta_t = (
                frame["delta_world_mm"][0] * u_t[0] + frame["delta_world_mm"][1] * u_t[1]
            ) / 1000.0
            thetas.append(math.degrees(delta_t / radius))
        if not thetas:
            continue
        all_thetas.extend(thetas)
        print(
            f"  {episode_dir.name}: n={len(thetas):3d} "
            f"median={np.median(thetas):+.2f}deg  mad={np.median(np.abs(thetas - np.median(thetas))):.2f}deg"
        )

    if not all_thetas:
        raise SystemExit("no frames with world-frame deltas found")
    arr = np.asarray(all_thetas)
    print(
        f"\noverall: n={len(arr)} median={np.median(arr):+.3f}deg "
        f"mean={arr.mean():+.3f}deg std={arr.std():.3f}deg"
    )
    print(f"suggested exporter flag: --joint-offsets-deg shoulder_pan={np.median(arr):.2f}")


if __name__ == "__main__":
    main()
