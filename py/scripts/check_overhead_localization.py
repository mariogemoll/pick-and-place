#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Check that simulated overhead perception misses by as much as the rig does.

Simulating the overhead loop honestly *removes* error: in a clean scene the
extrinsics are exact and the workspace frame is exactly where the model puts it,
so render-and-detect localizes the cube to a fraction of a millimetre while the
rig misses by six to nine. Demonstrations generated that way would never show
the planner correcting for a localization it should not have trusted.

So the causes are perturbed instead — a small residual gap between where the
overhead camera is and where its calibration says it is — and this is the
measurement that says whether that reproduces the rig or not. Run it after
changing any of the three sigmas, or the scene, or the detector.

    python scripts/check_overhead_localization.py --episodes 60

Reports the cube's planar and vertical miss, its yaw miss, and the drop plate's
planar miss, next to how often a scene could not be localized at all.
"""

from __future__ import annotations

import argparse
from functools import partial

from pick_and_place.core.miscalibration import OverheadCameraModel
from pick_and_place.plant.overhead import DETECTION_HEIGHT, DETECTION_WIDTH
from pick_and_place.plant.overhead_check import measure
from pick_and_place.rollout.sim import build_recording_scene

#: What SIM2REAL measured on the rig, and therefore what a simulated chain has
#: to land on to be worth generating demonstrations from.
MEASURED_CUBE_XY_M = (0.006, 0.009)
MEASURED_TARGET_XY_M = (0.006, 0.009)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=60, help="scenes to localize")
    parser.add_argument("--seed", type=int, default=7, help="root seed for the draws")
    parser.add_argument(
        "--no-calibration-error",
        action="store_true",
        help=(
            "zero the injected causes, to see what an honest render-and-detect "
            "achieves on its own. Expect a fraction of a millimetre, which is the "
            "whole reason the causes exist."
        ),
    )
    args = parser.parse_args()

    sigmas = OverheadCameraModel()
    if args.no_calibration_error:
        sigmas = OverheadCameraModel(0.0, 0.0, 0.0)
    summary = measure(
        partial(
            build_recording_scene,
            render_width=DETECTION_WIDTH,
            render_height=DETECTION_HEIGHT,
        ),
        episodes=args.episodes,
        seed=args.seed,
        model_sigmas=sigmas,
    )
    print(summary.summary())
    low, high = MEASURED_CUBE_XY_M
    print(
        f"\nmeasured on the rig: cube {low * 1000:.0f}-{high * 1000:.0f} mm planar, "
        f"target {MEASURED_TARGET_XY_M[0] * 1000:.0f}-"
        f"{MEASURED_TARGET_XY_M[1] * 1000:.0f} mm"
    )
    if args.no_calibration_error:
        return
    verdict = "in band" if low <= summary.cube_xy_median_m <= high else "OUT OF BAND"
    print(f"cube planar median is {verdict}")


if __name__ == "__main__":
    main()
