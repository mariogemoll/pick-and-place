# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export canonical grasp oracle samples for policy distillation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from pick_and_place.episodes import (
    PICKUP_YAW_DEVIATION,
    _build_model,
    pickup_yaw_from_azimuth,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.trajectory import grasp_candidates
from pick_and_place.workspace_overlays import (
    CANONICAL_PICKUP_OVERLAY,
    PAN_AXIS,
    is_cube_pickup_allowed,
)


_FIELDNAMES = (
    "index",
    "status",
    "x_m",
    "y_m",
    "z_m",
    "radius_m",
    "azimuth_deg",
    "yaw_deg",
    "yaw_mod_90_deg",
    "yaw_deviation_deg",
    "face",
    "closing_sign",
    "closing_azimuth_deg",
    "pitch_deg",
    "roll_offset_deg",
    "elbow",
    "camera_outward",
    "shoulder_pan_deg",
    "shoulder_lift_deg",
    "elbow_flex_deg",
    "wrist_flex_deg",
    "wrist_roll_deg",
)


def _normalize_angle(angle: float) -> float:
    result = angle % (2.0 * math.pi)
    if result > math.pi:
        result -= 2.0 * math.pi
    if result <= -math.pi:
        result += 2.0 * math.pi
    return result


def _mod_90(angle: float) -> float:
    quarter = math.pi / 2.0
    return (angle + quarter / 2.0) % quarter - quarter / 2.0


def _grid_poses(
    grid: int,
    yaw_count: int,
    *,
    radius_min: float,
    radius_max: float,
    azimuth_min: float,
    azimuth_max: float,
    yaw_min: float,
    yaw_max: float,
    clip_to_pickup: bool,
) -> Iterable[CubePose]:
    radii = np.linspace(radius_min, radius_max, grid)
    azimuths = np.linspace(azimuth_min, azimuth_max, grid)
    yaw_deviations = (
        np.array((0.0,))
        if yaw_count == 1
        else np.linspace(yaw_min, yaw_max, yaw_count)
    )
    for radius in radii:
        for azimuth in azimuths:
            x = PAN_AXIS[0] + float(radius) * math.cos(float(azimuth))
            y = PAN_AXIS[1] + float(radius) * math.sin(float(azimuth))
            if clip_to_pickup and not is_cube_pickup_allowed(x, y):
                continue
            for deviation in yaw_deviations:
                yield CubePose(
                    x=x,
                    y=y,
                    z=CUBE_HALF_SIZE,
                    yaw=pickup_yaw_from_azimuth(float(azimuth), float(deviation)),
                )


def _blank_candidate_fields() -> dict[str, str]:
    return {
        "face": "",
        "closing_sign": "",
        "closing_azimuth_deg": "",
        "pitch_deg": "",
        "roll_offset_deg": "",
        "elbow": "",
        "camera_outward": "",
        "shoulder_pan_deg": "",
        "shoulder_lift_deg": "",
        "elbow_flex_deg": "",
        "wrist_flex_deg": "",
        "wrist_roll_deg": "",
    }


def _row(index: int, kinematics, pose: CubePose) -> dict[str, object]:
    radius = math.hypot(pose.x - PAN_AXIS[0], pose.y - PAN_AXIS[1])
    azimuth = math.atan2(pose.y - PAN_AXIS[1], pose.x - PAN_AXIS[0])
    row: dict[str, object] = {
        "index": index,
        "status": "no-ik",
        "x_m": f"{pose.x:.9f}",
        "y_m": f"{pose.y:.9f}",
        "z_m": f"{pose.z:.9f}",
        "radius_m": f"{radius:.9f}",
        "azimuth_deg": f"{math.degrees(azimuth):.6f}",
        "yaw_deg": f"{math.degrees(pose.yaw):.6f}",
        "yaw_mod_90_deg": f"{math.degrees(_mod_90(pose.yaw)):.6f}",
        "yaw_deviation_deg": f"{math.degrees(_normalize_angle(pose.yaw - azimuth)):.6f}",
        **_blank_candidate_fields(),
    }

    grasp = next(grasp_candidates(kinematics, pose), None)
    if grasp is None:
        return row

    closing_delta = _normalize_angle(grasp.closing_azimuth - azimuth)
    row.update(
        {
            "status": "selected",
            "face": grasp.face,
            "closing_sign": "+90" if closing_delta > 0.0 else "-90",
            "closing_azimuth_deg": f"{math.degrees(grasp.closing_azimuth):.6f}",
            "pitch_deg": f"{math.degrees(grasp.pitch):.6f}",
            "roll_offset_deg": f"{math.degrees(grasp.roll_offset):.6f}",
            "elbow": grasp.elbow,
            "camera_outward": f"{grasp.camera_outward:.9f}",
            "shoulder_pan_deg": f"{math.degrees(grasp.grasp_joints['shoulder_pan']):.6f}",
            "shoulder_lift_deg": f"{math.degrees(grasp.grasp_joints['shoulder_lift']):.6f}",
            "elbow_flex_deg": f"{math.degrees(grasp.grasp_joints['elbow_flex']):.6f}",
            "wrist_flex_deg": f"{math.degrees(grasp.grasp_joints['wrist_flex']):.6f}",
            "wrist_roll_deg": f"{math.degrees(grasp.grasp_joints['wrist_roll']):.6f}",
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=21, help="radial/azimuth samples per axis")
    parser.add_argument(
        "--yaw-count",
        type=int,
        default=9,
        help="cube yaw deviation samples per grid point",
    )
    parser.add_argument(
        "--yaw-deviation-deg",
        type=float,
        default=None,
        help="maximum yaw deviation around azimuth; default is the episode sampler range",
    )
    parser.add_argument(
        "--radius-min-mm",
        type=float,
        default=None,
        help="minimum radius from the pan axis; defaults to the canonical pickup overlay",
    )
    parser.add_argument(
        "--radius-max-mm",
        type=float,
        default=None,
        help="maximum radius from the pan axis; defaults to the canonical pickup overlay",
    )
    parser.add_argument(
        "--azimuth-min-deg",
        type=float,
        default=None,
        help="minimum azimuth; defaults to the canonical pickup overlay",
    )
    parser.add_argument(
        "--azimuth-max-deg",
        type=float,
        default=None,
        help="maximum azimuth; defaults to the canonical pickup overlay",
    )
    parser.add_argument(
        "--include-frame-clipped",
        action="store_true",
        help="keep samples excluded by workspace frame bounds instead of clipping them",
    )
    parser.add_argument("--limit", type=int, default=0, help="optional cap on exported samples")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV output path; omit to write to stdout",
    )
    args = parser.parse_args()

    if args.grid < 1:
        parser.error("--grid must be at least 1")
    if args.yaw_count < 1:
        parser.error("--yaw-count must be at least 1")

    radius_min = (
        CANONICAL_PICKUP_OVERLAY.inner_radius
        if args.radius_min_mm is None
        else args.radius_min_mm / 1000.0
    )
    radius_max = (
        CANONICAL_PICKUP_OVERLAY.outer_radius
        if args.radius_max_mm is None
        else args.radius_max_mm / 1000.0
    )
    azimuth_min = (
        CANONICAL_PICKUP_OVERLAY.azimuth_min
        if args.azimuth_min_deg is None
        else math.radians(args.azimuth_min_deg)
    )
    azimuth_max = (
        CANONICAL_PICKUP_OVERLAY.azimuth_max
        if args.azimuth_max_deg is None
        else math.radians(args.azimuth_max_deg)
    )
    if radius_min > radius_max:
        parser.error("--radius-min-mm cannot be greater than --radius-max-mm")
    if azimuth_min > azimuth_max:
        parser.error("--azimuth-min-deg cannot be greater than --azimuth-max-deg")

    yaw_deviation = (
        PICKUP_YAW_DEVIATION
        if args.yaw_deviation_deg is None
        else math.radians(args.yaw_deviation_deg)
    )
    if yaw_deviation < 0.0:
        parser.error("--yaw-deviation-deg cannot be negative")

    poses = list(
        _grid_poses(
            args.grid,
            args.yaw_count,
            radius_min=radius_min,
            radius_max=radius_max,
            azimuth_min=azimuth_min,
            azimuth_max=azimuth_max,
            yaw_min=-yaw_deviation,
            yaw_max=yaw_deviation,
            clip_to_pickup=not args.include_frame_clipped,
        )
    )
    if args.limit > 0:
        poses = poses[: args.limit]
    if not poses:
        print("No poses sampled.", file=sys.stderr)
        return 2

    model, _ = _build_model(poses[0])
    kinematics = derive_kinematics(model)
    rows = [_row(index, kinematics, pose) for index, pose in enumerate(poses)]

    if args.out is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with args.out.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    selected = sum(1 for row in rows if row["status"] == "selected")
    print(f"exported={len(rows)} selected={selected} no_ik={len(rows) - selected}", file=sys.stderr)
    return 0 if selected else 1


if __name__ == "__main__":
    sys.exit(main())
