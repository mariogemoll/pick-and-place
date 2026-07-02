# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Shared episode-level metadata written into LeRobotDataset episode rows."""

from __future__ import annotations

from typing import Any

from pick_and_place.episodes import PlacementError
from pick_and_place.geometry import CubePose


def cube_pose_metadata(source: CubePose, target: CubePose) -> dict[str, float]:
    """Return scalar source/target pose metadata for one pick-and-place episode."""
    return {
        "cube_start_x": float(source.x),
        "cube_start_y": float(source.y),
        "cube_start_z": float(source.z),
        "cube_start_roll": float(source.roll),
        "cube_start_pitch": float(source.pitch),
        "cube_start_yaw": float(source.yaw),
        "cube_target_x": float(target.x),
        "cube_target_y": float(target.y),
        "cube_target_z": float(target.z),
        "cube_target_roll": float(target.roll),
        "cube_target_pitch": float(target.pitch),
        "cube_target_yaw": float(target.yaw),
    }


def placement_error_metadata(
    error: PlacementError | None,
    *,
    detected: bool,
    check_error: str = "",
) -> dict[str, Any]:
    """Return scalar final-placement metadata using the real-run column names."""
    if error is None:
        nan = float("nan")
        return {
            "placement_detected": bool(detected),
            "placement_check_error": check_error,
            "placement_cube_x": nan,
            "placement_cube_y": nan,
            "placement_cube_z": nan,
            "placement_target_x": nan,
            "placement_target_y": nan,
            "placement_target_z": nan,
            "placement_dx": nan,
            "placement_dy": nan,
            "placement_dz": nan,
            "placement_xy": nan,
        }

    return {
        "placement_detected": bool(detected),
        "placement_check_error": check_error,
        "placement_cube_x": error.cube_xyz[0],
        "placement_cube_y": error.cube_xyz[1],
        "placement_cube_z": error.cube_xyz[2],
        "placement_target_x": error.target_xyz[0],
        "placement_target_y": error.target_xyz[1],
        "placement_target_z": error.target_xyz[2],
        "placement_dx": error.dx,
        "placement_dy": error.dy,
        "placement_dz": error.dz,
        "placement_xy": error.xy,
    }
