# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Shared episode-level metadata written into LeRobotDataset episode rows."""

from __future__ import annotations

from typing import Any

from pick_and_place.episodes import PlacementError
from pick_and_place.geometry import CubePose


def driver_metadata(driver: str) -> dict[str, str]:
    """Episode metadata naming what produced the trajectory.

    Values are open-ended so the same dataset can mix collection sources:
    ``"teleop"`` (a human on the SO-101 leader), ``"analytic"`` (the planner in
    ``real.py``), and later e.g. an RL or VLA policy.
    """
    return {"driver": driver}


def cube_pose_metadata(source: CubePose, target: CubePose) -> dict[str, float]:
    """Return the planar pick pose and target point for one pick-and-place episode.

    The cube always rests flat on the table, so its z and roll/pitch are
    constants and only the ``(x, y, yaw)`` of the pick pose carries
    information. The target is the centre of the black-square marker, a bare
    ``(x, y)`` point with no meaningful orientation.
    """
    return {
        "cube_start_x": float(source.x),
        "cube_start_y": float(source.y),
        "cube_start_yaw": float(source.yaw),
        "target_x": float(target.x),
        "target_y": float(target.y),
    }


def placement_error_metadata(error: PlacementError | None, *, detected: bool) -> dict[str, Any]:
    """Return final-placement metadata: whether the cube was seen and where it landed.

    Only ``placement_detected`` and the measured ``(x, y)`` of the cube after
    release are stored. The placement error is a pure function of ``cube_end``
    and the episode's ``target``, so any consumer recomputes it (with whatever
    tolerance it wants) rather than risk a stored copy drifting out of sync.
    """
    if error is None:
        nan = float("nan")
        return {"placement_detected": bool(detected), "cube_end_x": nan, "cube_end_y": nan}

    return {
        "placement_detected": bool(detected),
        "cube_end_x": error.cube_xyz[0],
        "cube_end_y": error.cube_xyz[1],
    }
