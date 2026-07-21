# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Localize the cube and drop target from one overhead RGB observation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from pick_and_place.cube_detection import cube_pose_to_world, estimate_cube_pose
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.paper_detection import PaperTarget, PaperTracker, detect_paper_target


def localize_cube(
    frame_rgb: NDArray[np.uint8],
    detector,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    free_grasp: bool = False,
) -> CubePose | None:
    """Return the world-frame cube pose inferred from one RGB image."""
    estimate = estimate_cube_pose(frame_rgb, detector, camera_matrix)
    if estimate is None:
        return None

    rotation, position = cube_pose_to_world(
        estimate,
        camera_position,
        camera_rotation,
    )
    roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
    return CubePose(
        x=float(position[0]),
        y=float(position[1]),
        z=CUBE_HALF_SIZE,
        roll=float(roll) if free_grasp else 0.0,
        pitch=float(pitch) if free_grasp else 0.0,
        yaw=float(yaw),
    )


def localize_drop_target(
    frame_rgb: NDArray[np.uint8],
    tracker: PaperTracker,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    target_color: str,
    workspace_corners_world: NDArray,
) -> PaperTarget | None:
    """Return the tracked drop target inferred from one RGB image."""
    detection = detect_paper_target(
        frame_rgb,
        camera_matrix,
        camera_position,
        camera_rotation,
        target_color=target_color,
        workspace_corners_world=workspace_corners_world,
    )
    return tracker.update(detection)
