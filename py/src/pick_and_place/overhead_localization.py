# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Stateful overhead localization from RGB observations."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from pick_and_place.cube_detection import (
    cube_pose_to_world,
    estimate_cube_pose,
    make_cube_detector,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.paper_detection import PaperTarget, PaperTracker, detect_paper_target


class OverheadLocalizer:
    """Localize workspace objects using one fixed nominal camera calibration."""

    def __init__(
        self,
        camera_matrix: NDArray,
        camera_position: NDArray,
        camera_rotation: NDArray,
        *,
        detector_factory: Callable[[], Any] | None = None,
        paper_tracker: PaperTracker | None = None,
    ) -> None:
        self.camera_matrix = np.asarray(camera_matrix, dtype=float).copy()
        self.camera_position = np.asarray(camera_position, dtype=float).copy()
        self.camera_rotation = np.asarray(camera_rotation, dtype=float).copy()
        self._detector_factory = detector_factory or make_cube_detector
        self._paper_tracker = paper_tracker if paper_tracker is not None else PaperTracker()
        self._cube_detector = None
        self.reset()

    def reset(self) -> None:
        """Forget detections from the previous episode."""
        self._cube_detector = self._detector_factory()
        self._paper_tracker.reset()

    def localize_cube(
        self,
        frame_rgb: NDArray[np.uint8],
        *,
        free_grasp: bool = False,
    ) -> CubePose | None:
        """Return the cube pose inferred from one RGB observation."""
        return localize_cube(
            frame_rgb,
            self._cube_detector,
            self.camera_matrix,
            self.camera_position,
            self.camera_rotation,
            free_grasp=free_grasp,
        )

    def localize_drop_target(
        self,
        frame_rgb: NDArray[np.uint8],
        *,
        target_color: str,
        workspace_corners_world: NDArray,
    ) -> PaperTarget | None:
        """Return the tracked drop target inferred from one RGB observation."""
        return localize_drop_target(
            frame_rgb,
            self._paper_tracker,
            self.camera_matrix,
            self.camera_position,
            self.camera_rotation,
            target_color=target_color,
            workspace_corners_world=workspace_corners_world,
        )


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
