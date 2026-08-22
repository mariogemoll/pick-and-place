# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fit the projector-pixel to workspace-metre homography.

The floor is planar and the projector is a pinhole aimed at it, so the map from
panel pixels to workspace coordinates is a homography: eight numbers, exact for
*any* projector pose. Where the projector stands is never measured, only solved
for -- which is the whole reason this works from wherever it happens to be
propped.

It is solved as two planar maps composed, rather than one fit against the
camera's calibrated pose:

``projector -> camera``
    from Gray code correspondences, thousands of them spread over the field.

``camera -> workspace``
    from the four corner AprilTag plates, whose workspace positions are declared
    in :mod:`pick_and_place.spec.workspace` and are exact by construction.

Composing them needs no camera extrinsics file. That matters because the
extrinsics are a separate measurement that can drift, and because a projector
calibration that depends on them silently inherits their error; here the corner
plates are in the very frame the correspondences came from, so the two halves
are consistent by construction even if the rig has been nudged since the
extrinsics were solved.

**Both halves assume a pinhole, so frames must be undistorted first.** The
overhead module is strongly barrel-distorted -- its first radial coefficient is
about -0.43 -- and a homography has no term that can absorb that. Feeding raw
frames in fits a plausible-looking transform that is wrong towards the edges,
which is exactly where a drop target would be placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pick_and_place.spec.workspace import WORKSPACE_FRAME_APRILTAG_PLATES


@dataclass(frozen=True)
class HomographyFit:
    """A fitted planar map and how well it reproduces the points it was fitted to."""

    matrix: NDArray[np.float64]
    residual_mm: NDArray[np.float64]
    point_count: int

    @property
    def rms_mm(self) -> float:
        """Root-mean-square residual in millimetres."""
        return float(np.sqrt(np.mean(self.residual_mm**2)))

    @property
    def max_mm(self) -> float:
        """Worst single residual in millimetres."""
        return float(self.residual_mm.max())


def apply_homography(matrix: NDArray, points: NDArray) -> NDArray[np.float64]:
    """Map ``(N, 2)`` points through a 3x3 homography."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    mapped = homogeneous @ np.asarray(matrix, dtype=float).T
    return mapped[:, :2] / mapped[:, 2:3]


def workspace_plate_centers() -> dict[int, tuple[float, float]]:
    """Corner plate tag ids to their workspace-frame xy, in metres."""
    return {
        tag_id: (position[0], position[1])
        for tag_id, _, position in WORKSPACE_FRAME_APRILTAG_PLATES
    }


def camera_to_workspace(detections_by_id: dict[int, NDArray]) -> NDArray[np.float64]:
    """Homography from undistorted camera pixels to workspace metres.

    ``detections_by_id`` maps a corner plate's tag id to its detected centre in
    camera pixels. All four plates are required: three would still determine a
    homography, but with no redundancy left there would be nothing to check it
    against, and a misdetected plate would pass silently.
    """
    import cv2

    expected = workspace_plate_centers()
    missing = sorted(set(expected) - set(detections_by_id))
    if missing:
        raise ValueError(
            f"corner plates {missing} were not detected; the projector solve needs all four. "
            "Check the arm is not standing over one, and that the projected pattern is not "
            "washing one out."
        )

    ids = sorted(expected)
    image_points = np.array([detections_by_id[i] for i in ids], dtype=float).reshape(-1, 2)
    world_points = np.array([expected[i] for i in ids], dtype=float)

    matrix, _ = cv2.findHomography(image_points, world_points, 0)
    if matrix is None:
        raise ValueError("the four corner plates are degenerate -- they must not be collinear")
    return np.asarray(matrix, dtype=float)


def restrict_to_workspace(
    projector_xy: NDArray,
    camera_xy: NDArray,
    camera_to_workspace_matrix: NDArray,
    *,
    half_extent_m: float = 0.230,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Drop correspondences that landed off the workspace board.

    The projector throws wider than the board, and what spills over lands on the
    table, the frame rail, or the floor -- all at different heights. Those points
    are perfectly real and perfectly useless: a homography describes *one* plane,
    so a point on a different one cannot satisfy it at any pose.

    Leaving them in and trusting RANSAC to reject them is weaker than it looks.
    The spill can be a third of the points, and RANSAC finds the largest
    consistent set rather than the correct one, so a big enough off-board
    population can win the consensus outright. Filtering geometrically decides
    the question by where a point *is* rather than by how many friends it has.

    ``half_extent_m`` defaults to the corner plates' own offset, which is the
    largest square known to be on the board.
    """
    workspace_xy = apply_homography(camera_to_workspace_matrix, camera_xy)
    on_board = np.all(np.abs(workspace_xy) <= half_extent_m, axis=1)
    projector_xy = np.asarray(projector_xy, dtype=float).reshape(-1, 2)
    camera_xy = np.asarray(camera_xy, dtype=float).reshape(-1, 2)
    return projector_xy[on_board], camera_xy[on_board]


def solve_projector_to_workspace(
    projector_xy: NDArray,
    camera_xy: NDArray,
    camera_to_workspace_matrix: NDArray,
    *,
    cv2_module: Any,
    ransac_reproj_mm: float = 3.0,
) -> HomographyFit:
    """Fit projector pixels to workspace metres through the camera.

    RANSAC rather than a plain least squares: a Gray code cell that straddles a
    shadow edge decodes to a centroid pulled off the lit part, and a handful of
    those would otherwise tilt the whole fit. The threshold is in millimetres on
    the floor, which is the units the answer is actually judged in.
    """
    projector_xy = np.asarray(projector_xy, dtype=float).reshape(-1, 2)
    camera_xy = np.asarray(camera_xy, dtype=float).reshape(-1, 2)
    if len(projector_xy) < 4:
        raise ValueError(f"a homography needs at least 4 correspondences, got {len(projector_xy)}")

    workspace_xy = apply_homography(camera_to_workspace_matrix, camera_xy)

    matrix, inliers = cv2_module.findHomography(
        projector_xy,
        workspace_xy,
        cv2_module.RANSAC,
        ransac_reproj_mm / 1000.0,
    )
    if matrix is None:
        raise ValueError("could not fit a projector homography to these correspondences")

    predicted = apply_homography(matrix, projector_xy)
    residual_mm = np.linalg.norm(predicted - workspace_xy, axis=1) * 1000.0
    if inliers is not None:
        residual_mm = residual_mm[inliers.ravel().astype(bool)]

    return HomographyFit(
        matrix=np.asarray(matrix, dtype=float),
        residual_mm=residual_mm,
        point_count=len(residual_mm),
    )


def workspace_to_projector(matrix: NDArray) -> NDArray[np.float64]:
    """Invert the fit, which is the direction the target plate is actually placed in."""
    return np.linalg.inv(np.asarray(matrix, dtype=float))
