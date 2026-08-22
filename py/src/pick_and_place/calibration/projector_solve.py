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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pick_and_place.core.paths import REPO_ROOT
from pick_and_place.core.workspace_bounds import WORKSPACE_FRAME_INNER_HALF_EXTENT
from pick_and_place.spec.workspace import WORKSPACE_FRAME_APRILTAG_PLATES

#: Where ``pap calibrate-projector`` writes its answer. Machine-local and
#: gitignored, like the camera calibration it sits beside: it describes one
#: projector standing in one spot, not the project.
CALIBRATION_PATH = REPO_ROOT / "config" / "projector" / "overhead_projector.json"


@dataclass(frozen=True)
class ProjectorCalibration:
    """A solved projector, read back off disk."""

    projector_to_workspace: NDArray[np.float64]
    projector_size: tuple[int, int]
    solved: str
    rms_mm: float


def load_projector_calibration(path: Path | None = None) -> ProjectorCalibration:
    """Read a solved projector calibration, or say how to make one."""
    resolved = path if path is not None else CALIBRATION_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"no projector calibration at {resolved}. Run `pap calibrate-projector` first; "
            "it is invalidated by moving the projector, so this is expected after a remount."
        )
    data = json.loads(resolved.read_text())
    width, height = data["projector_size"]
    return ProjectorCalibration(
        projector_to_workspace=np.array(data["homography"], dtype=float),
        projector_size=(int(width), int(height)),
        solved=str(data.get("solved", "unknown")),
        rms_mm=float(data.get("rms_mm", float("nan"))),
    )


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


def image_to_workspace_square(
    image_size: tuple[int, int],
    *,
    half_extent_m: float = WORKSPACE_FRAME_INNER_HALF_EXTENT,
    stretch: bool = False,
) -> NDArray[np.float64]:
    """Map an image's pixels onto the square inside the workspace frame rails.

    The default extent is the frame's **inner rail**, not the corner plates'
    centres: the plates sit 32.6 mm inside the rails, so filling only as far as
    them leaves a visible unlit border all the way round.

    **The image's bottom edge faces the arm.** The arm stands on the frame's
    north side, and the workspace is habitually looked at with the robot at the
    bottom, so the picture is laid down turned to match that view rather than
    the frame's own axes. It is a half turn, not a mirror, so lettering still
    reads. A caller wanting some other orientation rotates the image itself.

    ``stretch`` fills the square regardless of the image's shape. The default
    instead fits the image inside it at its own aspect ratio and centres what is
    left over, because the square is 1:1 and almost no photograph is.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image must have positive extent, got {image_size}")

    half_x = half_extent_m
    half_y = half_extent_m
    if not stretch:
        aspect = width / height
        if aspect >= 1.0:
            half_y = half_extent_m / aspect
        else:
            half_x = half_extent_m * aspect

    source = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=float
    )
    # A half turn of the frame's own axes: the image's top edge runs se -> sw,
    # putting the arm's side of the frame along the picture's bottom.
    destination = np.array(
        [[half_x, -half_y], [-half_x, -half_y], [-half_x, half_y], [half_x, half_y]],
        dtype=float,
    )

    import cv2

    matrix, _ = cv2.findHomography(source, destination, 0)
    return np.asarray(matrix, dtype=float)
