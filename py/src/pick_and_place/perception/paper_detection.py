# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Detect a square drop-zone target and map its center into world XY."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from pick_and_place.core.camera_projection import pixel_to_world_plane, project_to_pixel
from pick_and_place.spec.drop_zone import PaperTarget


class PaperTracker:
    """Stabilize drop-zone target detection over time."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        self._smoothed_center: NDArray | None = None
        self._smoothed_yaw: float | None = None
        self._smoothed_size: NDArray | None = None
        self._last_target: PaperTarget | None = None

    def reset(self) -> None:
        """Forget prior detections so the next estimate must be observed anew."""
        self._smoothed_center = None
        self._smoothed_yaw = None
        self._smoothed_size = None
        self._last_target = None

    def update(self, target: PaperTarget | None) -> PaperTarget | None:
        """Update the estimate with a new detection. Returns the smoothed target."""
        if target is None:
            return self._last_target

        size = np.array(target.half_extent) * 2.0
        yaw = target.yaw

        if self._smoothed_center is None or self.alpha <= 0.0:
            self._smoothed_center = target.center_world
            self._smoothed_yaw = yaw
            self._smoothed_size = size
        else:
            a = self.alpha
            self._smoothed_center = (1.0 - a) * self._smoothed_center + a * target.center_world
            self._smoothed_size = (1.0 - a) * self._smoothed_size + a * size

            # Smooth yaw with pi/2 symmetry so square detections do not jump.
            diff = (yaw - self._smoothed_yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
            self._smoothed_yaw += a * diff

        c, s = np.cos(self._smoothed_yaw), np.sin(self._smoothed_yaw)
        rot = np.array([[c, -s], [s, c]])
        hw, hh = self._smoothed_size / 2.0
        local_corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        world_corners = np.zeros((4, 3))
        world_corners[:, :2] = self._smoothed_center[:2] + local_corners @ rot.T
        world_corners[:, 2] = target.corners_world[0, 2]

        self._last_target = PaperTarget(
            center_px=target.center_px,
            corners_px=target.corners_px,
            center_world=self._smoothed_center,
            corners_world=world_corners,
            area_px=target.area_px,
            rectangularity=target.rectangularity,
        )
        return self._last_target


def draw_paper_target(bgr: NDArray, target: PaperTarget, scale_x: float, scale_y: float) -> None:
    """Outline the drop-zone target and its orientation on a BGR frame."""
    scale = np.array([scale_x, scale_y])
    corners = (target.corners_px * scale).astype(int)
    cv2.polylines(bgr, [corners.reshape(-1, 1, 2)], True, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(bgr, tuple(corners[0]), tuple(corners[1]), (0, 255, 255), 3, cv2.LINE_AA)

    center = (target.center_px * scale).astype(int)
    cv2.circle(bgr, tuple(center), 4, (0, 0, 255), -1)
    mid_first = ((corners[0] + corners[1]) / 2).astype(int)
    cv2.line(bgr, tuple(center), tuple(mid_first), (0, 0, 255), 2, cv2.LINE_AA)


MIN_TARGET_AREA_FRACTION = 0.008
MAX_TARGET_AREA_FRACTION = 0.15
MAX_TARGET_ASPECT = 1.35
MIN_TARGET_RECTANGULARITY = 0.82


@dataclass(frozen=True)
class TargetCandidate:
    """One mask contour, measured against the drop-zone square filter."""

    center_px: NDArray
    box_px: NDArray
    area_px: float
    area_fraction: float
    corner_count: int
    convex: bool
    aspect: float
    rectangularity: float
    rejection: str | None


def target_mask(
    frame_rgb: NDArray,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    target_color: str = "black",
    workspace_corners_world: NDArray | None = None,
) -> NDArray:
    """Mark the pixels that could belong to a black or white drop-zone square.

    When ``workspace_corners_world`` is given, the search is restricted to that
    world-space quad projected into the image, so off-table clutter cannot be
    mistaken for or merged into the target.
    """
    image = np.asarray(frame_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame_rgb must have shape (height, width, 3)")
    height, width = image.shape[:2]

    roi: NDArray | None = None
    if workspace_corners_world is not None:
        quad_px = project_to_pixel(
            workspace_corners_world, camera_matrix, camera_position, camera_rotation
        )
        roi = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(roi, np.round(quad_px).astype(np.int32), 255)

    # Local (adaptive) thresholding keys on "darker/brighter than the immediate
    # surroundings" rather than an absolute cutoff. An uneven illumination
    # gradient across the table therefore does not lump the target in with a
    # dimly lit corner, and a uniformly dark region is not flagged at all. The
    # block spans a sizeable fraction of the frame so the local mean is set by
    # the table around the target rather than by the target itself.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    block = max(31, round(0.1 * width)) | 1
    if target_color == "black":
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15
        )
    elif target_color == "white":
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, 15
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_and(mask, cv2.inRange(hsv, (0, 0, 0), (180, 60, 255)))
    else:
        raise ValueError(f"Unknown target_color: {target_color!r}")

    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)

    # Open to sever thin bridges to neighbouring blobs, then close to fill
    # speckle and glare holes inside the target.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))


def target_rejection(
    *,
    area_fraction: float,
    corner_count: int,
    convex: bool,
    aspect: float,
    rectangularity: float,
    min_area_fraction: float = MIN_TARGET_AREA_FRACTION,
    max_area_fraction: float = MAX_TARGET_AREA_FRACTION,
) -> str | None:
    """Name the filter a measured contour fails, or ``None`` if it is a target."""
    if area_fraction < min_area_fraction:
        return "too small"
    if area_fraction > max_area_fraction:
        return "too large"
    if corner_count != 4 or not convex:
        return f"not a convex quadrilateral ({corner_count} corners)"
    if not np.isfinite(aspect):
        return "degenerate"
    if aspect > MAX_TARGET_ASPECT:
        return "not square enough"
    if rectangularity < MIN_TARGET_RECTANGULARITY:
        return "too ragged"
    return None


def target_candidates(
    mask: NDArray,
    *,
    min_area_fraction: float = MIN_TARGET_AREA_FRACTION,
    max_area_fraction: float = MAX_TARGET_AREA_FRACTION,
) -> list[TargetCandidate]:
    """Measure every contour in the mask and say why each is not the target."""
    image_area = float(mask.shape[0] * mask.shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    measured: list[TargetCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        corners = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        convex = bool(cv2.isContourConvex(corners))
        rect = cv2.minAreaRect(contour)
        side_a, side_b = rect[1]
        degenerate = min(side_a, side_b) <= 0.0
        aspect = np.inf if degenerate else max(side_a, side_b) / min(side_a, side_b)
        rectangularity = 0.0 if degenerate else area / (side_a * side_b)
        area_fraction = area / image_area
        measured.append(
            TargetCandidate(
                center_px=np.asarray(rect[0], dtype=float),
                box_px=cv2.boxPoints(rect).astype(float),
                area_px=area,
                area_fraction=area_fraction,
                corner_count=len(corners),
                convex=convex,
                aspect=float(aspect),
                rectangularity=float(rectangularity),
                rejection=target_rejection(
                    area_fraction=area_fraction,
                    corner_count=len(corners),
                    convex=convex,
                    aspect=float(aspect),
                    rectangularity=float(rectangularity),
                    min_area_fraction=min_area_fraction,
                    max_area_fraction=max_area_fraction,
                ),
            )
        )
    return measured


def _candidate_target(
    candidate: TargetCandidate,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    plane_z: float,
) -> PaperTarget | None:
    """Lift an accepted contour onto the table plane, or drop it if it misses."""
    center_world = pixel_to_world_plane(
        candidate.center_px,
        camera_matrix,
        camera_position,
        camera_rotation,
        plane_z=plane_z,
    )
    if center_world is None:
        return None

    world_corners = [
        pixel_to_world_plane(
            corner,
            camera_matrix,
            camera_position,
            camera_rotation,
            plane_z=plane_z,
        )
        for corner in candidate.box_px
    ]
    if any(corner is None for corner in world_corners):
        return None

    return PaperTarget(
        center_px=candidate.center_px,
        corners_px=candidate.box_px,
        center_world=center_world,
        corners_world=np.asarray(world_corners, dtype=float),
        area_px=candidate.area_px,
        rectangularity=candidate.rectangularity,
    )


def detect_paper_target(
    frame_rgb: NDArray,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    plane_z: float = 0.0,
    min_area_fraction: float = MIN_TARGET_AREA_FRACTION,
    max_area_fraction: float = MAX_TARGET_AREA_FRACTION,
    target_color: str = "black",
    workspace_corners_world: NDArray | None = None,
) -> PaperTarget | None:
    """Find the strongest black or white drop-zone square contour."""
    mask = target_mask(
        frame_rgb,
        camera_matrix,
        camera_position,
        camera_rotation,
        target_color=target_color,
        workspace_corners_world=workspace_corners_world,
    )
    scored: list[tuple[float, PaperTarget]] = []
    for candidate in target_candidates(
        mask,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
    ):
        if candidate.rejection is not None:
            continue
        target = _candidate_target(
            candidate,
            camera_matrix,
            camera_position,
            camera_rotation,
            plane_z=plane_z,
        )
        if target is None:
            continue
        scored.append(
            (candidate.area_px * candidate.rectangularity / candidate.aspect, target)
        )

    return max(scored, key=lambda item: item[0])[1] if scored else None
