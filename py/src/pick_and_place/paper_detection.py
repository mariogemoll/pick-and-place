# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Detect a square drop-zone target and map its center into world XY."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PAPER_TARGET_MARKER_NAME = "paper_target_marker"


@dataclass(frozen=True)
class PaperTarget:
    """Detected drop-zone geometry in the image and on a horizontal world plane."""

    center_px: NDArray
    corners_px: NDArray
    center_world: NDArray
    corners_world: NDArray
    area_px: float
    rectangularity: float

    @property
    def xy(self) -> tuple[float, float]:
        return float(self.center_world[0]), float(self.center_world[1])

    @property
    def yaw(self) -> float:
        """Yaw angle (radians) of the square's first edge in world XY."""
        edge = self.corners_world[1] - self.corners_world[0]
        return float(np.arctan2(edge[1], edge[0]))


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

        edge_x = target.corners_world[1] - target.corners_world[0]
        edge_y = target.corners_world[2] - target.corners_world[1]
        size = np.array([np.linalg.norm(edge_x[:2]), np.linalg.norm(edge_y[:2])])
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
    import cv2

    scale = np.array([scale_x, scale_y])
    corners = (target.corners_px * scale).astype(int)
    cv2.polylines(bgr, [corners.reshape(-1, 1, 2)], True, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(bgr, tuple(corners[0]), tuple(corners[1]), (0, 255, 255), 3, cv2.LINE_AA)

    center = (target.center_px * scale).astype(int)
    cv2.circle(bgr, tuple(center), 4, (0, 0, 255), -1)
    mid_first = ((corners[0] + corners[1]) / 2).astype(int)
    cv2.line(bgr, tuple(center), tuple(mid_first), (0, 0, 255), 2, cv2.LINE_AA)


def pixel_to_world_plane(
    pixel: NDArray,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    plane_z: float = 0.0,
) -> NDArray | None:
    """Intersect an undistorted image pixel's camera ray with ``z=plane_z``."""
    pixel_h = np.array((float(pixel[0]), float(pixel[1]), 1.0))
    ray_cv = np.linalg.inv(np.asarray(camera_matrix, dtype=float)) @ pixel_h
    ray_mj = np.array((ray_cv[0], -ray_cv[1], -ray_cv[2]))
    ray_world = np.asarray(camera_rotation, dtype=float) @ ray_mj
    origin = np.asarray(camera_position, dtype=float)
    if abs(ray_world[2]) < 1e-9:
        return None
    distance = (float(plane_z) - origin[2]) / ray_world[2]
    if distance <= 0.0:
        return None
    return origin + distance * ray_world


def project_to_pixel(
    points_world: NDArray,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
) -> NDArray:
    """Project world points onto the image plane (inverse of ``pixel_to_world_plane``)."""
    points = np.asarray(points_world, dtype=float).reshape(-1, 3)
    rays_mj = (points - np.asarray(camera_position, dtype=float)) @ np.asarray(
        camera_rotation, dtype=float
    )
    rays_cv = rays_mj * np.array((1.0, -1.0, -1.0))
    projected = rays_cv @ np.asarray(camera_matrix, dtype=float).T
    return projected[:, :2] / projected[:, 2:3]


def add_paper_target_marker(spec) -> None:
    """Add a hidden, non-colliding drop-zone marker to an ``MjSpec`` before compile."""
    import mujoco

    body = spec.worldbody.add_body(name=PAPER_TARGET_MARKER_NAME, pos=(0.0, 0.0, 0.0))
    body.add_geom(
        name=PAPER_TARGET_MARKER_NAME + "_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, 0.002),
        size=(0.05, 0.05, 0.001),
        rgba=(1.0, 1.0, 1.0, 0.0),
        contype=0,
        conaffinity=0,
        group=1,
    )


def set_paper_target_marker(model, data, target: PaperTarget, *, usable: bool) -> None:
    """Show ``target`` as a translucent square in an already compiled model."""
    import math
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAPER_TARGET_MARKER_NAME)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PAPER_TARGET_MARKER_NAME + "_geom")
    if body_id < 0 or geom_id < 0:
        return

    edge_x = target.corners_world[1] - target.corners_world[0]
    edge_y = target.corners_world[2] - target.corners_world[1]
    model.body_pos[body_id] = (*target.center_world[:2], 0.0)
    yaw = target.yaw
    model.body_quat[body_id] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    model.geom_size[geom_id] = (
        max(float(np.linalg.norm(edge_x[:2])) / 2.0, 0.001),
        max(float(np.linalg.norm(edge_y[:2])) / 2.0, 0.001),
        0.001,
    )
    model.geom_rgba[geom_id] = (
        (0.12, 0.12, 0.12, 0.72) if usable else (1.0, 0.45, 0.05, 0.65)
    )
    mujoco.mj_forward(model, data)


def detect_paper_target(
    frame_rgb: NDArray,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    plane_z: float = 0.0,
    min_area_fraction: float = 0.008,
    max_area_fraction: float = 0.15,
    target_color: str = "black",
    workspace_corners_world: NDArray | None = None,
) -> PaperTarget | None:
    """Find the strongest black or white drop-zone square contour.

    When ``workspace_corners_world`` is given, the search is restricted to that
    world-space quad projected into the image, so off-table clutter cannot be
    mistaken for or merged into the target.
    """
    import cv2

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))

    image_area = float(image.shape[0] * image.shape[1])
    min_area = min_area_fraction * image_area
    max_area = max_area_fraction * image_area
    candidates: list[tuple[float, PaperTarget]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(corners) != 4 or not cv2.isContourConvex(corners):
            continue

        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) <= 0.0:
            continue
        aspect = max(width, height) / min(width, height)
        rectangularity = area / (width * height)
        if aspect > 1.35 or rectangularity < 0.82:
            continue

        center_px = np.asarray(rect[0], dtype=float)
        center_world = pixel_to_world_plane(
            center_px,
            camera_matrix,
            camera_position,
            camera_rotation,
            plane_z=plane_z,
        )
        if center_world is None:
            continue

        box_corners = cv2.boxPoints(rect).astype(float)
        world_corners = [
            pixel_to_world_plane(
                corner,
                camera_matrix,
                camera_position,
                camera_rotation,
                plane_z=plane_z,
            )
            for corner in box_corners
        ]
        if any(corner is None for corner in world_corners):
            continue

        target = PaperTarget(
            center_px=center_px,
            corners_px=box_corners,
            center_world=center_world,
            corners_world=np.asarray(world_corners, dtype=float),
            area_px=area,
            rectangularity=float(rectangularity),
        )
        candidates.append((area * rectangularity / aspect, target))

    return max(candidates, key=lambda item: item[0])[1] if candidates else None
