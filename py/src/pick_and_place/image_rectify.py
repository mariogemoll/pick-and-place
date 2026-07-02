# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Undistort a real camera frame into the same rectified square pinhole view
the sim cameras render and the VLA policy is trained on.

A real camera stores frames at its native, lens-distorted resolution. The sim
recordings and the VLA input, by contrast, are square crops of an ideal
pinhole view. ``build_undistort_map`` derives that rectified pinhole from a
camera's calibrated intrinsics, scaled to the frame's actual resolution, using
focal length ``fy`` on both axes with the principal point at the image center
-- the same geometry ``SimCameraRig`` feeds MuJoCo. ``transform_frame`` then
applies the map, center-crops to a square (keeping the full image height), and
resizes to the policy's input size.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SQUARE_SIZE = 512


def build_undistort_map(
    intrinsics: dict[str, Any], frame_w: int, frame_h: int, cv2: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort/rectify map for a frame of size ``frame_w`` x ``frame_h``."""
    matrix = np.array(intrinsics["camera_matrix"], dtype=float)
    dist = np.array(intrinsics["dist_coeffs"], dtype=float)
    matrix[0, :] *= frame_w / float(intrinsics["width"])
    matrix[1, :] *= frame_h / float(intrinsics["height"])

    fy = float(matrix[1, 1])
    rect_matrix = np.array(
        [[fy, 0.0, frame_w / 2.0], [0.0, fy, frame_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return cv2.initUndistortRectifyMap(
        matrix, dist, None, rect_matrix, (frame_w, frame_h), cv2.CV_16SC2
    )


def transform_frame(
    rgb: np.ndarray, undistort_map: tuple[np.ndarray, np.ndarray], size: int, cv2: Any
) -> np.ndarray:
    """Undistort, center-crop to a square, and resize to ``size`` x ``size``."""
    rectified = cv2.remap(rgb, undistort_map[0], undistort_map[1], cv2.INTER_LINEAR)
    h, w = rectified.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    crop = rectified[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def rectified_square_camera_matrix(intrinsics: dict[str, Any], size: int) -> list[list[float]]:
    """The 3x3 camera matrix a frame processed by ``transform_frame`` obeys.

    Mirrors ``build_undistort_map``'s rectified pinhole (focal length ``fy`` on
    both axes, principal point at the frame center) without needing a frame or
    ``cv2``: the center-square crop keeps that already-centered principal point
    centered, and the final resize scales everything uniformly, so the result
    is an analytic function of the calibration alone.
    """
    matrix = np.array(intrinsics["camera_matrix"], dtype=float)
    width = float(intrinsics["width"])
    height = float(intrinsics["height"])
    fy = float(matrix[1, 1])
    side = min(width, height)
    scale = size / side
    f = fy * scale
    c = size / 2.0
    return [[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]]
