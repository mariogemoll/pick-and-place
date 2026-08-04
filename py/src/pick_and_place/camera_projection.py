# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Move points between an image and a horizontal world plane.

A pinhole camera with a known pose and matrix, and the flat table it looks at,
are enough to turn a pixel into a world point and back. Both directions are
plain linear algebra, so detectors, renderers and diagnostics share them rather
than each carrying their own sign conventions.

Those conventions are the one subtlety: OpenCV's camera frame looks down +z with
+y downward, MuJoCo's looks down -z with +y up, so the ray flips in y and z on
the way between them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pixel_to_world_plane(
    pixel: NDArray,
    camera_matrix: NDArray,
    camera_position: NDArray,
    camera_rotation: NDArray,
    *,
    plane_z: float = 0.0,
) -> NDArray | None:
    """Intersect an undistorted image pixel's camera ray with ``z=plane_z``.

    ``None`` when the ray runs parallel to the plane or hits it behind the
    camera.
    """
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
