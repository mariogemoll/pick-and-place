# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Minimal 4x4 transform helpers for the trajectory math.

Matrices are 4x4 ``numpy`` arrays in the column-vector convention: a point ``p``
transforms as ``M @ [x, y, z, 1]``. Rotations are right-handed and intrinsic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]
Mat4 = NDArray[np.float64]


def identity() -> Mat4:
    return np.eye(4, dtype=np.float64)


def translation(x: float, y: float, z: float) -> Mat4:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


def rot_x(theta: float) -> Mat4:
    c, s = np.cos(theta), np.sin(theta)
    m = np.eye(4, dtype=np.float64)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rot_y(theta: float) -> Mat4:
    c, s = np.cos(theta), np.sin(theta)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rot_z(theta: float) -> Mat4:
    c, s = np.cos(theta), np.sin(theta)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def rotation_zyx(roll: float, pitch: float, yaw: float) -> Mat4:
    """Intrinsic ``ZYX`` Euler rotation: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def make_basis(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Mat4:
    """Rotation whose columns are the given basis vectors."""
    m = np.eye(4, dtype=np.float64)
    m[:3, 0] = x_axis
    m[:3, 1] = y_axis
    m[:3, 2] = z_axis
    return m


def transform_point(m: Mat4, v: Vec3) -> Vec3:
    """Apply the full affine transform to a point."""
    p = m @ np.array((v[0], v[1], v[2], 1.0))
    return p[:3]


def transform_direction(m: Mat4, v: Vec3) -> Vec3:
    """Apply the rotation part and normalize."""
    d = m[:3, :3] @ np.asarray(v, dtype=np.float64)
    return d / np.linalg.norm(d)


def get_position(m: Mat4) -> Vec3:
    return m[:3, 3].copy()


def with_position(m: Mat4, p: Vec3) -> Mat4:
    out = m.copy()
    out[:3, 3] = p
    return out
