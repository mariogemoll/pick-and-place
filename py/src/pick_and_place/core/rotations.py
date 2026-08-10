# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Quaternion helpers in MuJoCo's ``wxyz`` order.

MuJoCo stores quaternions scalar-first while ``scipy`` stores them
scalar-last, so every conversion between the two has to reorder the
components. These helpers do it in one place, and keep the canonical sign
(non-negative scalar part) so two quaternions describing the same rotation
compare equal componentwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def mat_to_quat_wxyz(matrix: NDArray) -> NDArray:
    """Convert a 3x3 rotation matrix to a canonical MuJoCo wxyz quaternion."""
    xyzw = Rotation.from_matrix(matrix).as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
    return quat if quat[0] >= 0.0 else -quat


def quat_wxyz_to_rotation_6d(quaternion: NDArray) -> NDArray:
    """Convert wxyz quaternion(s) to the first two rotation-matrix columns.

    The output order is column-by-column: ``[r00, r10, r20, r01, r11, r21]``.
    This is deliberately not the row-major flattening of ``matrix[:, :2]``.
    """
    quaternion = np.asarray(quaternion)
    if quaternion.shape[-1:] != (4,):
        raise ValueError("quaternion must end in four wxyz components")
    xyzw = quaternion[..., [1, 2, 3, 0]]
    matrix = Rotation.from_quat(xyzw).as_matrix()
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def quat_angle_deg(q0: NDArray, q1: NDArray) -> float:
    """Return the shortest angular distance between two wxyz quaternions."""
    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    return float(np.degrees((r0.inv() * r1).magnitude()))


def average_quaternions_wxyz(quaternions: list[NDArray]) -> NDArray:
    """Average same-hemisphere wxyz quaternions and normalize the result."""
    if not quaternions:
        raise ValueError("cannot average zero quaternions")
    quats_xyzw = [[q[1], q[2], q[3], q[0]] for q in quaternions]
    xyzw = Rotation.from_quat(quats_xyzw).mean().as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
    return quat if quat[0] >= 0.0 else -quat


def pose_delta_mm_deg(
    pos_a: NDArray,
    quat_a: NDArray,
    pos_b: NDArray,
    quat_b: NDArray,
) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between two poses in the same frame."""
    mm = float(
        np.linalg.norm(np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float)) * 1000.0
    )
    deg = quat_angle_deg(np.asarray(quat_a, dtype=float), np.asarray(quat_b, dtype=float))
    return mm, deg
