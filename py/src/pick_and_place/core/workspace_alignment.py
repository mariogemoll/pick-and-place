# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Align the authored workspace frame with where the overhead camera sees it.

The four workspace-frame AprilTag plates (ids 12-15) have authored positions in
the scene model, but the overhead camera's real detections of them sit up to
~1.5 cm away — stale extrinsics, or the physical frame having moved since
calibration. Since the plates are visible in every overhead frame, that
discrepancy can be fit per episode as a rigid 2D transform (yaw about vertical
plus table-plane translation) and then removed from every pose the overhead
camera localizes (the pick cube, most importantly).

Conventions: ``fit_alignment`` returns the transform ``A`` with
``detected ~= A(authored)``; ``correct_point``/``correct_yaw`` apply ``A``-inverse
to a measured world pose, mapping it back into the authored frame that the sim
scene and the arm's kinematics assume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkspaceAlignment:
    """Rigid 2D transform from authored to detected table coordinates."""

    yaw: float
    tx: float
    ty: float
    num_tags: int
    residual_mm: float

    def correct_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a detected world point back into the authored frame."""
        dx = x - self.tx
        dy = y - self.ty
        cos = math.cos(-self.yaw)
        sin = math.sin(-self.yaw)
        return cos * dx - sin * dy, sin * dx + cos * dy

    def correct_yaw(self, yaw: float) -> float:
        return yaw - self.yaw


IDENTITY_ALIGNMENT = WorkspaceAlignment(yaw=0.0, tx=0.0, ty=0.0, num_tags=0, residual_mm=0.0)


def pixel_to_table_point(
    pixel: np.ndarray,
    camera_matrix: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    plane_z: float,
) -> tuple[float, float]:
    """Intersect a pixel's viewing ray with the horizontal plane ``z = plane_z``.

    ``cam_rot`` is the MuJoCo camera rotation (x right, y up, looking along -z);
    the pixel is in an undistorted image obeying ``camera_matrix`` (OpenCV
    convention, y down).
    """
    direction_cv = np.linalg.inv(camera_matrix) @ np.array([pixel[0], pixel[1], 1.0])
    rot_cv = cam_rot @ np.diag([1.0, -1.0, -1.0])
    direction = rot_cv @ direction_cv
    t = (plane_z - cam_pos[2]) / direction[2]
    point = cam_pos + t * direction
    return float(point[0]), float(point[1])


def fit_alignment(
    authored: dict[int, tuple[float, float]],
    detected: dict[int, tuple[float, float]],
) -> WorkspaceAlignment | None:
    """Least-squares rigid 2D fit ``detected ~= R(yaw) @ authored + t``.

    Needs at least two tags seen in both sets; returns None otherwise.
    """
    ids = sorted(set(authored) & set(detected))
    if len(ids) < 2:
        return None
    a = np.array([authored[i] for i in ids], dtype=float)
    d = np.array([detected[i] for i in ids], dtype=float)
    a_mean = a.mean(axis=0)
    d_mean = d.mean(axis=0)
    a_c = a - a_mean
    d_c = d - d_mean
    # 2D Kabsch: the optimal rotation angle from the cross-covariance.
    cov = a_c.T @ d_c
    yaw = math.atan2(cov[0, 1] - cov[1, 0], cov[0, 0] + cov[1, 1])
    cos, sin = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cos, -sin], [sin, cos]])
    translation = d_mean - rotation @ a_mean
    residuals = d - (a @ rotation.T + translation)
    return WorkspaceAlignment(
        yaw=yaw,
        tx=float(translation[0]),
        ty=float(translation[1]),
        num_tags=len(ids),
        residual_mm=float(np.sqrt((residuals**2).sum(axis=1)).mean() * 1000.0),
    )
