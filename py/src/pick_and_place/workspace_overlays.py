# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fixed MuJoCo meshes for the standard SO-101 workspace overlays."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

# These values are pinned from ts/src/visualizations/workspace-overlay.ts and
# ts/src/ik/workspace.ts for the stock SO-101 model. Recompute them when the
# robot kinematics or the TypeScript workspace definitions change.
_PAN_AXIS = (0.0388353, -8.97657e-09)
_AZIMUTH_MIN = -1.9198621771937634
_AZIMUTH_LENGTH = 3.839724354387525

WORKSPACE_OVERLAY_GROUP = 4
_SEGMENTS = 96
_HALF_THICKNESS = 0.00001
_RGBA = (1.0, 0.4667, 0.0, 0.22)


@dataclass(frozen=True)
class WorkspaceOverlay:
    """A fixed annular sector projected onto the floor."""

    name: str
    inner_radius: float
    outer_radius: float
    z: float


WORKSPACE_OVERLAYS = (
    WorkspaceOverlay("workspace_global", 0.0, 0.4418431804405771, 0.0002),
    WorkspaceOverlay("workspace_ground_height_arm", 0.0, 0.42911855464836285, 0.0004),
    WorkspaceOverlay(
        "workspace_ground_pregrasp",
        0.07330432949931037,
        0.25536116910332146,
        0.0006,
    ),
    WorkspaceOverlay(
        "workspace_clearance_pregrasp",
        0.08671592519116689,
        0.24194957341146492,
        0.0008,
    ),
)


def add_workspace_overlays(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    *,
    prefix: str = "",
) -> None:
    """Add standard non-colliding workspace overlays in ``parent`` coordinates."""
    for overlay in WORKSPACE_OVERLAYS:
        name = f"{prefix}{overlay.name}"
        vertices, faces = _annular_sector_mesh(
            overlay.inner_radius,
            overlay.outer_radius,
        )
        mesh = spec.add_mesh(name=name)
        mesh.uservert = vertices.flatten()
        mesh.userface = faces.flatten()
        parent.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=(*_PAN_AXIS, overlay.z),
            rgba=_RGBA,
            contype=0,
            conaffinity=0,
            group=WORKSPACE_OVERLAY_GROUP,
        )


def _annular_sector_mesh(
    inner_radius: float,
    outer_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed, thin annular-sector mesh centered at the origin."""
    angles = np.linspace(_AZIMUTH_MIN, _AZIMUTH_MIN + _AZIMUTH_LENGTH, _SEGMENTS + 1)
    outer_xy = np.column_stack((outer_radius * np.cos(angles), outer_radius * np.sin(angles)))
    if math.isclose(inner_radius, 0.0):
        contour_xy = np.vstack((outer_xy, (0.0, 0.0)))
    else:
        inner_xy = np.column_stack((inner_radius * np.cos(angles), inner_radius * np.sin(angles)))
        contour_xy = np.vstack((outer_xy, inner_xy[::-1]))

    count = len(contour_xy)
    bottom = np.column_stack((contour_xy, np.full(count, -_HALF_THICKNESS)))
    top = np.column_stack((contour_xy, np.full(count, _HALF_THICKNESS)))
    vertices = np.vstack((bottom, top))

    faces: list[tuple[int, int, int]] = []
    if math.isclose(inner_radius, 0.0):
        center = count - 1
        for i in range(_SEGMENTS):
            faces.extend(((center, i + 1, i), (center + count, i + count, i + 1 + count)))
    else:
        for i in range(_SEGMENTS):
            outer_a = i
            outer_b = i + 1
            inner_a = count - 1 - i
            inner_b = count - 2 - i
            faces.extend(
                (
                    (outer_a, inner_a, outer_b),
                    (outer_b, inner_a, inner_b),
                    (outer_a + count, outer_b + count, inner_a + count),
                    (outer_b + count, inner_b + count, inner_a + count),
                )
            )

    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, j, i + count), (j, j + count, i + count)))

    return vertices, np.asarray(faces, dtype=int)
