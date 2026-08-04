# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Draw the workspace sectors into a MuJoCo scene as thin floor decals.

One mesh per sector from :mod:`pick_and_place.workspace_bounds`, plus a box over
each corner AprilTag plate marking the cube exclusion zone. Everything added here
is non-colliding and lives in its own visual group, so a viewer can toggle it and
physics never sees it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pick_and_place.spec.workspace import WORKSPACE_FRAME_APRILTAG_PLATES
from pick_and_place.workspace_bounds import (
    CANONICAL_PICKUP_SECTOR,
    CUBE_APRILTAG_EXCLUSION_HALF_EXTENT,
    CUBE_PLACEMENT_BOUNDS,
    CUBE_PLACEMENT_SECTOR,
    PAN_AXIS,
    REACH_SECTORS,
    frame_to_world_xy,
)

WORKSPACE_OVERLAY_GROUP = 4
_SEGMENTS = 96
_HALF_THICKNESS = 0.00001
_RGBA = (1.0, 0.4667, 0.0, 0.22)
_CUBE_PLACEMENT_RGBA = (0.1333, 0.7725, 0.3686, 0.42)
_CUBE_EXCLUSION_RGBA = (0.9373, 0.2667, 0.2667, 0.62)
_CANONICAL_PICKUP_RGBA = (0.0, 0.45, 1.0, 0.28)


def add_workspace_overlays(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    *,
    prefix: str = "",
) -> None:
    """Add standard non-colliding workspace overlays in ``parent`` coordinates."""
    for sector in REACH_SECTORS:
        name = f"{prefix}{sector.name}"
        vertices, faces = _annular_sector_mesh(
            sector.inner_radius,
            sector.outer_radius,
            sector.azimuth_min,
            sector.azimuth_max,
        )
        mesh = spec.add_mesh(name=name)
        mesh.uservert = vertices.flatten()
        mesh.userface = faces.flatten()
        parent.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=(*PAN_AXIS, sector.z),
            rgba=_RGBA,
            contype=0,
            conaffinity=0,
            group=WORKSPACE_OVERLAY_GROUP,
        )

    for sector, rgba in (
        (CUBE_PLACEMENT_SECTOR, _CUBE_PLACEMENT_RGBA),
        (CANONICAL_PICKUP_SECTOR, _CANONICAL_PICKUP_RGBA),
    ):
        vertices, faces = _clipped_annular_sector_mesh(
            sector.inner_radius,
            sector.outer_radius,
            CUBE_PLACEMENT_BOUNDS,
            sector.azimuth_min,
            sector.azimuth_max,
        )
        mesh = spec.add_mesh(name=sector.name)
        mesh.uservert = vertices.flatten()
        mesh.userface = faces.flatten()
        parent.add_geom(
            name=sector.name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=(*PAN_AXIS, sector.z),
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=WORKSPACE_OVERLAY_GROUP,
        )

    for _, corner_name, tag_pos in WORKSPACE_FRAME_APRILTAG_PLATES:
        tag_x, tag_y = frame_to_world_xy(tag_pos[0], tag_pos[1])
        parent.add_geom(
            name=f"workspace_cube_exclusion_tag_{corner_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                CUBE_APRILTAG_EXCLUSION_HALF_EXTENT,
                CUBE_APRILTAG_EXCLUSION_HALF_EXTENT,
                _HALF_THICKNESS,
            ),
            pos=(tag_x, tag_y, CUBE_PLACEMENT_SECTOR.z + 0.00004),
            rgba=_CUBE_EXCLUSION_RGBA,
            contype=0,
            conaffinity=0,
            group=WORKSPACE_OVERLAY_GROUP,
        )


def _clipped_annular_sector_mesh(
    inner_radius: float,
    outer_radius: float,
    bounds: tuple[float, float, float, float],
    azimuth_min: float,
    azimuth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the annular sector clipped to an axis-aligned center rectangle."""
    x_min, x_max, y_min, y_max = bounds
    local_bounds = (
        x_min - PAN_AXIS[0],
        x_max - PAN_AXIS[0],
        y_min - PAN_AXIS[1],
        y_max - PAN_AXIS[1],
    )
    sections: list[tuple[float, float, float]] = []
    for angle in np.linspace(azimuth_min, azimuth_max, _SEGMENTS + 1):
        dx, dy = math.cos(angle), math.sin(angle)
        exits = []
        if dx > 0.0:
            exits.append(local_bounds[1] / dx)
        elif dx < 0.0:
            exits.append(local_bounds[0] / dx)
        if dy > 0.0:
            exits.append(local_bounds[3] / dy)
        elif dy < 0.0:
            exits.append(local_bounds[2] / dy)
        clipped_outer = min(outer_radius, *exits)
        if clipped_outer >= inner_radius:
            sections.append((float(angle), inner_radius, clipped_outer))

    if len(sections) < 2:
        raise ValueError("cube placement bounds do not intersect the workspace")

    vertices: list[tuple[float, float, float]] = []
    for z in (-_HALF_THICKNESS, _HALF_THICKNESS):
        for angle, inner, outer in sections:
            vertices.append((inner * math.cos(angle), inner * math.sin(angle), z))
            vertices.append((outer * math.cos(angle), outer * math.sin(angle), z))

    count = len(sections)
    layer = 2 * count
    faces: list[tuple[int, int, int]] = []
    for i in range(count - 1):
        bi, bo = 2 * i, 2 * i + 1
        ni, no = bi + 2, bo + 2
        faces.extend(((bi, bo, ni), (bo, no, ni)))
        faces.extend(((bi + layer, ni + layer, bo + layer), (bo + layer, ni + layer, no + layer)))
        faces.extend(((bi, ni, bi + layer), (ni, ni + layer, bi + layer)))
        faces.extend(((bo, bo + layer, no), (no, bo + layer, no + layer)))

    for inner_index, outer_index in ((0, 1), (2 * (count - 1), 2 * (count - 1) + 1)):
        faces.extend(
            (
                (inner_index, inner_index + layer, outer_index),
                (outer_index, inner_index + layer, outer_index + layer),
            )
        )
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def _annular_sector_mesh(
    inner_radius: float,
    outer_radius: float,
    azimuth_min: float,
    azimuth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed, thin annular-sector mesh centered at the origin."""
    angles = np.linspace(azimuth_min, azimuth_max, _SEGMENTS + 1)
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
