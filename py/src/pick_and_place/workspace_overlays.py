# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fixed MuJoCo meshes for the standard SO-101 workspace overlays."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from pick_and_place.environment import (
    WORKSPACE_FRAME_APRILTAG_PLATES,
    WORKSPACE_FRAME_POS,
    WORKSPACE_FRAME_QUAT,
)
from pick_and_place.geometry import CUBE_HALF_SIZE

WORKSPACE_FRAME_INNER_HALF_EXTENT = 0.2813 - 0.0187

# Fixed SO-101 floor-workspace bounds for the calibrated model. Recompute these
# when robot kinematics or workspace definitions change.
PAN_AXIS = (0.0388353, -8.97657e-09)
AZIMUTH_MIN = -1.9198621771937634
AZIMUTH_MAX = AZIMUTH_MIN + 3.839724354387525
CANONICAL_PICKUP_AZIMUTH_MIN = math.radians(-100.0)
CANONICAL_PICKUP_AZIMUTH_MAX = math.radians(100.0)

WORKSPACE_OVERLAY_GROUP = 4
_SEGMENTS = 96
_HALF_THICKNESS = 0.00001
_RGBA = (1.0, 0.4667, 0.0, 0.22)
_CUBE_PLACEMENT_RGBA = (0.1333, 0.7725, 0.3686, 0.42)
_CUBE_EXCLUSION_RGBA = (0.9373, 0.2667, 0.2667, 0.62)
_CANONICAL_PICKUP_RGBA = (0.0, 0.45, 1.0, 0.28)


@dataclass(frozen=True)
class WorkspaceOverlay:
    """A fixed annular sector projected onto the floor."""

    name: str
    inner_radius: float
    outer_radius: float
    z: float
    azimuth_min: float = AZIMUTH_MIN
    azimuth_max: float = AZIMUTH_MAX


WORKSPACE_OVERLAYS = (
    WorkspaceOverlay("workspace_global", 0.0, 0.4418431804405771, 0.0002),
    WorkspaceOverlay("workspace_ground_height_arm", 0.0, 0.42911855464836285, 0.0004),
    WorkspaceOverlay(
        "workspace_ground_grasp",
        0.07330432949931037,
        0.25536116910332146,
        0.0006,
    ),
    WorkspaceOverlay(
        "workspace_clearance_grasp",
        0.08671592519116689,
        0.24194957341146492,
        0.0008,
    ),
)

# Conservatively inset the frame interior by the cube's circumradius. This
# keeps every corner clear of the rails for every sampled yaw.
_CUBE_FRAME_MARGIN = math.sqrt(2.0) * CUBE_HALF_SIZE
# Recovery drops are intentionally aimed farther from the physical workspace
# frame than ordinary pickup starts. A recovery can miss its target by a few
# centimetres; keeping the target inset avoids landing the cube on the rails.
RECOVERY_TARGET_FRAME_BORDER_MARGIN = 0.06
_CUBE_FRAME_HALF_EXTENT = WORKSPACE_FRAME_INNER_HALF_EXTENT - _CUBE_FRAME_MARGIN
_APRILTAG_PLATE_HALF_SIZE = 0.03
TARGET_PLATE_HALF_SIZE = 0.05
TARGET_PLATE_CLEARANCE = 0.002
CUBE_APRILTAG_EXCLUSION_HALF_EXTENT = _APRILTAG_PLATE_HALF_SIZE + _CUBE_FRAME_MARGIN
CUBE_PLACEMENT_BOUNDS = (
    WORKSPACE_FRAME_POS[0] - _CUBE_FRAME_HALF_EXTENT,
    WORKSPACE_FRAME_POS[0] + _CUBE_FRAME_HALF_EXTENT,
    WORKSPACE_FRAME_POS[1] - _CUBE_FRAME_HALF_EXTENT,
    WORKSPACE_FRAME_POS[1] + _CUBE_FRAME_HALF_EXTENT,
)
CUBE_PLACEMENT_OVERLAY = WorkspaceOverlay(
    "workspace_cube_placement",
    WORKSPACE_OVERLAYS[-1].inner_radius,
    0.4310255047641903,
    0.0010,
)
CANONICAL_PICKUP_OVERLAY = WorkspaceOverlay(
    "workspace_canonical_pickup",
    0.11,
    0.426,
    0.0012,
    CANONICAL_PICKUP_AZIMUTH_MIN,
    CANONICAL_PICKUP_AZIMUTH_MAX,
)


def _is_cube_center_allowed(
    x: float, y: float, overlay: WorkspaceOverlay
) -> bool:
    x_min, x_max, y_min, y_max = CUBE_PLACEMENT_BOUNDS
    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        return False
    dx = x - PAN_AXIS[0]
    dy = y - PAN_AXIS[1]
    radius = math.hypot(dx, dy)
    azimuth = math.atan2(dy, dx)
    in_clearance_sector = (
        overlay.inner_radius <= radius <= overlay.outer_radius
        and overlay.azimuth_min <= azimuth <= overlay.azimuth_max
    )
    if not in_clearance_sector:
        return False

    local_x, local_y = _world_to_frame_xy(x, y)
    return not any(
        abs(local_x - tag_pos[0]) <= CUBE_APRILTAG_EXCLUSION_HALF_EXTENT
        and abs(local_y - tag_pos[1]) <= CUBE_APRILTAG_EXCLUSION_HALF_EXTENT
        for _, _, tag_pos in WORKSPACE_FRAME_APRILTAG_PLATES
    )


def is_cube_pickup_allowed(x: float, y: float) -> bool:
    """Return whether a floor cube can use the canonical pick-lift pose."""
    return _is_cube_center_allowed(x, y, CANONICAL_PICKUP_OVERLAY)


def is_vertical_grip_allowed(x: float, y: float) -> bool:
    """Return whether a floor cube can use the vertical gripper pose."""
    return is_cube_pickup_allowed(x, y)


def is_cube_drop_allowed(x: float, y: float) -> bool:
    """Return whether a cube-center drop target is in the broad arm workspace."""
    return _is_cube_center_allowed(x, y, CUBE_PLACEMENT_OVERLAY)


def _project_polygon(points: tuple[tuple[float, float], ...], axis: tuple[float, float]) -> tuple[float, float]:
    values = [px * axis[0] + py * axis[1] for px, py in points]
    return min(values), max(values)


def _polygons_overlap(
    a: tuple[tuple[float, float], ...],
    b: tuple[tuple[float, float], ...],
) -> bool:
    for polygon in (a, b):
        for index, p0 in enumerate(polygon):
            p1 = polygon[(index + 1) % len(polygon)]
            edge_x = p1[0] - p0[0]
            edge_y = p1[1] - p0[1]
            length = math.hypot(edge_x, edge_y)
            axis = (-edge_y / length, edge_x / length)
            a_min, a_max = _project_polygon(a, axis)
            b_min, b_max = _project_polygon(b, axis)
            if a_max <= b_min or b_max <= a_min:
                return False
    return True


def target_plate_corners_frame(
    x: float,
    y: float,
    yaw: float,
    half_size: float = TARGET_PLATE_HALF_SIZE,
) -> tuple[tuple[float, float], ...]:
    """Return the target plate's four corners in workspace-frame coordinates."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    corners = []
    for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        wx = x + (sx * c - sy * s) * half_size
        wy = y + (sx * s + sy * c) * half_size
        corners.append(_world_to_frame_xy(wx, wy))
    return tuple(corners)


def is_target_plate_allowed(
    x: float,
    y: float,
    yaw: float,
    *,
    half_size: float = TARGET_PLATE_HALF_SIZE,
    clearance: float = TARGET_PLATE_CLEARANCE,
) -> bool:
    """Return whether a black target plate clears frame rails and AprilTags."""
    corners = target_plate_corners_frame(x, y, yaw, half_size)
    frame_limit = WORKSPACE_FRAME_INNER_HALF_EXTENT - clearance
    if any(abs(cx) > frame_limit or abs(cy) > frame_limit for cx, cy in corners):
        return False

    tag_half = _APRILTAG_PLATE_HALF_SIZE + clearance
    for _, _, tag_pos in WORKSPACE_FRAME_APRILTAG_PLATES:
        tag_x, tag_y = tag_pos[:2]
        tag_corners = (
            (tag_x - tag_half, tag_y - tag_half),
            (tag_x + tag_half, tag_y - tag_half),
            (tag_x + tag_half, tag_y + tag_half),
            (tag_x - tag_half, tag_y + tag_half),
        )
        if _polygons_overlap(corners, tag_corners):
            return False
    return True


def is_target_plate_position_allowed(x: float, y: float) -> bool:
    """Return whether any square-plate yaw can fit at the target center."""
    return _first_allowed_target_plate_yaw(x, y) is not None


def _first_allowed_target_plate_yaw(
    x: float,
    y: float,
    half_size: float = TARGET_PLATE_HALF_SIZE,
) -> float | None:
    """Scan ``[0, 90)`` degrees for a fitting yaw; ``None`` if the center is unusable."""
    for step in range(90):
        yaw = step * (math.pi / 2.0) / 90.0
        if is_target_plate_allowed(x, y, yaw, half_size=half_size):
            return yaw
    return None


def sample_target_plate_yaw(
    rng: np.random.Generator,
    x: float,
    y: float,
    *,
    half_size: float = TARGET_PLATE_HALF_SIZE,
    max_attempts: int = 200,
) -> float:
    """Sample a plate yaw in ``[0, 90)`` degrees whose corners stay in bounds.

    The plate is square, so any yaw outside ``[0, 90)`` is equivalent to one
    inside it.

    Yaw 0 is *not* a safe fallback, despite being the axis-aligned minimum
    against the frame rails. Near an AprilTag plate offset diagonally from the
    target, yaw 0 is the worst case: the plate reaches toward the tag with its
    corner, ``half_size * sqrt(2)`` from the center. Rotating toward 45 degrees
    presents an edge midpoint instead, only ``half_size`` out, which can clear a
    tag the axis-aligned plate overlaps. Falling back to the discrete scan
    therefore preserves the guarantee that
    :func:`is_target_plate_position_allowed` makes about the center.
    """
    for _ in range(max_attempts):
        yaw = float(rng.uniform(0.0, math.pi / 2.0))
        if is_target_plate_allowed(x, y, yaw, half_size=half_size):
            return yaw
    fallback = _first_allowed_target_plate_yaw(x, y, half_size)
    return 0.0 if fallback is None else fallback


def is_cube_recovery_target_allowed(x: float, y: float) -> bool:
    """Return whether a recovery drop target leaves extra room around the frame."""
    if not is_cube_pickup_allowed(x, y):
        return False
    local_x, local_y = _world_to_frame_xy(x, y)
    half_extent = WORKSPACE_FRAME_INNER_HALF_EXTENT - RECOVERY_TARGET_FRAME_BORDER_MARGIN
    return abs(local_x) <= half_extent and abs(local_y) <= half_extent


# Backward-compatible name for callers that mean pickup placement.
is_cube_placement_allowed = is_cube_pickup_allowed


def workspace_interior_corners_world() -> np.ndarray:
    """World-space corners of the workspace-frame interior.

    Used to mask overhead detections to the table surface, excluding off-table
    clutter (keyboard, cables, the shadowed table border).
    """
    h = WORKSPACE_FRAME_INNER_HALF_EXTENT
    z = WORKSPACE_FRAME_POS[2]
    return np.array(
        [(*_frame_to_world_xy(fx, fy), z) for fx, fy in ((-h, -h), (h, -h), (h, h), (-h, h))],
        dtype=float,
    )


def _world_to_frame_xy(x: float, y: float) -> tuple[float, float]:
    """Transform a world XY point into the workspace-frame coordinate system."""
    w, qx, qy, qz = WORKSPACE_FRAME_QUAT
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * w)
    r10 = 2.0 * (qx * qy + qz * w)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    world_dx = x - WORKSPACE_FRAME_POS[0]
    world_dy = y - WORKSPACE_FRAME_POS[1]
    return (
        r00 * world_dx + r10 * world_dy,
        r01 * world_dx + r11 * world_dy,
    )


def _frame_to_world_xy(x: float, y: float) -> tuple[float, float]:
    """Transform a workspace-frame XY point into world coordinates."""
    w, qx, qy, qz = WORKSPACE_FRAME_QUAT
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * w)
    r10 = 2.0 * (qx * qy + qz * w)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    return (
        WORKSPACE_FRAME_POS[0] + r00 * x + r01 * y,
        WORKSPACE_FRAME_POS[1] + r10 * x + r11 * y,
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
            overlay.azimuth_min,
            overlay.azimuth_max,
        )
        mesh = spec.add_mesh(name=name)
        mesh.uservert = vertices.flatten()
        mesh.userface = faces.flatten()
        parent.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            pos=(*PAN_AXIS, overlay.z),
            rgba=_RGBA,
            contype=0,
            conaffinity=0,
            group=WORKSPACE_OVERLAY_GROUP,
        )

    vertices, faces = _clipped_annular_sector_mesh(
        CUBE_PLACEMENT_OVERLAY.inner_radius,
        CUBE_PLACEMENT_OVERLAY.outer_radius,
        CUBE_PLACEMENT_BOUNDS,
        CUBE_PLACEMENT_OVERLAY.azimuth_min,
        CUBE_PLACEMENT_OVERLAY.azimuth_max,
    )
    mesh = spec.add_mesh(name=CUBE_PLACEMENT_OVERLAY.name)
    mesh.uservert = vertices.flatten()
    mesh.userface = faces.flatten()
    parent.add_geom(
        name=CUBE_PLACEMENT_OVERLAY.name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        pos=(*PAN_AXIS, CUBE_PLACEMENT_OVERLAY.z),
        rgba=_CUBE_PLACEMENT_RGBA,
        contype=0,
        conaffinity=0,
        group=WORKSPACE_OVERLAY_GROUP,
    )

    vertices, faces = _clipped_annular_sector_mesh(
        CANONICAL_PICKUP_OVERLAY.inner_radius,
        CANONICAL_PICKUP_OVERLAY.outer_radius,
        CUBE_PLACEMENT_BOUNDS,
        CANONICAL_PICKUP_OVERLAY.azimuth_min,
        CANONICAL_PICKUP_OVERLAY.azimuth_max,
    )
    mesh = spec.add_mesh(name=CANONICAL_PICKUP_OVERLAY.name)
    mesh.uservert = vertices.flatten()
    mesh.userface = faces.flatten()
    parent.add_geom(
        name=CANONICAL_PICKUP_OVERLAY.name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        pos=(*PAN_AXIS, CANONICAL_PICKUP_OVERLAY.z),
        rgba=_CANONICAL_PICKUP_RGBA,
        contype=0,
        conaffinity=0,
        group=WORKSPACE_OVERLAY_GROUP,
    )

    for _, corner_name, tag_pos in WORKSPACE_FRAME_APRILTAG_PLATES:
        tag_x, tag_y = _frame_to_world_xy(tag_pos[0], tag_pos[1])
        parent.add_geom(
            name=f"workspace_cube_exclusion_tag_{corner_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                CUBE_APRILTAG_EXCLUSION_HALF_EXTENT,
                CUBE_APRILTAG_EXCLUSION_HALF_EXTENT,
                _HALF_THICKNESS,
            ),
            pos=(tag_x, tag_y, CUBE_PLACEMENT_OVERLAY.z + 0.00004),
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
