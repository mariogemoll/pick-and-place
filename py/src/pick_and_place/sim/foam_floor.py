# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The EVA foam sheet laid on the table inside the workspace frame square.

The real sheet is one piece of 3 mm grey foam cut to the square the frame rails
enclose, with the corner AprilTag plates and the robot's wedge-shaped base plate
cut out of it, so everything that has to sit on the bare table still does.

MuJoCo collides convex shapes, and the cut outline is not convex, so the sheet is
expressed as the convex pieces it decomposes into: three boxes across the body of
the square, and either side of the robot a wedge whose slanted edge follows the
base plate. They tile the outline exactly, meeting along shared faces.

Pieces are laid out in the workspace frame's local coordinates — +x east, +y
north, the robot at the north edge — and the sheet body carries the surveyed
frame pose, so it lands under the corner plates whether or not the frame itself
is in the scene.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pick_and_place.core.workspace_bounds import WORKSPACE_FRAME_INNER_HALF_EXTENT
from pick_and_place.spec.workspace import (
    FOAM_FLOOR_BASE_CUTOUT_FLARE,
    FOAM_FLOOR_BASE_CUTOUT_TIP_HALF_WIDTH,
    FOAM_FLOOR_BASE_CUTOUT_TIP_Y,
    FOAM_FLOOR_CORNER_CUTOUT_INNER,
    FOAM_FLOOR_THICKNESS,
    WORKSPACE_FRAME_POS,
    WORKSPACE_FRAME_QUAT,
)

FOAM_MATERIAL = "foam"
FOAM_FLOOR_BODY_NAME = "foam_floor"

#: A rectangular piece: its name and the rectangle it covers in frame-local
#: coordinates, as (x_min, x_max, y_min, y_max).
FoamBox = tuple[str, tuple[float, float, float, float]]

#: A wedge piece: its name and the four corners of its plan outline.
FoamWedge = tuple[str, tuple[tuple[float, float], ...]]


def base_cutout_half_width(y: float) -> float:
    """Half-width of the robot's base plate, and so of the cut, at *y*."""
    return FOAM_FLOOR_BASE_CUTOUT_TIP_HALF_WIDTH + FOAM_FLOOR_BASE_CUTOUT_FLARE * (
        y - FOAM_FLOOR_BASE_CUTOUT_TIP_Y
    )


def foam_floor_boxes() -> tuple[FoamBox, ...]:
    """Return the rectangular pieces: everything south of the robot's cut."""
    edge = WORKSPACE_FRAME_INNER_HALF_EXTENT
    corner = FOAM_FLOOR_CORNER_CUTOUT_INNER
    return (
        ("center", (-edge, edge, -corner, corner)),
        ("south", (-corner, corner, -edge, -corner)),
        ("north", (-corner, corner, corner, FOAM_FLOOR_BASE_CUTOUT_TIP_Y)),
    )


def foam_floor_wedges() -> tuple[FoamWedge, ...]:
    """Return the two pieces left either side of the robot's base plate.

    Each is a trapezoid: square where it meets the corner cut and the rest of the
    sheet, slanted where it follows the plate's edge out to the rail.
    """
    edge = WORKSPACE_FRAME_INNER_HALF_EXTENT
    corner = FOAM_FLOOR_CORNER_CUTOUT_INNER
    tip_y = FOAM_FLOOR_BASE_CUTOUT_TIP_Y
    return tuple(
        (
            f"north_{name}",
            (
                (side * base_cutout_half_width(tip_y), tip_y),
                (side * corner, tip_y),
                (side * corner, edge),
                (side * base_cutout_half_width(edge), edge),
            ),
        )
        for name, side in (("east", 1.0), ("west", -1.0))
    )


def add_foam_floor(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    *,
    collision_default: mujoco.MjsDefault | None = None,
) -> mujoco.MjsBody:
    """Add the foam sheet to *parent*, posed on the surveyed workspace frame.

    The sheet rests on the table, so each piece spans Z from 0 to the foam's
    thickness. The pieces collide: the cube and the gripper meet the foam's top
    face, 3 mm above the table, and not the table itself.
    """
    body = parent.add_body(
        name=FOAM_FLOOR_BODY_NAME, pos=WORKSPACE_FRAME_POS, quat=WORKSPACE_FRAME_QUAT
    )
    half_thickness = FOAM_FLOOR_THICKNESS / 2

    for name, (x_min, x_max, y_min, y_max) in foam_floor_boxes():
        _add_piece(
            body,
            name,
            collision_default,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=((x_min + x_max) / 2, (y_min + y_max) / 2, half_thickness),
            size=((x_max - x_min) / 2, (y_max - y_min) / 2, half_thickness),
        )

    for name, corners in foam_floor_wedges():
        mesh = spec.add_mesh(name=f"foam_floor_{name}")
        # Only the corners: the piece is convex, so the compiler's hull of them is
        # the prism itself.
        mesh.uservert = np.array(
            [(x, y, z) for z in (0.0, FOAM_FLOOR_THICKNESS) for x, y in corners],
            dtype=float,
        ).flatten()
        _add_piece(
            body,
            name,
            collision_default,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
        )
    return body


def _add_piece(
    body: mujoco.MjsBody,
    name: str,
    collision_default: mujoco.MjsDefault | None,
    **geom: object,
) -> None:
    """Add one piece of the sheet, as a visual geom and a colliding one."""
    body.add_geom(
        name=f"foam_floor_{name}_visual",
        material=FOAM_MATERIAL,
        contype=0,
        conaffinity=0,
        group=2,
        **geom,
    )
    body.add_geom(
        default=collision_default,
        name=f"foam_floor_{name}_collision",
        material="collision",
        group=3,
        **geom,
    )
