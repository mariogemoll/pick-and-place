# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reusable MuJoCo geometry for the UVC camera module used in the lab."""

from __future__ import annotations

import mujoco

# MuJoCo box/cylinder sizes are half-extents.
BOARD_HALF_SIZE = (0.016, 0.016, 0.001)
LENS_RADIUS = 0.007
LENS_HALF_LENGTH = 0.010
LENS_POS = (0.0, 0.0, -(BOARD_HALF_SIZE[2] + LENS_HALF_LENGTH))

MODULE_RGBA = (0.05, 0.05, 0.05, 1.0)
LENS_RGBA = (0.16, 0.16, 0.16, 1.0)
COLLISION_RGBA = (0.2, 0.8, 0.2, 0.5)


def add_camera_module(
    parent: mujoco.MjsBody,
    *,
    prefix: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision: bool = True,
    collision_default: mujoco.MjsDefault | None = None,
) -> mujoco.MjsBody:
    """Attach a camera board and lens to ``parent`` and return its child body.

    ``pos`` and ``quat`` place the module's board-center frame in the parent
    body. Names are prefixed so multiple instances can coexist in one model.
    The calibrated MuJoCo camera should be added separately by the caller.
    """
    body = parent.add_body(name=f"{prefix}camera_module", pos=pos, quat=quat)

    body.add_geom(
        name=f"{prefix}camera_board_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=BOARD_HALF_SIZE,
        rgba=MODULE_RGBA,
        contype=0,
        conaffinity=0,
        group=2,
    )
    body.add_geom(
        name=f"{prefix}camera_lens_visual",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=LENS_POS,
        size=(LENS_RADIUS, LENS_HALF_LENGTH, 0.0),
        rgba=LENS_RGBA,
        contype=0,
        conaffinity=0,
        group=2,
    )
    body.add_site(
        name=f"{prefix}camera_frame",
        size=(0.0015, 0.0015, 0.0015),
        group=3,
    )

    if collision:
        body.add_geom(
            default=collision_default,
            name=f"{prefix}camera_board_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=BOARD_HALF_SIZE,
            rgba=COLLISION_RGBA,
            group=3,
        )
        body.add_geom(
            default=collision_default,
            name=f"{prefix}camera_lens_collision",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=LENS_POS,
            size=(LENS_RADIUS, LENS_HALF_LENGTH, 0.0),
            rgba=COLLISION_RGBA,
            group=3,
        )

    return body
