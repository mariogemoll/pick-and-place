# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The drop-zone square, as a geom in a MuJoCo scene.

The real rig marks the drop zone with a sheet of black paper. The marker is
added hidden before compile and then positioned, sized and coloured per episode,
so one compiled model serves every target without a rebuild.
"""

from __future__ import annotations

import math

import mujoco

from pick_and_place.spec.workspace import WORKSPACE_FLOOR_Z

PAPER_TARGET_MARKER_NAME = "paper_target_marker"


def add_paper_target_marker(spec: mujoco.MjSpec) -> None:
    """Add a hidden, non-colliding drop-zone marker to an ``MjSpec`` before compile."""
    body = spec.worldbody.add_body(name=PAPER_TARGET_MARKER_NAME, pos=(0.0, 0.0, 0.0))
    body.add_geom(
        name=PAPER_TARGET_MARKER_NAME + "_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, WORKSPACE_FLOOR_Z + 0.002),
        size=(0.05, 0.05, 0.001),
        rgba=(1.0, 1.0, 1.0, 0.0),
        contype=0,
        conaffinity=0,
        group=1,
    )


def place_paper_target_marker(
    model: mujoco.MjModel,
    center_xy: tuple[float, float],
    yaw: float,
    half_extent: tuple[float, float],
    *,
    usable: bool,
    alpha: float = 0.72,
) -> None:
    """Position and show the drop-zone marker square in an already compiled model.

    ``half_extent`` is the half side length (metres) along the marker's local x/y
    axes. ``usable`` colours the square the standard black when the drop is allowed
    and orange when it falls outside the permitted drop zone. ``alpha`` is the
    opacity: the live viewer overlays a translucent square, while a sim recording
    uses a fully opaque square to look like real black paper on the table.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAPER_TARGET_MARKER_NAME)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PAPER_TARGET_MARKER_NAME + "_geom")
    if body_id < 0 or geom_id < 0:
        return

    model.body_pos[body_id] = (center_xy[0], center_xy[1], 0.0)
    model.body_quat[body_id] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    model.geom_size[geom_id] = (
        max(half_extent[0], 0.001),
        max(half_extent[1], 0.001),
        0.001,
    )
    rgb = (0.12, 0.12, 0.12) if usable else (1.0, 0.45, 0.05)
    model.geom_rgba[geom_id] = (*rgb, alpha)
