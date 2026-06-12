# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compose SO-101 MuJoCo models from the stock SO-ARM100 MJCF.

The vendored model in SO-ARM100/ stays untouched: it is loaded as an MjSpec,
its full-mesh collision geoms are replaced by the hand-tuned box model from
:mod:`pick_and_place.collision_boxes`.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from pick_and_place.collision_boxes import (
    COLLISION_BOXES,
    GRIP_CONDIM,
    GRIP_FRICTION,
    GRIP_SOLIMP,
    GRIP_SOLREF,
)
from pick_and_place.wrist_camera import add_wrist_camera

REPO_ROOT = Path(__file__).resolve().parents[3]
STOCK_XML = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.xml"
STOCK_ASSETS_DIR = STOCK_XML.parent / "assets"

#: Debug tint for collision geoms (group 3, hidden by default in viewers).
_COLLISION_RGBA = (0.2, 0.8, 0.2, 0.5)


def build_robot(*, wrist_camera: bool = True) -> mujoco.MjSpec:
    """Stock SO-101 with box collisions; call ``.compile()`` on the result.

    The hex-nut wrist-camera mount and 32x32 UVC module are included by
    default. Pass ``wrist_camera=False`` for the unmodified stock wrist.
    """
    spec = mujoco.MjSpec.from_file(str(STOCK_XML))
    spec.meshdir = str(STOCK_ASSETS_DIR)
    _strip_mesh_collisions(spec)
    _add_collision_boxes(spec)
    if wrist_camera:
        add_wrist_camera(spec)
    return spec


def _strip_mesh_collisions(spec: mujoco.MjSpec) -> None:
    for geom in [g for g in spec.geoms if g.classname and g.classname.name == "collision"]:
        spec.delete(geom)


def _add_collision_boxes(spec: mujoco.MjSpec) -> None:
    collision_default = spec.find_default("collision")
    for body_name, boxes in COLLISION_BOXES.items():
        body = spec.body(body_name)
        for box in boxes:
            grip_kwargs = (
                dict(
                    friction=GRIP_FRICTION,
                    condim=GRIP_CONDIM,
                    solref=GRIP_SOLREF,
                    solimp=GRIP_SOLIMP,
                )
                if box.grip
                else {}
            )
            body.add_geom(
                default=collision_default,
                name=box.name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=box.pos,
                quat=box.quat,
                size=box.size,
                rgba=_COLLISION_RGBA,
                **grip_kwargs,
            )
