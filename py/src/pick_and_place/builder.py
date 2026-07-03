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
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.materials import MaterialConfig, apply_materials
from pick_and_place.robot_dynamics import DEFAULT_ROBOT_DYNAMICS_PATH, load_robot_dynamics_config
from pick_and_place.wrist_camera import add_wrist_camera

REPO_ROOT = Path(__file__).resolve().parents[3]
STOCK_XML = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.xml"
STOCK_ASSETS_DIR = STOCK_XML.parent / "assets"

#: Debug tint for collision geoms (group 3, hidden by default in viewers).
_COLLISION_RGBA = (0.2, 0.8, 0.2, 0.5)


def build_robot(
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    robot_dynamics: bool | str | Path = True,
) -> mujoco.MjSpec:
    """Stock SO-101 with box collisions; call ``.compile()`` on the result.

    The hex-nut wrist-camera mount and 32x32 UVC module are included by
    default. Pass ``wrist_camera=False`` for the unmodified stock wrist.

    ``robot_dynamics`` applies fitted actuator time constants to the stock
    position actuators when a calibration file is available. Pass ``False`` for
    the raw upstream actuator dynamics, or a path to use a specific calibration.
    """
    spec = mujoco.MjSpec.from_file(str(STOCK_XML))
    spec.meshdir = str(STOCK_ASSETS_DIR)
    _apply_robot_dynamics(spec, robot_dynamics)
    _strip_mesh_collisions(spec)
    _add_collision_boxes(spec)
    _exclude_base_shoulder_contact(spec)
    if wrist_camera:
        add_wrist_camera(spec)
    apply_materials(spec, materials or MaterialConfig())
    return spec


def _apply_robot_dynamics(spec: mujoco.MjSpec, robot_dynamics: bool | str | Path) -> None:
    if robot_dynamics is False:
        return
    path = DEFAULT_ROBOT_DYNAMICS_PATH if robot_dynamics is True else Path(robot_dynamics)
    if not path.is_file():
        return

    config = load_robot_dynamics_config(path)
    for name in JOINT_NAMES:
        joint_config = config["joints"].get(name)
        if joint_config is None:
            continue
        time_constant = joint_config.get("time_constant_s")
        if time_constant is None:
            continue
        actuator = spec.actuator(name)
        actuator.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
        actuator.dynprm[0] = float(time_constant)


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


def _exclude_base_shoulder_contact(spec: mujoco.MjSpec) -> None:
    # MuJoCo intentionally allows contacts when the parent is welded to the
    # world, so mechanically adjacent static-base links need an explicit filter.
    spec.add_exclude(
        name="base_shoulder",
        bodyname1="base",
        bodyname2="shoulder",
    )
