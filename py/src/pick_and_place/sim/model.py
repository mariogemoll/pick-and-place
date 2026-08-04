# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compile a runnable model of the task scene, and move things around in it.

:mod:`pick_and_place.sim.scene` composes the scene as an ``MjSpec``; this is the
step after — place the cube, compile, and hand back the ``model``/``data`` pair
that everything which simulates the task starts from. The setters work on an
already-compiled pair, so one persistent model can be reused across episodes
rather than recompiled per cube pose (which is what lets a live viewer stay
bound to it).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose, PlacementError, cube_quat_from_pose
from pick_and_place.sim.paper_target_marker import add_paper_target_marker
from pick_and_place.sim.scene import build_scene
from pick_and_place.spec.workspace import CUBE_HALF_SIZE


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value


def get_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[jid]])


def set_cube_pose(model: mujoco.MjModel, data: mujoco.MjData, source: CubePose) -> None:
    """Move the freejoint ``pick_cube`` to ``source`` in an existing model's data.

    Lets a single persistent model be reused across episodes (so a live viewer can
    stay bound to it) instead of recompiling one per cube pose."""
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    jnt_adr = model.body_jntadr[cube_body_id]
    qpos_adr = model.jnt_qposadr[jnt_adr]
    qvel_adr = model.jnt_dofadr[jnt_adr]
    data.qpos[qpos_adr:qpos_adr + 3] = (source.x, source.y, source.z)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = cube_quat_from_pose(source)
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def placement_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: CubePose,
) -> PlacementError:
    """Measure the current cube-center offset from the target center."""
    mujoco.mj_forward(model, data)
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    cube_xyz = tuple(float(v) for v in data.xpos[cube_body_id])
    target_xyz = (float(target.x), float(target.y), float(CUBE_HALF_SIZE))
    dx = cube_xyz[0] - target_xyz[0]
    dy = cube_xyz[1] - target_xyz[1]
    dz = cube_xyz[2] - target_xyz[2]
    return PlacementError(
        cube_xyz=cube_xyz,
        target_xyz=target_xyz,
        dx=dx,
        dy=dy,
        dz=dz,
        xy=math.hypot(dx, dy),
    )


def build_model(
    source: CubePose,
    include_environment: bool = False,
    offwidth: int = 1280,
    offheight: int = 720,
    paper_target_marker: bool = False,
    background_panorama: Path | str | np.ndarray | None = None,
    table_texture: Path | str | np.ndarray | None = None,
    robot_dynamics: bool | str | Path = True,
    apriltag_cube: bool | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = build_scene(
        include_environment=include_environment,
        background_panorama=background_panorama,
        table_texture=table_texture,
        robot_dynamics=robot_dynamics,
        apriltag_cube=apriltag_cube,
    )
    if paper_target_marker:
        add_paper_target_marker(spec)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, offwidth)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, offheight)
    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    cube.quat = cube_quat_from_pose(source)
    cube.add_freejoint()  # make the cube a real dynamic body
    model = spec.compile()
    return model, mujoco.MjData(model)
