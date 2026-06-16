# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compose the SO-101 robot with a floor, workspace overlays, light, and cube."""

from __future__ import annotations

from pathlib import Path

import mujoco

from pick_and_place.builder import STOCK_ASSETS_DIR, build_robot
from pick_and_place.environment import add_overhead_camera_mount, add_workspace_frame
from pick_and_place.materials import MaterialConfig, apply_materials
from pick_and_place.workspace_overlays import add_workspace_overlays


def build_scene(
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    include_environment: bool = False,
) -> mujoco.MjSpec:
    """Return the composed robot with a floor, workspace overlays, light, and cube."""
    spec = build_robot(wrist_camera=wrist_camera, materials=materials)
    spec.modelname = "so101_with_cube"
    spec.worldbody.add_light(
        name="scene_light",
        pos=(0.0, 0.0, 1.0),
        dir=(0.0, 0.0, -1.0),
    )
    add_workspace_overlays(spec, spec.body("base"))
    _add_pick_cube(spec)

    if include_environment:
        collision_default = spec.find_default("collision")
        add_workspace_frame(spec, collision_default=collision_default)
        add_overhead_camera_mount(spec, collision_default=collision_default)

    apply_materials(spec, materials or MaterialConfig())
    _add_groundplane(spec)

    return spec


def build_environment(
    *,
    materials: MaterialConfig | None = None,
) -> mujoco.MjSpec:
    """Return only the environment, with no robot.

    Contains the floor, pick cube, calibration workspace frame, and overhead
    camera mount, all attached to the worldbody. The web viewer loads this on
    top of the standalone ``so101`` model so the robot is defined exactly once
    instead of being baked into the scene a second time.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "pick_and_place_environment"
    _add_pick_cube(spec)
    add_workspace_frame(spec)
    add_overhead_camera_mount(spec)
    apply_materials(spec, materials or MaterialConfig())
    _add_groundplane(spec)
    return spec


def _add_groundplane(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.2, 0.3),
        markrgb=(0.8, 0.8, 0.8),
        width=300,
        height=300,
    )
    groundplane = spec.add_material(
        name="groundplane",
        texuniform=True,
        texrepeat=(5.0, 5.0),
        reflectance=0.2,
    )
    groundplane.textures[1] = "groundplane"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05),
        material="groundplane",
    )


def export_scene(
    output: Path,
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    include_environment: bool = False,
) -> Path:
    """Write a standalone, machine-local XML file for the composed scene."""
    spec = build_scene(
        wrist_camera=wrist_camera,
        materials=materials,
        include_environment=include_environment,
    )
    spec.meshdir = str(STOCK_ASSETS_DIR)
    spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    return output


def _add_pick_cube(spec: mujoco.MjSpec) -> None:
    cube = spec.worldbody.add_body(name="pick_cube", pos=(0.2, -0.12, 0.015))
    cube.add_geom(
        name="pick_cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.015, 0.015, 0.015),
        rgba=(0.82, 0.12, 0.08, 1.0),
        mass=0.0137,
        priority=1,
        solref=(0.002, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    )
