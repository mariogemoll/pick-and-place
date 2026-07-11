# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compose the SO-101 robot with a floor, workspace overlays, light, and cube."""

from __future__ import annotations

from pathlib import Path

import mujoco

from pick_and_place.builder import STOCK_ASSETS_DIR, build_robot
from pick_and_place.environment import (
    APRILTAG_TEXTURE_DIR,
    add_overhead_camera_mount,
    add_workspace_frame,
    add_workspace_frame_apriltags,
)
from pick_and_place.materials import MaterialConfig, apply_materials
from pick_and_place.workspace_overlays import add_workspace_overlays

# Tag IDs stickered onto the pick cube's six faces, in MuJoCo cube-texture order
# (right, left, up, down, front, back). With the cube unrotated those map to the
# world directions -X, +X, -Y, +Y, +Z, -Z respectively.
PICK_CUBE_APRILTAG_IDS: tuple[int, int, int, int, int, int] = (0, 1, 2, 3, 4, 5)

# Half-edge of the pick cube; the 30 mm faces carry 30 mm AprilTag stickers.
PICK_CUBE_HALF_SIZE = 0.015

# Plain pick cube colour, used when the AprilTag faces are not requested.
PICK_CUBE_RGBA = (0.82, 0.12, 0.08, 1.0)

# The real robot is mounted on the workspace frame, elevating its base by the
# frame's thickness. The floor and cube remain at world Z=0.
ROBOT_BASE_Z_OFFSET = 0.0072

# Physics rate of the compiled scene. 600 Hz is an exact integer multiple of the
# 30 Hz control rate shared by the hardware runner and the RL env, so one control
# period is always a whole number of physics steps; the stock 500 Hz MuJoCo
# default is not.
SIMULATION_HZ = 600.0


def build_scene(
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    include_environment: bool = True,
    apriltags: bool | None = None,
) -> mujoco.MjSpec:
    """Return the composed robot with a floor, workspace overlays, light, and cube.

    ``apriltags`` selects the perception targets: the AprilTag-stickered pick
    cube and the workspace frame's calibration tag plates. Both are purely
    visual (texture files rendered by scripts/render_apriltag_textures.py), so
    scenes that never render camera images can drop them. When left ``None`` it
    follows ``include_environment``, so the simple scene gets the plain red cube
    and the standard scene the tagged one.
    """
    if apriltags is None:
        apriltags = include_environment

    spec = build_robot(wrist_camera=wrist_camera, materials=materials)
    spec.modelname = "so101_with_cube"
    spec.option.timestep = 1.0 / SIMULATION_HZ
    spec.worldbody.add_light(
        name="scene_light",
        pos=(0.0, 0.0, 1.0),
        dir=(0.0, 0.0, -1.0),
    )
    
    base = spec.body("base")
    base.pos = (0.0, 0.0, ROBOT_BASE_Z_OFFSET)

    # Attach overlays to worldbody so they stay on the floor.
    add_workspace_overlays(spec, spec.worldbody)
    _add_pick_cube(spec, apriltag=apriltags)

    if include_environment:
        collision_default = spec.find_default("collision")
        add_workspace_frame(spec, collision_default=collision_default)
        add_overhead_camera_mount(spec, collision_default=collision_default)

    apply_materials(spec, materials or MaterialConfig())
    if apriltags:
        _add_pick_cube_apriltags(spec)
        if include_environment:
            add_workspace_frame_apriltags(spec)
    _add_groundplane(spec)

    return spec


def build_environment(
    *,
    materials: MaterialConfig | None = None,
    apriltags: bool = True,
) -> mujoco.MjSpec:
    """Return only the environment, with no robot.

    Contains the floor, pick cube, calibration workspace frame, and overhead
    camera mount, all attached to the worldbody. The web viewer loads this on
    top of the standalone ``so101`` model so the robot is defined exactly once
    instead of being baked into the scene a second time. This is the standard
    scene, so the pick cube carries AprilTag faces by default.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "pick_and_place_environment"
    _add_pick_cube(spec, apriltag=apriltags)
    add_workspace_frame(spec)
    add_overhead_camera_mount(spec)
    apply_materials(spec, materials or MaterialConfig())
    if apriltags:
        _add_pick_cube_apriltags(spec)
        add_workspace_frame_apriltags(spec)
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
    include_environment: bool = True,
    apriltags: bool | None = None,
) -> Path:
    """Write a standalone, machine-local XML file for the composed scene."""
    spec = build_scene(
        wrist_camera=wrist_camera,
        materials=materials,
        include_environment=include_environment,
        apriltags=apriltags,
    )
    spec.meshdir = str(STOCK_ASSETS_DIR)
    spec.compile()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_xml())
    return output


def _add_pick_cube(spec: mujoco.MjSpec, *, apriltag: bool) -> None:
    cube = spec.worldbody.add_body(name="pick_cube", pos=(0.2, -0.12, PICK_CUBE_HALF_SIZE))
    half = PICK_CUBE_HALF_SIZE
    # The AprilTag stickers are white-backed; the material (added after
    # apply_materials) carries the per-face textures and tints them with this
    # white base so the tags render at full contrast. A plain cube keeps its
    # solid colour and no material.
    rgba = (1.0, 1.0, 1.0, 1.0) if apriltag else PICK_CUBE_RGBA
    cube.add_geom(
        name="pick_cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half, half, half),
        rgba=rgba,
        material="pick_cube_apriltags" if apriltag else "",
        mass=0.0137,
        priority=1,
        solref=(0.002, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    )


def _add_pick_cube_apriltags(spec: mujoco.MjSpec) -> None:
    """Texture the pick cube's six faces with their AprilTag stickers.

    Called after :func:`apply_materials`, which clears the spec's materials, so
    the cube texture and material survive into the compiled model (mirroring how
    the workspace-frame tags are added).
    """
    texture = spec.add_texture(
        name="pick_cube_apriltags",
        type=mujoco.mjtTexture.mjTEXTURE_CUBE,
    )
    texture.cubefiles = [
        str(APRILTAG_TEXTURE_DIR / f"tagStandard41h12_{tag_id:05d}_30x30mm_tag20mm.png")
        for tag_id in PICK_CUBE_APRILTAG_IDS
    ]
    material = spec.add_material(name="pick_cube_apriltags")
    material.textures[1] = "pick_cube_apriltags"
