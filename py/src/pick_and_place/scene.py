# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compose the SO-101 robot with a floor, workspace overlays, soft light, and cube."""

from __future__ import annotations

from pathlib import Path

import mujoco

from pick_and_place.builder import STOCK_ASSETS_DIR, build_robot
from pick_and_place.environment import (
    APRILTAG_TEXTURE_DIR,
    WORKSPACE_FRAME_POS,
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

# The north frame components end at local Y=300 mm.  In the calibrated world
# frame that is this X position; the tabletop meets their outer face exactly.
TABLE_NORTH_EDGE_X = WORKSPACE_FRAME_POS[0] - 0.3
TABLE_LENGTH = 20.0
# East in the workspace frame maps to +Y in world coordinates.  The tabletop
# deliberately extends well beyond the replay view in every horizontal direction.
TABLE_WEST_EDGE_Y = -10.0
TABLE_EAST_EDGE_Y = 10.0
TABLE_WIDTH = TABLE_EAST_EDGE_Y - TABLE_WEST_EDGE_Y
TABLE_THICKNESS = 0.04
TABLE_HEIGHT = 0.75
TABLE_LEG_WIDTH = 0.08
TABLE_LEG_INSET = 0.12
TABLE_RGBA = (0.56, 0.5, 0.4, 1.0)
# Neutral backdrop used outside the physical tabletop in camera-matched replays.
TABLE_BACKGROUND_RGBA = (0.58, 0.58, 0.58, 1.0)
BACKDROP_WALL_DISTANCE = 1.4
BACKDROP_WALL_THICKNESS = 0.04
BACKDROP_WALL_WIDTH = 30.0
BACKDROP_WALL_HEIGHT = 12.0


def build_scene(
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    include_environment: bool = True,
    tabletop: bool = False,
    apriltag_cube: bool | None = None,
    robot_dynamics: bool | str | Path = True,
) -> mujoco.MjSpec:
    """Return the composed robot with a floor, workspace overlays, soft light, and cube.

    ``apriltag_cube`` selects the pick cube's appearance: the plain red cube for
    the simple scene, or the AprilTag-stickered cube (a perception target) for
    the standard scene. When left ``None`` it follows ``include_environment``, so
    the simple scene gets the red cube and the standard scene the tagged one.
    """
    if apriltag_cube is None:
        apriltag_cube = include_environment

    spec = build_robot(
        wrist_camera=wrist_camera,
        materials=materials,
        robot_dynamics=robot_dynamics,
    )
    spec.modelname = "so101_with_cube"
    _add_scene_lighting(spec)

    base = spec.body("base")
    base.pos = (0.0, 0.0, ROBOT_BASE_Z_OFFSET)

    # Attach overlays to worldbody so they stay on the floor.
    add_workspace_overlays(spec, spec.worldbody)
    _add_pick_cube(spec, apriltag=apriltag_cube)

    if include_environment:
        collision_default = spec.find_default("collision")
        add_workspace_frame(spec, collision_default=collision_default)
        add_overhead_camera_mount(spec, collision_default=collision_default)

    apply_materials(spec, materials or MaterialConfig())
    if apriltag_cube:
        _add_pick_cube_apriltags(spec)
    if include_environment:
        add_workspace_frame_apriltags(spec)
    if tabletop:
        _add_tabletop(spec)
    else:
        _add_groundplane(spec)

    return spec


def build_environment(
    *,
    materials: MaterialConfig | None = None,
    apriltag_cube: bool = True,
    tabletop: bool = False,
) -> mujoco.MjSpec:
    """Return only the environment, with no robot.

    Contains the floor, pick cube, calibration workspace frame, and overhead
    camera mount, all attached to the worldbody. The web viewer loads this on
    top of the standalone ``so101`` model so the robot is defined exactly once
    instead of being baked into the scene a second time. Set ``tabletop`` for
    the lit finite table used by camera-matched replays. This is the standard
    scene, so the pick cube carries AprilTag faces by default.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "pick_and_place_environment"
    if tabletop:
        _add_scene_lighting(spec)
    _add_pick_cube(spec, apriltag=apriltag_cube)
    add_workspace_frame(spec)
    add_overhead_camera_mount(spec)
    apply_materials(spec, materials or MaterialConfig())
    if apriltag_cube:
        _add_pick_cube_apriltags(spec)
    add_workspace_frame_apriltags(spec)
    if tabletop:
        _add_tabletop(spec)
    else:
        _add_groundplane(spec)
    return spec


def _add_scene_lighting(spec: mujoco.MjSpec) -> None:
    """Configure the neutral headlight and overhead fill used by replay renders."""
    spec.visual.headlight.diffuse = (0.6, 0.6, 0.6)
    spec.visual.headlight.ambient = (0.3, 0.3, 0.3)
    spec.visual.headlight.specular = (0.0, 0.0, 0.0)
    scene_light = spec.worldbody.add_light(
        name="scene_light",
        pos=(0.0, 0.0, 1.0),
        dir=(0.0, 0.0, -1.0),
        diffuse=(0.35, 0.35, 0.35),
        ambient=(0.15, 0.15, 0.15),
        specular=(0.0, 0.0, 0.0),
    )
    scene_light.castshadow = False
    warm_spotlight = spec.worldbody.add_light(
        name="warm_spotlight",
        pos=(1.2, -0.8, 2.0),
        dir=(-0.9, 0.8, -2.0),
        cutoff=35.0,
        exponent=8.0,
        diffuse=(0.4, 0.28, 0.17),
        ambient=(0.04, 0.028, 0.017),
        specular=(0.0, 0.0, 0.0),
    )
    warm_spotlight.castshadow = False


def _add_groundplane(spec: mujoco.MjSpec) -> None:
    spec.add_material(
        name="groundplane",
        rgba=(0.82, 0.74, 0.6, 1.0),
        reflectance=0.0,
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05),
        material="groundplane",
    )


def _add_tabletop(spec: mujoco.MjSpec) -> None:
    """Add the finite tabletop and neutral background used by replay renders."""
    # The plane sits below the tabletop, so it is visible only beyond the table
    # edge without introducing a seam or z-fighting on the table surface.
    spec.worldbody.add_geom(
        name="table_background",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05),
        pos=(0.0, 0.0, -TABLE_HEIGHT),
        rgba=TABLE_BACKGROUND_RGBA,
        contype=0,
        conaffinity=0,
    )
    # The wall is 1.4 m north of the table edge. Its bottom meets the background
    # floor so the camera sees one continuous neutral backdrop behind the scene.
    spec.worldbody.add_geom(
        name="backdrop_wall",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(
            TABLE_NORTH_EDGE_X - BACKDROP_WALL_DISTANCE - BACKDROP_WALL_THICKNESS / 2,
            0.0,
            -TABLE_HEIGHT + BACKDROP_WALL_HEIGHT / 2,
        ),
        size=(BACKDROP_WALL_THICKNESS / 2, BACKDROP_WALL_WIDTH / 2, BACKDROP_WALL_HEIGHT / 2),
        rgba=TABLE_BACKGROUND_RGBA,
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_geom(
        name="tabletop",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(
            TABLE_NORTH_EDGE_X + TABLE_LENGTH / 2,
            (TABLE_WEST_EDGE_Y + TABLE_EAST_EDGE_Y) / 2,
            -TABLE_THICKNESS / 2,
        ),
        size=(TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_THICKNESS / 2),
        rgba=TABLE_RGBA,
        contype=0,
        conaffinity=0,
    )
    leg_height = TABLE_HEIGHT - TABLE_THICKNESS
    leg_z = -TABLE_THICKNESS - leg_height / 2
    for x, y in (
        (TABLE_NORTH_EDGE_X + TABLE_LEG_INSET, TABLE_WEST_EDGE_Y + TABLE_LEG_INSET),
        (TABLE_NORTH_EDGE_X + TABLE_LEG_INSET, TABLE_EAST_EDGE_Y - TABLE_LEG_INSET),
        (
            TABLE_NORTH_EDGE_X + TABLE_LENGTH - TABLE_LEG_INSET,
            TABLE_WEST_EDGE_Y + TABLE_LEG_INSET,
        ),
        (
            TABLE_NORTH_EDGE_X + TABLE_LENGTH - TABLE_LEG_INSET,
            TABLE_EAST_EDGE_Y - TABLE_LEG_INSET,
        ),
    ):
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, y, leg_z),
            size=(TABLE_LEG_WIDTH / 2, TABLE_LEG_WIDTH / 2, leg_height / 2),
            rgba=TABLE_RGBA,
            contype=0,
            conaffinity=0,
        )


def export_scene(
    output: Path,
    *,
    wrist_camera: bool = True,
    materials: MaterialConfig | None = None,
    include_environment: bool = True,
    apriltag_cube: bool | None = None,
    robot_dynamics: bool | str | Path = True,
) -> Path:
    """Write a standalone, machine-local XML file for the composed scene."""
    spec = build_scene(
        wrist_camera=wrist_camera,
        materials=materials,
        include_environment=include_environment,
        apriltag_cube=apriltag_cube,
        robot_dynamics=robot_dynamics,
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
