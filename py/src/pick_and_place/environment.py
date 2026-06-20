# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Environment objects for sim2real: calibration workspace_frame and overhead camera."""

from __future__ import annotations

from pathlib import Path

import mujoco

from pick_and_place.camera_module import add_camera_module
from pick_and_place.camera_intrinsics import OVERHEAD_CAMERA_INTRINSICS

REPO_ROOT = Path(__file__).resolve().parents[3]
APRILTAG_TEXTURE_DIR = REPO_ROOT / "assets" / "apriltags" / "textures"
WORKSPACE_FRAME_STL_DIR = REPO_ROOT / "stl" / "workspace_frame"
OVERHEAD_MOUNT_STL_DIR = (
    REPO_ROOT
    / "SO-ARM100"
    / "Optional"
    / "Overhead_Cam_Mount_32x32_UVC_Module"
    / "stl"
)

WORKSPACE_FRAME_POS = (0.279579, 0.0000305, 0.0)
WORKSPACE_FRAME_QUAT = (-0.707107, 0.0, 0.0, -0.707107)

OVERHEAD_CAMERA_MOUNT_POS = (0.316979, -0.0729945, 0.0)
OVERHEAD_CAMERA_MOUNT_QUAT = (-0.707107, 0.0, 0.0, -0.707107)

WORKSPACE_FRAME_RED = (0.91, 0.3, 0.24, 1.0)
WORKSPACE_FRAME_GRAY = (0.55, 0.55, 0.58, 1.0)
WORKSPACE_FRAME_YELLOW = (0.93, 0.67, 0.18, 1.0)
WORKSPACE_FRAME_BLUE = (0.18, 0.55, 0.91, 1.0)
# A brighter grey for the overhead mount (dark parts hierarchy: camera < motor < mount).
MOUNT_BRIGHT_GRAY = (0.3, 0.3, 0.3, 1.0)

WORKSPACE_FRAME_APRILTAG_PLATES: tuple[tuple[int, str, tuple[float, float, float]], ...] = (
    (12, "ne", (0.230, 0.230, 0.0025)),
    (13, "nw", (-0.230, 0.230, 0.0025)),
    (14, "sw", (-0.230, -0.230, 0.0025)),
    (15, "se", (0.230, -0.230, 0.0025)),
)


def _add_robot_arm_base(
    spec: mujoco.MjSpec,
    frame: mujoco.MjsBody,
    name: str,
    cx: float,
    collision_default: mujoco.MjsDefault | None,
) -> None:
    """Add one SO-101 arm-base plate to the workspace north edge at local x = ``cx``.

    The plate is the printed ``arm_base.stl`` mount; the mesh origin sits 0.0534 m
    in x and 0.0491 m in y from the collision-box centre, so both are offset from
    the nominal plate centre ``(cx, 0.2552)``.
    """
    if not any(m.name == "robot_arm_base" for m in spec.meshes):
        spec.add_mesh(
            name="robot_arm_base",
            file=str(OVERHEAD_MOUNT_STL_DIR / "arm_base.stl"),
            scale=(0.001, 0.001, 0.001),
        )
    frame.add_geom(
        name=f"workspace_frame_{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="robot_arm_base",
        pos=(cx - 0.0534305, 0.206079, 0.005),
        quat=(-0.5, -0.5, 0.5, 0.5),
        rgba=WORKSPACE_FRAME_YELLOW,
        material="plastic",
        contype=0,
        conaffinity=0,
        group=2,
    )
    frame.add_geom(
        default=collision_default,
        name=f"workspace_frame_{name}_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(cx, 0.2552, 0.0036),
        size=(0.0795196, 0.0448, 0.0036),
        material="collision",
        group=3,
    )


def _add_north_single_robot_plate(
    spec: mujoco.MjSpec,
    frame: mujoco.MjsBody,
    collision_default: mujoco.MjsDefault | None,
) -> None:
    """One robot arm-base plate at the north centre, flanked by grey plastic strips."""
    _add_workspace_frame_part(
        spec, frame, "north_02", WORKSPACE_FRAME_STL_DIR / "part_02_box_11p6.stl",
        pos=(-0.1375, 0.2813, 0), rgba=WORKSPACE_FRAME_GRAY, material="plastic",
        col_size=(0.053, 0.0187, 0.0036), col_pos=(-0.1325, 0.2813, 0.0036), collision_default=collision_default
    )
    _add_robot_arm_base(spec, frame, "north_03", cx=0.0, collision_default=collision_default)
    _add_workspace_frame_part(
        spec, frame, "north_04", WORKSPACE_FRAME_STL_DIR / "part_02_box_11p6.stl",
        pos=(0.1375, 0.2813, 0), quat=(0, 0, 0, 1), rgba=WORKSPACE_FRAME_GRAY, material="plastic",
        col_size=(0.053, 0.0187, 0.0036), col_pos=(0.1325, 0.2813, 0.0036), collision_default=collision_default
    )


def _add_north_dual_robot_plates(
    spec: mujoco.MjSpec,
    frame: mujoco.MjsBody,
    collision_default: mujoco.MjsDefault | None,
) -> None:
    """Two robot arm-base plates at local x = ±0.116 with a centre connector."""
    _add_workspace_frame_part(
        spec, frame, "north_03", WORKSPACE_FRAME_STL_DIR / "part_04_box_7p3.stl",
        pos=(0.0, 0.2813, 0), rgba=WORKSPACE_FRAME_GRAY, material="plastic",
        col_size=(0.0365, 0.0187, 0.0036), collision_default=collision_default
    )
    # local +x maps to world +y, so cx = +0.116 is the "left" plate, -0.116 "right".
    _add_robot_arm_base(spec, frame, "north_left", cx=0.116, collision_default=collision_default)
    _add_robot_arm_base(spec, frame, "north_right", cx=-0.116, collision_default=collision_default)


def add_workspace_frame(
    spec: mujoco.MjSpec,
    *,
    collision_default: mujoco.MjsDefault | None = None,
    dual_robot: bool = False,
) -> mujoco.MjsBody:
    """Add the 60cm calibration workspace_frame to the worldbody.

    With ``dual_robot`` the single centre arm-base plate on the north edge is
    replaced by two arm-base plates at local x = ±0.116 (the two robot mounts),
    matching the two-robot hackathon rig.
    """
    frame = spec.worldbody.add_body(name="workspace_frame_frame", pos=WORKSPACE_FRAME_POS, quat=WORKSPACE_FRAME_QUAT)

    # North side components. The corner flats (north_01/north_05) are shared; the
    # centre carries either one robot arm-base plate or two, depending on the rig.
    _add_workspace_frame_part(
        spec, frame, "north_01", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(-0.24775, 0.2813, 0), rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.05225, 0.0187, 0.0036), collision_default=collision_default
    )
    if dual_robot:
        _add_north_dual_robot_plates(spec, frame, collision_default)
    else:
        _add_north_single_robot_plate(spec, frame, collision_default)
    _add_workspace_frame_part(
        spec, frame, "north_05", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(0.24775, 0.2813, 0), quat=(0, 0, 0, 1), rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.05225, 0.0187, 0.0036), collision_default=collision_default
    )

    # East side (all MDF).
    side_quat = (0.707107, 0.0, 0.0, -0.707107)
    _add_workspace_frame_part(
        spec, frame, "east_01", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(0.2813, 0.24775, 0), quat=side_quat, rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.0187, 0.05225, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "east_02", WORKSPACE_FRAME_STL_DIR / "part_03_box_15p9.stl",
        pos=(0.2813, 0.116, 0), quat=side_quat, rgba=WORKSPACE_FRAME_YELLOW, material="mdf",
        col_size=(0.0187, 0.0695, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "east_03", WORKSPACE_FRAME_STL_DIR / "part_04_box_7p3.stl",
        pos=(0.2813, 0, 0), quat=side_quat, rgba=WORKSPACE_FRAME_BLUE, material="mdf",
        col_size=(0.0187, 0.0365, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "east_04", WORKSPACE_FRAME_STL_DIR / "part_03_box_15p9.stl",
        pos=(0.2813, -0.116, 0), quat=side_quat, rgba=WORKSPACE_FRAME_YELLOW, material="mdf",
        col_size=(0.0187, 0.0695, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "east_05", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(0.2813, -0.24775, 0), quat=(0.707107, 0.0, 0.0, 0.707107), rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.0187, 0.05225, 0.0036), collision_default=collision_default
    )

    # South side (all MDF).
    _add_workspace_frame_part(
        spec, frame, "south_01", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(0.24775, -0.2813, 0), quat=(0, 0, 0, 1), rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.05225, 0.0187, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "south_02", WORKSPACE_FRAME_STL_DIR / "part_03_box_15p9.stl",
        pos=(0.116, -0.2813, 0), quat=(0, 0, 0, 1), rgba=WORKSPACE_FRAME_YELLOW, material="mdf",
        col_size=(0.0695, 0.0187, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "south_04", WORKSPACE_FRAME_STL_DIR / "part_03_box_15p9.stl",
        pos=(-0.116, -0.2813, 0), quat=(0, 0, 0, 1), rgba=WORKSPACE_FRAME_YELLOW, material="mdf",
        col_size=(0.0695, 0.0187, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "south_05", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(-0.24775, -0.2813, 0), quat=(-1, 0, 0, 0), rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.05225, 0.0187, 0.0036), collision_default=collision_default
    )

    # West side (all MDF).
    west_quat = (0.707107, 0.0, 0.0, 0.707107)
    _add_workspace_frame_part(
        spec, frame, "west_01", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(-0.2813, -0.24775, 0), quat=west_quat, rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.0187, 0.05225, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "west_02", WORKSPACE_FRAME_STL_DIR / "part_03_box_15p9.stl",
        pos=(-0.2813, -0.116, 0), quat=west_quat, rgba=WORKSPACE_FRAME_YELLOW, material="mdf",
        col_size=(0.0187, 0.0695, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "west_03", WORKSPACE_FRAME_STL_DIR / "part_04_box_7p3.stl",
        pos=(-0.2813, 0, 0), quat=west_quat, rgba=WORKSPACE_FRAME_BLUE, material="mdf",
        col_size=(0.0187, 0.0365, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "west_04", WORKSPACE_FRAME_STL_DIR / "part_03_box_15p9.stl",
        pos=(-0.2813, 0.116, 0), quat=west_quat, rgba=WORKSPACE_FRAME_YELLOW, material="mdf",
        col_size=(0.0187, 0.0695, 0.0036), collision_default=collision_default
    )
    _add_workspace_frame_part(
        spec, frame, "west_05", WORKSPACE_FRAME_STL_DIR / "part_01_box_10p45_flat.stl",
        pos=(-0.2813, 0.24775, 0), quat=(-0.707107, 0.0, 0.0, 0.707107), rgba=WORKSPACE_FRAME_RED, material="mdf",
        col_size=(0.0187, 0.05225, 0.0036), collision_default=collision_default
    )

    return frame


def add_workspace_frame_apriltags(spec: mujoco.MjSpec) -> None:
    """Add the textured calibration AprilTag plates to the workspace frame."""
    frame = spec.body("workspace_frame_frame")
    for tag_id, corner_name, pos in WORKSPACE_FRAME_APRILTAG_PLATES:
        texture_name = f"workspace_frame_apriltag_{tag_id:02d}"
        material_name = f"{texture_name}_material"
        spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_CUBE,
            file=str(
                APRILTAG_TEXTURE_DIR
                / f"tagStandard41h12_{tag_id:05d}_60x60mm_tag40mm.png"
            ),
        )
        material = spec.add_material(name=material_name)
        material.textures[1] = texture_name
        frame.add_geom(
            name=f"workspace_frame_tag_{corner_name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.03, 0.03, 0.0025),
            pos=pos,
            material=material_name,
            contype=0,
            conaffinity=0,
            group=2,
        )


def _add_workspace_frame_part(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    name: str,
    mesh_path: Path,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float] = (1, 0, 0, 0),
    rgba: tuple[float, float, float, float] | None = None,
    material: str | None = None,
    col_size: tuple[float, float, float] | None = None,
    col_pos: tuple[float, float, float] | None = None,
    collision_default: mujoco.MjsDefault | None = None,
) -> None:
    mesh_name = f"workspace_frame_{mesh_path.stem}"
    if not any(m.name == mesh_name for m in spec.meshes):
        spec.add_mesh(name=mesh_name, file=str(mesh_path))

    parent.add_geom(
        name=f"workspace_frame_{name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        pos=pos,
        quat=quat,
        rgba=rgba,
        material=material,
        contype=0,
        conaffinity=0,
        group=2,
    )

    if col_size:
        parent.add_geom(
            default=collision_default,
            name=f"workspace_frame_{name}_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=col_pos or (pos[0], pos[1], 0.0036),
            size=col_size,
            material="collision",
            group=3,
        )


def add_overhead_camera_mount(
    spec: mujoco.MjSpec,
    *,
    collision_default: mujoco.MjsDefault | None = None,
) -> mujoco.MjsBody:
    """Add the overhead camera mount and the camera module."""
    mount = spec.worldbody.add_body(
        name="overhead_camera_mount",
        pos=OVERHEAD_CAMERA_MOUNT_POS,
        quat=OVERHEAD_CAMERA_MOUNT_QUAT
    )

    # Meshes for the mount.
    spec.add_mesh(
        name="overhead_mount_bottom",
        file=str(OVERHEAD_MOUNT_STL_DIR / "cam_mount_bottom.stl"),
        scale=(0.001, 0.001, 0.001)
    )
    spec.add_mesh(
        name="overhead_mount_middle",
        file=str(OVERHEAD_MOUNT_STL_DIR / "cam_mount_middle.stl"),
        scale=(0.001, 0.001, 0.001)
    )
    spec.add_mesh(
        name="overhead_mount_top",
        file=str(OVERHEAD_MOUNT_STL_DIR / "cam_mount_top.stl"),
        scale=(0.001, 0.001, 0.001)
    )

    # Mount parts are plastic, with a brighter gray than motors/cameras.
    # We set these RGBA values here; scene.py will call apply_materials which
    # we'll fix to NOT clear rgba if a material is already assigned.
    mount.add_geom(
        name="overhead_mount_bottom_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="overhead_mount_bottom",
        pos=(0.0365125, -0.2626, 0),
        quat=(0.5, 0.5, 0.5, 0.5),
        rgba=MOUNT_BRIGHT_GRAY,
        material="plastic",
        contype=0,
        conaffinity=0,
        group=2,
    )
    mount.add_geom(
        name="overhead_mount_middle_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="overhead_mount_middle",
        pos=(0.0730125, -0.2439, 0.188),
        quat=(0.5, 0.5, 0.5, 0.5),
        rgba=MOUNT_BRIGHT_GRAY,
        material="plastic",
        contype=0,
        conaffinity=0,
        group=2,
    )
    mount.add_geom(
        name="overhead_mount_top_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="overhead_mount_top",
        pos=(0.0730125, -0.2439, 0.188),
        quat=(0.5, 0.5, 0.5, 0.5),
        rgba=MOUNT_BRIGHT_GRAY,
        material="plastic",
        contype=0,
        conaffinity=0,
        group=2,
    )

    # Collision boxes for the mount.
    mount.add_geom(
        default=collision_default,
        name="overhead_mount_main_col",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0730125, -0.2439, 0.276),
        quat=(0.707107, 0, 0, 0.707107),
        size=(0.0128, 0.0184, 0.27),
        material="collision",
        group=3,
    )
    mount.add_geom(
        default=collision_default,
        name="overhead_mount_bottom_col",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0730245, -0.2439, 0.0036),
        quat=(0.707107, 0, 0, 0.707107),
        size=(0.0187, 0.046512, 0.0038),
        material="collision",
        group=3,
    )

    camera_module = add_camera_module(
        mount,
        prefix="overhead_",
        pos=(0.0730125, -0.21667, 0.5536),
        quat=(0.976299, 0.216427, 0, 0),
        camera_name="overhead_camera",
        fovy=OVERHEAD_CAMERA_INTRINSICS["fovy_deg"],
        collision_default=collision_default,
    )
    for geom in camera_module.geoms:
        if int(geom.group) == 2:
            geom.material = "camera"

    return mount
