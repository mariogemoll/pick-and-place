#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate simplified SO101 intermediary GLBs.

Decimates every source STL to RATIO of its faces (MeshLab quadric edge
collapse). If the result deviates from the original by more than TARGET_MM
(98th percentile of sampled two-way surface distance), the smallest ratio
that meets the budget is found by bisection instead.

Meshes are packed into three named-node GLBs matching the web viewer's three
independently-loadable scopes: ``arm.glb`` (everything but the gripper),
``gripper.glb`` (the ``gripper`` body subtree, for the gripper-only
visualization), and ``environment.glb`` (workspace frame + overhead mount).
The ``sts3215_03a_v1`` motor housing is reused by both the arm and the
gripper, so it is packed into both files.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from tqdm import tqdm

RATIO = 0.04
TARGET_MM = 0.5
MM_TO_M = 0.001
WRIST_CAMERA_MOUNT_SOURCE_TRANSFORM = np.array(
    (
        (0.0, 0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "SO-ARM100" / "Simulation" / "SO101" / "assets"
CAMERA_MODULE_MESH_NAME = "uvc_camera_module_32x32"
WRIST_CAMERA_MOUNT_STL = (
    ROOT
    / "SO-ARM100"
    / "Optional"
    / "SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module"
    / "stl"
    / "SO-ARM101_camera_wrist_mount.stl"
)
OVERHEAD_MOUNT_STL_DIR = (
    ROOT
    / "SO-ARM100"
    / "Optional"
    / "Overhead_Cam_Mount_32x32_UVC_Module"
    / "stl"
)
OVERHEAD_MOUNT_STLS = [
    OVERHEAD_MOUNT_STL_DIR / "arm_base.stl",
    OVERHEAD_MOUNT_STL_DIR / "cam_mount_bottom.stl",
    OVERHEAD_MOUNT_STL_DIR / "cam_mount_middle.stl",
    OVERHEAD_MOUNT_STL_DIR / "cam_mount_top.stl",
]
OVERHEAD_MOUNT_NAMES = {
    "arm_base.stl": "overhead_cam_arm_base",
    "cam_mount_bottom.stl": "overhead_mount_bottom",
    "cam_mount_middle.stl": "overhead_mount_middle",
    "cam_mount_top.stl": "overhead_mount_top",
}
WORKSPACE_FRAME_STL_DIR = ROOT / "stl" / "workspace_frame"
WORKSPACE_FRAME_STLS = sorted(WORKSPACE_FRAME_STL_DIR.glob("*.stl"))
WORKSPACE_FRAME_NAMES = {
    path.name: "workspace_frame_" + path.stem for path in WORKSPACE_FRAME_STLS
}
GLB_DIR = ROOT / "intermediary-glb"

# Mesh-name partition into the three packed GLBs (see module docstring).
ENVIRONMENT_MESH_NAMES = frozenset(OVERHEAD_MOUNT_NAMES.values()) | frozenset(
    WORKSPACE_FRAME_NAMES.values()
)
GRIPPER_MESH_NAMES = frozenset({
    "wrist_roll_follower_so101_v1",
    "moving_jaw_so101_v1",
    "SO-ARM101_camera_wrist_mount",
    CAMERA_MODULE_MESH_NAME,
})
SHARED_ARM_GRIPPER_MESH_NAMES = frozenset({"sts3215_03a_v1"})

# Keep these dimensions aligned with pick_and_place.camera_module. The generated
# mesh is visual-only; MuJoCo continues to use its primitive visual/collision geoms.
CAMERA_BOARD_EXTENTS = (0.032, 0.032, 0.002)
CAMERA_LENS_RADIUS = 0.007
CAMERA_LENS_LENGTH = 0.020
CAMERA_LENS_POS = (0.0, 0.0, -0.011)


def generate_camera_module_mesh() -> trimesh.Trimesh:
    """Generate the shared 32x32 UVC camera-module visual."""
    board = trimesh.creation.box(extents=CAMERA_BOARD_EXTENTS)
    barrel = trimesh.creation.cylinder(
        radius=CAMERA_LENS_RADIUS,
        height=CAMERA_LENS_LENGTH,
        sections=32,
    )
    barrel.apply_translation(CAMERA_LENS_POS)
    glass = trimesh.creation.cylinder(
        radius=CAMERA_LENS_RADIUS * 0.8,
        height=0.0004,
        sections=32,
    )
    glass.apply_translation((0.0, 0.0, CAMERA_LENS_POS[2] - CAMERA_LENS_LENGTH / 2 - 0.0002))

    board.visual.face_colors = (13, 13, 13, 255)
    barrel.visual.face_colors = (40, 40, 40, 255)
    glass.visual.face_colors = (30, 70, 95, 255)
    return trimesh.util.concatenate((board, barrel, glass))


def load_source_mesh(path: Path):
    """Load a source mesh in the canonical web coordinate system: meters."""
    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    # The SO-ARM mount STLs are authored in millimeters; the workspace-frame
    # STLs are already in meters (MuJoCo loads them unscaled), so they must not
    # be rescaled here.
    if path == WRIST_CAMERA_MOUNT_STL or path in OVERHEAD_MOUNT_STLS:
        mesh.apply_scale(MM_TO_M)
    if path == WRIST_CAMERA_MOUNT_STL:
        mesh.apply_transform(WRIST_CAMERA_MOUNT_SOURCE_TRANSFORM)
    return mesh


def decimate(mesh, ratio: float):
    reduced = []
    for component in mesh.split(only_watertight=False):
        target = max(4, math.ceil(len(component.faces) * ratio))
        if target >= len(component.faces):
            reduced.append(component.copy())
            continue
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(component.vertices, component.faces))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target,
            preserveboundary=True,
            boundaryweight=2.0,
            preservenormal=True,
            preservetopology=True,
        )
        result = ms.current_mesh()
        reduced.append(trimesh.Trimesh(result.vertex_matrix(), result.face_matrix()))
    return trimesh.util.concatenate(reduced)


def deviation_mm(original, reduced, samples: int = 1500) -> float:
    to_reduced = trimesh.proximity.ProximityQuery(reduced).on_surface(original.sample(samples))[1]
    to_original = trimesh.proximity.ProximityQuery(original).on_surface(reduced.sample(samples))[1]
    return float(np.percentile(np.concatenate([to_reduced, to_original]), 98)) * 1000


def destination_scenes(
    name: str, arm: trimesh.Scene, gripper: trimesh.Scene, environment: trimesh.Scene
) -> list[trimesh.Scene]:
    if name in ENVIRONMENT_MESH_NAMES:
        return [environment]
    if name in SHARED_ARM_GRIPPER_MESH_NAMES:
        return [arm, gripper]
    if name in GRIPPER_MESH_NAMES:
        return [gripper]
    return [arm]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-wrist-camera-mount",
        action="store_true",
        help="omit the optional SO-101 wrist-camera mount and camera-module visual",
    )
    parser.add_argument(
        "--no-overhead-camera-mount",
        action="store_true",
        help="omit the optional SO-101 overhead-camera mount",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(0)
    GLB_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(INPUT_DIR.glob("*.stl"))
    if not paths:
        raise SystemExit(f"no STL meshes found in: {INPUT_DIR}")

    gripper_scene = trimesh.Scene()
    if not args.no_wrist_camera_mount:
        if not WRIST_CAMERA_MOUNT_STL.is_file():
            raise SystemExit(f"wrist-camera mount STL not found: {WRIST_CAMERA_MOUNT_STL}")
        paths.append(WRIST_CAMERA_MOUNT_STL)
        gripper_scene.add_geometry(
            generate_camera_module_mesh(),
            node_name=CAMERA_MODULE_MESH_NAME,
            geom_name=CAMERA_MODULE_MESH_NAME,
        )

    if not args.no_overhead_camera_mount:
        for path in OVERHEAD_MOUNT_STLS:
            if not path.is_file():
                raise SystemExit(f"overhead-camera mount STL not found: {path}")
            paths.append(path)

    for path in WORKSPACE_FRAME_STLS:
        paths.append(path)

    arm_scene = trimesh.Scene()
    environment_scene = trimesh.Scene()
    name_map = {**OVERHEAD_MOUNT_NAMES, **WORKSPACE_FRAME_NAMES}

    print(f"{'mesh':42} {'orig':>7} {'ratio':>6} {'faces':>6} {'p98':>7}")
    for path in tqdm(paths, unit="mesh"):
        mesh = load_source_mesh(path)
        ratio = RATIO
        reduced = decimate(mesh, ratio)
        error = deviation_mm(mesh, reduced)
        if error > TARGET_MM:
            low, high = RATIO, 1.0
            for _ in range(8):
                mid = math.sqrt(low * high)
                candidate = decimate(mesh, mid)
                candidate_error = deviation_mm(mesh, candidate)
                if candidate_error <= TARGET_MM:
                    ratio, reduced, error, high = mid, candidate, candidate_error, mid
                else:
                    low = mid

        name = name_map.get(path.name, path.stem)
        for scene in destination_scenes(name, arm_scene, gripper_scene, environment_scene):
            scene.add_geometry(reduced, node_name=name, geom_name=name)
        tqdm.write(f"{path.name:42} {len(mesh.faces):7d} {ratio:6.3f} {len(reduced.faces):6d} "
                   f"{error:5.2f}mm")

    arm_scene.export(GLB_DIR / "arm.glb")
    gripper_scene.export(GLB_DIR / "gripper.glb")
    environment_scene.export(GLB_DIR / "environment.glb")
    print(f"Wrote arm.glb, gripper.glb, environment.glb to {GLB_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
