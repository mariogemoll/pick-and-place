#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate a simplified intermediary GLB for a robot_descriptions model.

Only meshes used by visual geoms (MuJoCo geom group 2 or 4) are exported;
collision-only meshes never reach the web viewer.

Meshes are simplified to the finest shared error tolerance whose packed,
meshopt-compressed GLB fits --target-kb (see ``simplification``).

Individual parts can be held to a tighter tolerance than the rest of the
robot with ``--detail GLOB=FACTOR`` (a matched mesh's tolerance is divided
by FACTOR), e.g. to keep an emblem crisp::

    python scripts/simplify_generic_meshes.py panda_mj_description \\
        --detail 'link6_*=4'

An end effector's meshes can be simplified alongside the base robot's, into
the same output file, prefixed to match the mesh names produced by
``export_generic_robot.py --gripper``::

    python scripts/simplify_generic_meshes.py ur5e_mj_description \\
        --gripper robotiq_2f85_mj_description
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path

import mujoco
import trimesh

from simplification import (
    MeshCurve,
    build_curves,
    compressed_kb,
    detail_factor_for,
    find_tolerance,
    parse_detail,
    print_selection,
)

DEFAULT_TARGET_KB = 300.0
GRIPPER_PREFIX = "gripper_"

ROOT = Path(__file__).resolve().parents[2]
GLB_ROOT = ROOT / "intermediary-glb-generic"


def mjcf_path_for(robot: str) -> Path:
    module = importlib.import_module(f"robot_descriptions.{robot}")
    return Path(module.MJCF_PATH)


def visual_mesh_names(spec: mujoco.MjSpec) -> frozenset[str]:
    """Names of meshes referenced by at least one visual geom (group 2 or 4).

    Falls back to all meshes for models that don't separate visual and
    collision geometry into groups.
    """
    model = spec.copy().compile()
    visual: set[str] = set()
    referenced: set[str] = set()
    for geom_id in range(model.ngeom):
        if mujoco.mjtGeom(int(model.geom_type[geom_id])) != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_name = model.mesh(int(model.geom_dataid[geom_id])).name
        referenced.add(mesh_name)
        if int(model.geom_group[geom_id]) in (2, 4):
            visual.add(mesh_name)
    return frozenset(visual or referenced)


def load_meshes(robot: str, *, name_prefix: str = "") -> list[tuple[str, trimesh.Trimesh]]:
    mjcf_path = mjcf_path_for(robot)
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    mesh_dir = mjcf_path.parent / spec.meshdir
    visual = visual_mesh_names(spec)

    meshes: list[tuple[str, trimesh.Trimesh]] = []
    for mesh_spec in spec.meshes:
        mesh_name = mesh_spec.name or Path(mesh_spec.file).stem
        if mesh_name not in visual:
            continue
        mesh = trimesh.load_mesh(mesh_dir / mesh_spec.file, force="mesh", process=True)
        # OBJ visual meshes are often vertex-split along normal/UV seams, which
        # makes every smoothing island a separate connected component and
        # defeats the per-component decimation. Merge purely by position.
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.apply_scale(mesh_spec.scale)
        meshes.append((f"{name_prefix}{Path(mesh_spec.file).stem}", mesh))
    return meshes


def pack_scene(curves: list[MeshCurve], tolerance_mm: float) -> trimesh.Scene:
    scene = trimesh.Scene()
    for curve in curves:
        scene.add_geometry(
            curve.select(tolerance_mm).mesh, node_name=curve.name, geom_name=curve.name
        )
    return scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "robot",
        help="robot_descriptions module name, e.g. ur5e_mj_description",
    )
    parser.add_argument(
        "--gripper",
        help="robot_descriptions module name for an end effector to simplify alongside, e.g. robotiq_2f85_mj_description",
    )
    parser.add_argument(
        "--target-kb",
        type=float,
        default=DEFAULT_TARGET_KB,
        help="size budget for the meshopt-compressed GLB (default %(default)s)",
    )
    parser.add_argument(
        "--detail",
        type=parse_detail,
        action="append",
        default=[],
        metavar="GLOB=FACTOR",
        help="hold meshes matching GLOB (fnmatch on the node name, including any "
        f"'{GRIPPER_PREFIX}' prefix) to a FACTOR-times tighter tolerance; repeatable",
    )
    args = parser.parse_args()

    glb_dir = GLB_ROOT / args.robot
    glb_dir.mkdir(parents=True, exist_ok=True)
    name = args.robot.removesuffix("_mj_description")

    meshes = load_meshes(args.robot)
    if args.gripper:
        meshes += load_meshes(args.gripper, name_prefix=GRIPPER_PREFIX)

    curves = build_curves([
        (mesh, mesh_name, detail_factor_for(mesh_name, args.detail))
        for mesh_name, mesh in meshes
    ])

    with tempfile.TemporaryDirectory() as workdir:
        tolerance = find_tolerance(
            lambda tol: [pack_scene(curves, tol)], args.target_kb, Path(workdir)
        )
        scene = pack_scene(curves, tolerance)
        size = compressed_kb([scene], Path(workdir))

    print(f"\ntolerance {tolerance:.3f}mm -> {size:.1f}KB compressed")
    print_selection(curves, tolerance)

    out_path = glb_dir / f"{name}.glb"
    scene.export(out_path)
    print(f"Wrote {len(curves)} meshes as named nodes to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
