#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate simplified intermediary GLBs for an arbitrary robot_descriptions model.

Decimates every mesh referenced by the model's MJCF (MeshLab quadric edge
collapse), searching by bisection for the smallest face ratio whose result
deviates from the original by at most the target budget (98th percentile of
sampled two-way surface distance, --target-mm).

An end effector's meshes can be simplified alongside the base robot's, into
the same output directory, prefixed to match the mesh names produced by
``export_generic_robot.py --gripper``::

    python scripts/simplify_generic_meshes.py ur5e_mj_description \\
        --gripper robotiq_2f85_mj_description
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

import mujoco
import numpy as np
import pymeshlab
import trimesh
from tqdm import tqdm

MIN_RATIO = 0.002
DEFAULT_TARGET_MM = 0.5
GRIPPER_PREFIX = "gripper_"

ROOT = Path(__file__).resolve().parents[2]
GLB_ROOT = ROOT / "intermediary-glb-generic"


def decimate(mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
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


def deviation_mm(original: trimesh.Trimesh, reduced: trimesh.Trimesh, samples: int = 1500) -> float:
    to_reduced = trimesh.proximity.ProximityQuery(reduced).on_surface(original.sample(samples))[1]
    to_original = trimesh.proximity.ProximityQuery(original).on_surface(reduced.sample(samples))[1]
    return float(np.percentile(np.concatenate([to_reduced, to_original]), 98)) * 1000


def simplify(
    mesh: trimesh.Trimesh, target_mm: float
) -> tuple[float, trimesh.Trimesh, float]:
    """Find the smallest decimation ratio whose result meets the deviation budget."""
    reduced = decimate(mesh, MIN_RATIO)
    error = deviation_mm(mesh, reduced)
    if error <= target_mm:
        return MIN_RATIO, reduced, error

    ratio, reduced, error = 1.0, mesh.copy(), 0.0
    low, high = MIN_RATIO, 1.0
    for _ in range(8):
        mid = math.sqrt(low * high)
        candidate = decimate(mesh, mid)
        candidate_error = deviation_mm(mesh, candidate)
        if candidate_error <= target_mm:
            ratio, reduced, error, high = mid, candidate, candidate_error, mid
        else:
            low = mid
    return ratio, reduced, error


def mjcf_path_for(robot: str) -> Path:
    module = importlib.import_module(f"robot_descriptions.{robot}")
    return Path(module.MJCF_PATH)


def simplify_meshes(
    robot: str, glb_dir: Path, target_mm: float, *, name_prefix: str = ""
) -> int:
    mjcf_path = mjcf_path_for(robot)
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    mesh_dir = mjcf_path.parent / spec.meshdir

    print(f"{'mesh':42} {'orig':>7} {'ratio':>6} {'faces':>6} {'p98':>7}")
    for mesh_spec in tqdm(spec.meshes, unit="mesh"):
        source = mesh_dir / mesh_spec.file
        name = f"{name_prefix}{Path(mesh_spec.file).stem}"
        mesh = trimesh.load_mesh(source, force="mesh", process=True)
        # OBJ visual meshes are often vertex-split along normal/UV seams, which
        # makes every smoothing island a separate connected component and
        # defeats the per-component decimation below. Merge purely by position.
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.apply_scale(mesh_spec.scale)

        ratio, reduced, error = simplify(mesh, target_mm)
        reduced.export(glb_dir / f"{name}.glb")
        tqdm.write(
            f"{name:42} {len(mesh.faces):7d} {ratio:6.3f} {len(reduced.faces):6d} {error:5.2f}mm"
        )

    return len(spec.meshes)


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
        "--target-mm",
        type=float,
        default=DEFAULT_TARGET_MM,
        help="deviation budget in mm (98th percentile of sampled two-way surface distance)",
    )
    args = parser.parse_args()

    glb_dir = GLB_ROOT / args.robot
    glb_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    count = simplify_meshes(args.robot, glb_dir, args.target_mm)
    if args.gripper:
        count += simplify_meshes(
            args.gripper, glb_dir, args.target_mm, name_prefix=GRIPPER_PREFIX
        )

    print(f"Wrote {count} GLBs to {glb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
