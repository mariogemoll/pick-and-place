#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate simplified SO101 intermediary GLBs.

Decimates every source STL to RATIO of its faces (MeshLab quadric edge
collapse). If the result deviates from the original by more than TARGET_MM
(98th percentile of sampled two-way surface distance), the smallest ratio
that meets the budget is found by bisection instead.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from tqdm import tqdm

RATIO = 0.04
TARGET_MM = 0.5

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "SO-ARM100" / "Simulation" / "SO101" / "assets"
GLB_DIR = ROOT / "intermediary-glb"


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


def main() -> int:
    np.random.seed(0)
    GLB_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(INPUT_DIR.glob("*.stl"))
    if not paths:
        raise SystemExit(f"no STL meshes found in: {INPUT_DIR}")

    print(f"{'mesh':42} {'orig':>7} {'ratio':>6} {'faces':>6} {'p98':>7}")
    for path in tqdm(paths, unit="mesh"):
        mesh = trimesh.load_mesh(path, force="mesh", process=True)
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
        reduced.export(GLB_DIR / path.with_suffix(".glb").name)
        tqdm.write(f"{path.name:42} {len(mesh.faces):7d} {ratio:6.3f} {len(reduced.faces):6d} "
                   f"{error:5.2f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
