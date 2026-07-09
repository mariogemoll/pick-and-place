# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Size-budgeted mesh simplification shared by the GLB generation scripts.

Each mesh is decimated (MeshLab quadric edge collapse) at a fixed grid of
face ratios, recording the geometric error of each variant (98th percentile
of sampled two-way surface distance). Given an error tolerance, every mesh
independently picks its smallest variant within tolerance, so faces go to
the parts that need them. ``find_tolerance`` bisects for the finest
tolerance whose packed, meshopt-compressed GLBs still fit a size budget.
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pymeshlab
import trimesh
from tqdm import tqdm

MIN_RATIO = 0.002
RATIO_GRID = np.geomspace(MIN_RATIO, 1.0, 16)
MIN_TOLERANCE_MM = 0.01
MAX_TOLERANCE_MM = 20.0
SIZE_SEARCH_STEPS = 10

ROOT = Path(__file__).resolve().parents[2]
GLTF_TRANSFORM = ROOT / "ts" / "node_modules" / ".bin" / "gltf-transform"


@dataclass
class Variant:
    ratio: float
    error_mm: float
    mesh: trimesh.Trimesh


@dataclass
class MeshCurve:
    """A source mesh with its precomputed decimation variants, coarse to fine."""

    name: str
    original_faces: int
    detail_factor: float
    variants: list[Variant]

    def select(self, tolerance_mm: float) -> Variant:
        tolerance = tolerance_mm / self.detail_factor
        for variant in self.variants:
            if variant.error_mm <= tolerance:
                return variant
        return self.variants[-1]


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


def build_curve(mesh: trimesh.Trimesh, name: str, detail_factor: float) -> MeshCurve:
    """Decimate at every grid ratio, coarse to fine, until within the finest tolerance."""
    # Deviation sampling draws from the global RNG; reseed per mesh so results
    # don't depend on processing order or worker-process assignment.
    np.random.seed(0)
    variants: list[Variant] = []
    for ratio in RATIO_GRID:
        reduced = decimate(mesh, float(ratio))
        error = 0.0 if len(reduced.faces) >= len(mesh.faces) else deviation_mm(mesh, reduced)
        variants.append(Variant(float(ratio), error, reduced))
        if error <= MIN_TOLERANCE_MM / detail_factor:
            break
    return MeshCurve(name, len(mesh.faces), detail_factor, variants)


def build_curves(specs: list[tuple[trimesh.Trimesh, str, float]]) -> list[MeshCurve]:
    """Build curves for (mesh, name, detail_factor) specs in parallel processes."""
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(build_curve, *spec) for spec in specs]
        for _ in tqdm(as_completed(futures), total=len(futures), unit="mesh"):
            pass
        return [future.result() for future in futures]


def detail_factor_for(name: str, details: list[tuple[str, float]]) -> float:
    factor = 1.0
    for pattern, pattern_factor in details:
        if fnmatch.fnmatch(name, pattern):
            factor = max(factor, pattern_factor)
    return factor


def parse_detail(value: str) -> tuple[str, float]:
    pattern, sep, factor = value.partition("=")
    if not sep or not pattern:
        raise argparse.ArgumentTypeError(f"expected GLOB=FACTOR, got: {value!r}")
    try:
        parsed = float(factor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid factor in: {value!r}") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"factor must be positive: {value!r}")
    return pattern, parsed


def compressed_kb(scenes: list[trimesh.Scene], workdir: Path) -> float:
    total = 0.0
    for index, scene in enumerate(scenes):
        raw = workdir / f"raw_{index}.glb"
        optimized = workdir / f"optimized_{index}.glb"
        scene.export(raw)
        subprocess.run(
            [str(GLTF_TRANSFORM), "meshopt", str(raw), str(optimized)],
            check=True,
            capture_output=True,
        )
        total += optimized.stat().st_size / 1024
    return total


def find_tolerance(
    pack: Callable[[float], list[trimesh.Scene]], target_kb: float, workdir: Path
) -> float:
    """Bisect for the finest tolerance whose compressed GLBs fit the budget."""
    low, high = MIN_TOLERANCE_MM, MAX_TOLERANCE_MM

    size = compressed_kb(pack(low), workdir)
    print(f"tolerance {low:6.3f}mm -> {size:7.1f}KB")
    if size <= target_kb:
        return low

    tolerance = high
    for _ in range(SIZE_SEARCH_STEPS):
        mid = math.sqrt(low * high)
        size = compressed_kb(pack(mid), workdir)
        print(f"tolerance {mid:6.3f}mm -> {size:7.1f}KB")
        if size <= target_kb:
            tolerance, high = mid, mid
        else:
            low = mid
    if tolerance == MAX_TOLERANCE_MM:
        size = compressed_kb(pack(tolerance), workdir)
        if size > target_kb:
            print(f"warning: {size:.1f}KB exceeds the {target_kb:.0f}KB budget even at "
                  f"the coarsest tolerance ({tolerance}mm)")
    return tolerance


def print_selection(curves: list[MeshCurve], tolerance: float) -> None:
    print(f"{'mesh':42} {'orig':>7} {'ratio':>6} {'faces':>6} {'p98':>7} {'detail':>6}")
    for curve in curves:
        variant = curve.select(tolerance)
        print(
            f"{curve.name:42} {curve.original_faces:7d} {variant.ratio:6.3f} "
            f"{len(variant.mesh.faces):6d} {variant.error_mm:5.2f}mm "
            f"{curve.detail_factor:5.1f}x"
        )
