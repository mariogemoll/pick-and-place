# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The declared reset distribution shared by evaluation manifests and RL resets.

``py/scripts/generate_scenario_manifest.py`` freezes draws from this
distribution into immutable benchmark manifests; DPPO fine-tuning draws from it
live on a disjoint seed stream. Keeping one implementation is what makes
"training and evaluation come from the same distribution" a fact about the code
rather than a claim about two copies of it.

The draw order is part of the contract: cube, then target, then plate yaw. A
frozen manifest reproduces only while that order and these constraints hold.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pick_and_place.scripted.episode_sampling import sample_cube, sample_target
from pick_and_place.core.geometry import CubePose
from pick_and_place.core.workspace_bounds import (
    CANONICAL_PICKUP_SECTOR,
    is_cube_drop_allowed,
    is_cube_pickup_allowed,
    PAN_AXIS,
    sample_target_plate_yaw,
)
from pick_and_place.spec.workspace import DROP_ZONE_HALF_SIZE

# Minimum cube-centre to target-centre distance. The plate is a 0.10 m square
# (DROP_ZONE_HALF_SIZE = 0.05) and the cube is 0.03 m wide, so anything under
# ~0.065 m puts the cube on the plate; 0.10 m clears it with margin.
MIN_TARGET_SEPARATION_M = 0.10
MAX_TARGET_ATTEMPTS = 1000
# Keep the cube/target this far inside their zone boundary. The samplers only
# check the true centre, but the scripted policy acts on the *localised* pose and
# overhead localisation drifts up to ~1.5 cm; a pose nearer than this to the edge
# can localise outside the zone (cube rejected by the planner) or fail to localise
# at all (target). Screening against a box of this half-width keeps every scene
# reliably pickable and placeable.
SOURCE_INTERIOR_MARGIN_M = 0.02
TARGET_INTERIOR_MARGIN_M = 0.02
MAX_SOURCE_ATTEMPTS = 1000


@dataclass(frozen=True)
class SceneDraw:
    """One sampled scene: where the cube starts and where it must end up."""

    source: CubePose
    target: CubePose
    plate_yaw_rad: float


def comfortably_interior(
    x: float, y: float, margin: float, in_zone: Callable[[float, float], bool]
) -> bool:
    """True when (x, y) and a box of +/- ``margin`` around it are all in-zone, so
    localisation drift cannot push the pose across the boundary."""
    return all(
        in_zone(x + dx, y + dy)
        for dx in (-margin, 0.0, margin)
        for dy in (-margin, 0.0, margin)
    )


def sample_scene(rng: np.random.Generator) -> SceneDraw:
    """Draw one well-posed scene from the declared distribution."""
    # Redraw the cube until it sits comfortably inside the pickup zone (not just
    # its centre, per sample_cube). Scenes whose first draw already clears the
    # margin keep it, so only edge cubes change.
    for _ in range(MAX_SOURCE_ATTEMPTS):
        source = sample_cube(rng)
        if comfortably_interior(
            source.x, source.y, SOURCE_INTERIOR_MARGIN_M, is_cube_pickup_allowed
        ):
            break
    else:
        raise RuntimeError(
            f"no cube >= {SOURCE_INTERIOR_MARGIN_M} m inside the pickup zone after "
            f"{MAX_SOURCE_ATTEMPTS} attempts"
        )
    # Redraw the target until it clears the cube and sits comfortably inside the
    # drop zone. Scenes whose first draw already satisfies both keep that draw
    # (same RNG state), so only the offending ones change.
    for _ in range(MAX_TARGET_ATTEMPTS):
        target = sample_target(rng)
        far_enough = (
            math.hypot(source.x - target.x, source.y - target.y) >= MIN_TARGET_SEPARATION_M
        )
        if far_enough and comfortably_interior(
            target.x, target.y, TARGET_INTERIOR_MARGIN_M, is_cube_drop_allowed
        ):
            break
    else:
        raise RuntimeError(
            f"no target >= {MIN_TARGET_SEPARATION_M} m from the cube and "
            f">= {TARGET_INTERIOR_MARGIN_M} m inside the drop zone after "
            f"{MAX_TARGET_ATTEMPTS} attempts"
        )
    # Drawn last (like the recorder) so it does not perturb the pose stream: the
    # source/target an index gets are unchanged by adding the plate yaw.
    plate_yaw = sample_target_plate_yaw(rng, target.x, target.y, half_size=DROP_ZONE_HALF_SIZE)
    return SceneDraw(source=source, target=target, plate_yaw_rad=float(plate_yaw))


def workspace_region(source: CubePose) -> str:
    """Name the third of the pickup zone the cube starts in."""
    radius = math.hypot(source.x - PAN_AXIS[0], source.y - PAN_AXIS[1])
    inner = CANONICAL_PICKUP_SECTOR.inner_radius
    outer = CANONICAL_PICKUP_SECTOR.outer_radius
    third = (outer - inner) / 3.0
    if radius < inner + third:
        return "near"
    if radius < inner + 2.0 * third:
        return "mid"
    return "far"
