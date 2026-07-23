# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate a canonical scenario manifest.

Each scenario draws its source cube from ``sample_cube`` and its target from
``sample_target`` using a per-scenario seeded RNG (seed = ``seed_base + index``).
Two constraints keep every scenario well-posed for the scripted controller (the
gold standard), on top of what ``sample_cube``/``sample_target`` already enforce:

- The cube is redrawn until it is at least ``SOURCE_INTERIOR_MARGIN_M`` inside
  the pickup-zone boundary. ``sample_cube`` only checks the true centre, but the
  policy plans from the *localised* cube and overhead localisation drifts ~1.5 cm,
  so an edge cube can localise outside the zone and be rejected by the planner.
- The target is redrawn until it is (a) at least ``MIN_TARGET_SEPARATION_M`` from
  the cube -- the two are sampled independently and their sectors overlap, so
  otherwise the cube can start on the drop-zone plate (a 0.10 m square) and occlude
  target localization -- and (b) at least ``TARGET_INTERIOR_MARGIN_M`` inside the
  drop-zone boundary, so the plate is not so near the edge that the policy cannot
  localise it (the target-side analog of the cube margin).

The drop-plate yaw is sampled per scenario (in [0, pi/2), the square plate's
distinct orientations) with the recorder's ``sample_target_plate_yaw``, so eval
plates are oriented like the training data rather than always axis-aligned.

By default scenarios are canonical: no domain randomization, no joint
miscalibration. Passing ``--domain-randomization-preset`` instead produces a
domain-randomized suite (e.g. ``dr_100_v1``) that reuses the canonical
positions and plate yaw for a given ``--seed-base`` and layers a frozen
``DomainSample`` (visual/lighting/camera perturbation) plus the joint-offset
miscalibration drawn with it. The randomization is drawn from an independent
seed stream (``domain_seed``), so it does not perturb the position/yaw draws:
run this with the same ``--seed-base`` as the canonical suite and every scenario
is the canonical scene with only the DR/miscalibration layer added, making
DR-vs-canonical a clean paired comparison.

Floating-point values are rounded to six decimals, which makes the frozen
manifest reproducible across platforms while retaining precision well beyond
the simulator's meaningful resolution.

The suite is policy-agnostic: the same scenarios are used to evaluate every
controller (scripted, ACT, diffusion, ...), so the name describes the scenes,
not the policy.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
from pathlib import Path

import numpy as np

from pick_and_place.domain_randomization import (
    DomainRandomizationPreset,
    domain_seed,
)
from pick_and_place.episodes import (
    CANONICAL_PICKUP_OVERLAY,
    PAN_AXIS,
    cube_quat_from_pose,
    sample_cube,
    sample_target,
)
from pick_and_place.paper_detection import DROP_ZONE_HALF_SIZE
from pick_and_place.policy_evaluation import SCENARIO_MANIFEST_VERSION
from pick_and_place.workspace_overlays import (
    is_cube_drop_allowed,
    is_cube_placement_allowed,
    sample_target_plate_yaw,
)

INITIAL_ROBOT_STATE_REAL = [0.0, 0.0, 0.0, 0.0, -90.0, 39.3]
CONTROL_HZ = 30.0
MAX_STEPS = 450
FLOAT_DECIMALS = 6
# Minimum cube-centre to target-centre distance. The plate is a 0.10 m square
# (DROP_ZONE_HALF_SIZE = 0.05) and the cube is 0.03 m wide, so anything under
# ~0.065 m puts the cube on the plate; 0.10 m clears it with margin.
MIN_TARGET_SEPARATION_M = 0.10
MAX_TARGET_ATTEMPTS = 1000
# Keep the cube/target this far inside their zone boundary. The samplers only
# check the true centre, but the scripted policy acts on the *localised* pose and
# overhead localisation drifts up to ~1.5 cm; a pose nearer than this to the edge
# can localise outside the zone (cube rejected by the planner) or fail to localise
# at all (target). Screening against a box of this half-width keeps every scenario
# reliably pickable and placeable.
SOURCE_INTERIOR_MARGIN_M = 0.02
TARGET_INTERIOR_MARGIN_M = 0.02
MAX_SOURCE_ATTEMPTS = 1000


def _round(values) -> list[float]:
    return [round(float(value), FLOAT_DECIMALS) for value in values]


def _round_sample(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), FLOAT_DECIMALS)
    if isinstance(value, dict):
        return {key: _round_sample(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_sample(item) for item in value]
    return value


def _comfortably_interior(x: float, y: float, margin: float, in_zone) -> bool:
    """True when (x, y) and a box of +/- ``margin`` around it are all in-zone, so
    localisation drift cannot push the pose across the boundary."""
    return all(
        in_zone(x + dx, y + dy)
        for dx in (-margin, 0.0, margin)
        for dy in (-margin, 0.0, margin)
    )


def _domain_layer(
    preset: DomainRandomizationPreset | None, seed_base: int, index: int
) -> dict:
    """The scenario's domain-randomization and miscalibration fields.

    Canonical (``preset is None``) yields a disabled sample and empty joint
    offsets. Otherwise a full ``DomainSample`` is drawn from ``preset`` on an
    independent ``domain_seed`` stream -- so it never perturbs the position/yaw
    draws -- and serialized as the env's :func:`_domain_sample_from_scenario`
    expects: every ``DomainSample`` field except ``miscalibration``, plus
    ``enabled``. The joint-offset miscalibration drawn alongside it becomes the
    separate ``miscalibration_sample`` (the env consumes only the joint offsets;
    pan jitter and belief errors are not applied during eval)."""
    if preset is None:
        return {
            "domain_randomization_preset": None,
            "domain_randomization_sample": {"enabled": False},
            "miscalibration_sample": {"joint_offsets_deg": {}},
        }
    sample = preset.sample(domain_seed(seed_base, index))
    domain_sample = _round_sample(
        {name: value for name, value in sample.__dict__.items() if name != "miscalibration"}
    )
    domain_sample["enabled"] = True
    return {
        "domain_randomization_preset": preset.name,
        "domain_randomization_sample": domain_sample,
        "miscalibration_sample": {
            "joint_offsets_deg": _round_sample(sample.miscalibration.base_offsets_deg),
        },
    }


def _write_payload(path: Path, payload: dict) -> None:
    if path.suffix == ".xz":
        serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        path.write_bytes(
            lzma.compress(
                serialized,
                format=lzma.FORMAT_XZ,
                preset=9 | lzma.PRESET_EXTREME,
            )
        )
        return
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _region(radius_from_pan: float) -> str:
    inner = CANONICAL_PICKUP_OVERLAY.inner_radius
    outer = CANONICAL_PICKUP_OVERLAY.outer_radius
    third = (outer - inner) / 3.0
    if radius_from_pan < inner + third:
        return "near"
    if radius_from_pan < inner + 2.0 * third:
        return "mid"
    return "far"


def _scenario(
    suite: str,
    index: int,
    seed: int,
    seed_base: int,
    preset: DomainRandomizationPreset | None,
) -> dict:
    rng = np.random.default_rng(seed)
    # Redraw the cube until it sits comfortably inside the pickup zone (not just
    # its centre, per sample_cube). Scenarios whose first draw already clears the
    # margin keep it, so only edge cubes change.
    for _ in range(MAX_SOURCE_ATTEMPTS):
        source = sample_cube(rng)
        if _comfortably_interior(
            source.x, source.y, SOURCE_INTERIOR_MARGIN_M, is_cube_placement_allowed
        ):
            break
    else:
        raise RuntimeError(
            f"scenario {index}: no cube >= {SOURCE_INTERIOR_MARGIN_M} m inside the pickup "
            f"zone after {MAX_SOURCE_ATTEMPTS} attempts"
        )
    # Redraw the target until it clears the cube and sits comfortably inside the
    # drop zone. Scenarios whose first draw already satisfies both keep that draw
    # (same RNG state), so only the offending ones change.
    for _ in range(MAX_TARGET_ATTEMPTS):
        target = sample_target(rng)
        far_enough = (
            math.hypot(source.x - target.x, source.y - target.y) >= MIN_TARGET_SEPARATION_M
        )
        if far_enough and _comfortably_interior(
            target.x, target.y, TARGET_INTERIOR_MARGIN_M, is_cube_drop_allowed
        ):
            break
    else:
        raise RuntimeError(
            f"scenario {index}: no target >= {MIN_TARGET_SEPARATION_M} m from the cube and "
            f">= {TARGET_INTERIOR_MARGIN_M} m inside the drop zone after "
            f"{MAX_TARGET_ATTEMPTS} attempts"
        )
    # Drawn last (like the recorder) so it does not perturb the pose stream: the
    # source/target an index gets are unchanged by adding the plate yaw.
    plate_yaw = sample_target_plate_yaw(rng, target.x, target.y, half_size=DROP_ZONE_HALF_SIZE)
    radius = math.hypot(source.x - PAN_AXIS[0], source.y - PAN_AXIS[1])
    return {
        "scenario_id": f"{suite}-{index:03d}",
        "group": "canonical" if preset is None else "domain_randomized",
        "workspace_region": _region(radius),
        "seed": seed,
        "source_position_m": _round([source.x, source.y, source.z]),
        "source_orientation_wxyz": _round(cube_quat_from_pose(source)),
        "target_position_m": _round([target.x, target.y, target.z]),
        "initial_robot_state_real": list(INITIAL_ROBOT_STATE_REAL),
        **_domain_layer(preset, seed_base, index),
        "control_hz": CONTROL_HZ,
        "max_steps": MAX_STEPS,
        "target_plate_yaw_rad": round(float(plate_yaw), FLOAT_DECIMALS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="suite name; defaults to canonical_100_v1, or dr_100_v1 with a preset",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=1701)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output manifest; use an .xz suffix for compressed JSON",
    )
    parser.add_argument(
        "--domain-randomization-preset",
        type=Path,
        default=None,
        help=(
            "path to a domain-randomization preset (e.g. "
            "config/domain_randomization/act_mild_v1.json); when set, layers a "
            "frozen DR + miscalibration draw onto the canonical scenes. Use the "
            "same --seed-base as the canonical suite for a paired comparison."
        ),
    )
    args = parser.parse_args()

    preset = (
        DomainRandomizationPreset.load(args.domain_randomization_preset)
        if args.domain_randomization_preset is not None
        else None
    )
    suite = args.suite or ("dr_100_v1" if preset is not None else "canonical_100_v1")

    scenarios = [
        _scenario(suite, index, args.seed_base + index, args.seed_base, preset)
        for index in range(args.count)
    ]
    payload = {
        "schema_version": SCENARIO_MANIFEST_VERSION,
        "suite": suite,
        "scenarios": scenarios,
    }
    _write_payload(args.output, payload)
    print(f"Wrote {args.output}: {len(scenarios)} scenarios")


if __name__ == "__main__":
    main()
