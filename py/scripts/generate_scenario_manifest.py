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

By default scenarios are canonical. Optional switches freeze the same per-episode
draws used by randomized scripted recording: a domain preset supplies appearance,
material, camera, cube-orientation and full miscalibration draws; the physics dial
supplies an independently seeded arm draw; and near-neutral starts reuse the
recorder's joint sampler. Each stream is independent of scene geometry, so a
randomized suite can stay paired with the canonical scene stream.

Floating-point values are rounded to six decimals, which makes the frozen
manifest reproducible across platforms while retaining precision well beyond
the simulator's meaningful resolution.

The suite is policy-agnostic: the same scenarios are used to evaluate every
controller (scripted, ACT, diffusion, ...), so the name describes the scenes,
not the policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import lzma
import math
from pathlib import Path

import numpy as np

from pick_and_place.sim.domain_randomization import (
    DomainSample,
    DomainRandomizationPreset,
    domain_sample_payload,
    domain_seed,
    orient_cube,
)
from pick_and_place.core.geometry import cube_quat_from_pose
from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.core.miscalibration import (
    DEFAULT_PAN_JITTER_SIGMA_DEG,
    DEFAULT_PAN_JITTER_TAU_S,
)
from pick_and_place.core.physics import PhysicsModel
from pick_and_place.policies.policy_evaluation import SCENARIO_MANIFEST_VERSION
from pick_and_place.rollout.episode_setup import physics_rng
from pick_and_place.scripted.episode_sampling import sample_near_neutral
from pick_and_place.scripted.scenario_sampling import sample_scene, workspace_region

INITIAL_ROBOT_STATE_REAL = [0.0, 0.0, 0.0, 0.0, -90.0, 39.3]
CONTROL_HZ = 30.0
MAX_STEPS = 450
FLOAT_DECIMALS = 6
INITIAL_STATE_SEED_SALT = 0x53544152


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


def _domain_layer(
    preset: DomainRandomizationPreset | None,
    sample: DomainSample | None,
    sample_seed: int | None,
) -> dict:
    """The scenario's domain-randomization and miscalibration fields.

    Canonical (``preset is None``) yields explicit nominal randomization and
    miscalibration values. Otherwise a full ``DomainSample`` is drawn from
    ``preset`` on an independent ``domain_seed`` stream -- so it never perturbs
    the position/yaw draws -- and serialized as the env's
    :func:`_domain_sample_from_scenario`
    expects: every ``DomainSample`` field except ``miscalibration``, plus
    ``enabled``. The joint-offset miscalibration drawn alongside it becomes the
    separate, fully materialized ``miscalibration_sample``."""
    if preset is None or sample is None or sample_seed is None:
        return {
            "domain_randomization_preset": None,
            "domain_randomization_sample": {"enabled": False},
            "miscalibration_sample": {
                "joint_offsets_deg": {},
                "pan_jitter": None,
                "cube_belief_error": [0.0, 0.0, 0.0, 0.0],
                "target_belief_error": [0.0, 0.0],
            },
        }
    domain_sample = _round_sample(domain_sample_payload(sample))
    domain_sample["enabled"] = True
    return {
        "domain_randomization_preset": preset.name,
        "domain_randomization_sample": domain_sample,
        "miscalibration_sample": {
            "joint_offsets_deg": _round_sample(sample.miscalibration.base_offsets_deg),
            "pan_jitter": {
                "sigma_deg": DEFAULT_PAN_JITTER_SIGMA_DEG,
                "tau_s": DEFAULT_PAN_JITTER_TAU_S,
                "seed": int(
                    np.random.default_rng(
                        np.random.SeedSequence([sample_seed, 0x50414E])
                    ).integers(2**63)
                ),
            },
            "cube_belief_error": _round_sample(sample.miscalibration.cube_belief_error),
            "target_belief_error": _round_sample(sample.miscalibration.target_belief_error),
        },
    }


def _initial_robot_state(seed_base: int, index: int, randomized: bool) -> list[float]:
    if not randomized:
        return list(INITIAL_ROBOT_STATE_REAL)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed_base, index, INITIAL_STATE_SEED_SALT])
    )
    joints, gripper = sample_near_neutral(rng)
    return _round(sim_frame_to_real(joints, gripper))


def _write_payload(path: Path, payload: object) -> None:
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


def _write_manifest(path: Path, payload: dict, scenarios_per_file: int | None) -> None:
    if scenarios_per_file is None:
        _write_payload(path, payload)
        return
    if path.suffix == ".xz":
        raise ValueError("a sharded manifest output must be an uncompressed JSON file")
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = payload.pop("scenarios")
    shard_names = []
    for start in range(0, len(scenarios), scenarios_per_file):
        shard_name = f"scenarios-{start // scenarios_per_file:03d}.json.xz"
        _write_payload(path.parent / shard_name, scenarios[start : start + scenarios_per_file])
        shard_names.append(shard_name)
    payload["scenario_files"] = shard_names
    _write_payload(path, payload)


def _scenario(
    suite: str,
    index: int,
    seed: int,
    seed_base: int,
    preset: DomainRandomizationPreset | None,
    physics_model: PhysicsModel,
    randomize_initial_robot_state: bool,
    control_hz: float,
    max_steps: int,
) -> dict:
    rng = np.random.default_rng(seed)
    try:
        scene = sample_scene(rng)
    except RuntimeError as error:
        raise RuntimeError(f"scenario {index}: {error}") from error
    sample_seed = domain_seed(seed_base, index) if preset is not None else None
    sample = preset.sample(sample_seed) if preset is not None else None
    source = (
        orient_cube(scene.source, sample.cube_orientation_index)
        if sample is not None
        else scene.source
    )
    target = scene.target
    return {
        "scenario_id": f"{suite}-{index:03d}",
        "group": "canonical" if preset is None else "domain_randomized",
        "workspace_region": workspace_region(source),
        "seed": seed,
        "source_position_m": _round([source.x, source.y, source.z]),
        "source_orientation_wxyz": _round(cube_quat_from_pose(source)),
        "target_position_m": _round([target.x, target.y, target.z]),
        "initial_robot_state_real": _initial_robot_state(
            seed_base, index, randomize_initial_robot_state
        ),
        **_domain_layer(preset, sample, sample_seed),
        "physics_sample": _round_sample(
            asdict(physics_model.sample(physics_rng(seed_base, index)))
        ),
        "control_hz": control_hz,
        "max_steps": max_steps,
        "target_plate_yaw_rad": round(scene.plate_yaw_rad, FLOAT_DECIMALS),
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
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--physics-randomization",
        type=float,
        default=0.0,
        metavar="AMOUNT",
        help="physics-randomization amount, using the scripted recorder's draw stream",
    )
    parser.add_argument(
        "--randomize-initial-robot-state",
        action="store_true",
        help="freeze a varied start drawn from the recorder's near-neutral envelope",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output manifest; use an .xz suffix for compressed JSON",
    )
    parser.add_argument(
        "--scenarios-per-file",
        type=int,
        default=None,
        help="write a sharded manifest with at most this many scenarios per compressed file",
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
    if args.count < 1:
        parser.error("--count must be at least 1")
    if not math.isfinite(args.control_hz) or args.control_hz <= 0.0:
        parser.error("--control-hz must be positive and finite")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if not math.isfinite(args.physics_randomization) or args.physics_randomization < 0.0:
        parser.error("--physics-randomization must be finite and nonnegative")
    if args.scenarios_per_file is not None and args.scenarios_per_file < 1:
        parser.error("--scenarios-per-file must be at least 1")

    preset = (
        DomainRandomizationPreset.load(args.domain_randomization_preset)
        if args.domain_randomization_preset is not None
        else None
    )
    suite = args.suite or ("dr_100_v1" if preset is not None else "canonical_100_v1")
    physics_model = PhysicsModel(amount=args.physics_randomization)

    scenarios = [
        _scenario(
            suite,
            index,
            args.seed_base + index,
            args.seed_base,
            preset,
            physics_model,
            args.randomize_initial_robot_state,
            args.control_hz,
            args.max_steps,
        )
        for index in range(args.count)
    ]
    payload = {
        "schema_version": SCENARIO_MANIFEST_VERSION,
        "suite": suite,
        "scenarios": scenarios,
    }
    _write_manifest(args.output, payload, args.scenarios_per_file)
    print(f"Wrote {args.output}: {len(scenarios)} scenarios")


if __name__ == "__main__":
    main()
