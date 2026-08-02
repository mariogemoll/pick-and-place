#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turn a scene-difficulty sweep into scene phenotypes.

``scene_difficulty_sweep.py`` records which scenes the policy solves and how
reliably. This reconstructs each scene's geometry from its ``scenario_id`` --
scenes are a pure function of ``seed_base + index``, so nothing has to be stored
alongside the outcomes -- and asks which geometric features separate the scenes
it always solves from the ones it never does.

Two questions, in order:

1. *Is difficulty a property of the scene at all?* If per-scene success rates
   pile up at 0 and 1, the outcome was settled at reset and the policy's own
   sampling noise barely moves it. If they spread across the middle, the same
   scene goes either way and the action is what decides.
2. *Which scenes?* Reported as per-feature separation between the scenes the
   policy reliably solves and the ones it reliably fails, plus the milestone at
   which the failures stop.

Needs no GPU: it reads the sweep JSON and re-derives scenes on the CPU.

    python py/scripts/analyze_scene_difficulty.py \\
      --sweep outputs/scene-difficulty/sweep-stochastic.json \\
      --output outputs/scene-difficulty/phenotypes.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Milestones in the order the task passes through them, so "where did it stop"
# is the first one that is False.
MILESTONE_ORDER = (
    "pickup_contact_attempted",
    "cube_lifted",
    "stable_carry",
    "target_reached_while_holding",
    "cube_released",
    "cube_settled",
    "successful_placement",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.25,
        help="per-scene success rate at or below which a scene counts as hard",
    )
    parser.add_argument(
        "--easy-threshold",
        type=float,
        default=0.75,
        help="per-scene success rate at or above which a scene counts as easy",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    """Rotation about world z, in radians."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scene_features(scenario) -> dict[str, float]:
    """Geometry a failure could plausibly be a function of.

    Everything here is known at reset, which is the point: a feature that
    separates hard from easy scenes is difficulty the policy cannot influence.
    """
    from pick_and_place.workspace_overlays import PAN_AXIS

    cube_x, cube_y, _ = scenario.source_position_m
    target_x, target_y, _ = scenario.target_position_m
    cube_yaw = yaw_from_quat(*scenario.source_orientation_wxyz)

    # Polar coordinates about the pan axis are the arm's natural ones: radius is
    # reach, angle is the base joint.
    cube_radius = math.hypot(cube_x - PAN_AXIS[0], cube_y - PAN_AXIS[1])
    cube_bearing = math.atan2(cube_y - PAN_AXIS[1], cube_x - PAN_AXIS[0])
    target_radius = math.hypot(target_x - PAN_AXIS[0], target_y - PAN_AXIS[1])
    target_bearing = math.atan2(target_y - PAN_AXIS[1], target_x - PAN_AXIS[0])

    # A parallel gripper is symmetric under 90 degrees of cube yaw, and the grasp
    # that matters is yaw relative to the arm's approach, not to the world.
    yaw_vs_approach = (cube_yaw - cube_bearing + math.pi / 4) % (math.pi / 2) - math.pi / 4

    return {
        "cube_x": cube_x,
        "cube_y": cube_y,
        "cube_radius_m": cube_radius,
        "cube_bearing_deg": math.degrees(cube_bearing),
        "cube_yaw_deg": math.degrees(cube_yaw),
        "grasp_yaw_offset_deg": math.degrees(yaw_vs_approach),
        "abs_grasp_yaw_offset_deg": abs(math.degrees(yaw_vs_approach)),
        "target_x": target_x,
        "target_y": target_y,
        "target_radius_m": target_radius,
        "target_bearing_deg": math.degrees(target_bearing),
        "carry_distance_m": math.hypot(cube_x - target_x, cube_y - target_y),
        "carry_bearing_change_deg": abs(
            math.degrees((target_bearing - cube_bearing + math.pi) % (2 * math.pi) - math.pi)
        ),
        "plate_yaw_deg": math.degrees(scenario.target_plate_yaw_rad),
        "radius_change_m": target_radius - cube_radius,
    }


def first_failed_milestone(runs: list[dict]) -> str:
    """The earliest milestone a majority of this scene's failures never reached."""
    failures = [run for run in runs if not run["success"]]
    if not failures:
        return "none"
    for milestone in MILESTONE_ORDER:
        reached = sum(run.get(milestone, False) for run in failures)
        if reached < len(failures) / 2:
            return milestone
    return "settled_but_off_target"


def main() -> None:
    args = _parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pick_and_place.dppo_rl.scenes import training_scenario

    payload = json.loads(args.sweep.read_text())
    seed_base = int(payload["summary"]["config"]["scene_seed_base"])
    repeats = int(payload["summary"]["repeats"])

    runs_by_scene: dict[str, list[dict]] = defaultdict(list)
    for episode in payload["episodes"]:
        runs_by_scene[episode["scenario_id"]].append(episode)
    scenes = {
        scene: runs for scene, runs in runs_by_scene.items() if len(runs) == repeats
    }

    records = []
    for scene, runs in sorted(scenes.items()):
        index = int(scene.rsplit("-", 1)[1])
        scenario = training_scenario(index, seed_base=seed_base)
        rate = sum(run["success"] for run in runs) / len(runs)
        records.append({
            "scenario_id": scene,
            "index": index,
            "success_rate": rate,
            "workspace_region": scenario.workspace_region,
            "stopped_at": first_failed_milestone(runs),
            "median_final_xy_error_m": float(
                np.median([
                    run["final_xy_error_m"]
                    for run in runs
                    if not math.isnan(run["final_xy_error_m"])
                ])
            )
            if any(not math.isnan(run["final_xy_error_m"]) for run in runs)
            else None,
            **scene_features(scenario),
        })

    rates = np.array([record["success_rate"] for record in records])
    # Scene features only. The rate and the placement error are outcomes, not
    # things known at reset, so separating hard from easy by them would be
    # circular -- and the error is None on scenes that never placed the cube.
    outcomes = {"success_rate", "median_final_xy_error_m"}
    feature_names = [
        key
        for key, value in records[0].items()
        if isinstance(value, float) and key not in outcomes
    ]

    hard = [r for r in records if r["success_rate"] <= args.hard_threshold]
    easy = [r for r in records if r["success_rate"] >= args.easy_threshold]

    # How much of the variance in outcome is between scenes rather than within?
    # Under a "no scene effect" null every scene shares one Bernoulli p, so the
    # per-scene counts are Binomial(repeats, p) and their variance is fixed.
    # Excess variance over that is the scene effect.
    overall = float(rates.mean())
    null_var = overall * (1.0 - overall) / repeats
    observed_var = float(rates.var())
    scene_variance_share = (
        max(0.0, (observed_var - null_var) / observed_var) if observed_var > 0 else 0.0
    )

    separations = []
    for name in feature_names:
        hard_values = np.array([r[name] for r in hard if r[name] is not None])
        easy_values = np.array([r[name] for r in easy if r[name] is not None])
        if len(hard_values) < 3 or len(easy_values) < 3:
            continue
        pooled = math.sqrt((hard_values.var() + easy_values.var()) / 2.0)
        # Point-biserial correlation against the continuous rate is the honest
        # summary; Cohen's d on the hard/easy split says how visible it would be.
        column = np.array([r[name] for r in records])
        correlation = (
            float(np.corrcoef(column, rates)[0, 1]) if column.std() > 0 else 0.0
        )
        separations.append({
            "feature": name,
            "hard_mean": float(hard_values.mean()),
            "easy_mean": float(easy_values.mean()),
            "cohens_d": float((easy_values.mean() - hard_values.mean()) / pooled)
            if pooled > 0
            else 0.0,
            "correlation_with_success": correlation,
        })
    separations.sort(key=lambda item: -abs(item["cohens_d"]))

    stopped_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record["success_rate"] < 1.0:
            stopped_counts[record["stopped_at"]] += 1

    region_rates = {
        region: float(
            np.mean([r["success_rate"] for r in records if r["workspace_region"] == region])
        )
        for region in sorted({r["workspace_region"] for r in records})
    }

    summary = {
        "scenes": len(records),
        "repeats": repeats,
        "scene_seed_base": seed_base,
        "overall_success": overall,
        "histogram": {
            f"{k}/{repeats}": int(np.sum(np.isclose(rates, k / repeats)))
            for k in range(repeats + 1)
        },
        "never_solved": float(np.mean(rates == 0.0)),
        "always_solved": float(np.mean(rates == 1.0)),
        "mixed": float(np.mean((rates > 0.0) & (rates < 1.0))),
        "observed_variance": observed_var,
        "binomial_null_variance": null_var,
        "scene_variance_share": scene_variance_share,
        "hard_scenes": len(hard),
        "easy_scenes": len(easy),
        "failure_stage_counts": dict(stopped_counts),
        "success_by_workspace_region": region_rates,
        "feature_separation": separations,
    }
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "scenes": records}, indent=2) + "\n"
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
