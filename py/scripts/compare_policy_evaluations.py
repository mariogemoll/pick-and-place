#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Compare evaluation runs that share a scenario manifest.

Two runs of the same manifest are *paired*: every scenario appears in both, so
the informative quantity is not the difference of two independent rates but the
scenarios where the two arms disagree. This reports both, and refuses to
compare runs whose scenario sets or rollout settings differ, because a
difference in those is a difference in the experiment rather than in the
policy.

The rates reported are deliberately more than the oracle's ``success``: that
flag requires the cube to be placed within 4 cm *and* released *and* settled,
which is the right headline but a coarse instrument for ranking checkpoints.
``placed_6cm`` matches the threshold used in the closed-loop handoff tables and
``cube_lifted`` isolates the grasp from the transport.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PLACED_TOLERANCE_M = 0.06

#: Rollout settings that must agree before two runs can be compared. A
#: difference in any of them means the arms did not face the same task.
COMPARABLE_FIELDS = (
    ("scenario_manifest", "sha256"),
    ("scenario_manifest", "selected_scenario_ids"),
    ("environment", "scene_appearance"),
    ("environment", "control_hz"),
    ("environment", "episode_step_limits"),
    ("environment", "image_height"),
    ("environment", "image_width"),
    ("environment", "oracle"),
    ("controller", "executed_action_steps"),
    ("controller", "sampler"),
    ("controller", "denoising_steps"),
    ("controller", "sampling_seed"),
    ("controller", "policy_hz"),
    ("controller", "weights"),
    # Different normalization statistics would mean the two policies were
    # driving the arm through different action scales.
    ("controller", "normalization", "sha256"),
    # Flow-image runs do not expose the Diffusion Policy sampler or its
    # normalization object.  Bind their equivalent rollout and export
    # contracts directly, otherwise two Euler settings could be reported as a
    # paired comparison merely because the absent Diffusion Policy fields both
    # resolve to None.
    ("controller", "integration"),
    ("controller", "integration_steps"),
    ("controller", "noise_correlation"),
    ("controller", "export", "manifest_sha256"),
    ("controller", "export", "normalization_sha256"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="evaluation run directories, each containing run.json and episodes.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="run to pair every other run against (default: the first run given)",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="report rates even when the runs did not face the same scenarios or settings",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the comparison here")
    return parser.parse_args()


def _get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _read_episodes(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]


class EvaluationRun:
    """One arm of a comparison, from a single run directory or a set of shards.

    A shard is a worker that ran a disjoint slice of the same suite under the
    same settings (``eval_policy_sim.py --offset/--limit``). Reassembling them
    here rather than at write time keeps every shard's own ``run.json``
    inspectable, and lets the settings guard run *between* the shards too: a
    shard launched with the wrong flag is caught rather than averaged in.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.name = directory.name
        shards = sorted(
            path.parent for path in directory.glob("shard-*/run.json")
        )
        if not (directory / "run.json").exists() and shards:
            self.shards = shards
            self.run = json.loads((shards[0] / "run.json").read_text())
            self.episodes = []
            for shard in shards:
                shard_settings = _settings_of(json.loads((shard / "run.json").read_text()))
                differing = {
                    key
                    for key, value in self.settings().items()
                    if value != shard_settings[key]
                    and key != "scenario_manifest/selected_scenario_ids"
                }
                if differing:
                    raise SystemExit(
                        f"{directory.name}: shard {shard.name} does not match "
                        f"{shards[0].name} in {sorted(differing)}"
                    )
                self.episodes.extend(_read_episodes(shard))
            seen = [episode["scenario_id"] for episode in self.episodes]
            if len(seen) != len(set(seen)):
                duplicated = sorted({name for name in seen if seen.count(name) > 1})
                raise SystemExit(
                    f"{directory.name}: shards overlap on {duplicated}; "
                    "the same scenario would be counted twice"
                )
        else:
            self.shards = [directory]
            self.run = json.loads((directory / "run.json").read_text())
            self.episodes = _read_episodes(directory)
        self.episodes.sort(key=lambda episode: episode["scenario_id"])
        self.by_scenario = {episode["scenario_id"]: episode for episode in self.episodes}

    @property
    def scenario_ids(self) -> list[str]:
        return [episode["scenario_id"] for episode in self.episodes]

    def outcomes(self, metric: str) -> dict[str, bool]:
        return {
            scenario_id: _metric(episode, metric)
            for scenario_id, episode in self.by_scenario.items()
        }

    def settings(self) -> dict[str, Any]:
        return _settings_of(self.run)


def _settings_of(run: dict[str, Any]) -> dict[str, Any]:
    return {"/".join(path): _get(run, path) for path in COMPARABLE_FIELDS}


def _metric(episode: dict[str, Any], metric: str) -> bool:
    if metric == "success":
        return bool(episode["success"])
    if metric == "placed_6cm":
        return float(episode["final_xy_error_m"]) <= PLACED_TOLERANCE_M
    return bool(episode["milestones"][metric])


METRICS = ("success", "placed_6cm", "cube_lifted", "pickup_contact_attempted")


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0, 1] at rates near 0 and 1."""
    if total == 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _binomial_tail(k: int, n: int) -> float:
    """Two-sided exact binomial p at p=0.5, used for McNemar's paired test."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def mcnemar(
    a: dict[str, bool], b: dict[str, bool], scenario_ids: list[str]
) -> dict[str, Any]:
    """Paired comparison over the scenarios both arms ran.

    Concordant scenarios carry no information about which arm is better, so the
    test looks only at the two discordant counts.
    """
    only_a = [s for s in scenario_ids if a[s] and not b[s]]
    only_b = [s for s in scenario_ids if b[s] and not a[s]]
    discordant = len(only_a) + len(only_b)
    return {
        "both": sum(1 for s in scenario_ids if a[s] and b[s]),
        "neither": sum(1 for s in scenario_ids if not a[s] and not b[s]),
        "only_first": len(only_a),
        "only_second": len(only_b),
        "discordant": discordant,
        "p_value_exact": _binomial_tail(len(only_a), discordant),
        "only_first_scenarios": only_a,
        "only_second_scenarios": only_b,
    }


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _rate_row(run: EvaluationRun, metric: str) -> dict[str, Any]:
    outcomes = run.outcomes(metric)
    count = sum(outcomes.values())
    total = len(outcomes)
    low, high = wilson_interval(count, total)
    return {"count": count, "total": total, "rate": count / total if total else 0.0,
            "ci95": [low, high]}


def main() -> None:
    args = _parse_args()
    runs = [EvaluationRun(directory) for directory in args.runs]
    baseline = (
        EvaluationRun(args.baseline)
        if args.baseline is not None
        else runs[0]
    )

    comparison: dict[str, Any] = {"arms": {}, "paired_against": baseline.name}

    print(f"{'arm':<32}  {'n':>3}  " + "  ".join(f"{m:>22}" for m in METRICS))
    for run in runs:
        cells = []
        for metric in METRICS:
            row = _rate_row(run, metric)
            cells.append(
                f"{row['count']:>3}/{row['total']:<3} "
                f"{row['rate']:>5.1%} "
                f"[{row['ci95'][0]:.0%}-{row['ci95'][1]:.0%}]"
            )
        print(f"{run.name:<32}  {len(run.episodes):>3}  " + "  ".join(f"{c:>22}" for c in cells))
        comparison["arms"][run.name] = {
            "episode_count": len(run.episodes),
            "rates": {metric: _rate_row(run, metric) for metric in METRICS},
            "final_xy_error_cm_median": 100.0
            * _median([e["final_xy_error_m"] for e in run.episodes]),
            "min_tcp_to_cube_cm_median": 100.0
            * _median([e["min_tcp_to_cube_distance_m"] for e in run.episodes]),
            "settings": run.settings(),
        }

    print()
    print(f"{'arm':<32}  {'median final XY':>16}  {'median closest TCP':>19}")
    for run in runs:
        arm = comparison["arms"][run.name]
        print(
            f"{run.name:<32}  {arm['final_xy_error_cm_median']:>13.1f} cm  "
            f"{arm['min_tcp_to_cube_cm_median']:>16.1f} cm"
        )

    for run in runs:
        if run.name == baseline.name:
            continue
        shared = sorted(set(baseline.by_scenario) & set(run.by_scenario))
        differences = {
            key: (value, run.settings()[key])
            for key, value in baseline.settings().items()
            if value != run.settings()[key] and key != "scenario_manifest/selected_scenario_ids"
        }
        if differences and not args.allow_mismatch:
            raise SystemExit(
                f"{run.name} and {baseline.name} did not face the same experiment: "
                + ", ".join(
                    f"{key} {first!r} vs {second!r}" for key, (first, second) in differences.items()
                )
                + " (pass --allow-mismatch to compare anyway)"
            )
        print()
        header = f"paired: {run.name} vs {baseline.name} over {len(shared)} shared scenarios"
        print(header)
        if differences:
            print("  WARNING, the arms differ in: " + ", ".join(sorted(differences)))
        paired: dict[str, Any] = {}
        print(
            f"  {'metric':<24} {'only ' + run.name:>34} {'only ' + baseline.name:>34}"
            f" {'both':>5} {'neither':>8} {'p':>7}"
        )
        for metric in METRICS:
            result = mcnemar(run.outcomes(metric), baseline.outcomes(metric), shared)
            paired[metric] = result
            print(
                f"  {metric:<24} {result['only_first']:>34} {result['only_second']:>34}"
                f" {result['both']:>5} {result['neither']:>8}"
                f" {result['p_value_exact']:>7.3f}"
            )
        comparison["arms"][run.name]["paired_vs_baseline"] = paired

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
