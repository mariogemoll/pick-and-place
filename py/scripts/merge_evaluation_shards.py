#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Merge sharded ``eval_policy_sim.py`` runs into one evaluation result.

``eval_policy_sim.py`` takes ``--offset`` and ``--limit``, which split one
manifest across concurrent workers. Scenarios are independent and each carries
its own seed, so the shards are comparable and their union is exactly the
result a single serial run would have produced -- but each shard writes its own
``summary.json`` over its own slice, and a suite-level number cannot be
recovered by averaging those.

This rebuilds the episode records and re-runs the same aggregation the serial
path uses, rather than reimplementing it, so a merged summary and a serial one
are computed by identical code.

Sharding is what makes a checkpoint *ladder* affordable. Scoring is CPU and
render bound -- a few percent of the GPU -- so a hundred scenarios take about an
hour serially but a few minutes across a dozen workers. The previous SmolVLA
run could only afford eight scenarios per checkpoint, which is too few to rank
them; that is why "is 20,000 steps the ceiling?" went unanswered.

    python py/scripts/merge_evaluation_shards.py \
        --output eval/headline-050000 eval/shards/050000-*
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from pick_and_place.policies.policy_evaluation import (
    EpisodeResult,
    FailureFlags,
    ScenarioManifest,
    TaskMilestones,
    write_evaluation_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+", help="shard result directories")
    parser.add_argument("--output", type=Path, required=True, help="merged result directory")
    return parser.parse_args()


def resolve_manifest(recorded: str) -> Path:
    """Find the manifest a shard scored, from the path it recorded.

    ``run.json`` stores the manifest's absolute path *on the machine that
    scored it* -- ``/workspace/pick-and-place/config/evaluation/...`` for a
    rented pod -- so merging anywhere else cannot open it. The manifests are
    frozen and version controlled, and the shards' agreed sha256 has already
    been checked, so falling back to this checkout's copy of the same file name
    resolves to identical content rather than merely a similar one.
    """
    path = Path(recorded)
    if path.exists():
        return path
    local = Path(__file__).resolve().parents[2] / "config" / "evaluation" / path.name
    if local.exists():
        return local
    raise SystemExit(f"cannot find manifest {path.name}: not at {recorded}, not at {local}")


def episode_from_dict(payload: dict[str, Any]) -> EpisodeResult:
    """Rebuild an EpisodeResult from the dict `to_dict` wrote."""
    known = {field.name for field in fields(EpisodeResult)}
    unknown = set(payload) - known
    if unknown:
        raise SystemExit(
            f"episode record carries unknown fields {sorted(unknown)}; "
            "the schema moved and this merge would silently drop them"
        )
    values = dict(payload)
    values["milestones"] = TaskMilestones(**payload["milestones"])
    values["failures"] = FailureFlags(**payload["failures"])
    return EpisodeResult(**values)


def main() -> None:
    args = parse_args()

    runs: list[dict[str, Any]] = []
    results: list[EpisodeResult] = []
    seen: dict[str, Path] = {}
    for shard in args.shards:
        episodes = shard / "episodes.jsonl"
        if not episodes.exists():
            raise SystemExit(f"{shard} has no episodes.jsonl; the shard did not finish")
        runs.append(json.loads((shard / "run.json").read_text()))
        for line in episodes.read_text().splitlines():
            if not line.strip():
                continue
            result = episode_from_dict(json.loads(line))
            # Overlapping --offset/--limit windows would double-count scenarios
            # and inflate the denominator, which is invisible in the summary.
            if result.scenario_id in seen:
                raise SystemExit(
                    f"scenario {result.scenario_id} appears in both {seen[result.scenario_id]} "
                    f"and {shard}; the shard windows overlap"
                )
            seen[result.scenario_id] = shard
            results.append(result)

    # A merged number is only meaningful if every shard scored the same
    # checkpoint against the same manifest. Comparing the recorded fingerprint
    # and manifest hash catches a stale shard directory left by an earlier run,
    # which would otherwise merge silently.
    def agreed(path: list[str]) -> Any:
        values = []
        for run in runs:
            node: Any = run
            for key in path:
                node = (node or {}).get(key) if isinstance(node, dict) else None
            values.append(json.dumps(node, sort_keys=True))
        if len(set(values)) != 1:
            raise SystemExit(f"shards disagree on {'.'.join(path)}: {sorted(set(values))}")
        return json.loads(values[0])

    agreed(["checkpoint", "fingerprint"])
    agreed(["scenario_manifest", "sha256"])
    suite = agreed(["scenario_manifest", "suite"])

    merged = dict(runs[0])
    results.sort(key=lambda result: result.scenario_id)
    selected = [result.scenario_id for result in results]
    manifest = dict(merged["scenario_manifest"])
    manifest["selected_scenario_ids"] = selected
    # Recomputed rather than inherited: every shard is a slice and so records
    # complete_suite false, while their union is usually the whole suite. The
    # count is not in run.json, so the manifest itself is the only source --
    # and its sha256 was just checked against what the shards scored.
    manifest["complete_suite"] = len(selected) == len(
        ScenarioManifest.load(resolve_manifest(manifest["path"])).scenarios
    )
    merged["scenario_manifest"] = manifest
    merged["sharded_from"] = [str(shard) for shard in args.shards]
    merged["started_at_utc"] = min(run["started_at_utc"] for run in runs)
    merged["finished_at_utc"] = max(run["finished_at_utc"] for run in runs)

    summary = write_evaluation_artifacts(args.output, merged, results)
    print(
        f"Merged {len(args.shards)} shards of {suite} into {args.output}: "
        f"{summary['success_count']}/{summary['episode_count']} successes "
        f"({summary['success_rate']:.1%})."
    )


if __name__ == "__main__":
    main()
