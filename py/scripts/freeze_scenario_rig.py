#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Rewrite a scenario suite so every scene faces the *same* simulated rig.

A randomized suite redraws the robot, the cameras and the physics for every
scenario, so a score over it is a score against the whole envelope: the policy
must hedge across every rig it might meet, and no single rig is ever measured.
That is the right question for robustness and the wrong one for "how well does
this work on *a* robot", which is what a deployment faces and what most reported
numbers describe.

This freezes one draw -- the domain-randomization, miscalibration and physics
blocks together -- across every scenario, leaving the scene geometry alone. The
result is the same task suite seen by one fixed robot with one fixed camera
setup under one fixed lighting condition.

``--from-scenario`` picks which scenario's draw becomes the rig, so a rig is
named by where it came from and any of them can be reproduced. The rig may come
from a *different* suite via ``--rig-from``, which is how the authored rig is
applied: take it from a canonical suite, which records nominal as ``enabled:
false`` and empty offset maps rather than as zeros. Synthesizing "no
randomization" by flattening a randomized draw does not work -- zero light
intensity renders a black scene -- so the nominal rig is always copied from a
suite that already has one.

    # a rig the envelope actually produced
    python py/scripts/freeze_scenario_rig.py \\
        config/evaluation/randomized_selection_200_v1/manifest.json \\
        --from-scenario 7 \\
        --output /tmp/rig-007/manifest.json --scenarios-per-file 50

    # the authored rig, on the same 200 scenes
    python py/scripts/freeze_scenario_rig.py \\
        config/evaluation/randomized_selection_200_v1/manifest.json \\
        --rig-from config/evaluation/canonical_100_v1.json.xz --from-scenario 0 \\
        --label nominal \\
        --output /tmp/rig-nominal/manifest.json --scenarios-per-file 50
"""

from __future__ import annotations

import argparse
import copy
import json
import lzma
from pathlib import Path
from typing import Any

#: The three blocks that together are "the rig". Everything else in a scenario
#: -- the cube pose, the target, the workspace region, the step budget -- is the
#: task, and is left exactly as it was so suites stay paired scene by scene.
RIG_FIELDS = ("domain_randomization_sample", "miscalibration_sample", "physics_sample")


def _read_payload(path: Path) -> Any:
    if path.suffix == ".xz":
        return json.loads(lzma.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def _write_payload(path: Path, payload: object) -> None:
    if path.suffix == ".xz":
        serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        path.write_bytes(
            lzma.compress(serialized, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
        )
        return
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_suite(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a suite's header and its scenarios, sharded or not."""
    payload = _read_payload(manifest_path)
    if "scenarios" in payload:
        scenarios = payload.pop("scenarios")
        return payload, scenarios
    scenarios = []
    for name in payload.pop("scenario_files"):
        scenarios.extend(_read_payload(manifest_path.parent / name))
    return payload, scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="suite to rewrite")
    parser.add_argument(
        "--from-scenario",
        type=int,
        required=True,
        help="index of the scenario whose rig every scene should face",
    )
    parser.add_argument(
        "--rig-from",
        type=Path,
        default=None,
        help="take the rig from this suite instead of the one being rewritten",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="name for the rig in the suite name and header (default: its scenario index)",
    )
    parser.add_argument("--output", type=Path, required=True, help="output manifest path")
    parser.add_argument("--suite", default=None, help="suite name (default: derived from the source)")
    parser.add_argument("--scenarios-per-file", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header, scenarios = load_suite(args.manifest)
    if not scenarios:
        raise SystemExit(f"{args.manifest} holds no scenarios")

    rig_scenarios = scenarios
    if args.rig_from is not None:
        _, rig_scenarios = load_suite(args.rig_from)
    if not 0 <= args.from_scenario < len(rig_scenarios):
        raise SystemExit(
            f"--from-scenario {args.from_scenario} is outside the "
            f"{len(rig_scenarios)}-scenario rig source"
        )
    source = rig_scenarios[args.from_scenario]
    missing = [field for field in RIG_FIELDS if field not in source]
    if missing:
        raise SystemExit(f"the rig source scenario has no {missing}")
    rig = {field: copy.deepcopy(source[field]) for field in RIG_FIELDS}
    label = args.label or f"scenario{args.from_scenario:03d}"

    frozen = []
    for scenario in scenarios:
        scene = copy.deepcopy(scenario)
        for field in RIG_FIELDS:
            scene[field] = copy.deepcopy(rig[field])
        frozen.append(scene)

    suite = args.suite or f"{header.get('suite', args.manifest.parent.name)}-rig-{label}"
    header["suite"] = suite
    # The rig is the whole point of this suite, so record it where a reader
    # looks rather than making them diff two scenario files to find it.
    header["frozen_rig"] = {"source": label, **rig}
    header["scenarios"] = frozen

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.scenarios_per_file is None:
        _write_payload(args.output, header)
    else:
        if args.output.suffix == ".xz":
            raise SystemExit("a sharded manifest output must be an uncompressed JSON file")
        shard_names = []
        payload_scenarios = header.pop("scenarios")
        for start in range(0, len(payload_scenarios), args.scenarios_per_file):
            name = f"scenarios-{start // args.scenarios_per_file:03d}.json.xz"
            _write_payload(
                args.output.parent / name,
                payload_scenarios[start : start + args.scenarios_per_file],
            )
            shard_names.append(name)
        header["scenario_files"] = shard_names
        _write_payload(args.output, header)
    print(f"{suite}: {len(frozen)} scenarios on one rig from {label} -> {args.output}")


if __name__ == "__main__":
    main()
