#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Re-summarize an existing DPPO imitation audit by task phase.

Reads a completed audit output directory (``examples.jsonl`` holds one record
per evaluated chunk with its full per-step errors) and labels every example
with the coarse task phase of its chunk start. Older episodes carry no recorded
phase ground truth, so phases are reconstructed from each episode's gripper
command trace. No policy inference is run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from pick_and_place.dppo_imitation_audit import FRAME_STRIDE, StoredNpz
from pick_and_place.task_phases import PHASES, segment_phases

# +/- 0.5 s of grasp window on either timeline.
GRASP_HALFWIDTH_10HZ = 5
GRASP_HALFWIDTH_30HZ = 15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True, help="canonical train.npz")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--held-out-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _training_phase_labels(
    training_data: Path, normalization: Path
) -> dict[str, np.ndarray]:
    archive = StoredNpz(training_data)
    actions = archive.array("actions")
    traj_lengths = archive.array("traj_lengths")
    with np.load(normalization, allow_pickle=False) as bounds:
        gripper_min = float(bounds["action_min"][-1])
        gripper_max = float(bounds["action_max"][-1])
    offsets = np.concatenate(([0], np.cumsum(traj_lengths, dtype=np.int64)))
    labels = {}
    for episode, length in enumerate(traj_lengths):
        normalized = np.asarray(actions[offsets[episode] : offsets[episode] + int(length), -1])
        trace = (normalized + 1.0) / 2.0 * (gripper_max - gripper_min) + gripper_min
        labels[str(episode)] = segment_phases(
            trace, grasp_halfwidth=GRASP_HALFWIDTH_10HZ
        ).labels()
    return labels


def _held_out_phase_labels(root: Path, episodes: set[str]) -> dict[str, np.ndarray]:
    labels = {}
    for name in sorted(episodes):
        paths = sorted((root / name / "data").glob("chunk-*/file-*.parquet"))
        if len(paths) != 1:
            raise ValueError(f"{name} must contain one data parquet")
        table = pq.read_table(paths[0], columns=["frame_index", "action"]).sort_by(
            "frame_index"
        )
        trace = np.asarray(table["action"].to_pylist(), dtype=np.float64)[:, -1]
        labels[name] = segment_phases(trace, grasp_halfwidth=GRASP_HALFWIDTH_30HZ).labels()
    return labels


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    arm_l2 = np.asarray([record["arm_vector_l2"] for record in records], dtype=np.float64)
    return {
        "num_examples": len(records),
        "arm_l2_all_16": _distribution(arm_l2.mean(axis=1)),
        "arm_l2_first_8": _distribution(arm_l2[:, :8].mean(axis=1)),
        "arm_l2_step_0": _distribution(arm_l2[:, 0]),
        "arm_l2_step_7": _distribution(arm_l2[:, 7]),
    }


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    records = [
        json.loads(line)
        for line in (args.audit_dir / "examples.jsonl").read_text().splitlines()
        if line
    ]
    if not records:
        raise ValueError("audit directory contains no examples")

    held_out_episodes = {
        record["episode"] for record in records if record["split"] == "held_out"
    }
    label_sources = {
        "training": _training_phase_labels(args.training_data, args.normalization),
        "held_out": _held_out_phase_labels(args.held_out_root, held_out_episodes),
    }

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        labels = label_sources[record["split"]][record["episode"]]
        index = record["index_10hz"]
        if record["split"] == "held_out":
            index *= FRAME_STRIDE
        grouped[record["split"]][str(labels[index])].append(record)

    summary: dict[str, Any] = {
        "audit_dir": str(args.audit_dir.resolve()),
        "phase_source": "gripper-trace reconstruction (segment_phases)",
        "splits": {},
    }
    for split, by_phase in grouped.items():
        summary["splits"][split] = {
            phase: _summarize(by_phase[phase]) for phase in PHASES if phase in by_phase
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")

    for split, phases in summary["splits"].items():
        print(f"{split}:")
        for phase in PHASES:
            if phase not in phases:
                continue
            stats = phases[phase]
            print(
                f"  {phase:12s} n={stats['num_examples']:3d}"
                f"  all16 {stats['arm_l2_all_16']['mean']:6.3f}"
                f"  first8 {stats['arm_l2_first_8']['mean']:6.3f}"
                f"  step0 {stats['arm_l2_step_0']['mean']:6.3f}"
                f"  first8 p95 {stats['arm_l2_first_8']['p95']:6.3f}"
            )


if __name__ == "__main__":
    main()
