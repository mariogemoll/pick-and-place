#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a reproducible DPPO imitation audit across discovered checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHECKPOINT_PATTERN = re.compile(r"state_(\d+)\.pt$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = Path(__file__).with_name("audit_dppo_imitation.py")
DEFAULT_CONFIG = REPOSITORY_ROOT / "config/diffusion_policy/pretrain_so101_unet_img.yaml"


def discover_checkpoints(
    directory: Path,
    *,
    minimum_epoch: int | None = None,
    maximum_epoch: int | None = None,
) -> list[tuple[int, Path]]:
    """Return valid ``state_N.pt`` checkpoints in ascending epoch order."""
    checkpoints = []
    for path in directory.glob("state_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file() or path.stat().st_size == 0:
            continue
        epoch = int(match.group(1))
        if minimum_epoch is not None and epoch < minimum_epoch:
            continue
        if maximum_epoch is not None and epoch > maximum_epoch:
            continue
        checkpoints.append((epoch, path.resolve()))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {directory}")
    epochs = [epoch for epoch, _ in checkpoints]
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"duplicate checkpoint epochs under {directory}")
    return checkpoints


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--training-export", type=Path, required=True)
    parser.add_argument("--held-out-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dppo-python", type=Path, default=os.environ.get("DPPO_PYTHON"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--panels-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument("--min-epoch", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    args = parser.parse_args()
    if args.dppo_python is None:
        parser.error("--dppo-python or DPPO_PYTHON is required")
    if args.ddim_steps < 1:
        parser.error("--ddim-steps must be positive")
    if args.samples_per_split < 1:
        parser.error("--samples-per-split must be positive")
    if not 0 <= args.panels_per_split <= args.samples_per_split:
        parser.error("--panels-per-split must be between zero and the sample count")
    return args


def trend_rows(checkpoint_outputs: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    """Flatten checkpoint summaries into stable CSV/JSON trend rows."""
    rows = []
    for epoch, output in checkpoint_outputs:
        with (output / "run.json").open() as file:
            run = json.load(file)
        with (output / "summary.json").open() as file:
            summary = json.load(file)
        for split in ("training", "held_out"):
            split_summary = summary[split]
            all_steps = split_summary["all_steps"]
            row = {
                "epoch": epoch,
                "split": split,
                "sampler": run["sampler"],
                "sampling_seed": run["sampling_seed"],
                "example_seed": run["example_seed"],
                "num_examples": split_summary["num_examples"],
                "arm_l2_mean": all_steps["arm_vector_l2"]["mean"],
                "arm_l2_median": all_steps["arm_vector_l2"]["median"],
                "arm_l2_p90": all_steps["arm_vector_l2"]["p90"],
                "arm_l2_p95": all_steps["arm_vector_l2"]["p95"],
            }
            for joint, statistics in all_steps["joint_absolute_error"].items():
                row[f"{joint}_mae"] = statistics["mean"]
            for step in (0, 1, 3, 7, 15):
                statistics = split_summary["selected_steps"][str(step)]["arm_vector_l2"]
                row[f"step_{step}_arm_l2_mean"] = statistics["mean"]
                row[f"step_{step}_arm_l2_p95"] = statistics["p95"]
            rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def _run_manifest(args: argparse.Namespace, checkpoints: list[tuple[int, Path]]) -> dict[str, Any]:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "checkpoint_discovery": {
            "directory": str(args.checkpoints_dir.resolve()),
            "minimum_epoch": args.min_epoch,
            "maximum_epoch": args.max_epoch,
            "snapshot": [
                {
                    "epoch": epoch,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified_ns": path.stat().st_mtime_ns,
                }
                for epoch, path in checkpoints
            ],
        },
        "audit_parameters": {
            "sampler": f"ddim-{args.ddim_steps}",
            "sampling_seed": args.seed,
            "example_seed": args.example_seed,
            "samples_per_split": args.samples_per_split,
            "panels_per_split": args.panels_per_split,
            "device": args.device,
            "training_data": str(args.training_data.resolve()),
            "training_export": str(args.training_export.resolve()),
            "normalization": str(args.normalization.resolve()),
            "held_out_root": str(args.held_out_root.resolve()),
            "config": str(args.config.resolve()),
            "dppo_python": str(args.dppo_python),
        },
    }


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    building = output.with_name(f"{output.name}.building")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if building.exists():
        raise FileExistsError(f"incomplete output already exists: {building}")

    checkpoints = discover_checkpoints(
        args.checkpoints_dir.resolve(),
        minimum_epoch=args.min_epoch,
        maximum_epoch=args.max_epoch,
    )
    building.mkdir(parents=True)
    checkpoint_root = building / "checkpoints"
    checkpoint_root.mkdir()
    manifest = _run_manifest(args, checkpoints)
    _write_json(building / "sweep.json", manifest)

    checkpoint_outputs = []
    for position, (epoch, checkpoint) in enumerate(checkpoints, start=1):
        checkpoint_output = checkpoint_root / f"epoch-{epoch:04d}"
        print(
            f"[{position}/{len(checkpoints)}] epoch {epoch}: "
            f"{manifest['audit_parameters']['sampler']}",
            flush=True,
        )
        command = [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--training-data",
            str(args.training_data.resolve()),
            "--normalization",
            str(args.normalization.resolve()),
            "--training-export",
            str(args.training_export.resolve()),
            "--held-out-root",
            str(args.held_out_root.resolve()),
            "--output",
            str(checkpoint_output),
            "--dppo-python",
            str(args.dppo_python.absolute()),
            "--config",
            str(args.config.resolve()),
            "--device",
            args.device,
            "--ddim-steps",
            str(args.ddim_steps),
            "--samples-per-split",
            str(args.samples_per_split),
            "--panels-per-split",
            str(args.panels_per_split),
            "--seed",
            str(args.seed),
            "--example-seed",
            str(args.example_seed),
        ]
        subprocess.run(command, cwd=REPOSITORY_ROOT / "py", check=True)
        checkpoint_outputs.append((epoch, checkpoint_output))

    rows = trend_rows(checkpoint_outputs)
    fieldnames = list(rows[0])
    with (building / "trend.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    held_out_ranking = sorted(
        (
            {
                "epoch": row["epoch"],
                "arm_l2_mean": row["arm_l2_mean"],
                "arm_l2_p95": row["arm_l2_p95"],
            }
            for row in rows
            if row["split"] == "held_out"
        ),
        key=lambda item: (item["arm_l2_mean"], item["epoch"]),
    )
    trend = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "num_checkpoints": len(checkpoints),
        "epochs": [epoch for epoch, _ in checkpoints],
        "rows": rows,
        "held_out_ranking_by_arm_l2_mean": held_out_ranking,
    }
    _write_json(building / "trend.json", trend)
    manifest["completed_at"] = trend["completed_at"]
    manifest["result_files"] = ["trend.json", "trend.csv"]
    _write_json(building / "sweep.json", manifest)
    building.rename(output)

    print("held-out ranking by all-horizon arm L2 mean:")
    for item in held_out_ranking:
        print(
            f"  epoch {item['epoch']}: mean={item['arm_l2_mean']:.3f}, p95={item['arm_l2_p95']:.3f}"
        )


if __name__ == "__main__":
    main()
