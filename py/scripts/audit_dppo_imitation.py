#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure DPPO action-chunk imitation on training and held-out examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.dppo_imitation_audit import (
    FRAME_STRIDE,
    HISTORY_STEPS,
    HORIZON_STEPS,
    POLICY_HZ,
    SOURCE_HZ,
    HeldOutSplit,
    TrainingArtifactSplit,
    example_record,
    sample_refs,
    save_panel,
    summarize_errors,
)
from pick_and_place.dppo_policy import DppoPolicyController

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config/diffusion_policy/pretrain_so101_unet_img.yaml"
AUDIT_MODULE = REPOSITORY_ROOT / "py/src/pick_and_place/dppo_imitation_audit.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True, help="canonical train.npz")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--training-export", type=Path, required=True, help="canonical export.json")
    parser.add_argument(
        "--held-out-root",
        type=Path,
        required=True,
        help="directory containing ep001017 through ep001117",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dppo-python", type=Path, default=os.environ.get("DPPO_PYTHON"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--panels-per-split", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="fixed policy sampling seed")
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument(
        "--held-out-max-episodes",
        type=int,
        default=None,
        help="testing only: use the first N held-out episodes instead of all 100",
    )
    args = parser.parse_args()
    if args.dppo_python is None:
        parser.error("--dppo-python or DPPO_PYTHON is required")
    if args.panels_per_split < 0:
        parser.error("--panels-per-split must be nonnegative")
    if args.panels_per_split > args.samples_per_split:
        parser.error("--panels-per-split cannot exceed --samples-per-split")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=REPOSITORY_ROOT, check=False).returncode != 0
    )
    staged_dirty = (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY_ROOT, check=False
        ).returncode
        != 0
    )
    return {
        "git_commit": commit,
        "git_tracked_or_staged_changes": tracked_dirty or staged_dirty,
        "audit_script_sha256": _sha256(Path(__file__)),
        "audit_module_sha256": _sha256(AUDIT_MODULE),
    }


def _validate_training_export(path: Path) -> dict[str, Any]:
    with path.open() as file:
        manifest = json.load(file)
    required = {
        "num_episodes": 1000,
        "num_frames": 96765,
        "fps": POLICY_HZ,
        "source_fps": SOURCE_HZ,
        "frame_stride": FRAME_STRIDE,
        "frame_selection": "episode-relative indices 0, frame_stride, 2 * frame_stride, ...",
        "camera_features": [
            "observation.images.overhead",
            "observation.images.wrist",
        ],
        "image_transform": "aspect-fill resize followed by center crop",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"training export violates the audit contract: {mismatches}")
    return manifest


def _evaluate_split(
    name: str,
    split: Any,
    controller: DppoPolicyController,
    *,
    count: int,
    example_seed: int,
    sampling_seed: int,
    panel_count: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    refs = sample_refs(split.refs(), count, seed=example_seed)
    examples = split.load_examples(refs)
    predictions = []
    records = []
    sampler = str(controller.handshake["sampler"])
    for index, example in enumerate(examples):
        prediction = controller.predict_horizon_from_history(
            list(example.observations), sampling_seed=sampling_seed
        )
        predictions.append(prediction)
        panel_relative = None
        if index < panel_count:
            panel_relative = f"examples/{name}-{index:03d}.png"
            save_panel(output_dir / panel_relative, name, example, prediction)
        records.append(
            example_record(
                name,
                example,
                prediction,
                sampler=sampler,
                sampling_seed=sampling_seed,
                panel_path=panel_relative,
            )
        )
    targets = np.stack([example.target_actions for example in examples])
    return summarize_errors(np.stack(predictions), targets), records


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    building = output.with_name(f"{output.name}.building")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if building.exists():
        raise FileExistsError(f"incomplete output already exists: {building}")

    manifest = _validate_training_export(args.training_export)
    training = TrainingArtifactSplit(args.training_data, args.normalization)
    held_out = HeldOutSplit(args.held_out_root, max_episodes=args.held_out_max_episodes)
    building.mkdir(parents=True)
    (building / "examples").mkdir()

    controller = None
    try:
        controller = DppoPolicyController.launch(
            python=args.dppo_python,
            checkpoint=args.checkpoint,
            config=args.config,
            normalization=args.normalization,
            device=args.device,
            seed=args.seed,
            ddim_steps=args.ddim_steps,
        )
        expected_handshake = {
            "horizon_steps": HORIZON_STEPS,
            "cond_steps": HISTORY_STEPS,
            "img_cond_steps": HISTORY_STEPS,
            "policy_hz": float(POLICY_HZ),
            "image_height": 96,
            "image_width": 96,
        }
        mismatches = {
            key: {"expected": expected, "actual": controller.handshake.get(key)}
            for key, expected in expected_handshake.items()
            if controller.handshake.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"checkpoint/config violates the audit contract: {mismatches}")

        summaries = {}
        records = []
        for split_index, (name, split) in enumerate(
            (("training", training), ("held_out", held_out))
        ):
            summary, split_records = _evaluate_split(
                name,
                split,
                controller,
                count=args.samples_per_split,
                example_seed=args.example_seed + split_index,
                sampling_seed=args.seed,
                panel_count=args.panels_per_split,
                output_dir=building,
            )
            summaries[name] = summary
            records.extend(split_records)

        checkpoint = args.checkpoint.resolve()
        run = {
            "checkpoint": {
                "path": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
                "epoch": int(controller.handshake["epoch"]),
            },
            "config": {
                "path": str(args.config.resolve()),
                "sha256": _sha256(args.config),
            },
            "normalization": {
                "path": str(args.normalization.resolve()),
                "sha256": _sha256(args.normalization),
            },
            "splits": {
                "training": {
                    "source": str(args.training_data.resolve()),
                    "source_dataset_sha256": manifest["source_sha256"],
                    "episodes": "0-999",
                    "in_sample_diagnostic": True,
                },
                "held_out": {
                    "source": str(args.held_out_root.resolve()),
                    "episodes": list(held_out.episodes),
                    "excluded_failed_episode": "ep001110",
                    "generalization_result": args.held_out_max_episodes is None,
                },
            },
            "sampler": str(controller.handshake["sampler"]),
            "sampling_seed": args.seed,
            "example_seed": args.example_seed,
            "samples_per_split": args.samples_per_split,
            "timing": {
                "source_hz": SOURCE_HZ,
                "policy_hz": POLICY_HZ,
                "frame_stride": FRAME_STRIDE,
                "frame_selection_30hz": "episode-relative 0, 3, 6, ... for state, action, and both images",
                "history_at_index_t": ["observation[t - 1]", "observation[t]"],
                "target_at_index_t": "actions[t:t + 16]",
                "action_convention": "position command at t; approximately reached by state[t + 1] on the 10 Hz timeline",
            },
            "server": controller.handshake,
            "code_provenance": _git_provenance(),
        }
        with (building / "run.json").open("w") as file:
            json.dump(run, file, indent=2, sort_keys=True)
            file.write("\n")
        with (building / "summary.json").open("w") as file:
            json.dump(summaries, file, indent=2, sort_keys=True)
            file.write("\n")
        with (building / "examples.jsonl").open("w") as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True))
                file.write("\n")
    except Exception:
        shutil.rmtree(building)
        raise
    finally:
        if controller is not None:
            controller.close()

    building.rename(output)
    for name in ("training", "held_out"):
        all_steps = summaries[name]["all_steps"]
        print(f"{name}: {summaries[name]['num_examples']} examples")
        for joint, stats in all_steps["joint_absolute_error"].items():
            print(
                f"  {joint}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
                f"p90={stats['p90']:.3f}, p95={stats['p95']:.3f}"
            )
        stats = all_steps["arm_vector_l2"]
        print(
            f"  arm_vector_l2: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
            f"p90={stats['p90']:.3f}, p95={stats['p95']:.3f}"
        )


if __name__ == "__main__":
    main()
