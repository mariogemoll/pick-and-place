#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure how much a DPPO checkpoint's predictions depend on each input.

Evaluates the same fixed examples as the imitation audit under controlled
input corruptions and reports the resulting physical-unit action error per
condition:

- ``intact``: the normal two-step history (audit baseline).
- ``shuffled_images``: images swapped in from a different example while the
  joint states stay correct. Error staying low means the policy acts largely
  from joint state.
- ``blank_images``: both cameras replaced by mid-gray (state-only input).
- ``overhead_only`` / ``wrist_only``: the other camera blanked.

With ``--ground-truth-root`` (staged episodes that carry per-frame true cube
pose and phase spans) it additionally evaluates object-masking conditions in
which exactly the cube's or the plate's pixels are inpainted with local
background, using segmentation re-renders of the recorded scene:

- ``cube_masked`` / ``plate_masked``: one object visually removed.

If masking the cube changes predictions no more than shuffling noise, the
policy was not using the cube's pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq

from pick_and_place.dppo_imitation_audit import (
    AuditExample,
    HeldOutSplit,
    TrainingArtifactSplit,
    sample_refs,
    summarize_errors,
)
from pick_and_place.dppo_policy import DppoPolicyController
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    WRIST_FEATURE,
    PolicyObservation,
)
from pick_and_place.scene_visibility import SceneMeasurer, inpaint_object
from pick_and_place.sim_recorder import OVERHEAD_CAMERA, WRIST_CAMERA

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config/diffusion_policy/pretrain_so101_unet_img.yaml"

BLANK_LEVEL = 128
CAMERA_BY_FEATURE = {OVERHEAD_FEATURE: OVERHEAD_CAMERA, WRIST_FEATURE: WRIST_CAMERA}

Condition = Callable[[int, AuditExample, list[AuditExample]], list[PolicyObservation]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--held-out-root", type=Path, required=True)
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=None,
        help="staged episodes with per-frame cube pose, for object-masking conditions",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dppo-python", type=Path, default=os.environ.get("DPPO_PYTHON"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--example-seed", type=int, default=42)
    args = parser.parse_args()
    if args.dppo_python is None:
        parser.error("--dppo-python or DPPO_PYTHON is required")
    return args


def _copy_observations(example: AuditExample) -> list[PolicyObservation]:
    return [
        {name: np.array(value, copy=True) for name, value in observation.items()}
        for observation in example.observations
    ]


def _blank(observations: list[PolicyObservation], features: tuple[str, ...]) -> None:
    for observation in observations:
        for feature in features:
            observation[feature] = np.full_like(observation[feature], BLANK_LEVEL)


def condition_intact(
    index: int, example: AuditExample, examples: list[AuditExample]
) -> list[PolicyObservation]:
    return _copy_observations(example)


def condition_shuffled_images(
    index: int, example: AuditExample, examples: list[AuditExample]
) -> list[PolicyObservation]:
    donor = examples[(index + len(examples) // 2) % len(examples)]
    observations = _copy_observations(example)
    for observation, donor_observation in zip(observations, donor.observations, strict=True):
        for feature in (OVERHEAD_FEATURE, WRIST_FEATURE):
            observation[feature] = np.array(donor_observation[feature], copy=True)
    return observations


def condition_blank_images(
    index: int, example: AuditExample, examples: list[AuditExample]
) -> list[PolicyObservation]:
    observations = _copy_observations(example)
    _blank(observations, (OVERHEAD_FEATURE, WRIST_FEATURE))
    return observations


def condition_overhead_only(
    index: int, example: AuditExample, examples: list[AuditExample]
) -> list[PolicyObservation]:
    observations = _copy_observations(example)
    _blank(observations, (WRIST_FEATURE,))
    return observations


def condition_wrist_only(
    index: int, example: AuditExample, examples: list[AuditExample]
) -> list[PolicyObservation]:
    observations = _copy_observations(example)
    _blank(observations, (OVERHEAD_FEATURE,))
    return observations


BASE_CONDITIONS: dict[str, Condition] = {
    "intact": condition_intact,
    "shuffled_images": condition_shuffled_images,
    "blank_images": condition_blank_images,
    "overhead_only": condition_overhead_only,
    "wrist_only": condition_wrist_only,
}


@dataclass(frozen=True)
class GroundTruthEpisode:
    states: np.ndarray  # (N, 6) at 30 Hz
    cube_poses: np.ndarray  # (N, 7) at 30 Hz
    target_xy: tuple[float, float]
    target_plate_yaw: float


def _load_ground_truth(root: Path, names: tuple[str, ...]) -> dict[str, GroundTruthEpisode]:
    episodes = {}
    for name in names:
        meta_paths = sorted((root / name / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        row = pq.read_table(meta_paths[0]).to_pylist()[0]
        data_paths = sorted((root / name / "data").glob("chunk-*/file-*.parquet"))
        table = pq.read_table(
            data_paths[0],
            columns=["frame_index", "observation.state", "observation.environment_state"],
        ).sort_by("frame_index")
        episodes[name] = GroundTruthEpisode(
            states=np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
            cube_poses=np.asarray(
                table["observation.environment_state"].to_pylist(), dtype=np.float64
            ),
            target_xy=(float(row["target_x"]), float(row["target_y"])),
            target_plate_yaw=float(row["target_plate_yaw"]),
        )
    return episodes


def _video_render_hw(root: Path) -> tuple[int, int]:
    with (root / "meta" / "info.json").open() as file:
        info = json.load(file)
    shapes = {
        tuple(feature["shape"][:2])
        for feature in info["features"].values()
        if feature.get("dtype") == "video"
    }
    if len(shapes) != 1:
        raise ValueError(f"{root.name} cameras must share one resolution, got {shapes}")
    return next(iter(shapes))


def _masking_condition(
    target: str,
    episodes: dict[str, GroundTruthEpisode],
    measurer: SceneMeasurer,
) -> Condition:
    def condition(
        index: int, example: AuditExample, examples: list[AuditExample]
    ) -> list[PolicyObservation]:
        if example.source_frame_indices is None:
            raise ValueError("masking conditions need source frame indices")
        episode = episodes[example.ref.episode]
        measurer.set_target_plate(episode.target_xy, episode.target_plate_yaw)
        observations = _copy_observations(example)
        for observation, source_index in zip(
            observations, example.source_frame_indices, strict=True
        ):
            measurer.set_frame(
                episode.states[source_index], episode.cube_poses[source_index]
            )
            for feature, camera in CAMERA_BY_FEATURE.items():
                coverage = measurer.coverage_maps(camera)[target]
                observation[feature], _ = inpaint_object(observation[feature], coverage)
        return observations

    return condition


def _evaluate(
    controller: DppoPolicyController,
    examples: list[AuditExample],
    conditions: dict[str, Condition],
    *,
    sampling_seed: int,
) -> dict[str, Any]:
    intact_predictions: list[np.ndarray] | None = None
    results: dict[str, Any] = {}
    for name, condition in conditions.items():
        predictions = [
            controller.predict_horizon_from_history(
                condition(index, example, examples), sampling_seed=sampling_seed
            )
            for index, example in enumerate(examples)
        ]
        targets = np.stack([example.target_actions for example in examples])
        stacked = np.stack(predictions)
        summary = summarize_errors(stacked, targets)
        entry: dict[str, Any] = {
            "arm_l2_all_16": summary["all_steps"]["arm_vector_l2"],
            "arm_l2_first_8": {
                key: float(
                    np.mean(
                        [
                            summary["per_step"][str(step)]["arm_vector_l2"][key]
                            for step in range(8)
                        ]
                    )
                )
                for key in ("mean", "median", "p95")
            },
        }
        if name == "intact":
            intact_predictions = predictions
        else:
            assert intact_predictions is not None, "intact must be the first condition"
            deltas = np.stack(
                [
                    np.linalg.norm(np.asarray(a)[:, :5] - np.asarray(b)[:, :5], axis=-1)
                    for a, b in zip(predictions, intact_predictions, strict=True)
                ]
            )
            entry["arm_l2_delta_from_intact"] = {
                "mean": float(deltas.mean()),
                "median": float(np.median(deltas)),
                "p95": float(np.quantile(deltas, 0.95)),
            }
        results[name] = entry
    return results


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    building = output.with_name(f"{output.name}.building")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if building.exists():
        raise FileExistsError(f"incomplete output already exists: {building}")

    splits: dict[str, tuple[Any, dict[str, Condition]]] = {
        "training": (
            TrainingArtifactSplit(args.training_data, args.normalization),
            dict(BASE_CONDITIONS),
        ),
        "held_out": (HeldOutSplit(args.held_out_root), dict(BASE_CONDITIONS)),
    }

    measurer = None
    if args.ground_truth_root is not None:
        names = tuple(
            sorted(
                path.name
                for path in args.ground_truth_root.iterdir()
                if (path / "meta" / "info.json").is_file()
            )
        )
        ground_truth_split = HeldOutSplit(args.ground_truth_root, episode_names=names)
        episodes = _load_ground_truth(args.ground_truth_root, names)
        measurer = SceneMeasurer(
            _video_render_hw(args.ground_truth_root / names[0]), 96
        )
        splits["ground_truth"] = (
            ground_truth_split,
            {
                **BASE_CONDITIONS,
                "cube_masked": _masking_condition("cube", episodes, measurer),
                "plate_masked": _masking_condition("plate", episodes, measurer),
            },
        )

    building.mkdir(parents=True)
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
        report: dict[str, Any] = {
            "checkpoint": str(args.checkpoint.resolve()),
            "epoch": int(controller.handshake["epoch"]),
            "sampler": str(controller.handshake["sampler"]),
            "sampling_seed": args.seed,
            "example_seed": args.example_seed,
            "samples_per_split": args.samples_per_split,
            "blank_level": BLANK_LEVEL,
            "splits": {},
        }
        for split_index, (split_name, (split, conditions)) in enumerate(splits.items()):
            refs = sample_refs(
                split.refs(), args.samples_per_split, seed=args.example_seed + split_index
            )
            examples = split.load_examples(refs)
            report["splits"][split_name] = _evaluate(
                controller, examples, conditions, sampling_seed=args.seed
            )
            print(f"{split_name}:")
            for condition_name, entry in report["splits"][split_name].items():
                line = (
                    f"  {condition_name:16s} first8 mean "
                    f"{entry['arm_l2_first_8']['mean']:7.3f}"
                    f"  all16 mean {entry['arm_l2_all_16']['mean']:7.3f}"
                )
                if "arm_l2_delta_from_intact" in entry:
                    line += (
                        "  delta-from-intact mean "
                        f"{entry['arm_l2_delta_from_intact']['mean']:7.3f}"
                    )
                print(line)
        with (building / "summary.json").open("w") as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")
    except Exception:
        shutil.rmtree(building)
        raise
    finally:
        if controller is not None:
            controller.close()
        if measurer is not None:
            measurer.close()

    building.rename(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
