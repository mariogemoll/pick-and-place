# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure the visibility gates proposed by EPISODE_SPEC.

The full scene-visibility report renders two RGB images and two segmentation
images at every policy tick.  This pass only needs overhead segmentation, so it
samples each phase and evaluates the renderer-free centre-ray gate alongside
it.  An optional smaller detector sample compares both gates with the actual
overhead localization pipeline.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from pick_and_place.analysis.scene_visibility import (
    OBJECT_COVERAGE,
    SceneMeasurer,
    load_episode_truth,
    video_render_hw,
)
from pick_and_place.cli.common import add_output_argument
from pick_and_place.cli.dataset import add_episodes_root_argument, add_max_episodes_argument
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.plant.overhead import SimOverheadPerception
from pick_and_place.rollout.sim import OVERHEAD_CAMERA


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the episode-visibility measurement."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_episodes_root_argument(parser, help="staged episodes directory containing ep*/")
    add_output_argument(parser, required=True, help="visibility report JSON")
    add_max_episodes_argument(parser, help="read only the first N episodes")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=30,
        help="source-frame sampling stride; 30 is one sample per second",
    )
    parser.add_argument(
        "--detector-episodes",
        type=int,
        default=0,
        help="also run the detector at the start and first tick of each phase",
    )
    return parser


def validate(parser: SuggestingArgumentParser, args: argparse.Namespace) -> None:
    """Reject the strides and counts argparse's own types cannot check."""
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    if args.detector_episodes < 0:
        parser.error("--detector-episodes must be nonnegative")


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _agreement(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    pairs = [(bool(row[left]), bool(row[right])) for row in rows]
    return {
        "count": len(pairs),
        "agreement_fraction": (
            float(np.mean([a == b for a, b in pairs])) if pairs else None
        ),
        "both_visible": sum(a and b for a, b in pairs),
        "left_only": sum(a and not b for a, b in pairs),
        "right_only": sum(not a and b for a, b in pairs),
        "both_hidden": sum(not a and not b for a, b in pairs),
    }


def _sample_indices(phases: np.ndarray, stride: int) -> list[int]:
    indices = set(range(0, len(phases), stride))
    indices.add(0)
    indices.update(int(index) for index in np.flatnonzero(phases[1:] != phases[:-1]) + 1)
    return sorted(indices)


def _detector_indices(phases: np.ndarray) -> set[int]:
    indices = {0}
    for phase in dict.fromkeys(str(value) for value in phases):
        indices.add(int(np.flatnonzero(phases == phase)[0]))
    return indices


def _summarize(rows: list[dict[str, Any]], episodes: list[str]) -> dict[str, Any]:
    starts = [row for row in rows if row["source_frame"] == 0]
    phases: dict[str, Any] = {}
    for phase in dict.fromkeys(row["coarse_phase"] for row in rows):
        phase_rows = [row for row in rows if row["coarse_phase"] == phase]
        phases[phase] = {
            "samples": len(phase_rows),
            "episodes": len({row["episode"] for row in phase_rows}),
        }
        for object_name in ("cube", "plate"):
            pixels = [float(row[f"{object_name}_pixels"]) for row in phase_rows]
            phases[phase][object_name] = {
                "visible_fraction": float(np.mean([value > 0 for value in pixels])),
                "pixels": _distribution(pixels),
                "center_ray_visible_fraction": float(
                    np.mean([row[f"{object_name}_center_ray"] for row in phase_rows])
                ),
                "ray_vs_mask": _agreement(
                    phase_rows, f"{object_name}_center_ray", f"{object_name}_mask_visible"
                ),
            }

    detector_rows = [row for row in rows if "cube_detector" in row]
    detector: dict[str, Any] = {"looks": len(detector_rows), "objects": {}}
    for object_name in ("cube", "plate"):
        detector["objects"][object_name] = {
            "detector_vs_mask": _agreement(
                detector_rows, f"{object_name}_detector", f"{object_name}_mask_visible"
            ),
            "center_ray_vs_detector": _agreement(
                detector_rows, f"{object_name}_center_ray", f"{object_name}_detector"
            ),
        }
    detector["by_trajectory_phase"] = {}
    for phase in dict.fromkeys(row["trajectory_phase"] for row in detector_rows):
        phase_rows = [row for row in detector_rows if row["trajectory_phase"] == phase]
        detector["by_trajectory_phase"][phase] = {}
        for object_name in ("cube", "plate"):
            detector["by_trajectory_phase"][phase][object_name] = {
                "detector_vs_mask": _agreement(
                    phase_rows,
                    f"{object_name}_detector",
                    f"{object_name}_mask_visible",
                ),
                "center_ray_vs_detector": _agreement(
                    phase_rows,
                    f"{object_name}_center_ray",
                    f"{object_name}_detector",
                ),
            }

    trajectory_phases: dict[str, Any] = {}
    for phase in dict.fromkeys(row["trajectory_phase"] for row in rows):
        phase_rows = [row for row in rows if row["trajectory_phase"] == phase]
        trajectory_phases[phase] = {
            "samples": len(phase_rows),
            "episodes": len({row["episode"] for row in phase_rows}),
        }
        for object_name in ("cube", "plate"):
            pixels = [float(row[f"{object_name}_pixels"]) for row in phase_rows]
            trajectory_phases[phase][object_name] = {
                "visible_fraction": float(np.mean([value > 0 for value in pixels])),
                "pixels": _distribution(pixels),
                "center_ray_visible_fraction": float(
                    np.mean([row[f"{object_name}_center_ray"] for row in phase_rows])
                ),
                "ray_vs_mask": _agreement(
                    phase_rows,
                    f"{object_name}_center_ray",
                    f"{object_name}_mask_visible",
                ),
            }

    return {
        "episodes": len(episodes),
        "episode_names": episodes,
        "start": {
            "cube_mask_visible_fraction": float(
                np.mean([row["cube_mask_visible"] for row in starts])
            ),
            "cube_center_ray_visible_fraction": float(
                np.mean([row["cube_center_ray"] for row in starts])
            ),
            "cube_ray_vs_mask": _agreement(starts, "cube_center_ray", "cube_mask_visible"),
        },
        "phases": phases,
        "trajectory_phases": trajectory_phases,
        "detector": detector,
    }


def run(args: argparse.Namespace) -> None:
    """Measure visibility across the staged episodes and write the report."""
    roots = sorted(
        path for path in args.episodes_root.iterdir() if (path / "meta" / "info.json").is_file()
    )
    if args.max_episodes is not None:
        roots = roots[: args.max_episodes]
    if not roots:
        raise FileNotFoundError(f"no complete episodes under {args.episodes_root}")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    measurer = SceneMeasurer(video_render_hw(roots[0]), 96)
    perception = (
        SimOverheadPerception(measurer.model, measurer.data)
        if args.detector_episodes
        else None
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode_number, root in enumerate(roots):
            episode = load_episode_truth(root)
            measurer.set_target_plate(episode.target_xy, episode.target_plate_yaw)
            detector_indices = (
                _detector_indices(episode.trajectory_phases)
                if episode_number < args.detector_episodes
                else set()
            )
            for index in _sample_indices(episode.trajectory_phases, args.frame_stride):
                measurer.set_frame(episode.states[index], episode.cube_poses[index])
                coverage = measurer.coverage_maps(OVERHEAD_CAMERA)
                row: dict[str, Any] = {
                    "episode": episode.name,
                    "source_frame": index,
                    "coarse_phase": str(episode.coarse_phases[index]),
                    "trajectory_phase": str(episode.trajectory_phases[index]),
                }
                for object_name in ("cube", "plate"):
                    pixels = int((coverage[object_name] >= OBJECT_COVERAGE).sum())
                    row[f"{object_name}_pixels"] = pixels
                    row[f"{object_name}_mask_visible"] = pixels > 0
                    row[f"{object_name}_center_ray"] = measurer.center_ray_visible(
                        OVERHEAD_CAMERA, object_name
                    )
                if perception is not None and index in detector_indices:
                    perception.reset()
                    reading = perception.look()
                    row["cube_detector"] = reading.cube is not None
                    row["plate_detector"] = reading.target is not None
                rows.append(row)
            if (episode_number + 1) % 100 == 0:
                print(f"measured {episode_number + 1}/{len(roots)} episodes", flush=True)
    finally:
        if perception is not None:
            perception.close()
        measurer.close()

    summary = _summarize(rows, [root.name for root in roots])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps({key: summary[key] for key in ("episodes", "start", "detector")}, indent=2))
    for phase, stats in summary["trajectory_phases"].items():
        print(
            f"{phase}: plate visible {stats['plate']['visible_fraction']:.1%}, "
            f"center ray {stats['plate']['center_ray_visible_fraction']:.1%}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    run(args)


if __name__ == "__main__":
    main()
