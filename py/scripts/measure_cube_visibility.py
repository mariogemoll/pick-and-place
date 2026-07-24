#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure cube and target-plate visibility in the policy's actual 96x96 view.

For staged simulation episodes that carry per-frame ground truth (the true cube
pose in ``observation.environment_state`` and exact ``phase_spans`` metadata),
this reconstructs every 10 Hz policy tick in MuJoCo — robot at the recorded
joints, cube at its recorded pose, target plate at its recorded pose and yaw —
renders segmentation masks through the same camera pipeline as recording, and
applies the exact aspect-fill/center-crop transform used for training images.

It reports, per camera and task phase, how many effective 96x96 pixels the
source cube and the target plate actually occupy and how much luminance
contrast they have against their local background. This quantifies whether the
cube is even represented in the pixels the policy sees.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.scene_visibility import (
    OBJECT_COVERAGE,
    SceneMeasurer,
    contrast,
    load_episode_truth,
    video_render_hw,
)
from pick_and_place.sim_recorder import OVERHEAD_CAMERA, WRIST_CAMERA
from pick_and_place.task_phases import PHASES

FRAME_STRIDE = 3  # 30 Hz source frames per 10 Hz policy tick
CAMERAS = {"overhead": OVERHEAD_CAMERA, "wrist": WRIST_CAMERA}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-root",
        type=Path,
        required=True,
        help="staged episodes directory containing ep*/ with per-frame ground truth",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--panels",
        type=int,
        default=12,
        help="overlay panels to save, spread across phases (default: 12)",
    )
    args = parser.parse_args()
    if args.image_size < 8:
        parser.error("--image-size must be at least 8")
    if args.panels < 0:
        parser.error("--panels must be nonnegative")
    return args


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _save_panel(
    path: Path, rgb: np.ndarray, coverage: dict[str, np.ndarray], title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4, 4), constrained_layout=True)
    axis.imshow(rgb)
    for name, color in (("cube", "red"), ("plate", "deepskyblue")):
        axis.contour(coverage[name], levels=[OBJECT_COVERAGE], colors=color, linewidths=1.0)
    axis.set_title(title, fontsize=8)
    axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    output = args.output.resolve()
    building = output.with_name(f"{output.name}.building")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if building.exists():
        raise FileExistsError(f"incomplete output already exists: {building}")

    episode_roots = sorted(
        path for path in args.episodes_root.iterdir() if (path / "meta" / "info.json").is_file()
    )
    if args.max_episodes is not None:
        episode_roots = episode_roots[: args.max_episodes]
    if not episode_roots:
        raise FileNotFoundError(f"no complete episodes under {args.episodes_root}")

    render_hw = video_render_hw(episode_roots[0])
    measurer = SceneMeasurer(render_hw, args.image_size)
    building.mkdir(parents=True)
    (building / "panels").mkdir()

    tick_rows: list[dict[str, Any]] = []
    panel_budget: dict[str, int] = dict.fromkeys(PHASES, 0)
    panels_per_phase = args.panels // len(PHASES) if args.panels else 0
    try:
        for episode_root in episode_roots:
            episode = load_episode_truth(episode_root)
            if video_render_hw(episode_root) != render_hw:
                raise ValueError(f"{episode.name} has a different camera resolution")
            measurer.set_target_plate(episode.target_xy, episode.target_plate_yaw)
            for source_index in range(0, len(episode.states), FRAME_STRIDE):
                phase = str(episode.coarse_phases[source_index])
                measurer.set_frame(
                    episode.states[source_index], episode.cube_poses[source_index]
                )
                for camera_label, camera in CAMERAS.items():
                    rgb, coverage = measurer.render(camera)
                    cube_contrast = contrast(rgb, coverage["cube"], coverage["plate"])
                    plate_contrast = contrast(rgb, coverage["plate"], coverage["cube"])
                    tick_rows.append(
                        {
                            "episode": episode.name,
                            "source_frame": source_index,
                            "tick_10hz": source_index // FRAME_STRIDE,
                            "phase": phase,
                            "camera": camera_label,
                            "cube_area_px": float(coverage["cube"].sum()),
                            "cube_pixels": int((coverage["cube"] >= OBJECT_COVERAGE).sum()),
                            "cube_contrast": cube_contrast,
                            "plate_area_px": float(coverage["plate"].sum()),
                            "plate_pixels": int((coverage["plate"] >= OBJECT_COVERAGE).sum()),
                            "plate_contrast": plate_contrast,
                        }
                    )
                    if (
                        camera_label == "overhead"
                        and panel_budget[phase] < panels_per_phase
                        and source_index % (5 * FRAME_STRIDE) == 0
                    ):
                        panel_budget[phase] += 1
                        for label in CAMERAS:
                            image, cov = (rgb, coverage) if label == "overhead" else (
                                measurer.render(CAMERAS[label])
                            )
                            _save_panel(
                                building
                                / "panels"
                                / f"{episode.name}-f{source_index:04d}-{phase}-{label}.png",
                                image,
                                cov,
                                f"{episode.name} frame {source_index} {phase} {label}",
                            )

        fields = list(tick_rows[0].keys())
        with (building / "ticks.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(tick_rows)

        summary: dict[str, Any] = {
            "episodes": [root.name for root in episode_roots],
            "render_hw": list(render_hw),
            "image_size": args.image_size,
            "frame_stride": FRAME_STRIDE,
            "object_coverage_threshold": OBJECT_COVERAGE,
            "per_camera_phase": {},
        }
        for camera_label in CAMERAS:
            summary["per_camera_phase"][camera_label] = {}
            for phase in PHASES:
                rows = [
                    row
                    for row in tick_rows
                    if row["camera"] == camera_label and row["phase"] == phase
                ]
                if not rows:
                    continue
                summary["per_camera_phase"][camera_label][phase] = {
                    "ticks": len(rows),
                    "cube_area_px": _distribution([row["cube_area_px"] for row in rows]),
                    "cube_pixels": _distribution([float(row["cube_pixels"]) for row in rows]),
                    "cube_contrast": _distribution(
                        [row["cube_contrast"] for row in rows if row["cube_contrast"] is not None]
                    ),
                    "cube_invisible_fraction": float(
                        np.mean([row["cube_pixels"] == 0 for row in rows])
                    ),
                    "plate_area_px": _distribution([row["plate_area_px"] for row in rows]),
                    "plate_pixels": _distribution(
                        [float(row["plate_pixels"]) for row in rows]
                    ),
                    "plate_contrast": _distribution(
                        [
                            row["plate_contrast"]
                            for row in rows
                            if row["plate_contrast"] is not None
                        ]
                    ),
                }
        with (building / "summary.json").open("w") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
            file.write("\n")
    except Exception:
        shutil.rmtree(building)
        raise
    finally:
        measurer.close()

    building.rename(output)
    for camera_label, phases in summary["per_camera_phase"].items():
        print(f"{camera_label}:")
        for phase, stats in phases.items():
            cube = stats["cube_area_px"]
            plate = stats["plate_area_px"]
            print(
                f"  {phase}: cube median {cube['median']:.1f} px"
                f" (invisible {stats['cube_invisible_fraction']:.0%}),"
                f" plate median {plate['median']:.1f} px, {stats['ticks']} ticks"
            )


if __name__ == "__main__":
    main()
