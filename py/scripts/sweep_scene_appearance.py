#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Sweep floor and drop-zone-target colours and measure what the policy would see.

The DPPO vision diagnosis established that the pick cube carries 4-5x less
luminance contrast than the drop-zone target in the 96x96 images the policy
consumes, and that the encoder consequently represents the target and not the
cube. The cube itself is hard to recolour because its AprilTag faces drive the
descent visual servo and the real-robot cube tracker, so this sweeps the
*background* instead: the floor and the target square.

For each (floor colour, target colour) combination this re-renders the same
recorded ground-truth ticks through the same camera pipeline and 96x96
aspect-fill/center-crop transform as training images, and reports per camera and
task phase how many pixels the cube and target occupy, how much contrast each
has against its local background, and — because a dark floor risks pushing the
target into the cube's brightness band — how separable the cube and target are
from each other.

Object segmentation does not depend on colour, so masks are rendered once per
tick and shared across every variant: the pixel sets being compared are
identical, and only the RGB render is repeated.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.sim.scene_appearance import (
    CUBE_COLOURS,
    FLOOR_COLOURS,
    TARGET_COLOURS,
    SceneAppearance,
    SceneAppearanceOverride,
)
from pick_and_place.analysis.scene_visibility import (
    OBJECT_COVERAGE,
    SceneMeasurer,
    contrast,
    load_episode_truth,
    video_render_hw,
)
from pick_and_place.core.image_ops import resize_and_center_crop
from pick_and_place.runtime.sim_recorder import OVERHEAD_CAMERA, WRIST_CAMERA
from pick_and_place.core.task_phases import PHASES

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
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5,
        help="episodes to sample (default: 5; the sweep compares variants, not absolutes)",
    )
    parser.add_argument(
        "--tick-stride",
        type=int,
        default=3,
        help="use every Nth 10 Hz policy tick (default: 3)",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument(
        "--floors",
        nargs="+",
        default=list(FLOOR_COLOURS),
        choices=list(FLOOR_COLOURS),
        help="floor colours to sweep (default: all)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGET_COLOURS),
        choices=list(TARGET_COLOURS),
        help="target colours to sweep (default: all)",
    )
    parser.add_argument(
        "--cubes",
        nargs="+",
        default=["apriltag"],
        choices=list(CUBE_COLOURS),
        help=(
            "cube appearances to sweep (default: apriltag, the current cube). "
            "Solid colours are only reachable by re-rendering tagged recordings."
        ),
    )
    parser.add_argument(
        "--frame-tags",
        nargs="+",
        default=["on"],
        choices=["on", "off"],
        help=(
            "whether the workspace frame keeps its AprilTag calibration plates "
            "(default: on, the current scene). Pass 'on off' to measure what "
            "peeling them off buys."
        ),
    )
    parser.add_argument(
        "--panel-ticks",
        type=int,
        default=4,
        help="ticks to save overlay panels for, per camera, per variant (default: 4)",
    )
    args = parser.parse_args()
    if args.image_size < 8:
        parser.error("--image-size must be at least 8")
    if args.tick_stride < 1:
        parser.error("--tick-stride must be at least 1")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be at least 1")
    if args.panel_ticks < 0:
        parser.error("--panel-ticks must be nonnegative")
    return args


def _mean_rgb(image: np.ndarray, coverage: np.ndarray) -> np.ndarray | None:
    mask = coverage >= OBJECT_COVERAGE
    if not mask.any():
        return None
    return image[mask].astype(np.float64).mean(axis=0)


def _luminance(rgb: np.ndarray) -> float:
    return float(rgb @ (0.2126, 0.7152, 0.0722))


def _separability(
    image: np.ndarray, coverage: dict[str, np.ndarray], other: str
) -> tuple[float | None, float | None]:
    """RGB distance and luminance gap between the cube's and another object's pixels.

    Both are ``None`` when either object is absent from the view. A dark floor
    lifts cube contrast but can leave the cube looking like whatever else in the
    scene is bright — the target, or the white arm reaching for it — which is
    what these two numbers detect.
    """
    cube = _mean_rgb(image, coverage["cube"])
    rival = _mean_rgb(image, coverage[other])
    if cube is None or rival is None:
        return None, None
    return float(np.linalg.norm(cube - rival)), abs(_luminance(cube) - _luminance(rival))


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


def _save_panel(path: Path, rgb: np.ndarray, coverage: dict[str, np.ndarray], title: str) -> None:
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


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for camera_label in CAMERAS:
        summary[camera_label] = {}
        for phase in PHASES:
            phase_rows = [
                row for row in rows if row["camera"] == camera_label and row["phase"] == phase
            ]
            if not phase_rows:
                continue

            def column(key: str, source: list[dict[str, Any]] = phase_rows) -> list[float]:
                return [row[key] for row in source if row[key] is not None]

            summary[camera_label][phase] = {
                "ticks": len(phase_rows),
                "cube_pixels": _distribution(column("cube_pixels")),
                "cube_contrast": _distribution(column("cube_contrast")),
                "cube_invisible_fraction": float(
                    np.mean([row["cube_pixels"] == 0 for row in phase_rows])
                ),
                "plate_pixels": _distribution(column("plate_pixels")),
                "plate_contrast": _distribution(column("plate_contrast")),
                "cube_plate_rgb_distance": _distribution(column("cube_plate_rgb_distance")),
                "cube_plate_luminance_delta": _distribution(
                    column("cube_plate_luminance_delta")
                ),
                "cube_robot_rgb_distance": _distribution(column("cube_robot_rgb_distance")),
                "cube_robot_luminance_delta": _distribution(
                    column("cube_robot_luminance_delta")
                ),
                "cube_frametag_rgb_distance": _distribution(
                    column("cube_frametag_rgb_distance")
                ),
                "image_mean_luminance": _distribution(column("image_mean_luminance")),
            }
    return summary


def _comparison_table(summaries: dict[str, dict[str, Any]]) -> str:
    """Rank variants by overhead acquisition cube contrast, the diagnosed gap."""
    lines = [
        "# Floor and target colour sweep",
        "",
        "Overhead camera, acquisition phase — the view and phase where the",
        "diagnosis located the failure. The `sep` columns are mean RGB distances",
        "between the cube's pixels and the target's or the arm's; low values mean",
        "a dark floor has lifted cube contrast only to leave it looking like the",
        "other bright things in the scene.",
        "",
        "| Variant | Cube px | Cube contrast | Plate contrast | Cube/plate sep |"
        " Cube/robot sep | Cube/frametag sep | Image luminance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    listed = []
    for name, summary in summaries.items():
        stats = summary.get("overhead", {}).get("acquisition")
        if stats is None:
            continue
        listed.append((name, stats))
    listed.sort(key=lambda item: -(item[1]["cube_contrast"] or {}).get("median", 0.0))
    for name, stats in listed:
        def median(key: str) -> str:
            value = stats[key]
            return f"{value['median']:.3f}" if value else "—"

        lines.append(
            f"| {name} | {median('cube_pixels')} | {median('cube_contrast')} |"
            f" {median('plate_contrast')} | {median('cube_plate_rgb_distance')} |"
            f" {median('cube_robot_rgb_distance')} |"
            f" {median('cube_frametag_rgb_distance')} | {median('image_mean_luminance')} |"
        )
    return "\n".join(lines) + "\n"


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

    variants = [
        SceneAppearance(floor, target, cube, tags == "on")
        for floor, target, cube, tags in itertools.product(
            args.floors, args.targets, args.cubes, args.frame_tags
        )
    ]
    render_hw = video_render_hw(episode_roots[0])
    measurer = SceneMeasurer(render_hw, args.image_size)
    override = SceneAppearanceOverride(measurer.model)
    building.mkdir(parents=True)
    for variant in variants:
        (building / "panels" / variant.name).mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    try:
        for episode_root in episode_roots:
            episode = load_episode_truth(episode_root)
            if video_render_hw(episode_root) != render_hw:
                raise ValueError(f"{episode.name} has a different camera resolution")
            measurer.set_target_plate(episode.target_xy, episode.target_plate_yaw)
            # The marker placement decides this episode's plate colour.
            override.refresh_plate_baseline()
            tick_indices = list(range(0, len(episode.states), FRAME_STRIDE * args.tick_stride))
            panel_ticks = {
                tick_indices[round(fraction * (len(tick_indices) - 1))]
                for fraction in np.linspace(0.0, 1.0, args.panel_ticks)
            } if args.panel_ticks and tick_indices else set()
            for source_index in tick_indices:
                phase = str(episode.coarse_phases[source_index])
                measurer.set_frame(episode.states[source_index], episode.cube_poses[source_index])
                # Segmentation is appearance independent: render the masks once
                # so every variant is scored over an identical pixel set.
                coverage = {
                    label: measurer.coverage_maps(
                        camera, include_robot=True, include_frame_tags=True
                    )
                    for label, camera in CAMERAS.items()
                }
                for variant in variants:
                    override.apply(variant)
                    for camera_label, camera in CAMERAS.items():
                        measurer.rgb_renderer.update_scene(measurer.data, camera=camera)
                        rgb = resize_and_center_crop(
                            measurer.rgb_renderer.render(), args.image_size, args.image_size
                        )
                        maps = coverage[camera_label]
                        rgb_distance, luminance_delta = _separability(rgb, maps, "plate")
                        robot_distance, robot_luminance = _separability(rgb, maps, "robot")
                        tag_distance, _ = _separability(rgb, maps, "frame_tags")
                        rows.append(
                            {
                                "variant": variant.name,
                                "floor": variant.floor,
                                "target": variant.target,
                                "cube": variant.cube,
                                "frame_tags": variant.frame_tags,
                                "episode": episode.name,
                                "source_frame": source_index,
                                "phase": phase,
                                "camera": camera_label,
                                "cube_pixels": int((maps["cube"] >= OBJECT_COVERAGE).sum()),
                                "cube_contrast": contrast(rgb, maps["cube"], maps["plate"]),
                                "plate_pixels": int((maps["plate"] >= OBJECT_COVERAGE).sum()),
                                "plate_contrast": contrast(rgb, maps["plate"], maps["cube"]),
                                "cube_plate_rgb_distance": rgb_distance,
                                "cube_plate_luminance_delta": luminance_delta,
                                "cube_robot_rgb_distance": robot_distance,
                                "cube_robot_luminance_delta": robot_luminance,
                                "cube_frametag_rgb_distance": tag_distance,
                                "image_mean_luminance": _luminance(
                                    rgb.astype(np.float64).reshape(-1, 3).mean(axis=0)
                                ),
                            }
                        )
                        if source_index in panel_ticks:
                            stem = f"{episode.name}-f{source_index:04d}-{phase}-{camera_label}"
                            _save_panel(
                                building / "panels" / variant.name / f"{stem}.png",
                                rgb,
                                maps,
                                f"{variant.name}\n{stem}",
                            )

        with (building / "ticks.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summaries = {
            variant.name: _summarize([row for row in rows if row["variant"] == variant.name])
            for variant in variants
        }
        with (building / "summary.json").open("w") as file:
            json.dump(
                {
                    "episodes": [root.name for root in episode_roots],
                    "render_hw": list(render_hw),
                    "image_size": args.image_size,
                    "tick_stride": args.tick_stride,
                    "floor_colours": {name: list(FLOOR_COLOURS[name]) for name in args.floors},
                    "target_colours": {name: list(TARGET_COLOURS[name]) for name in args.targets},
                    "cubes": args.cubes,
                    "frame_tags": args.frame_tags,
                    "variants": summaries,
                },
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")
        table = _comparison_table(summaries)
        (building / "comparison.md").write_text(table)
    except Exception:
        shutil.rmtree(building)
        raise
    finally:
        measurer.close()

    building.rename(output)
    print(table)


if __name__ == "__main__":
    main()
