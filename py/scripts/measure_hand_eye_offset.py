#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure the hand-eye error between real and sim from exported frame pairs.

The pairs written by ``export_sim_real_pairs`` place the sim cube where the
overhead camera saw it and pose the sim arm at the measured joint angles. Any
offset between the cube in the real wrist image and in the sim wrist render is
therefore the accumulated hand-eye error of the overhead localization ->
world -> arm kinematics -> wrist camera chain — the same error that forces the
real robot into a visual-servoing phase before grasping.

For every stable-cube pair frame this script estimates the cube pose from
AprilTag faces in both the real and the sim wrist image and reports the offset
distribution: per-episode bias and a global summary, in world coordinates
(millimeters, using the sim wrist camera pose recorded with each pair) plus
image pixels. Pair exports that predate the recorded camera pose fall back to
wrist-camera-frame offsets. The distribution is what sim training should
randomize over so that open-loop reaching in sim misses the way real reaching
misses.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.cube_detection import (
    cube_pose_to_world,
    estimate_cube_pose,
    make_cube_detector,
)
from pick_and_place.image_rectify import rectified_camera_matrix

MAX_CUBE_DISTANCE = 0.5


def _detect(image_path: Path, detector, camera_matrix: np.ndarray):
    """Cube pose estimate in the wrist camera frame, or None."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    estimate = estimate_cube_pose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), detector, camera_matrix)
    if estimate is None:
        return None
    if float(np.linalg.norm(estimate.position)) > MAX_CUBE_DISTANCE:
        return None
    return estimate


def _pixel(position: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    projected = camera_matrix @ position
    return projected[:2] / projected[2]


def _summarize(name: str, deltas: list[np.ndarray], pixel_deltas: list[np.ndarray]) -> dict:
    deltas_mm = np.asarray(deltas) * 1000.0
    pixels = np.asarray(pixel_deltas)
    summary = {
        "name": name,
        "n": len(deltas),
        "bias_mm": [float(v) for v in deltas_mm.mean(axis=0)],
        "std_mm": [float(v) for v in deltas_mm.std(axis=0)],
        "norm_mm_median": float(np.median(np.linalg.norm(deltas_mm, axis=1))),
        "pixel_bias": [float(v) for v in pixels.mean(axis=0)],
        "pixel_norm_median": float(np.median(np.linalg.norm(pixels, axis=1))),
    }
    print(
        f"{name}: n={summary['n']} "
        f"bias=({summary['bias_mm'][0]:+.1f}, {summary['bias_mm'][1]:+.1f}, "
        f"{summary['bias_mm'][2]:+.1f})mm "
        f"std=({summary['std_mm'][0]:.1f}, {summary['std_mm'][1]:.1f}, "
        f"{summary['std_mm'][2]:.1f})mm "
        f"|median|={summary['norm_mm_median']:.1f}mm / {summary['pixel_norm_median']:.0f}px"
    )
    return summary


def _measure_episode(
    episode_dir: Path, detector, camera_matrix: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    with (episode_dir / "pairs.json").open() as f:
        index = json.load(f)
    deltas: list[np.ndarray] = []
    pixel_deltas: list[np.ndarray] = []
    per_frame: list[dict] = []
    image_ext = next((p.suffix for p in (episode_dir / "wrist_real").glob("*")), None)
    if image_ext is None:
        return [], [], []
    for frame in index["frames"]:
        if frame["cube_tracking"] != "stable":
            continue
        name = f"{frame['frame']:06d}{image_ext}"
        real = _detect(episode_dir / "wrist_real" / name, detector, camera_matrix)
        sim = _detect(episode_dir / "wrist_sim" / name, detector, camera_matrix)
        if real is None or sim is None:
            continue
        pixel_delta = _pixel(real.position, camera_matrix) - _pixel(sim.position, camera_matrix)
        pixel_deltas.append(pixel_delta)
        record = {
            "frame": frame["frame"],
            "delta_cam_mm": [float(v * 1000.0) for v in real.position - sim.position],
            "delta_px": [float(v) for v in pixel_delta],
        }
        wrist_cam = frame.get("wrist_cam")
        if wrist_cam is not None:
            cam_pos = np.asarray(wrist_cam["pos"], dtype=float)
            cam_rot = np.asarray(wrist_cam["mat"], dtype=float).reshape(3, 3)
            _, world_real = cube_pose_to_world(real, cam_pos, cam_rot)
            _, world_sim = cube_pose_to_world(sim, cam_pos, cam_rot)
            delta = np.asarray(world_real, dtype=float) - np.asarray(world_sim, dtype=float)
            record["delta_world_mm"] = [float(v * 1000.0) for v in delta]
        else:
            delta = real.position - sim.position
        deltas.append(delta)
        per_frame.append(record)
    return deltas, pixel_deltas, per_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs_roots",
        type=Path,
        nargs="+",
        help="pair export root(s), dataset dirs, or single episode dirs",
    )
    parser.add_argument("--output", type=Path, default=None, help="write the summary JSON here")
    args = parser.parse_args()

    episode_dirs: list[Path] = []
    for root in args.pairs_roots:
        if (root / "pairs.json").is_file():
            episode_dirs.append(root)
        else:
            episode_dirs.extend(sorted(root.glob("**/episode_*/")))
    episode_dirs = [d for d in episode_dirs if (d / "pairs.json").is_file()]
    if not episode_dirs:
        parser.error("no episode directories with pairs.json found")

    with (episode_dirs[0] / "pairs.json").open() as f:
        first_index = json.load(f)
    intrinsics = load_local_camera_intrinsics()["wrist_camera"]
    camera_matrix = np.asarray(
        rectified_camera_matrix(intrinsics, first_index["width"], first_index["height"]),
        dtype=float,
    )
    detector = make_cube_detector()

    all_deltas: list[np.ndarray] = []
    all_pixel_deltas: list[np.ndarray] = []
    episode_summaries: list[dict] = []
    for episode_dir in episode_dirs:
        deltas, pixel_deltas, per_frame = _measure_episode(episode_dir, detector, camera_matrix)
        if not deltas:
            print(f"{episode_dir}: no frames with the cube detected in both images")
            continue
        summary = _summarize(str(episode_dir), deltas, pixel_deltas)
        summary["frames"] = per_frame
        episode_summaries.append(summary)
        all_deltas.extend(deltas)
        all_pixel_deltas.extend(pixel_deltas)

    if not all_deltas:
        raise SystemExit("no measurable frames found")

    overall = _summarize("overall", all_deltas, all_pixel_deltas)
    episode_biases = [s["bias_mm"] for s in episode_summaries]
    overall["episode_bias_spread_mm"] = [
        float(statistics.pstdev(axis_values))
        for axis_values in zip(*episode_biases)
    ] if len(episode_biases) > 1 else None

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump({"overall": overall, "episodes": episode_summaries}, f, indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
