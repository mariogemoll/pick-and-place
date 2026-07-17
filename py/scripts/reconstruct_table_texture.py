#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reconstruct a top-down table texture from overhead-camera footage.

The overhead camera is fixed and calibrated, so recovering the table surface is
far simpler than the wrist panorama: no ray accumulation over camera poses is
needed. Two steps:

1. **Clean plate by temporal median.** Sample one mid-episode overhead frame from
   each of many episodes and take the per-pixel median. The cube, arm, gripper,
   operator hands, and drop-zone paper are transient and land in different places
   every episode, so the median rejects them and leaves the static table surface.
   Frames are taken from the middle of each episode (not the boundaries), where the
   arm is actively working and spread across the workspace, so it never dwells in
   its rest pose long enough to survive the median.

2. **Rectify to top-down orthographic.** The table is the world Z=0 plane and the
   camera pose + intrinsics are known, so there is an exact plane->image mapping.
   For each texel of an (X, Y) grid over the finite-floor square we project its
   world point into the undistorted image and sample, giving a top-down texture
   registered 1:1 to the ``floor`` geom that ``build_scene`` textures.

Writes the rectified texture and, beside it, the median plate for inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq

from pick_and_place import build_scene
from pick_and_place.camera_compare import load_intrinsics
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.environment import WORKSPACE_FRAME_POS
from pick_and_place.paper_detection import project_to_pixel
from pick_and_place.scene import WORKSPACE_FLOOR_HALF

OVERHEAD_CAMERA = "overhead_camera"
OVERHEAD_W, OVERHEAD_H = 1920, 1080
OVERHEAD_FEATURE = "observation.images.overhead"


def _calibrated_overhead_pose(camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation of the calibrated overhead camera.

    The camera is rigidly mounted to the worldbody, so its pose is independent of
    the arm configuration; a single ``mj_forward`` on the default state suffices.
    """
    spec = build_scene(include_environment=True, robot_dynamics=False)
    model = spec.compile()
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise SystemExit(f"unknown camera {camera_name!r}")
    return data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].reshape(3, 3).copy()


def _episode_frames(dataset_roots: list[Path]) -> list[tuple[Path, int, int]]:
    """Per-episode ``(overhead_video_path, start_frame, length)`` across datasets."""
    prefix = f"videos/{OVERHEAD_FEATURE}"
    episodes: list[tuple[Path, int, int]] = []
    for root in dataset_roots:
        with (root / "meta" / "info.json").open() as f:
            fps = float(json.load(f).get("fps", 30))
        for parquet_path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
            for row in pq.read_table(parquet_path).to_pylist():
                chunk = int(row[f"{prefix}/chunk_index"])
                file = int(row[f"{prefix}/file_index"])
                start = round(float(row[f"{prefix}/from_timestamp"]) * fps)
                path = root / "videos" / OVERHEAD_FEATURE / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
                if path.is_file():
                    episodes.append((path, start, int(row["length"])))
    if not episodes:
        raise SystemExit(f"no overhead episodes under {dataset_roots}")
    return episodes


def _sample_frames(
    episodes: list[tuple[Path, int, int]],
    max_frames: int,
    mid_range: tuple[float, float],
    scale: float,
    undistort_map,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Read one mid-episode frame from up to ``max_frames`` episodes.

    Returns the undistorted, downscaled RGB frames and their mean brightness (a
    cheap lighting proxy used to split frames into brightness tiers).
    """
    order = rng.permutation(len(episodes))[:max_frames]
    lo, hi = mid_range
    # Read sequentially per video file to avoid reopening the same file repeatedly.
    wanted: dict[Path, list[int]] = {}
    for i in order:
        path, start, length = episodes[i]
        frac = rng.uniform(lo, hi)
        wanted.setdefault(path, []).append(start + int(frac * max(1, length)))

    frames: list[np.ndarray] = []
    for path, idxs in wanted.items():
        cap = cv2.VideoCapture(str(path))
        for idx in sorted(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
            if scale != 1.0:
                rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        cap.release()

    if not frames:
        raise SystemExit("could not read any overhead frames")
    brightness = np.array([float(f.mean()) for f in frames])
    print(f"sampled {len(frames)} mid-episode frames from {len(wanted)} video files")
    return frames, brightness


def _brightness_bins(brightness: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each frame to one of ``n_bins`` lighting tiers by brightness quantile."""
    if n_bins <= 1:
        return np.zeros(len(brightness), dtype=int)
    edges = np.quantile(brightness, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.digitize(brightness, edges[1:-1]), 0, n_bins - 1)


def _rectify_floor_square(
    plate: np.ndarray,
    camera_matrix: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Sample the median plate into a top-down texture over the finite-floor square.

    Output orientation: row 0 is the +X (north) edge and decreases downward; column
    0 is the -Y (west) edge and increases rightward -- the map view looking straight
    down with world +X up and world +Y right. ``build_scene`` maps this onto the
    ``floor`` box so the texel at world (x, y) sits exactly over that point.
    """
    cx, cy = WORKSPACE_FRAME_POS[0], WORKSPACE_FRAME_POS[1]
    half = WORKSPACE_FLOOR_HALF
    xs = np.linspace(cx + half, cx - half, resolution)  # row 0 = +X edge
    ys = np.linspace(cy - half, cy + half, resolution)  # col 0 = -Y edge
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    world = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=-1)

    px = project_to_pixel(world, camera_matrix, cam_pos, cam_rot)
    map_x = px[:, 0].reshape(resolution, resolution).astype(np.float32)
    map_y = px[:, 1].reshape(resolution, resolution).astype(np.float32)
    return cv2.remap(plate, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="LeRobotDataset root(s)")
    parser.add_argument("--out", type=Path, required=True, help="output texture PNG path")
    parser.add_argument("--resolution", type=int, default=1024, help="output texture edge (texels)")
    parser.add_argument("--max-frames", type=int, default=300, help="episodes to median (one frame each)")
    parser.add_argument(
        "--brightness-bins",
        type=int,
        default=1,
        help="split frames into N lighting tiers, one table texture each (mirrors the panorama)",
    )
    parser.add_argument(
        "--mid-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=(0.25, 0.75),
        help="sample each frame from this fraction of its episode (avoids rest pose)",
    )
    parser.add_argument(
        "--plate-scale",
        type=float,
        default=0.5,
        help="downscale factor for the median plate (memory vs. sharpness)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for frame sampling")
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path(LOCAL_CAMERA_INTRINSICS_DIR) / f"{OVERHEAD_CAMERA}.json",
    )
    args = parser.parse_args()

    camera_matrix, undistort_map = load_intrinsics(args.intrinsics, OVERHEAD_W, OVERHEAD_H, cv2)
    episodes = _episode_frames(args.datasets)
    print(f"{len(episodes)} episodes with overhead video across {len(args.datasets)} dataset(s)")
    rng = np.random.default_rng(args.seed)
    frames, brightness = _sample_frames(
        episodes, args.max_frames, tuple(args.mid_range), args.plate_scale, undistort_map, rng
    )
    n_bins = max(1, args.brightness_bins)
    bin_of = _brightness_bins(brightness, n_bins)

    cam_pos, cam_rot = _calibrated_overhead_pose(OVERHEAD_CAMERA)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for b in range(n_bins):
        members = [frames[i] for i in range(len(frames)) if bin_of[i] == b]
        if not members:
            continue
        in_bin = brightness[bin_of == b]
        print(f"bin {b}: {len(members)} frames, brightness {in_bin.min():.0f}..{in_bin.max():.0f}")
        plate = np.median(np.stack(members, axis=0), axis=0).astype(np.uint8)

        # Scale the intrinsics to the (downscaled) plate resolution.
        scaled_matrix = camera_matrix.copy()
        scaled_matrix[0] *= plate.shape[1] / OVERHEAD_W
        scaled_matrix[1] *= plate.shape[0] / OVERHEAD_H
        texture = _rectify_floor_square(plate, scaled_matrix, cam_pos, cam_rot, args.resolution)

        out = args.out if n_bins == 1 else args.out.with_name(f"{args.out.stem}_b{b}{args.out.suffix}")
        cv2.imwrite(str(out), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(out.with_name(out.stem + "_plate.png")), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
        )
        print(f"  wrote {out.name} ({args.resolution}x{args.resolution}) and {out.stem}_plate.png")


if __name__ == "__main__":
    main()
