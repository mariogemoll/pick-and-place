#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Backfill cube source/target pose metadata into recorded LeRobotDatasets.

The script reads ``meta/episodes/chunk-*/file-*.parquet`` files in one or more
dataset roots. It preserves existing columns, adds missing ``cube_start_x/y/yaw``
and ``target_x/y`` scalar metadata, and writes the parquet file only with
``--write``.

Missing targets and sources are recovered from the beginning of each episode's
overhead video using the current local camera calibration. A parquet file is
written only if every episode can be converted to the required pose-column
schema.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from pick_and_place.camera_compare import load_intrinsics
from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.cube_detection import (
    cube_pose_to_world,
    estimate_cube_pose,
    make_cube_detector,
)
from pick_and_place.episodes import _build_model
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.paper_detection import detect_paper_target
from pick_and_place.workspace_overlays import PAN_AXIS, workspace_interior_corners_world


POSE_COLUMNS = ("cube_start_x", "cube_start_y", "cube_start_yaw", "target_x", "target_y")


def _source_row(pose: CubePose) -> dict[str, Any]:
    return {
        "cube_start_x": float(pose.x),
        "cube_start_y": float(pose.y),
        "cube_start_yaw": float(pose.yaw),
    }


def _target_row(pose: CubePose) -> dict[str, Any]:
    return {"target_x": float(pose.x), "target_y": float(pose.y)}


def _build_calibrated_overhead(camera_name: str) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    dummy_source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(dummy_source, include_environment=True, paper_target_marker=True)
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise SystemExit(f"unknown camera {camera_name!r}")
    return model, data, data.cam_xpos[camera_id].copy(), data.cam_xmat[camera_id].reshape(3, 3).copy()


def _video_path(dataset_root: Path, row: pd.Series) -> Path | None:
    chunk_col = "videos/observation.images.overhead/chunk_index"
    file_col = "videos/observation.images.overhead/file_index"
    if chunk_col not in row or file_col not in row:
        return None
    chunk = int(row[chunk_col])
    file_index = int(row[file_col])
    return (
        dataset_root
        / "videos"
        / "observation.images.overhead"
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def _read_episode_frames(
    cap: Any,
    start_s: float,
    *,
    max_frames: int,
) -> list[np.ndarray]:
    fps = cap.get(5) or 30.0
    frames = []
    for i in range(max_frames):
        cap.set(0, max(0.0, (start_s + i / fps) * 1000.0))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(bgr)
    return frames


def _detect_source(
    frames_bgr: list[np.ndarray],
    *,
    cv2: Any,
    camera_matrix: np.ndarray,
    undistort_map: tuple[np.ndarray, np.ndarray] | None,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
) -> CubePose | None:
    detector = make_cube_detector()
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
        estimate = estimate_cube_pose(rgb, detector, camera_matrix)
        if estimate is None:
            continue
        rotation, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
        return CubePose(
            x=float(position[0]),
            y=float(position[1]),
            z=CUBE_HALF_SIZE,
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )
    return None


def _detect_target(
    frames_bgr: list[np.ndarray],
    *,
    cv2: Any,
    camera_matrix: np.ndarray,
    undistort_map: tuple[np.ndarray, np.ndarray] | None,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    target_color: str,
) -> CubePose | None:
    workspace_corners = workspace_interior_corners_world()
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
        target = detect_paper_target(
            rgb,
            camera_matrix,
            cam_pos,
            cam_rot,
            target_color=target_color,
            workspace_corners_world=workspace_corners,
        )
        if target is not None:
            return CubePose(
                x=target.xy[0],
                y=target.xy[1],
                z=CUBE_HALF_SIZE,
                yaw=target.yaw,
            )
    return None


def backfill_file(
    dataset_root: Path,
    parquet_path: Path,
    *,
    cv2: Any,
    camera_matrix: np.ndarray,
    undistort_map: tuple[np.ndarray, np.ndarray] | None,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    target_color: str,
    max_frames: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, int, int, int]:
    df = pd.read_parquet(parquet_path)
    for column in POSE_COLUMNS:
        if column not in df.columns:
            df[column] = math.nan

    recovered_sources = 0
    recovered_targets = 0
    missing_rows = 0
    caps: dict[Path, Any] = {}
    try:
        for index, row in df.iterrows():
            need_source = overwrite or not pd.notna(row.get("cube_start_x", math.nan))
            need_target = overwrite or not pd.notna(row.get("target_x", math.nan))

            if not need_source and not need_target:
                continue

            video = _video_path(dataset_root, row)
            if video is None or not video.exists():
                missing_rows += 1
                continue
            cap = caps.get(video)
            if cap is None:
                cap = cv2.VideoCapture(str(video))
                caps[video] = cap
            if not cap.isOpened():
                missing_rows += 1
                continue

            start_s = float(row.get("videos/observation.images.overhead/from_timestamp", 0.0))
            frames = _read_episode_frames(cap, start_s, max_frames=max_frames)

            if need_source:
                source = _detect_source(
                    frames,
                    cv2=cv2,
                    camera_matrix=camera_matrix,
                    undistort_map=undistort_map,
                    cam_pos=cam_pos,
                    cam_rot=cam_rot,
                )
                if source is not None:
                    for key, value in _source_row(source).items():
                        df.at[index, key] = value
                    recovered_sources += 1
                else:
                    missing_rows += 1

            if need_target:
                target = _detect_target(
                    frames,
                    cv2=cv2,
                    camera_matrix=camera_matrix,
                    undistort_map=undistort_map,
                    cam_pos=cam_pos,
                    cam_rot=cam_rot,
                    target_color=target_color,
                )
                if target is not None:
                    for key, value in _target_row(target).items():
                        df.at[index, key] = value
                    recovered_targets += 1
                else:
                    missing_rows += 1
    finally:
        for cap in caps.values():
            cap.release()

    required_missing = int(df[list(POSE_COLUMNS)].isna().any(axis=1).sum())
    return df, recovered_sources, recovered_targets, max(missing_rows, required_missing)


def episode_parquets(root: Path) -> list[Path]:
    return sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_roots",
        nargs="*",
        type=Path,
        default=[Path("datasets")],
        help="dataset root(s), or a parent directory containing dataset roots",
    )
    parser.add_argument("--camera-name", default="overhead_camera")
    parser.add_argument("--target-color", choices=("black", "white"), default="black")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write", action="store_true", help="write converted parquet files")
    args = parser.parse_args()

    import cv2

    intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
    if not intrinsics.exists():
        raise SystemExit(f"missing intrinsics: {intrinsics}")
    camera_matrix, undistort_map = load_intrinsics(intrinsics, 1920, 1080, cv2)
    _, _, cam_pos, cam_rot = _build_calibrated_overhead(args.camera_name)

    roots: list[Path] = []
    for root in args.dataset_roots:
        if episode_parquets(root):
            roots.append(root)
        else:
            roots.extend(sorted(p.parent.parent for p in root.glob("*/meta/info.json")))

    total_sources = 0
    total_targets = 0
    touched = 0
    for root in roots:
        for parquet_path in episode_parquets(root):
            df, sources, targets, missing = backfill_file(
                root,
                parquet_path,
                cv2=cv2,
                camera_matrix=camera_matrix,
                undistort_map=undistort_map,
                cam_pos=cam_pos,
                cam_rot=cam_rot,
                target_color=args.target_color,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
            if sources or targets:
                if missing:
                    print(
                        f"cannot fully convert {parquet_path}: "
                        f"{sources} source pose(s), {targets} target pose(s), "
                        f"{missing} episode row(s) still missing required pose data"
                    )
                    continue
                touched += 1
                total_sources += sources
                total_targets += targets
                action = "updated" if args.write else "would update"
                print(
                    f"{action} {parquet_path}: "
                    f"{sources} source pose(s), {targets} target pose(s)"
                )
                if args.write:
                    df.to_parquet(parquet_path, index=False)

    mode = "Wrote" if args.write else "Dry run:"
    print(
        f"{mode} {touched} parquet file(s), "
        f"{total_sources} source pose(s), {total_targets} target pose(s)."
    )


if __name__ == "__main__":
    main()
