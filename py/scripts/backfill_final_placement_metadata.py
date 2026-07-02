#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Backfill final measured cube placement metadata into LeRobotDatasets.

The script reads ``meta/episodes/chunk-*/file-*.parquet`` files in one or more
dataset roots. It preserves existing columns, adds missing ``placement_*``
episode metadata, and writes parquet files only with ``--write``.

Each episode must already have ``cube_target_*`` metadata. The final cube pose is
recovered from the end of the overhead video using the current local overhead
camera calibration.
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
from pick_and_place.dataset_metadata import placement_error_metadata
from pick_and_place.episodes import PlacementError, _build_model
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.workspace_overlays import PAN_AXIS


PLACEMENT_COLUMNS = (
    "placement_detected",
    "placement_check_error",
    "placement_cube_x",
    "placement_cube_y",
    "placement_cube_z",
    "placement_target_x",
    "placement_target_y",
    "placement_target_z",
)


def _build_calibrated_overhead(camera_name: str) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    dummy_source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(dummy_source, include_environment=True, paper_target_marker=True)
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise SystemExit(f"unknown camera {camera_name!r}")
    return (
        model,
        data,
        data.cam_xpos[camera_id].copy(),
        data.cam_xmat[camera_id].reshape(3, 3).copy(),
    )


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


def _target_pose(row: pd.Series) -> CubePose | None:
    if "cube_target_x" not in row or not pd.notna(row["cube_target_x"]):
        return None
    return CubePose(
        x=float(row["cube_target_x"]),
        y=float(row["cube_target_y"]),
        z=float(row.get("cube_target_z", CUBE_HALF_SIZE)),
        roll=float(row.get("cube_target_roll", 0.0) or 0.0),
        pitch=float(row.get("cube_target_pitch", 0.0) or 0.0),
        yaw=float(row.get("cube_target_yaw", 0.0) or 0.0),
    )


def _read_final_frames(
    cap: Any,
    from_s: float,
    to_s: float,
    *,
    max_frames: int,
) -> list[np.ndarray]:
    fps = cap.get(5) or 30.0
    if not math.isfinite(to_s) or to_s <= 0.0:
        to_s = from_s
    start_s = max(from_s, to_s - max_frames / fps)
    frames = []
    for i in range(max_frames):
        timestamp_s = start_s + i / fps
        if timestamp_s > to_s:
            break
        cap.set(0, max(0.0, timestamp_s * 1000.0))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(bgr)
    return frames


def _detect_final_cube(
    frames_bgr: list[np.ndarray],
    *,
    cv2: Any,
    camera_matrix: np.ndarray,
    undistort_map: tuple[np.ndarray, np.ndarray] | None,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
) -> CubePose | None:
    detector = make_cube_detector()
    last_pose = None
    prior_rotation = None
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
        estimate = estimate_cube_pose(rgb, detector, camera_matrix, prior_rotation=prior_rotation)
        if estimate is None:
            continue
        prior_rotation = estimate.rotation
        rotation, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
        last_pose = CubePose(
            x=float(position[0]),
            y=float(position[1]),
            z=CUBE_HALF_SIZE,
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )
    return last_pose


def _placement_row(
    cube: CubePose | None, target: CubePose, check_error: str = ""
) -> dict[str, Any]:
    if cube is None:
        row = placement_error_metadata(None, detected=False, check_error=check_error)
        row.update(
            {
                "placement_target_x": float(target.x),
                "placement_target_y": float(target.y),
                "placement_target_z": float(CUBE_HALF_SIZE),
            }
        )
        return row

    cube_xyz = (float(cube.x), float(cube.y), float(cube.z))
    target_xyz = (float(target.x), float(target.y), float(CUBE_HALF_SIZE))
    error = PlacementError(
        cube_xyz=cube_xyz,
        target_xyz=target_xyz,
        dx=cube_xyz[0] - target_xyz[0],
        dy=cube_xyz[1] - target_xyz[1],
        dz=cube_xyz[2] - target_xyz[2],
        xy=math.hypot(cube_xyz[0] - target_xyz[0], cube_xyz[1] - target_xyz[1]),
    )
    return placement_error_metadata(error, detected=True, check_error=check_error)


def backfill_file(
    dataset_root: Path,
    parquet_path: Path,
    *,
    cv2: Any,
    camera_matrix: np.ndarray,
    undistort_map: tuple[np.ndarray, np.ndarray] | None,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    max_frames: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, int, int]:
    df = pd.read_parquet(parquet_path)
    for column in PLACEMENT_COLUMNS:
        if column not in df.columns:
            df[column] = False if column == "placement_detected" else math.nan
    if "placement_check_error" in df.columns:
        df["placement_check_error"] = df["placement_check_error"].fillna("")

    recovered = 0
    missing = 0
    caps: dict[Path, Any] = {}
    try:
        for index, row in df.iterrows():
            need_placement = overwrite or not pd.notna(row.get("placement_cube_x", math.nan))
            if not need_placement:
                continue

            target = _target_pose(row)
            if target is None:
                missing += 1
                continue

            video = _video_path(dataset_root, row)
            if video is None or not video.exists():
                missing += 1
                continue
            cap = caps.get(video)
            if cap is None:
                cap = cv2.VideoCapture(str(video))
                caps[video] = cap
            if not cap.isOpened():
                missing += 1
                continue

            from_s = float(row.get("videos/observation.images.overhead/from_timestamp", 0.0))
            to_s = float(row.get("videos/observation.images.overhead/to_timestamp", from_s))
            frames = _read_final_frames(cap, from_s, to_s, max_frames=max_frames)
            cube = _detect_final_cube(
                frames,
                cv2=cv2,
                camera_matrix=camera_matrix,
                undistort_map=undistort_map,
                cam_pos=cam_pos,
                cam_rot=cam_rot,
            )
            if cube is None:
                missing += 1
                for key, value in _placement_row(None, target, "cube not detected").items():
                    df.at[index, key] = value
                continue

            for key, value in _placement_row(cube, target).items():
                df.at[index, key] = value
            recovered += 1
    finally:
        for cap in caps.values():
            cap.release()

    required_missing = int(df["placement_cube_x"].isna().sum())
    return df, recovered, max(missing, required_missing)


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
    parser.add_argument("--max-frames", type=int, default=45)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write recovered rows even when other rows in the same file remain undetected",
    )
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

    total_recovered = 0
    touched = 0
    for root in roots:
        for parquet_path in episode_parquets(root):
            df, recovered, missing = backfill_file(
                root,
                parquet_path,
                cv2=cv2,
                camera_matrix=camera_matrix,
                undistort_map=undistort_map,
                cam_pos=cam_pos,
                cam_rot=cam_rot,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
            if not recovered and not missing:
                continue
            if missing:
                if not args.allow_partial or not recovered:
                    print(
                        f"cannot fully convert {parquet_path}: "
                        f"{recovered} final placement pose(s), "
                        f"{missing} episode row(s) still missing measured placement"
                    )
                    continue
                action = "partially updated" if args.write else "would partially update"
                print(
                    f"{action} {parquet_path}: "
                    f"{recovered} final placement pose(s), "
                    f"{missing} episode row(s) still missing measured placement"
                )
                touched += 1
                total_recovered += recovered
                if args.write:
                    df.to_parquet(parquet_path, index=False)
                continue
            touched += 1
            total_recovered += recovered
            action = "updated" if args.write else "would update"
            print(f"{action} {parquet_path}: {recovered} final placement pose(s)")
            if args.write:
                df.to_parquet(parquet_path, index=False)

    mode = "Wrote" if args.write else "Dry run:"
    print(f"{mode} {touched} parquet file(s), {total_recovered} final placement pose(s).")


if __name__ == "__main__":
    main()
