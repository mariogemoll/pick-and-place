#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Convert a recorded real LeRobotDataset to the 512x512 square image format.

A real recording stores each camera at its native, lens-distorted resolution
(overhead 1920x1080, wrist 1280x720). The sim recordings and the VLA input, by
contrast, are 512x512 squares of an ideal pinhole view. This script bridges the
two: every camera frame is undistorted with its calibrated intrinsics,
center-cropped to a square (keeping the full image height), and resized to
512x512, so a converted real frame matches a sim frame pixel-geometry for
pixel-geometry. State, action, task and timing are copied through unchanged,
as is every other per-episode metadata column already present on the source
dataset (pickup/placement checks, cube start/target pose, success, ...).

The geometry mirrors how the sim sets its camera field of view: the rectified
pinhole uses focal length ``fy`` on both axes with the principal point at the
image center, and a center square crop keeps the full height, giving a vertical
(and, being square, horizontal) FOV of ``2*atan((h/2)/fy)`` -- exactly the
angle ``SimCameraRig`` feeds MuJoCo.

Videos are decoded sequentially with PyAV (one straight pass per file rather
than a per-frame seek), so the run is bound by raw decode speed. The v3 dataset
stores each episode's frames as one contiguous, in-order segment of its video
file, so decoding the files in order yields frames in lockstep with the numeric
rows read from the data parquet.

Example:

    python py/scripts/convert_dataset_to_square.py \
        --src datasets/20260702 --dst datasets/20260702-512
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.executor import RecordingSession
from pick_and_place.image_rectify import SQUARE_SIZE, build_undistort_map, transform_frame

# Which calibrated camera each dataset image feature was captured with.
FEATURE_TO_CAMERA = {
    "observation.images.overhead": "overhead_camera",
    "observation.images.wrist": "wrist_camera",
}

# Episode-metadata columns that are LeRobot bookkeeping (file layout, video
# spans, per-feature stats) rather than data recorded by this project. Every
# other column on the source episode table is treated as project metadata and
# carried through to the converted dataset unchanged, so this script does not
# need updating each time a new metadata column is added upstream.
BOOKKEEPING_COLUMNS = {"episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"}
BOOKKEEPING_PREFIXES = ("data/", "videos/", "stats/", "meta/episodes/")


def ordered_unique_files(df: Any, chunk_col: str, file_col: str) -> list[tuple[int, int]]:
    """Distinct ``(chunk_index, file_index)`` pairs in first-appearance order."""
    pairs: list[tuple[int, int]] = []
    for chunk, file in zip(df[chunk_col], df[file_col]):
        key = (int(chunk), int(file))
        if key not in pairs:
            pairs.append(key)
    return pairs


def episode_metadata_columns(episodes: Any) -> list[str]:
    return [
        c
        for c in episodes.columns
        if c not in BOOKKEEPING_COLUMNS and not c.startswith(BOOKKEEPING_PREFIXES)
    ]


def decode_frames(paths: list[Path]) -> Iterator[np.ndarray]:
    """Yield every frame of ``paths`` in order as ``(H, W, 3)`` uint8 RGB arrays."""
    import av

    for path in paths:
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            # LeRobot's background video encoder restores FFmpeg's default stderr
            # callback when it starts, so re-assert the quiet level on each frame
            # to keep libswscale's benign per-frame "no SIMD path for yuv420p->rgb24"
            # note out of the output.
            av.logging.set_level(av.logging.ERROR)
            yield frame.to_ndarray(format="rgb24")
        container.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source LeRobotDataset root")
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="output dataset root (default: <src>-512 alongside the source)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="repo id for the output dataset (default: <src dir name>-512)",
    )
    parser.add_argument("--size", type=int, default=SQUARE_SIZE, help="square output side (px)")
    parser.add_argument(
        "--vcodec",
        default="auto",
        help="LeRobot video codec (default: auto = best available HW encoder)",
    )
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode video during capture; --no-streaming-encoding falls back to PNG-then-encode",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="background image-writer threads for PNG-then-encode mode",
    )
    args = parser.parse_args()

    import cv2
    import pandas as pd
    from tqdm import tqdm

    dst_root = args.dst if args.dst is not None else args.src.with_name(f"{args.src.name}-512")
    if dst_root.exists():
        raise SystemExit(f"output {dst_root} already exists; remove it or pick another --dst")

    info = json.loads((args.src / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    data_path = info["data_path"]
    video_path = info["video_path"]

    intrinsics_by_camera = load_local_camera_intrinsics()
    missing = [cam for cam in FEATURE_TO_CAMERA.values() if cam not in intrinsics_by_camera]
    if missing:
        raise SystemExit(f"no calibrated intrinsics for {missing}; cannot undistort")

    tasks = pd.read_parquet(args.src / "meta" / "tasks.parquet")
    # tasks.parquet is indexed by the task string with a ``task_index`` column.
    task_by_index = {int(row.task_index): str(name) for name, row in tasks.iterrows()}

    episodes = pd.concat(
        pd.read_parquet(p) for p in sorted((args.src / "meta" / "episodes").rglob("*.parquet"))
    ).sort_values("episode_index")
    metadata_columns = episode_metadata_columns(episodes)
    # A column with missing values on some episodes (e.g. a check that only
    # newer episodes have) mixes real values with pandas' float NaN, which
    # pyarrow then rejects when episode metadata rows are later batched into
    # one table (a NaN sitting in an otherwise-string column, say). Normalize
    # every metadata column's missing values to plain ``None`` so pyarrow
    # always sees one consistent null representation.
    for col in metadata_columns:
        episodes[col] = episodes[col].astype(object).where(episodes[col].notna(), None)
    metadata_by_episode = {
        int(row["episode_index"]): {col: row[col] for col in metadata_columns}
        for row in episodes.to_dict("records")
    }

    def src_path(template: str, **fields: Any) -> Path:
        return args.src / template.format(**fields)

    data_files = [
        src_path(data_path, chunk_index=c, file_index=f)
        for c, f in ordered_unique_files(episodes, "data/chunk_index", "data/file_index")
    ]
    rows = pd.concat(pd.read_parquet(p) for p in data_files).sort_values("index")
    states = rows["observation.state"].to_numpy()
    actions = rows["action"].to_numpy()
    episode_indices = rows["episode_index"].to_numpy()
    task_indices = rows["task_index"].to_numpy()

    streams = {
        feature: decode_frames(
            [
                src_path(
                    video_path,
                    video_key=feature,
                    chunk_index=c,
                    file_index=f,
                )
                for c, f in ordered_unique_files(
                    episodes, f"videos/{feature}/chunk_index", f"videos/{feature}/file_index"
                )
            ]
        )
        for feature in FEATURE_TO_CAMERA
    }

    print(
        f"Converting {episodes.shape[0]} episode(s), {len(rows)} frame(s) "
        f"from {args.src} -> {dst_root}"
    )

    recording = RecordingSession(
        repo_id=args.repo_id or f"{args.src.name}-512",
        root=dst_root,
        task=task_by_index[0],
        fps=fps,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        image_writer_threads=args.image_writer_threads,
    )
    recording.create_dataset((args.size, args.size, 3), (args.size, args.size, 3))

    # Built lazily once the first frame reveals each camera's stored resolution.
    undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    current_episode = 0

    try:
        for i in tqdm(range(len(states)), desc="Converting frames", unit="frame"):
            episode = int(episode_indices[i])
            if episode != current_episode:
                recording.save_episode(metadata_by_episode[current_episode])
                current_episode = episode

            frame: dict[str, Any] = {
                "observation.state": np.asarray(states[i], np.float32),
                "action": np.asarray(actions[i], np.float32),
                "task": task_by_index[int(task_indices[i])],
            }
            for feature, camera in FEATURE_TO_CAMERA.items():
                rgb = next(streams[feature])
                if camera not in undistort_maps:
                    h, w = rgb.shape[:2]
                    undistort_maps[camera] = build_undistort_map(
                        intrinsics_by_camera[camera], w, h, cv2
                    )
                frame[feature] = transform_frame(rgb, undistort_maps[camera], args.size, cv2)

            recording.dataset.add_frame(frame)

            dropped = recording.dropped_frame_count()
            if dropped:
                raise RuntimeError(
                    f"Streaming video encoder dropped {dropped} frame(s); the video would "
                    "desync from the recorded rows. Use --vcodec auto or "
                    "--no-streaming-encoding."
                )

        recording.save_episode(metadata_by_episode[current_episode])
    finally:
        recording.finalize()

    print(f"Done. Wrote {dst_root}")


if __name__ == "__main__":
    main()
