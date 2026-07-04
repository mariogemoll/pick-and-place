#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Convert a recorded real LeRobotDataset to a rectified fixed-resolution format.

A real recording stores each camera at its native, lens-distorted resolution
(overhead 1920x1080, wrist 1280x720). Trained policies instead expect a fixed,
rectified pinhole view: the sim recordings and the VLA input are 512x512
squares, while an ACT policy typically wants 640x480. This script bridges the
two: every camera frame is undistorted with its calibrated intrinsics,
center-cropped to the requested output aspect ratio (keeping the full image
height when the output is no taller than it is wide), and resized to
``--width`` x ``--height``, so a converted real frame matches the target
pinhole geometry pixel for pixel. State, action, task and timing are copied
through unchanged, as is every other per-episode metadata column already
present on the source dataset (pickup/placement checks, cube start/target pose,
success, ...).

The geometry mirrors how the sim sets its camera field of view: the rectified
pinhole uses focal length ``fy`` on both axes with the principal point at the
image center, so the vertical FOV is ``2*atan((h/2)/fy)`` -- exactly the angle
``SimCameraRig`` feeds MuJoCo. A square output keeps that full vertical FOV on
both axes; a wider output (e.g. 640x480) keeps the same vertical FOV and shows
a proportionally wider horizontal slice.

Videos are decoded sequentially with PyAV (one straight pass per file rather
than a per-frame seek), so the run is bound by raw decode speed. The v3 dataset
stores each episode's frames as one contiguous, in-order segment of its video
file, so decoding the files in order yields frames in lockstep with the numeric
rows read from the data parquet.

Pass ``--episodes-file`` to convert only a subset of the source episodes
(their output indices renumber accordingly); produce that list with
``select_episodes.py`` to export and convert, say, only the successful
episodes in a single pass rather than filtering into an intermediate dataset
first.

Examples:

    # 512x512 square (VLA)
    python py/scripts/convert_dataset_resolution.py \
        --src datasets/20260702 --width 512 --height 512

    # 640x480 (ACT)
    python py/scripts/convert_dataset_resolution.py \
        --src datasets/20260702 --width 640 --height 480

    # Only the successful episodes, converted to 512x512
    python py/scripts/select_episodes.py --src datasets/20260702 \
        | python py/scripts/convert_dataset_resolution.py \
            --src datasets/20260702 --width 512 --height 512 --episodes-file -
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Generator
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


def read_episode_indices(path: Path) -> set[int]:
    """Parse source ``episode_index`` values from ``path`` (or stdin for ``-``)."""
    import re
    import sys

    text = sys.stdin.read() if str(path) == "-" else path.read_text()
    return {int(tok) for tok in re.split(r"[\s,]+", text.strip()) if tok}


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


def video_frames(paths: list[Path]) -> Generator[Any, None, None]:
    """Yield raw PyAV video frames from ``paths`` in order."""
    import av

    for path in paths:
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                # LeRobot's background video encoder restores FFmpeg's default stderr
                # callback when it starts, so re-assert the quiet level on each frame
                # to keep libswscale's benign per-frame "no SIMD path for yuv420p->rgb24"
                # note out of the output.
                av.logging.set_level(av.logging.ERROR)
                yield frame
        finally:
            container.close()


class VideoFrameReader:
    """Sequential reader that can cheaply advance past frames we will drop."""

    def __init__(self, paths: list[Path]) -> None:
        self._frames = video_frames(paths)

    def read_rgb(self) -> np.ndarray:
        return next(self._frames).to_ndarray(format="rgb24")

    def skip(self) -> None:
        next(self._frames)

    def close(self) -> None:
        self._frames.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source LeRobotDataset root")
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="output dataset root (default: <src>-<width>x<height> alongside the source)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="repo id for the output dataset (default: <src dir name>-<width>x<height>)",
    )
    parser.add_argument("--width", type=int, default=SQUARE_SIZE, help="output width (px)")
    parser.add_argument("--height", type=int, default=SQUARE_SIZE, help="output height (px)")
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
    parser.add_argument(
        "--encoder-queue-maxsize",
        type=int,
        default=3000,
        help=(
            "frames the streaming encoder may buffer per camera before raising on a drop "
            "(default: 3000, much deeper than live recording's 300 since this is an "
            "offline batch job with no real-time constraint)"
        ),
    )
    parser.add_argument(
        "--episodes-file",
        type=Path,
        default=None,
        help=(
            "file listing the source episode_index values to keep (whitespace/comma/newline "
            "separated, '-' for stdin); only those episodes are converted and their output "
            "indices renumber accordingly. Default: convert every episode. Produce this list "
            "with select_episodes.py to export only e.g. the successful episodes."
        ),
    )
    args = parser.parse_args()
    include_episodes = (
        read_episode_indices(args.episodes_file) if args.episodes_file is not None else None
    )
    suffix = f"{args.width}x{args.height}"

    import cv2
    import pandas as pd
    from tqdm import tqdm

    dst_root = args.dst if args.dst is not None else args.src.with_name(f"{args.src.name}-{suffix}")
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
    # A string column with missing values on some episodes (e.g. a check that
    # only newer episodes have) stores those as pandas' float NaN, not a
    # string. Episode metadata is written to the output dataset in batches of
    # consecutive episodes, and pyarrow infers each batch's column type from
    # its own values: a batch mixing NaN with real strings infers a type
    # pyarrow rejects outright, while a batch that is *entirely* NaN would
    # infer ``null``, which then conflicts with the ``string`` type an
    # earlier, non-empty batch already established for the same column.
    # Filling with an empty string instead of ``None`` sidesteps both cases,
    # since every value is then a real ``str`` and the column always infers
    # as ``string`` no matter how episodes land in a batch. Numeric columns
    # don't need this: their native NaN already infers as ``float64``
    # consistently whether or not a batch happens to be all-missing.
    for col in metadata_columns:
        if pd.api.types.is_string_dtype(episodes[col]):
            episodes[col] = episodes[col].astype(object).where(episodes[col].notna(), "")
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
    row_indices = rows["index"].to_numpy()

    streams = {
        feature: VideoFrameReader(
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

    available = {int(e) for e in episodes["episode_index"]}
    if include_episodes is not None:
        unknown = sorted(include_episodes - available)
        if unknown:
            raise SystemExit(f"--episodes-file lists episode(s) not in {args.src}: {unknown}")
        included_count = len(include_episodes)
        include_note = f" (keeping {included_count} of {len(available)} episode(s))"
    else:
        included_count = len(available)
        include_note = ""
    keep_rows = np.array(
        [
            include_episodes is None or int(episode) in include_episodes
            for episode in episode_indices
        ],
        dtype=bool,
    )
    kept_frame_count = int(keep_rows.sum())
    frame_note = (
        "" if kept_frame_count == len(rows) else f" (scanning {len(rows)} source frame(s))"
    )
    print(
        f"Converting {included_count} episode(s), {kept_frame_count} frame(s) to {suffix} "
        f"from {args.src} -> {dst_root}{include_note}{frame_note}"
    )

    recording = RecordingSession(
        repo_id=args.repo_id or f"{args.src.name}-{suffix}",
        root=dst_root,
        task=task_by_index[0],
        fps=fps,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        image_writer_threads=args.image_writer_threads,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
    )
    image_shape = (args.height, args.width, 3)
    recording.create_dataset(image_shape, image_shape)

    # Built lazily once the first frame reveals each camera's stored resolution.
    undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    current_episode = int(episode_indices[0])
    episode_has_frames = False
    # A source episode can have more rows than real video frames (e.g. a
    # corrupted merge that duplicated some rows under the same global
    # ``index``); the video was only ever encoded once per unique ``index``.
    # Pulling a stream frame only when ``index`` actually advances keeps the
    # two camera streams correctly positioned for every later episode
    # regardless of such duplicates, instead of desyncing partway through.
    last_row_index: int | None = None
    last_processed: dict[str, np.ndarray] = {}

    try:
        progress_desc = (
            "Converting frames" if kept_frame_count == len(rows) else "Scanning frames"
        )
        for i in tqdm(range(len(states)), desc=progress_desc, unit="frame"):
            episode = int(episode_indices[i])
            if episode != current_episode:
                if episode_has_frames:
                    recording.save_episode(metadata_by_episode[current_episode])
                current_episode = episode
                episode_has_frames = False

            row_index = int(row_indices[i])
            keep_row = bool(keep_rows[i])
            if row_index != last_row_index:
                pulled: dict[str, np.ndarray] = {}
                for feature, camera in FEATURE_TO_CAMERA.items():
                    if not keep_row:
                        streams[feature].skip()
                        continue
                    rgb = streams[feature].read_rgb()
                    if camera not in undistort_maps:
                        h, w = rgb.shape[:2]
                        undistort_maps[camera] = build_undistort_map(
                            intrinsics_by_camera[camera], w, h, cv2
                        )
                    pulled[feature] = transform_frame(
                        rgb, undistort_maps[camera], args.width, args.height, cv2
                    )
                last_processed = pulled
                last_row_index = row_index

            if not keep_row:
                continue

            frame: dict[str, Any] = {
                "observation.state": np.asarray(states[i], np.float32),
                "action": np.asarray(actions[i], np.float32),
                "task": task_by_index[int(task_indices[i])],
                **last_processed,
            }
            recording.dataset.add_frame(frame)
            episode_has_frames = True

            dropped = recording.dropped_frame_count()
            if dropped:
                raise RuntimeError(
                    f"Streaming video encoder dropped {dropped} frame(s); the video would "
                    "desync from the recorded rows. Use --vcodec auto or "
                    "--no-streaming-encoding."
                )

        if episode_has_frames:
            recording.save_episode(metadata_by_episode[current_episode])
    finally:
        for stream in streams.values():
            stream.close()
        recording.finalize()

    print(f"Done. Wrote {dst_root}")


if __name__ == "__main__":
    main()
