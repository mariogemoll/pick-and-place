#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Index recorded episodes for wrist-camera panorama building (fast first pass).

Walks one or more LeRobotDatasets and, for every episode, collects what the
accumulation pass needs: the per-frame joint states, the wrist-video path and its
start frame, and a brightness sample. Episodes are split into brightness bins
(lighting groups) so each panorama can be built from a single lighting condition.

The result is a single ``.npz`` handoff file consumed by
``accumulate_wrist_panorama.py``. This pass reads only a few frames per episode,
so it is quick; the slow ray-casting lives in the second script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

WRIST_FEATURE = "observation.images.wrist"


def _read_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def _read_episode_row(root: Path, episode_index: int) -> dict:
    for parquet_path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path)
        filtered = table.filter(pc.equal(table["episode_index"], episode_index))
        if filtered.num_rows:
            return filtered.slice(0, 1).to_pylist()[0]
    raise ValueError(f"episode {episode_index} not found under {root}")


def _episode_states(root: Path, info: dict, row: dict) -> np.ndarray:
    data_path = root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )
    table = pq.read_table(data_path, columns=["episode_index", "observation.state"])
    table = table.filter(pc.equal(table["episode_index"], row["episode_index"]))
    return np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)


def _wrist_video(root: Path, row: dict, fps: float) -> tuple[Path, int]:
    prefix = f"videos/{WRIST_FEATURE}"
    chunk = int(row[f"{prefix}/chunk_index"])
    file = int(row[f"{prefix}/file_index"])
    from_ts = float(row[f"{prefix}/from_timestamp"])
    path = root / "videos" / WRIST_FEATURE / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    return path, round(from_ts * fps)


def _episode_brightness(video_path: Path, start: int, length: int, samples: int = 5) -> float:
    """Mean luminance over a few evenly spaced frames — a cheap lighting proxy."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        idxs = np.linspace(start, start + max(0, length - 1), samples).round().astype(int)
        vals = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                vals.append(float(frame.mean()))
    finally:
        cap.release()
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="LeRobotDataset root(s)")
    parser.add_argument("--out", type=Path, required=True, help="output handoff .npz path")
    parser.add_argument(
        "--brightness-bins",
        type=int,
        default=1,
        help="split episodes into N brightness groups, one panorama each",
    )
    args = parser.parse_args()
    n_bins = max(1, args.brightness_bins)

    states: list[np.ndarray] = []
    video_paths: list[str] = []
    start_frames: list[int] = []
    brightness: list[float] = []

    for root in args.datasets:
        info = _read_info(root)
        fps = float(info.get("fps", 30))
        for ep in range(int(info.get("total_episodes", 0))):
            row = _read_episode_row(root, ep)
            ep_states = _episode_states(root, info, row)
            video_path, start = _wrist_video(root, row, fps)
            if not video_path.is_file():
                print(f"skip {root.name} ep {ep}: missing {video_path}")
                continue
            states.append(ep_states)
            video_paths.append(str(video_path))
            start_frames.append(start)
            brightness.append(_episode_brightness(video_path, start, len(ep_states)))
            print(f"{root.name} ep {ep}: {len(ep_states)} frames, brightness {brightness[-1]:.0f}")

    if not states:
        raise SystemExit("no episodes found")

    bright = np.array(brightness)
    if n_bins > 1:
        edges = np.quantile(bright, np.linspace(0, 1, n_bins + 1))
        bin_of = np.clip(np.digitize(bright, edges[1:-1]), 0, n_bins - 1)
    else:
        bin_of = np.zeros(len(states), dtype=int)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        states=np.array(states, dtype=object),
        video_paths=np.array(video_paths),
        start_frames=np.array(start_frames),
        brightness=bright,
        bin=bin_of,
        n_bins=n_bins,
    )

    print(f"\nIndexed {len(states)} episodes → {args.out}")
    print(f"brightness range {bright.min():.0f}..{bright.max():.0f}")
    for b in range(n_bins):
        in_bin = bright[bin_of == b]
        if in_bin.size:
            print(f"  bin {b}: {in_bin.size} eps, brightness {in_bin.min():.0f}..{in_bin.max():.0f}")


if __name__ == "__main__":
    main()
