# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Write MP4s of the frames a Diffusion Policy export actually trains on.

    python scripts/render_diffusion_policy_videos.py <dp_export> <output_dir> [--episodes 0-9]

The exported tensor is what the policy sees: decimated to the export frame
rate, resized to 96x96 and stitched into one channel-stacked block per frame.
Watching it back is the only way to check that end of the pipeline -- a camera
mixed up, a crop that cuts the cube off, or an aspect-fill that squashes the
table are all invisible in the source recordings and obvious here.

Each episode becomes one video with the cameras side by side, upscaled by
``--scale`` with nearest-neighbor so the pixels stay honest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pick_and_place.data.stored_npz import episode_bounds, memmap_stored_npz


def parse_episode_selection(text: str | None, count: int) -> list[int]:
    """Turn a ``0-9,12`` style selection into episode indices."""
    if text is None:
        return list(range(count))
    selected: list[int] = []
    for part in text.split(","):
        start, _, stop = part.partition("-")
        first = int(start)
        last = int(stop) if stop else first
        selected.extend(range(first, last + 1))
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError(f"episode selection {text!r} is out of range 0-{count - 1}")
    return selected


def split_cameras(frames: np.ndarray, camera_count: int) -> np.ndarray:
    """Lay a stack of NCHW channel-stacked frames out as one wide RGB strip."""
    count, channels, height, width = frames.shape
    if channels != 3 * camera_count:
        raise ValueError(f"{channels} channels do not split into {camera_count} RGB cameras")
    per_camera = frames.reshape(count, camera_count, 3, height, width)
    return np.ascontiguousarray(
        per_camera.transpose(0, 3, 1, 4, 2).reshape(count, height, camera_count * width, 3)
    )


def write_video(path: Path, frames: np.ndarray, fps: float, scale: int) -> None:
    """Encode an RGB frame stack, magnified by an integer factor."""
    import imageio_ffmpeg

    height, width = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width * scale, height * scale),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1))
    finally:
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="a Diffusion Policy export directory")
    parser.add_argument("output", type=Path, help="directory to write the MP4s into")
    parser.add_argument(
        "--episodes",
        help="which episodes to render, e.g. 0-9,42 (default: all of them)",
    )
    parser.add_argument("--scale", type=int, default=4, help="integer upscale factor (default: 4)")
    parser.add_argument("--fps", type=float, help="override the export frame rate")
    args = parser.parse_args()

    metadata = json.loads((args.export / "export.json").read_text())
    cameras = metadata["camera_features"]
    fps = args.fps if args.fps is not None else float(metadata["fps"])

    arrays = memmap_stored_npz(args.export / "train.npz")
    bounds = episode_bounds(arrays["traj_lengths"])
    episodes = parse_episode_selection(args.episodes, len(bounds))

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{' | '.join(cameras)} at {fps:g} fps, scale {args.scale}x -> {args.output}")
    total_frames = sum(int(stop - start) for start, stop in bounds[episodes])
    with tqdm(
        total=total_frames,
        desc="Render episodes",
        unit="frame",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for episode in episodes:
            start, stop = bounds[episode]
            frames = split_cameras(np.asarray(arrays["images"][start:stop]), len(cameras))
            write_video(args.output / f"episode_{episode:04d}.mp4", frames, fps, args.scale)
            progress.update(len(frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
