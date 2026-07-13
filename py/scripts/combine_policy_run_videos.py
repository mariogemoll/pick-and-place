#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tile the live camera views recorded by ``run_policy_real.py --record-video``.

A ``--record-video`` run writes one flat directory holding ``wrist_live.mp4``,
``overhead_live.mp4`` and, with ``--workspace-camera``, ``workspace_live.mp4``.
The views share a clock, and each already carries the run's audio track (when
recorded with ``--record-audio``). This script scales the available views into a
single side-by-side row, trimmed to the shortest view, with one copy of the audio.

Example:
    python scripts/combine_policy_run_videos.py episodes/20260712_212322 \\
        --out episodes/20260712_212322/combined.mp4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg


DEFAULT_CAMERAS = ("wrist", "overhead", "workspace")


def video_duration(path: Path) -> float:
    """Return the video-stream duration of a live camera file in seconds.

    The muxed audio track runs a little longer than the frames, so the
    container duration would overshoot; the video stream is what the tiling
    should be trimmed to.
    """
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None or stream.duration is None:
            raise ValueError(f"{path}: could not determine video duration")
        return float(stream.duration * stream.time_base)


def has_audio(path: Path) -> bool:
    """Return whether a live camera file carries an audio stream."""
    with av.open(str(path)) as container:
        return any(stream.type == "audio" for stream in container.streams)


def parse_size(value: str) -> tuple[int, int]:
    """Parse an even ``WIDTHxHEIGHT`` cell size suitable for yuv420p."""
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("width and height must be positive even numbers")
    return width, height


def build_command(
    videos: list[tuple[str, Path]],
    cell_size: tuple[int, int],
    fps: float,
    duration: float,
    audio_index: int | None,
    output: Path,
) -> list[str]:
    """Build the ffmpeg invocation that scales, tiles and trims the views."""
    cell_width, cell_height = cell_size
    layout = "|".join(f"{index * cell_width}_0" for index in range(len(videos)))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y"]
    for _, path in videos:
        command.extend(("-i", str(path)))
    filters: list[str] = []
    view_labels: list[str] = []
    for index, _ in enumerate(videos):
        label = f"v{index}"
        filters.append(
            f"[{index}:v]fps={fps:g},"
            f"scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,"
            f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,setpts=PTS-STARTPTS[{label}]"
        )
        view_labels.append(f"[{label}]")
    filters.append(
        "".join(view_labels)
        + f"xstack=inputs={len(view_labels)}:layout={layout}:fill=black,"
        + f"trim=duration={duration:.9f},setpts=PTS-STARTPTS[video]"
    )
    if audio_index is not None:
        filters.append(
            f"[{audio_index}:a]atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[audio]"
        )
    command.extend(("-filter_complex", ";".join(filters), "-map", "[video]"))
    if audio_index is not None:
        command.extend(("-map", "[audio]", "-c:a", "aac"))
    command.extend(
        (
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        )
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="run_policy_real --record-video directory holding the *_live.mp4 files"
    )
    parser.add_argument("--out", type=Path, default=None, help="output MP4 (default: RUN/combined.mp4)")
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="comma-separated camera views in tile order (default: wrist,overhead,workspace)",
    )
    parser.add_argument("--cell", type=parse_size, default=(960, 540), help="tile size (default: 960x540)")
    parser.add_argument("--fps", type=float, default=30.0, help="output frame rate (default: 30)")
    parser.add_argument(
        "--no-audio", action="store_true", help="do not include the recorded audio in the combined video"
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    if not camera_names:
        parser.error("--cameras must name at least one camera")
    if len(set(camera_names)) != len(camera_names):
        parser.error("--cameras must not repeat a camera")

    videos: list[tuple[str, Path]] = []
    for name in camera_names:
        path = args.run / f"{name}_live.mp4"
        if path.is_file():
            videos.append((name, path))
    if not videos:
        parser.error(f"No *_live.mp4 files for {', '.join(camera_names)} found in {args.run}")

    duration = min(video_duration(path) for _, path in videos)
    audio_index = None
    if not args.no_audio:
        audio_index = next((index for index, (_, path) in enumerate(videos) if has_audio(path)), None)

    output = args.out if args.out is not None else args.run / "combined.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(videos, args.cell, args.fps, duration, audio_index, output)
    subprocess.run(command, check=True)
    audio_note = "" if audio_index is not None else " (no audio)"
    print(f"Wrote {output} ({len(videos)} view(s): {', '.join(name for name, _ in videos)}){audio_note}")


if __name__ == "__main__":
    main()
