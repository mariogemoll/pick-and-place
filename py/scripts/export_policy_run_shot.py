#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export one still per camera from a ``run_policy_real.py --record-video`` run.

The run directory contains one ``*_live.mp4`` per camera.  This script takes
the first frame at or after the requested shared-clock time and writes one
native-resolution PNG per selected camera.

Example:
    python scripts/export_policy_run_shot.py episodes/20260712_212322 \\
        --time 12.5 --out episodes/20260712_212322/shots
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg


DEFAULT_CAMERAS = ("workspace", "overhead", "wrist")


def video_duration(path: Path) -> float:
    """Return the video-stream duration, excluding any longer audio stream."""
    with av.open(str(path)) as container:
        stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if stream is None or stream.duration is None:
            raise ValueError(f"{path}: could not determine video duration")
        return float(stream.duration * stream.time_base)


def build_command(
    video: Path,
    time: float,
    output: Path,
) -> list[str]:
    """Build an ffmpeg command that extracts the first frame at ``time``."""
    # trim makes this a post-input seek, so the selected frame is aligned to
    # the recorded timestamp instead of the nearest preceding keyframe.
    filter_graph = (
        f"[0:v]trim=start={time:.9f}:end={time + 1:.9f},"
        "select='eq(n\\,0)',setpts=PTS-STARTPTS[shot]"
    )
    return [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[shot]",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run_policy_real --record-video directory")
    parser.add_argument("--time", type=float, required=True, help="shared-camera time in seconds")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: RUN/shots_TIME)",
    )
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="comma-separated camera views (default: workspace,overhead,wrist)",
    )
    args = parser.parse_args()

    if args.time < 0:
        parser.error("--time must be non-negative")
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    if not camera_names:
        parser.error("--cameras must name at least one camera")
    if len(set(camera_names)) != len(camera_names):
        parser.error("--cameras must not repeat a camera")
    videos = [(name, args.run / f"{name}_live.mp4") for name in camera_names]
    missing = [name for name, path in videos if not path.is_file()]
    if missing:
        parser.error(f"{args.run}: missing video(s) for {', '.join(missing)}")

    duration = min(video_duration(path) for _, path in videos)
    if args.time >= duration:
        parser.error(f"--time {args.time:g}s is outside the shortest camera video ({duration:.3f}s)")
    output_dir = args.out if args.out is not None else args.run / f"shots_{args.time:.3f}s"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, video in videos:
        output = output_dir / f"{name}.png"
        subprocess.run(build_command(video, args.time, output), check=True)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
