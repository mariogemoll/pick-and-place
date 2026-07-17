#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tile the live camera views recorded by ``run_policy_real.py --record-video``.

A ``--record-video`` run writes one flat directory holding ``wrist_live.mp4``,
``overhead_live.mp4`` and, with ``--workspace-camera``, ``workspace_live.mp4``.
The views share a clock. With the default three cameras, this script uses the
workspace view as a 480x360 main view and stacks the wrist and overhead views at
240x180 beside it. Other camera selections use a side-by-side row. Multiple runs
are joined in the supplied order.

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


DEFAULT_CAMERAS = ("workspace", "overhead", "wrist")


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


def parse_size(value: str) -> tuple[int, int]:
    """Parse an even ``WIDTHxHEIGHT`` cell size suitable for yuv420p."""
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("width and height must be positive even numbers")
    return width, height


def parse_trim(value: str) -> tuple[float, float]:
    """Parse a ``START:END`` trim range in seconds."""
    try:
        start, end = (float(part) for part in value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be START:END in seconds") from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("must have 0 <= START < END")
    return start, end


def build_command(
    runs: list[list[tuple[str, Path]]],
    cell_size: tuple[int, int],
    workspace_size: tuple[int, int],
    fps: float,
    trim_windows: list[tuple[float, float]],
    crf: int,
    preset: str,
    output: Path,
    *,
    still: bool = False,
) -> list[str]:
    """Build the ffmpeg invocation that tiles and joins one or more runs.

    With ``still``, write a poster from the first frame after the trim window.
    """
    videos = runs[0]
    cell_width, cell_height = cell_size
    workspace_width, workspace_height = workspace_size
    workspace_layout = len(videos) == 3 and any(name == "workspace" for name, _ in videos)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y"]
    for run in runs:
        for _, path in run:
            command.extend(("-i", str(path)))
    filters: list[str] = []
    run_view_labels: list[list[str]] = []
    for run_index, run in enumerate(runs):
        labels: list[str] = []
        for camera_index, (name, _) in enumerate(run):
            label = f"r{run_index}v{camera_index}"
            width, height = (
                (workspace_width, workspace_height)
                if workspace_layout and name == "workspace"
                else (cell_width, cell_height)
            )
            input_index = run_index * len(videos) + camera_index
            filters.append(
                f"[{input_index}:v]fps={fps:g},"
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1,"
                f"trim=start={trim_windows[run_index][0]:.9f}:end={trim_windows[run_index][1]:.9f},"
                f"setpts=PTS-STARTPTS[{label}]"
            )
            labels.append(f"[{label}]")
        run_view_labels.append(labels)
    view_labels = []
    for camera_index in range(len(videos)):
        label = f"v{camera_index}"
        filters.append(
            "".join(labels[camera_index] for labels in run_view_labels)
            + f"concat=n={len(runs)}:v=1:a=0[{label}]"
        )
        view_labels.append(f"[{label}]")
    if workspace_layout:
        small_index = 0
        layout_parts = []
        for name, _ in videos:
            if name == "workspace":
                layout_parts.append("0_0")
            else:
                layout_parts.append(f"{workspace_width}_{small_index * cell_height}")
                small_index += 1
        layout = "|".join(layout_parts)
    else:
        layout = "|".join(f"{index * cell_width}_0" for index in range(len(videos)))
    filters.append(
        "".join(view_labels)
        + f"xstack=inputs={len(view_labels)}:layout={layout}:fill=black,"
        + "setpts=PTS-STARTPTS[video]"
    )
    command.extend(("-filter_complex", ";".join(filters), "-map", "[video]"))
    if still:
        command.extend(("-frames:v", "1", "-c:v", "png", "-update", "1", str(output)))
    else:
        command.extend(
            (
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-profile:v",
                "high",
                "-level:v",
                "4.1",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            )
        )
    return command


def scaled_size(
    size: tuple[int, int], source_cell: tuple[int, int], target_cell: tuple[int, int]
) -> tuple[int, int]:
    """Scale a layout dimension to follow a requested secondary-view size."""
    width = max(2, round(size[0] * target_cell[0] / source_cell[0] / 2) * 2)
    height = max(2, round(size[1] * target_cell[1] / source_cell[1] / 2) * 2)
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="run_policy_real --record-video directories, joined in this order",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output MP4 (required for multiple runs)"
    )
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="comma-separated camera views; with workspace plus two others, the others stack in this order (default: workspace,overhead,wrist)",
    )
    parser.add_argument(
        "--cell",
        type=parse_size,
        default=(240, 180),
        help="secondary-view size in the default three-camera layout (default: 240x180)",
    )
    parser.add_argument(
        "--workspace-size",
        type=parse_size,
        default=(480, 360),
        help="workspace-view size in the default three-camera layout (default: 480x360)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="output frame rate (default: 30)")
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="H.264 constant-quality value, 0-51; lower is higher quality and larger (default: 20)",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        help="H.264 encoder speed/compression tradeoff (default: medium)",
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="also write lossless poster.png (the tiled first frame) beside the output",
    )
    parser.add_argument(
        "--poster-cell",
        type=parse_size,
        default=None,
        help="secondary-view size for --poster (default: --cell)",
    )
    parser.add_argument(
        "--trim",
        type=parse_trim,
        action="append",
        metavar="START:END",
        help="trim one run to this time range in seconds; repeat once per run",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    if not camera_names:
        parser.error("--cameras must name at least one camera")
    if len(set(camera_names)) != len(camera_names):
        parser.error("--cameras must not repeat a camera")

    runs: list[list[tuple[str, Path]]] = []
    for run in args.runs:
        videos = [(name, run / f"{name}_live.mp4") for name in camera_names]
        missing = [name for name, path in videos if not path.is_file()]
        if missing:
            parser.error(f"{run}: missing video(s) for {', '.join(missing)}")
        runs.append(videos)

    durations = [min(video_duration(path) for _, path in videos) for videos in runs]
    if args.trim is None:
        trim_windows = [(0.0, duration) for duration in durations]
    elif len(args.trim) != len(runs):
        parser.error("--trim must be supplied once for each run")
    else:
        trim_windows = args.trim
        for run, (start, end), duration in zip(args.runs, trim_windows, durations):
            if end > duration:
                parser.error(
                    f"{run}: trim end {end:g}s exceeds the shortest camera video ({duration:.3f}s)"
                )
    if args.out is None and len(args.runs) > 1:
        parser.error("--out is required when combining multiple runs")
    output = args.out if args.out is not None else args.runs[0] / "combined.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        runs,
        args.cell,
        args.workspace_size,
        args.fps,
        trim_windows,
        args.crf,
        args.preset,
        output,
    )
    subprocess.run(command, check=True)
    print(f"Wrote {output} ({len(args.runs)} run(s), {len(camera_names)} view(s))")
    if args.poster:
        poster_path = output.with_name("poster.png")
        poster_cell = args.poster_cell if args.poster_cell is not None else args.cell
        poster_workspace_size = scaled_size(args.workspace_size, args.cell, poster_cell)
        subprocess.run(
            build_command(
                runs,
                poster_cell,
                poster_workspace_size,
                args.fps,
                trim_windows,
                args.crf,
                args.preset,
                poster_path,
                still=True,
            ),
            check=True,
        )
        print(f"Wrote {poster_path}")


if __name__ == "__main__":
    main()
