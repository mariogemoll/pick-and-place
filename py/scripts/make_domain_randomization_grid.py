# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Render synchronized canonical-versus-randomized episode video grids.

Each grid tile represents one shared episode index. Its columns are wrist and
overhead views; its blue-bordered top row is canonical and its orange-bordered
bottom row is randomized. This makes a small fixed-seed recording suitable for
inspecting the visual randomization before generating a larger training dataset.

Example:
    python scripts/make_domain_randomization_grid.py \\
        --canonical-dataset datasets/sim_canonical \\
        --randomized-dataset datasets/sim_act_mild_v1 \\
        --episodes 0,1,2,3 --cols 2 --out domain_randomization_grid.mp4
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CAMERAS = ("observation.images.wrist", "observation.images.overhead")


@dataclass(frozen=True)
class EpisodeClip:
    """The slice of a shared LeRobot video containing one episode."""

    path: Path
    start: float
    duration: float


def find_ffmpeg() -> str:
    """Return a working ffmpeg binary (preferring the local Homebrew builds)."""
    for candidate in (
        "/opt/homebrew/opt/ffmpeg@8/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg/bin/ffmpeg",
        shutil.which("ffmpeg"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("No ffmpeg binary found")


def load_episodes(dataset: Path) -> pd.DataFrame:
    """Load LeRobot v3 episode metadata indexed by global episode index."""
    parts = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No episode metadata under {dataset}/meta/episodes")
    episodes = pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)
    if episodes["episode_index"].duplicated().any():
        raise ValueError(f"{dataset}: duplicate episode_index values")
    return episodes.set_index("episode_index", drop=False)


def episode_clip(dataset: Path, row: pd.Series, camera: str) -> EpisodeClip:
    """Return the video slice for one camera view of an episode."""
    prefix = f"videos/{camera}"
    path = (
        dataset
        / "videos"
        / camera
        / f"chunk-{int(row[f'{prefix}/chunk_index']):03d}"
        / f"file-{int(row[f'{prefix}/file_index']):03d}.mp4"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    start = float(row[f"{prefix}/from_timestamp"])
    end = float(row[f"{prefix}/to_timestamp"])
    if end <= start:
        raise ValueError(f"{path}: non-positive episode duration ({start} to {end})")
    return EpisodeClip(path=path, start=start, duration=end - start)


def choose_episodes(
    canonical: pd.DataFrame,
    randomized: pd.DataFrame,
    requested: str | None,
    count: int,
    seed: int | None,
) -> list[int]:
    """Choose shared episode indices, preserving an explicit requested order."""
    shared = set(canonical.index) & set(randomized.index)
    if requested:
        selected = [int(value) for value in requested.split(",")]
        missing = [index for index in selected if index not in shared]
        if missing:
            raise ValueError(f"Episodes not present in both datasets: {missing}")
        return selected
    if len(shared) < count:
        raise ValueError(
            f"Only {len(shared)} episode index(es) are shared by the datasets; requested {count}"
        )
    pool = sorted(shared)
    random.Random(seed).shuffle(pool)
    return pool[:count]


def encode_args(out: Path, crf: int | None) -> list[str]:
    """Return codec arguments selected by the output extension."""
    if out.suffix.lower() == ".webm":
        return [
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            str(crf if crf is not None else 34),
            "-row-mt",
            "1",
        ]
    return [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf if crf is not None else 23),
        "-movflags",
        "+faststart",
    ]


def build_command(
    ffmpeg: str,
    canonical_root: Path,
    canonical: pd.DataFrame,
    randomized_root: Path,
    randomized: pd.DataFrame,
    episode_indices: list[int],
    rows: int,
    cols: int,
    cell_width: int,
    cell_height: int,
    duration: float,
    fps: int,
    crf: int | None,
    out: Path,
) -> list[str]:
    """Build the ffmpeg command for the four-view tile of every selected episode."""
    inputs: list[str] = []
    filters: list[str] = []
    tile_labels: list[str] = []
    input_index = 0

    for tile_index, episode_index in enumerate(episode_indices):
        views: list[str] = []
        for domain, root, episodes in (
            ("canonical", canonical_root, canonical),
            ("randomized", randomized_root, randomized),
        ):
            for camera in CAMERAS:
                clip = episode_clip(root, episodes.loc[episode_index], camera)
                inputs.extend(
                    ("-ss", f"{clip.start:.6f}", "-t", f"{clip.duration:.6f}", "-i", str(clip.path))
                )
                view = f"tile{tile_index}_{domain}_{camera.rsplit('.', maxsplit=1)[-1]}"
                border = "dodgerblue" if domain == "canonical" else "darkorange"
                filters.append(
                    f"[{input_index}:v]scale={cell_width}:{cell_height},setsar=1,fps={fps},"
                    f"trim=duration={duration:.6f},tpad=stop_mode=clone:stop_duration={duration:.6f},"
                    f"drawbox=x=0:y=0:w=iw:h=ih:color={border}:thickness=4[{view}]"
                )
                views.append(f"[{view}]")
                input_index += 1
        top, bottom = f"tile{tile_index}_top", f"tile{tile_index}_bottom"
        tile = f"tile{tile_index}"
        filters.extend((
            f"{views[0]}{views[1]}hstack=inputs=2[{top}]",
            f"{views[2]}{views[3]}hstack=inputs=2[{bottom}]",
            f"[{top}][{bottom}]vstack=inputs=2[{tile}]",
        ))
        tile_labels.append(f"[{tile}]")

    layout = "|".join(
        f"{(index % cols) * cell_width * 2}_{(index // cols) * cell_height * 2}"
        for index in range(len(episode_indices))
    )
    filters.append(
        "".join(tile_labels) + f"xstack=inputs={len(tile_labels)}:layout={layout}:fill=black[grid]"
    )
    return [
        ffmpeg,
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[grid]",
        "-t",
        f"{duration:.6f}",
        "-r",
        str(fps),
        *encode_args(out, crf),
        str(out),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--randomized-dataset", type=Path, required=True)
    parser.add_argument("--episodes", help="Comma-separated shared episode indices")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--cell", default="320x240", help="Per-camera cell size WxH")
    parser.add_argument("--duration", type=float, default=15.0, help="Grid duration in seconds")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0, help="Episode selection seed")
    parser.add_argument("--crf", type=int, help="Constant-quality level (23 mp4, 34 webm by default)")
    parser.add_argument("--out", type=Path, default=Path("domain_randomization_grid.mp4"))
    args = parser.parse_args()

    if args.rows < 1 or args.cols < 1 or args.duration <= 0 or args.fps < 1:
        parser.error("--rows, --cols, --duration, and --fps must be positive")
    try:
        cell_width, cell_height = (int(value) for value in args.cell.lower().split("x"))
    except ValueError as exc:
        raise ValueError("--cell must be WIDTHxHEIGHT") from exc
    if cell_width < 2 or cell_height < 2 or cell_width % 2 or cell_height % 2:
        parser.error("--cell dimensions must be positive even integers")

    canonical = load_episodes(args.canonical_dataset)
    randomized = load_episodes(args.randomized_dataset)
    selected = choose_episodes(
        canonical, randomized, args.episodes, args.rows * args.cols, args.seed
    )
    print(f"Rendering shared episode indices: {', '.join(str(index) for index in selected)}")
    command = build_command(
        find_ffmpeg(), args.canonical_dataset, canonical, args.randomized_dataset, randomized,
        selected, args.rows, args.cols, cell_width, cell_height, args.duration, args.fps,
        args.crf, args.out,
    )
    subprocess.run(command, check=True)
    print(
        f"Wrote {args.out}; each tile is blue canonical above orange randomized, "
        "wrist left of overhead."
    )


if __name__ == "__main__":
    main()
