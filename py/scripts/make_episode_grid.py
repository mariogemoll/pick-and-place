# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tile dataset episodes into a continuously-playing grid video.

Episodes in a LeRobot v3 dataset are concatenated into shared video files; each
episode's slice is described by a from/to timestamp in the episodes metadata.

Each grid slot plays a never-repeating stream of episodes back to back: the
first episode in a slot starts at a random point in time, and whenever an
episode ends the next one takes over, until the requested overall duration is
reached. Only the overhead camera is used.

Example:
    python scripts/make_episode_grid.py \
        --dataset ../datasets/640x480/combined \
        --rows 3 --cols 3 --cell 160x120 --duration 20 \
        --out episode_grid.mp4
"""

import argparse
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CAMERA = "observation.images.overhead"


@dataclass
class Clip:
    path: Path
    seek: float       # seek point within the shared video file (seconds)
    duration: float   # how much of the file to play from `seek`


def find_ffmpeg() -> str:
    """Return a working ffmpeg binary (prefers a non-broken Homebrew build)."""
    for candidate in (
        "/opt/homebrew/opt/ffmpeg@8/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg/bin/ffmpeg",
        shutil.which("ffmpeg"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("No ffmpeg binary found")


def load_episodes(dataset: Path) -> pd.DataFrame:
    parts = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No episode metadata under {dataset}/meta/episodes")
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def _flatten(obj):
    """Yield every scalar in an arbitrarily/raggedly nested array or list."""
    if isinstance(obj, np.ndarray):
        for e in obj.ravel():
            yield from _flatten(e)
    elif isinstance(obj, (list, tuple)):
        for e in obj:
            yield from _flatten(e)
    else:
        yield float(obj)


def episode_brightness(df: pd.DataFrame) -> pd.Series:
    """Mean overhead-image brightness per episode (0..1), from stored stats.

    The per-channel means are stored as raggedly nested object arrays, so
    flatten to scalars before averaging.
    """
    col = f"stats/{CAMERA}/mean"
    return df[col].apply(lambda v: float(np.mean(list(_flatten(v)))))


def video_path(dataset: Path, chunk_index: int, file_index: int) -> Path:
    return (
        dataset
        / "videos"
        / CAMERA
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def episode_span(dataset: Path, row) -> tuple[Path, float, float]:
    """Return (video path, from_ts, to_ts) for an episode's overhead slice."""
    path = video_path(
        dataset,
        int(row[f"videos/{CAMERA}/chunk_index"]),
        int(row[f"videos/{CAMERA}/file_index"]),
    )
    return path, float(row[f"videos/{CAMERA}/from_timestamp"]), \
        float(row[f"videos/{CAMERA}/to_timestamp"])


def build_slot_playlist(
    dataset: Path,
    rows_by_idx: pd.DataFrame,
    order: list[int],
    duration: float,
    rng: random.Random,
) -> list[Clip]:
    """Build a back-to-back clip stream for one slot, covering `duration`.

    The first episode starts at a random offset (so slots are out of phase);
    subsequent episodes play in full until the stream is long enough.
    """
    clips: list[Clip] = []
    total = 0.0
    first = True
    i = 0
    while total < duration:
        ep = order[i % len(order)]
        i += 1
        path, start, end = episode_span(dataset, rows_by_idx.loc[ep])
        length = end - start
        if first:
            # Start somewhere in the first ~70% so the opening clip isn't a
            # tiny sliver, then play to the end of that episode.
            start += rng.uniform(0.0, length * 0.7)
            length = end - start
            first = False
        clips.append(Clip(path, start, length))
        total += length
    return clips


def encode_args(out: Path, crf: int | None) -> list[str]:
    """Codec/quality flags chosen from the output file extension.

    .webm -> VP9 (smaller for web); anything else -> H.264 mp4. In both cases
    CRF is constant-quality: lower = better quality and larger file.
    """
    if out.suffix.lower() == ".webm":
        return [
            "-c:v", "libvpx-vp9",
            "-b:v", "0",
            "-crf", str(crf if crf is not None else 34),
            "-row-mt", "1",
            "-pix_fmt", "yuv420p",
        ]
    return [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf if crf is not None else 23),
        "-movflags", "+faststart",
    ]


def build_command(
    ffmpeg: str,
    slots: list[list[Clip]],
    rows: int,
    cols: int,
    cell_w: int,
    cell_h: int,
    fps: int,
    duration: float,
    crf: int | None,
    out: Path,
) -> list[str]:
    inputs: list[str] = []
    filters: list[str] = []
    slot_labels: list[str] = []
    idx = 0  # running ffmpeg input index

    for s, clips in enumerate(slots):
        seg_labels = []
        for clip in clips:
            # -ss/-t as input options => frame-accurate cut, PTS reset to 0.
            inputs += [
                "-ss", f"{clip.seek:.6f}",
                "-t", f"{clip.duration:.6f}",
                "-i", str(clip.path),
            ]
            seg = f"s{s}p{len(seg_labels)}"
            filters.append(
                f"[{idx}:v]scale={cell_w}:{cell_h},setsar=1,fps={fps}[{seg}]"
            )
            seg_labels.append(f"[{seg}]")
            idx += 1
        # Concatenate this slot's episodes, then trim to the shared duration so
        # every slot ends at the same instant.
        slot = f"slot{s}"
        filters.append(
            "".join(seg_labels)
            + f"concat=n={len(seg_labels)}:v=1:a=0,"
            + f"trim=duration={duration:.6f},setpts=PTS-STARTPTS[{slot}]"
        )
        slot_labels.append(f"[{slot}]")

    layout = "|".join(
        f"{(i % cols) * cell_w}_{(i // cols) * cell_h}" for i in range(len(slots))
    )
    filters.append(
        "".join(slot_labels)
        + f"xstack=inputs={len(slots)}:layout={layout}:fill=black[grid]"
    )

    return [
        ffmpeg,
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map", "[grid]",
        "-r", str(fps),
        *encode_args(out, crf),
        str(out),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Path to a LeRobot v3 dataset directory")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cell", default="160x120",
                    help="Cell size WxH; source is 4:3 so keep e.g. 160x120")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="Overall grid duration in seconds")
    ap.add_argument("--episodes", default=None,
                    help="Comma-separated episode pool; default is all episodes")
    ap.add_argument("--min-brightness", type=float, default=0.43,
                    help="Drop episodes dimmer than this mean brightness (0..1); "
                         "0 keeps all")
    ap.add_argument("--success-only", action="store_true",
                    help="Keep only episodes whose cube landed within 4cm of target")
    ap.add_argument("--driver", default=None,
                    help="Comma-separated driver(s) to keep, e.g. 'analytic' "
                         "(scripted) or 'teleop' (human); default keeps all")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=None,
                    help="Constant-quality level (lower=better/larger); default "
                         "23 for .mp4/H.264, 34 for .webm/VP9")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible randomization")
    ap.add_argument("--out", type=Path, default=Path("episode_grid.mp4"),
                    help="Output file; .webm => VP9 (smaller for web), else mp4")
    args = ap.parse_args()

    cell_w, cell_h = (int(v) for v in args.cell.lower().split("x"))
    if cell_w % 2 or cell_h % 2:
        raise ValueError("Cell width and height must be even for yuv420p")
    n_slots = args.rows * args.cols

    df = load_episodes(args.dataset)
    rows_by_idx = df.set_index("episode_index")

    keep = pd.Series(True, index=df.index)
    if args.min_brightness > 0:
        keep &= episode_brightness(df) >= args.min_brightness
    if args.success_only:
        placed = np.hypot(df["cube_end_x"] - df["target_x"],
                          df["cube_end_y"] - df["target_y"]) <= 0.04
        keep &= placed
    if args.driver:
        wanted = {d.strip() for d in args.driver.split(",")}
        keep &= df["driver"].isin(wanted)
    eligible = set(df.loc[keep, "episode_index"].tolist())

    if args.episodes:
        pool = [int(x) for x in args.episodes.split(",")]
        missing = set(pool) - set(df["episode_index"].tolist())
        if missing:
            raise ValueError(f"Episodes not in dataset: {sorted(missing)}")
        pool = [e for e in pool if e in eligible]
    else:
        pool = [e for e in df["episode_index"].tolist() if e in eligible]

    if len(pool) < n_slots:
        raise ValueError(f"Only {len(pool)} episodes pass filters but "
                         f"{n_slots} slots requested; relax --min-brightness")
    print(f"Pool: {len(pool)} episodes after filters")

    rng = random.Random(args.seed)
    # Each slot gets its own shuffled play order so streams stay out of sync.
    slots = [
        build_slot_playlist(
            args.dataset, rows_by_idx,
            rng.sample(pool, len(pool)), args.duration, rng,
        )
        for _ in range(n_slots)
    ]
    for s, clips in enumerate(slots):
        print(f"slot {s}: {len(clips)} episodes")

    cmd = build_command(
        find_ffmpeg(), slots, args.rows, args.cols,
        cell_w, cell_h, args.fps, args.duration, args.crf, args.out,
    )
    subprocess.run(cmd, check=True)
    print(f"Wrote {args.out} ({args.rows}x{args.cols}, "
          f"{cell_w}x{cell_h} cells, {args.duration:.0f}s)")


if __name__ == "__main__":
    main()
