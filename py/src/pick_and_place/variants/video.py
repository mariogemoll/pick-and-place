# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Writing a variant's videos so they sit beside a recording as equals.

A re-render is only comparable to the recording it came from if it is encoded
the way that recording was, so an episode's own video is asked what settings it
was written with rather than assumed to have used the current defaults. The same
reasoning drives :func:`encode_decode`, which measures the codec's own noise
floor: a recorded video is lossy, so even a pixel-perfect re-render differs from
it, and the difference that matters is the part above that floor.

The states, actions and every other recorded column are copied through
untouched. Only pixels change.
"""

from __future__ import annotations

import io
import json
import re
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from pick_and_place.rollout.sim import (
    OVERHEAD_CAMERA,
    WRIST_CAMERA,
)

#: Dataset feature name of each camera, and the MuJoCo camera it renders from.
CAMERA_FEATURES: dict[str, str] = {
    "observation.images.wrist": WRIST_CAMERA,
    "observation.images.overhead": OVERHEAD_CAMERA,
}

#: H.264 settings assumed when an episode's video carries no x264 options
#: string. The recorded datasets are written by LeRobot's streaming encoder at
#: these values; they are read back per episode rather than trusted blindly.
DEFAULT_X264_CRF = 30.0
DEFAULT_X264_KEYINT = 2

#: LeRobot samples every 6th pixel in each direction for image statistics,
#: which is what makes its recorded ``count`` exactly ``frames * 120 * 160``
#: for a 720x960 video. Matching the stride keeps recomputed statistics
#: comparable with the recorded ones.
STATS_PIXEL_STRIDE = 6


@dataclass(frozen=True)
class X264Settings:
    """The rate control an episode's videos were encoded with."""

    crf: float
    keyint: int

    def encoder_options(self) -> dict[str, str]:
        return {"crf": f"{self.crf:g}"}


@dataclass(frozen=True)
class FrameDiff:
    """Absolute per-pixel difference between two uint8 RGB images."""

    mean: float
    p99: float
    max: float

    @staticmethod
    def between(first: np.ndarray, second: np.ndarray) -> FrameDiff:
        if first.shape != second.shape:
            raise ValueError(f"cannot diff {first.shape} against {second.shape}")
        delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
        return FrameDiff(
            mean=float(delta.mean()),
            p99=float(np.percentile(delta, 99.0)),
            max=float(delta.max()),
        )


def episode_video_paths(episode_root: Path) -> dict[str, Path]:
    """Return each camera feature's single video file in a staged episode."""
    paths: dict[str, Path] = {}
    for feature in CAMERA_FEATURES:
        found = sorted((episode_root / "videos" / feature).glob("chunk-*/file-*.mp4"))
        if len(found) != 1:
            raise ValueError(
                f"{episode_root.name} must hold exactly one {feature} video, found {len(found)}"
            )
        paths[feature] = found[0]
    return paths


def episode_video_fps(episode_root: Path) -> int:
    """Frames per second the episode's videos were written at."""
    with (episode_root / "meta" / "info.json").open() as file:
        info = json.load(file)
    return int(info["fps"])


def x264_settings(video_path: Path, *, head_bytes: int = 512_000) -> X264Settings:
    """Read the rate control out of a video's embedded x264 options string.

    libx264 writes its full option list into the stream as user data, so an
    episode's own file states the CRF and keyframe interval it was produced
    with. Reading them back is what keeps a re-encode comparable to the
    recording even if the encoder's defaults change.
    """
    head = video_path.open("rb").read(head_bytes)
    crf = re.search(rb"[ -]crf=([0-9.]+)", head)
    keyint = re.search(rb"[ -]keyint=(\d+)", head)
    return X264Settings(
        crf=float(crf.group(1)) if crf else DEFAULT_X264_CRF,
        keyint=int(keyint.group(1)) if keyint else DEFAULT_X264_KEYINT,
    )


def decode_video(path: Path) -> Iterator[np.ndarray]:
    """Yield a video's frames as ``(H, W, 3)`` uint8 RGB arrays."""
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def video_frame_count(path: Path) -> int:
    """Frames the container reports, without decoding them."""
    import av

    with av.open(str(path)) as container:
        return container.streams.video[0].frames


class VideoWriter:
    """Write uint8 RGB frames as an H.264 MP4 matching a recorded episode's."""

    def __init__(
        self,
        target: Path | io.BytesIO,
        *,
        width: int,
        height: int,
        fps: int,
        settings: X264Settings,
    ) -> None:
        import av

        self._container = av.open(
            target if isinstance(target, io.BytesIO) else str(target),
            mode="w",
            format="mp4" if isinstance(target, io.BytesIO) else None,
        )
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.codec_context.time_base = Fraction(1, fps)
        self._stream.codec_context.gop_size = settings.keyint
        self._stream.codec_context.options = settings.encoder_options()
        self._frames = 0

    def write(self, rgb: np.ndarray) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = self._frames
        frame.time_base = self._stream.codec_context.time_base
        self._container.mux(self._stream.encode(frame))
        self._frames += 1

    @property
    def frames_written(self) -> int:
        return self._frames

    def close(self) -> None:
        self._container.mux(self._stream.encode(None))
        self._container.close()

    def __enter__(self) -> VideoWriter:  # noqa: PYI034 - a context manager, not a factory
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def encode_decode(
    frames: Sequence[np.ndarray], *, fps: int, settings: X264Settings
) -> list[np.ndarray]:
    """Round-trip frames through the recorded codec settings, in memory.

    The result minus the input is the codec's own noise floor: the part of a
    re-render's difference from a recorded video that no amount of rendering
    fidelity can remove, because the recording is itself lossy.
    """
    if not frames:
        return []
    buffer = io.BytesIO()
    height, width = frames[0].shape[:2]
    with VideoWriter(buffer, width=width, height=height, fps=fps, settings=settings) as writer:
        for frame in frames:
            writer.write(frame)
    buffer.seek(0)
    import av

    decoded: list[np.ndarray] = []
    with av.open(buffer) as container:
        for frame in container.decode(video=0):
            decoded.append(frame.to_ndarray(format="rgb24"))
    return decoded


class ImageStatsAccumulator:
    """Accumulate LeRobot-shaped image statistics over re-rendered frames.

    Only every ``STATS_PIXEL_STRIDE``-th pixel is kept, both to match the
    recorded statistics' sample size and to keep a full episode's pixels in
    memory at a few tens of megabytes.
    """

    def __init__(self) -> None:
        self._samples: list[np.ndarray] = []

    def add(self, rgb: np.ndarray) -> None:
        stride = STATS_PIXEL_STRIDE
        self._samples.append(rgb[::stride, ::stride].reshape(-1, 3).copy())

    def result(self) -> dict[str, object]:
        if not self._samples:
            raise ValueError("no frames accumulated")
        sample = np.concatenate(self._samples).astype(np.float64) / 255.0

        def per_channel(values: np.ndarray) -> list[list[list[float]]]:
            return [[[float(value)]] for value in values]

        stats: dict[str, object] = {
            "min": per_channel(sample.min(axis=0)),
            "max": per_channel(sample.max(axis=0)),
            "mean": per_channel(sample.mean(axis=0)),
            "std": per_channel(sample.std(axis=0)),
            "count": [int(sample.shape[0])],
        }
        for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
            stats[f"q{int(quantile * 100):02d}"] = per_channel(
                np.quantile(sample, quantile, axis=0)
            )
        return stats


def rewrite_image_stats(stats_path: Path, stats_by_feature: dict[str, dict[str, object]]) -> None:
    """Replace the camera features' statistics; leave every other one alone.

    The recorded statistics describe the recorded pixels, so a re-render with a
    different appearance — a dark floor above all — invalidates exactly the
    image entries and nothing else.
    """
    with stats_path.open() as file:
        stats = json.load(file)
    for feature, feature_stats in stats_by_feature.items():
        if feature in stats:
            stats[feature] = feature_stats
    with stats_path.open("w") as file:
        json.dump(stats, file, indent=4)
        file.write("\n")


def copy_episode_scaffold(source: Path, destination: Path) -> None:
    """Copy an episode's parquet data and metadata; leave the videos to the caller."""
    destination.mkdir(parents=True)
    for name in ("data", "meta"):
        shutil.copytree(source / name, destination / name)
    for extra in source.iterdir():
        if extra.is_file():
            shutil.copy2(extra, destination / extra.name)


def directory_size(path: Path) -> int:
    """Total bytes of every file under ``path``."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def evenly_spaced(count: int, wanted: int) -> list[int]:
    """Indices of ``wanted`` evenly spaced items out of ``count`` (all if <= 0)."""
    if wanted <= 0 or wanted >= count:
        return list(range(count))
    return sorted({round(index) for index in np.linspace(0, count - 1, wanted)})


def mean_of(values: Iterable[float]) -> float:
    collected = list(values)
    return float(np.mean(collected)) if collected else float("nan")
