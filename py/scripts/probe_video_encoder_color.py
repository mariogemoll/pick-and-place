#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure what each video writer does to a frame's colour on the way to disk.

Verifying the episode re-renderer (`rerender_episodes.py --verify`) found that
recorded episode videos hold roughly 0.7-0.8x the chroma of the frames the
renderer produced, with luminance preserved — a global colour transform, not a
rendering difference. This isolates where that happens by writing one known test
frame through both writers and decoding it back:

* the **LeRobot** streaming encoder that produced every recorded dataset, and
* the **PyAV** writer the re-renderer uses.

A luma gain near 1.0 with a chroma gain near 1.0 means the writer is faithful.
A chroma gain well below 1.0 means that writer desaturates, and every dataset
written through it holds less colour than the simulator rendered — which matters
directly for a policy that is evaluated on live renders it never round-tripped
through video.

Run from ``py/`` in the project environment (LeRobot required)::

    python scripts/probe_video_encoder_color.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from pick_and_place.episode_rerender import (
    VideoWriter,
    X264Settings,
    decode_video,
    episode_video_paths,
    x264_settings,
)
from pick_and_place.executor import CONTROL_HZ

WIDTH, HEIGHT = 960, 720
FRAMES = 30

# BT.601 full-range RGB -> YCbCr, the convention both writers nominally use.
RGB_TO_YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)


def test_frame() -> np.ndarray:
    """A frame spanning the scene's colours: tan floor, plate, cube, saturated bars."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    # The scene's actual floor tan under its lighting, which is where the
    # measured discrepancy lives, plus a vertical gradient like the light falloff.
    gradient = np.linspace(0.75, 1.0, HEIGHT)[:, None, None]
    image[:] = (np.array([209, 189, 153]) * gradient).clip(0, 255).astype(np.uint8)
    bars = [
        (255, 255, 255),
        (128, 128, 128),
        (26, 51, 217),  # the blue cube
        (31, 31, 31),  # the drop-zone plate
        (242, 204, 26),
        (217, 26, 20),
    ]
    band = HEIGHT // (2 * len(bars))
    for index, colour in enumerate(bars):
        image[index * band : (index + 1) * band, WIDTH // 2 :] = colour
    return image


def ycbcr(rgb: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float64) @ RGB_TO_YCBCR.T
    out[..., 1:] += 128.0
    return out


def gains(original: np.ndarray, decoded: np.ndarray) -> tuple[float, float, float]:
    """Fit luma and chroma gains of ``decoded`` against ``original``."""
    first, second = ycbcr(original), ycbcr(decoded)
    luma_gain, luma_offset = np.polyfit(first[..., 0].ravel(), second[..., 0].ravel(), 1)
    chroma_first = np.concatenate(
        [(first[..., 1] - 128).ravel(), (first[..., 2] - 128).ravel()]
    )
    chroma_second = np.concatenate(
        [(second[..., 1] - 128).ravel(), (second[..., 2] - 128).ravel()]
    )
    chroma_gain, _ = np.polyfit(chroma_first, chroma_second, 1)
    return float(luma_gain), float(luma_offset), float(chroma_gain)


def write_with_pyav(frame: np.ndarray, path: Path, settings: X264Settings) -> Path:
    with VideoWriter(
        path, width=WIDTH, height=HEIGHT, fps=int(CONTROL_HZ), settings=settings
    ) as writer:
        for _ in range(FRAMES):
            writer.write(frame)
    return path


def write_with_lerobot(frame: np.ndarray, root: Path) -> Path:
    """Write one episode through the recorder's own dataset writer."""
    from datasets.utils.logging import disable_progress_bar

    from pick_and_place.recording import RecordingSession

    disable_progress_bar()
    recording = RecordingSession(
        repo_id="local/encoder-colour-probe",
        root=root,
        task="probe",
        fps=CONTROL_HZ,
        vcodec="h264",
        streaming_encoding=True,
    )
    shape = (HEIGHT, WIDTH, 3)
    recording.create_dataset(shape, shape, environment_state_names=("x",))
    for _ in range(FRAMES):
        recording.dataset.add_frame(
            {
                "observation.state": np.zeros(6, dtype=np.float32),
                "action": np.zeros(6, dtype=np.float32),
                "observation.environment_state": np.zeros(1, dtype=np.float32),
                "observation.images.wrist": frame,
                "observation.images.overhead": frame,
                "task": "probe",
            }
        )
    recording.save_episode({})
    recording.finalize()
    return episode_video_paths(root)["observation.images.wrist"]


def middle_frame(path: Path) -> np.ndarray:
    """Decode a frame from the middle of the clip, away from the first keyframe."""
    frames = list(decode_video(path))
    return frames[len(frames) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-episode",
        type=Path,
        default=None,
        help="staged episode whose x264 settings the PyAV writer should copy",
    )
    parser.add_argument("--keep", type=Path, default=None, help="keep the probe output here")
    args = parser.parse_args()

    settings = X264Settings(crf=30.0, keyint=2)
    if args.reference_episode is not None:
        settings = x264_settings(episode_video_paths(args.reference_episode)["observation.images.wrist"])
    print(f"encoding at crf={settings.crf:g} keyint={settings.keyint}")

    frame = test_frame()
    workspace = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="encoder-probe-"))
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        results = {
            "pyav (re-renderer)": middle_frame(
                write_with_pyav(frame, workspace / "pyav.mp4", settings)
            ),
            "lerobot (recorder)": middle_frame(
                write_with_lerobot(frame, workspace / "lerobot")
            ),
        }
        print(f"\n{'writer':22s} {'luma gain':>10s} {'luma off':>9s} {'chroma gain':>12s}")
        for label, decoded in results.items():
            luma_gain, luma_offset, chroma_gain = gains(frame, decoded)
            print(f"{label:22s} {luma_gain:10.4f} {luma_offset:+9.2f} {chroma_gain:12.4f}")
        print(
            "\nA chroma gain near 1.0 is a faithful writer. The recorded datasets "
            "measure ~0.7-0.8 against the frames the renderer produced."
        )
    finally:
        if args.keep is None:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
