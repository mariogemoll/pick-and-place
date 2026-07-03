#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure the frame rate a camera actually delivers at various settings.

Sweeps one or more resolutions (and optionally a pixel format) on a single
camera and reports, for each, what the driver *claims* (``CAP_PROP_FPS`` and the
granted frame size) versus the rate frames actually arrive when read in a tight
loop. Because ``VideoCapture.read()`` blocks until the next frame is available,
counting successful reads over a fixed window gives the true delivered rate.

The delivered rate is the ceiling for a recording pipeline: asking the control
loop to run faster than this only logs duplicate stale frames.

At 1080p most USB webcams top out near 30 fps in raw formats; ``--fourcc MJPG``
(compressed) is usually required to unlock 60 fps at higher resolutions, so this
probe lets you set it and compare.

Example:
    python py/scripts/camera_fps_probe.py 0 \
        --resolutions 1920x1080,1280x720,640x480 --fps 60 --fourcc MJPG
"""

from __future__ import annotations

import argparse
import time

import cv2

from pick_and_place.cam_align_solve import parse_index_or_path


def parse_resolutions(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        w, _, h = chunk.partition("x")
        out.append((int(w), int(h)))
    return out


def measure(cap, seconds: float) -> tuple[int, float, tuple[int, int]]:
    """Read as fast as the camera delivers for ``seconds`` and count frames.

    Returns ``(frame_count, elapsed, (width, height))`` of the frames that
    actually arrived. The first read after a settings change is discarded so the
    reconfigure stall isn't counted against the rate.
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        return 0, 0.0, (0, 0)
    size = (frame.shape[1], frame.shape[0])
    count = 0
    start = time.perf_counter()
    deadline = start + seconds
    while time.perf_counter() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            count += 1
    elapsed = time.perf_counter() - start
    return count, elapsed, size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera", help="OpenCV camera index or device path")
    parser.add_argument(
        "--resolutions",
        default="1920x1080,1280x720,640x480",
        help="comma-separated WxH list to try (default: 1920x1080,1280x720,640x480)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="frame rate to request from the driver (default: 60)",
    )
    parser.add_argument(
        "--fourcc",
        default="auto",
        help="pixel format FOURCC to request, e.g. MJPG or YUYV (default: auto = leave as-is)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="measurement window per resolution (default: 3)",
    )
    args = parser.parse_args()

    resolutions = parse_resolutions(args.resolutions)
    if not resolutions:
        parser.error("no resolutions to test")

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    cap = cv2.VideoCapture(parse_index_or_path(args.camera), backend)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Could not open camera {args.camera!r}.")

    if args.fourcc.lower() != "auto":
        if len(args.fourcc) != 4:
            cap.release()
            parser.error("--fourcc must be a 4-character code like MJPG or YUYV")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))

    print(
        f"Requesting fps={args.fps:g}, fourcc={args.fourcc}, "
        f"{args.seconds:g}s per resolution.\n"
    )
    header = f"{'requested':>12}  {'granted':>10}  {'drv fps':>8}  {'fourcc':>7}  {'measured':>9}"
    print(header)
    print("-" * len(header))

    try:
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, args.fps)

            drv_fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = (
                "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
                if fourcc_int
                else "?"
            )

            count, elapsed, size = measure(cap, args.seconds)
            measured = count / elapsed if elapsed > 0 else float("nan")
            print(
                f"{width:>5}x{height:<6}  {size[0]:>4}x{size[1]:<5}  "
                f"{drv_fps:>8.1f}  {fourcc_str:>7}  {measured:>7.1f}fps"
            )
    finally:
        cap.release()


if __name__ == "__main__":
    main()
