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
    pap camera-fps-probe 0 \
        --resolutions 1920x1080,1280x720,640x480 --fps 60 --fourcc MJPG
"""

from __future__ import annotations

import argparse
import time

import cv2

from pick_and_place.calibration.cam_align_solve import parse_index_or_path
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.runtime.frame_reader import capture_backend


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


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the camera probe."""
    parser = SuggestingArgumentParser(description=__doc__)
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
    return parser


def validate(parser: SuggestingArgumentParser, args: argparse.Namespace) -> None:
    """Reject arguments the parser's own types cannot check.

    Both are decidable from the command line alone, so they belong here rather
    than half-way through opening a camera -- which is where --fourcc was
    checked, after the device had already been opened and had to be released
    again to report a typo.
    """
    if not parse_resolutions(args.resolutions):
        parser.error("no resolutions to test")
    if args.fourcc.lower() != "auto" and len(args.fourcc) != 4:
        parser.error("--fourcc must be a 4-character code like MJPG or YUYV")


def run(args: argparse.Namespace) -> None:
    """Probe the camera and print what each resolution achieved."""
    resolutions = parse_resolutions(args.resolutions)

    backend = capture_backend(cv2)
    cap = cv2.VideoCapture(parse_index_or_path(args.camera), backend)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Could not open camera {args.camera!r}.")

    if args.fourcc.lower() != "auto":
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    run(args)


if __name__ == "__main__":
    main()
