# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Project a photo or a video so it lands inside the workspace frame square.

The solved homography says where each projector pixel falls on the floor, so
running it backwards says which projector pixel to light to put an image pixel
at a chosen spot. Composing that with a map from the image onto the square the
four corner plates mark out warps a picture into exactly that square, however
obliquely the projector happens to be aimed.

The whole warp is one 3x3 matrix, so a frame costs a single ``warpPerspective``
no matter how severe the keystone is. Video is therefore just the same warp
applied per frame; what limits the rate is pushing 8 MB to the framebuffer each
time, not the geometry.

Needs ``pap calibrate-projector`` to have been run, and is invalidated by moving
the projector -- there is no way to notice that from here, so a picture that
lands crooked means recalibrate rather than adjust anything in this command.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pick_and_place.calibration.projector_solve import (
    image_to_workspace_square,
    load_projector_calibration,
    workspace_to_projector,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.workspace_bounds import WORKSPACE_FRAME_INNER_HALF_EXTENT
from pick_and_place.hardware.projector import blank, read_framebuffer_info, show

_ROTATIONS = {90: 0, 180: 1, 270: 2}  # cv2.ROTATE_* ordinals


def build_parser() -> SuggestingArgumentParser:
    """Flags for ``pap project-media``."""
    parser = SuggestingArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("media", type=Path, help="image or video file to project")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="projector calibration JSON (default: config/projector/overhead_projector.json)",
    )
    parser.add_argument(
        "--half-extent",
        type=float,
        default=WORKSPACE_FRAME_INNER_HALF_EXTENT,
        help="half-width in metres of the square to fill; the frame's inner rail",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="turn the image before projecting, for when it reads upside down",
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="fill the square rather than preserving the image's aspect ratio",
    )
    parser.add_argument("--loop", action="store_true", help="repeat a video until interrupted")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="hold a still image this long; 0 waits for Ctrl-C",
    )
    parser.add_argument("--fb-index", type=int, default=0, help="framebuffer index (/dev/fb<N>)")
    return parser


def _warp_matrix(args: argparse.Namespace, frame_size: tuple[int, int]) -> np.ndarray:
    """Compose image pixels -> workspace metres -> projector pixels."""
    calibration = load_projector_calibration(args.calibration)
    to_square = image_to_workspace_square(
        frame_size, half_extent_m=args.half_extent, stretch=args.stretch
    )
    return workspace_to_projector(calibration.projector_to_workspace) @ to_square


def _oriented(frame: np.ndarray, rotate: int, cv2) -> np.ndarray:
    """Turn a frame by a whole number of right angles."""
    if rotate == 0:
        return frame
    return cv2.rotate(frame, _ROTATIONS[rotate])


def run(args: argparse.Namespace) -> None:
    """Warp the media into the workspace square and project it."""
    import cv2

    info = read_framebuffer_info(args.fb_index)

    still = cv2.imread(str(args.media), cv2.IMREAD_COLOR)
    if still is not None:
        frame = _oriented(still, args.rotate, cv2)
        height, width = frame.shape[:2]
        warped = cv2.warpPerspective(
            frame, _warp_matrix(args, (width, height)), (info.width, info.height)
        )
        print(f"{args.media.name}: {width}x{height} -> workspace square")
        show(warped, info)
        try:
            if args.seconds > 0:
                time.sleep(args.seconds)
            else:
                print("holding; press Ctrl-C to blank and exit")
                while True:
                    time.sleep(3600)
        except KeyboardInterrupt:
            print()
        finally:
            blank(info)
        return

    video = cv2.VideoCapture(str(args.media))
    if not video.isOpened():
        raise SystemExit(f"{args.media} is neither an image nor a video this build can read")

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    period = 1.0 / fps
    matrix = None
    shown = 0
    print(f"{args.media.name}: {fps:.1f} fps, Ctrl-C to stop")

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                if args.loop:
                    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            started = time.monotonic()
            frame = _oriented(frame, args.rotate, cv2)
            if matrix is None:
                height, width = frame.shape[:2]
                matrix = _warp_matrix(args, (width, height))
            show(cv2.warpPerspective(frame, matrix, (info.width, info.height)), info)
            shown += 1
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print()
    finally:
        video.release()
        blank(info)
        print(f"projected {shown} frames")
