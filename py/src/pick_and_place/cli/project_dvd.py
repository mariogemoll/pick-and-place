# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Bounce a logo around the workspace square, the way the DVD player did.

Not only a toy. It is the most legible check of the projector calibration there
is: the logo is animated in **workspace metres** and only warped into projector
pixels at the last step, so if the calibration is right it turns exactly on the
frame rails and never overlaps them. A logo that clips through an edge, or
bounces short of one, is a calibration that has drifted -- which is otherwise
hard to notice, because a still image projected through a stale homography looks
perfectly fine until you measure it.

Animating in metres rather than in projector pixels is the whole point. Bouncing
in pixel space would look square on the panel and come out skewed on the floor,
since the projector is aimed obliquely; in metres it is square where it matters
and the keystone falls out of the same warp everything else here uses.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pick_and_place.calibration.projector_solve import (
    apply_homography,
    image_to_workspace_square,
    load_projector_calibration,
    workspace_to_projector,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.workspace_bounds import WORKSPACE_FRAME_INNER_HALF_EXTENT
from pick_and_place.hardware.projector import blank, read_framebuffer_info, show

#: Canvas resolution standing in for the workspace square. Generous enough that
#: the warp up to projector pixels never has to invent detail.
CANVAS_PX = 1000

#: Logo width divided by height. One constant, because the bounce limit and
#: the drawn mask must agree or the logo turns before it reaches the rail.
LOGO_ASPECT = 2.6

#: Cycled on every bounce, as the original did.
PALETTE = (
    (80, 80, 240),
    (80, 220, 90),
    (240, 180, 60),
    (220, 90, 220),
    (90, 220, 230),
    (240, 120, 80),
)


def build_parser() -> SuggestingArgumentParser:
    """Flags for ``pap project-dvd``."""
    parser = SuggestingArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", default=None, help="projector calibration JSON")
    parser.add_argument(
        "--half-extent",
        type=float,
        default=WORKSPACE_FRAME_INNER_HALF_EXTENT,
        help="half-width in metres of the square to bounce inside; the frame's inner rail",
    )
    parser.add_argument("--speed", type=float, default=0.12, help="logo speed in metres per second")
    parser.add_argument("--size", type=float, default=0.085, help="logo width in metres")
    parser.add_argument(
        "--logo",
        type=Path,
        default=None,
        help="image to bounce instead of the built-in wordmark; alpha is used as its shape",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="animation rate")
    parser.add_argument(
        "--seconds", type=float, default=0.0, help="run this long; 0 waits for Ctrl-C"
    )
    parser.add_argument("--fb-index", type=int, default=0, help="framebuffer index (/dev/fb<N>)")
    return parser


def _wordmark_mask(width_px: int, height_px: int, cv2) -> np.ndarray:
    """Draw the stand-in wordmark as a coverage mask.

    Deliberately a generic ellipse-and-letters shape rather than the real disc
    logo, which is a registered trademark. ``--logo`` takes any image, so nothing
    that is not ours has to live in this repository to get the effect.
    """
    mask = np.zeros((height_px, width_px), dtype=np.uint8)
    cx, cy = width_px // 2, height_px // 2
    cv2.ellipse(mask, (cx, int(cy * 1.35)), (width_px // 2 - 1, height_px // 5), 0, 0, 360, 255, -1)
    scale = width_px / 95.0
    thickness = max(1, int(scale * 2))
    (text_w, text_h), _ = cv2.getTextSize("DVD", cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    cv2.putText(
        mask,
        "DVD",
        (cx - text_w // 2, cy + text_h // 2 - height_px // 8),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        255,
        thickness,
        cv2.LINE_AA,
    )
    return mask.astype(np.float32) / 255.0


def _image_mask(path, width_px: int, height_px: int, cv2) -> np.ndarray:
    """Coverage mask from an image file: its alpha if it has one, else its brightness."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"could not read a logo image from {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        coverage = image[:, :, 3]
    elif image.ndim == 3:
        coverage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        coverage = image
    resized = cv2.resize(coverage, (width_px, height_px), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _blit(canvas: np.ndarray, mask: np.ndarray, center_px, color) -> None:
    """Paint ``color`` through ``mask``, centred, clipped to the canvas."""
    height, width = mask.shape
    left = int(round(center_px[0])) - width // 2
    top = int(round(center_px[1])) - height // 2
    x0, y0 = max(0, left), max(0, top)
    x1 = min(canvas.shape[1], left + width)
    y1 = min(canvas.shape[0], top + height)
    if x0 >= x1 or y0 >= y1:
        return
    window = mask[y0 - top : y1 - top, x0 - left : x1 - left, None]
    canvas[y0:y1, x0:x1] = (window * np.array(color, dtype=np.float32)).astype(np.uint8)


def advance(
    position: np.ndarray, velocity: np.ndarray, limit: np.ndarray, period: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Move the logo one tick and reflect it off the rails.

    A bounce clamps the centre to exactly ``limit`` before reversing, rather than
    letting it overshoot and turn wherever the next tick happens to land. Since
    ``limit`` is the rail inset by the logo's own half-extent, that puts the
    logo's *edge* exactly on the rail at the moment it turns -- which is the
    property that makes this a calibration check and not just a decoration.
    Without the clamp the turning point would drift with frame rate and speed.

    Returns the new position and velocity, and how many axes bounced: two means
    a corner.
    """
    position = position + velocity * period
    velocity = velocity.copy()
    bounced = 0
    for axis in (0, 1):
        if abs(position[axis]) > limit[axis]:
            position[axis] = np.sign(position[axis]) * limit[axis]
            velocity[axis] = -velocity[axis]
            bounced += 1
    return position, velocity, bounced


def run(args: argparse.Namespace) -> None:
    """Animate the logo in workspace metres and warp each frame to the projector."""
    import cv2

    info = read_framebuffer_info(args.fb_index)
    calibration = load_projector_calibration(args.calibration)
    canvas_to_square = image_to_workspace_square(
        (CANVAS_PX, CANVAS_PX), half_extent_m=args.half_extent, stretch=True
    )
    warp = workspace_to_projector(calibration.projector_to_workspace) @ canvas_to_square
    # Place the logo by inverting the very map that lays the canvas down, rather
    # than by rewriting its axis conventions here. Those conventions have already
    # changed once; a second copy of them is a second thing to forget.
    square_to_canvas = np.linalg.inv(canvas_to_square)

    metres_to_px = CANVAS_PX / (2.0 * args.half_extent)
    half_w = args.size / 2.0
    half_h = half_w / LOGO_ASPECT
    limit = np.array([args.half_extent - half_w, args.half_extent - half_h])
    if np.any(limit <= 0):
        raise SystemExit(f"a {args.size:.3f} m logo does not fit in the square")

    logo_w_px = max(8, int(round(args.size * metres_to_px)))
    logo_h_px = max(8, int(round(logo_w_px / LOGO_ASPECT)))
    mask = (
        _image_mask(args.logo, logo_w_px, logo_h_px, cv2)
        if args.logo is not None
        else _wordmark_mask(logo_w_px, logo_h_px, cv2)
    )

    position = np.zeros(2)
    heading = np.array([1.0, 0.72])
    velocity = heading / np.linalg.norm(heading) * args.speed
    color_index = 0
    corners = 0
    period = 1.0 / args.fps
    started_at = time.monotonic()

    print(
        f"calibration solved {calibration.solved} ({calibration.rms_mm:.2f} mm rms); "
        f"Ctrl-C to stop"
    )
    try:
        while True:
            frame_started = time.monotonic()
            canvas = np.zeros((CANVAS_PX, CANVAS_PX, 3), dtype=np.uint8)
            center_px = apply_homography(square_to_canvas, position[None, :])[0]
            _blit(canvas, mask, center_px, PALETTE[color_index % len(PALETTE)])
            show(cv2.warpPerspective(canvas, warp, (info.width, info.height)), info)

            position, velocity, bounced = advance(position, velocity, limit, period)
            if bounced:
                color_index += 1
            if bounced == 2:
                corners += 1
                print(f"  corner hit #{corners}")

            if args.seconds and time.monotonic() - started_at >= args.seconds:
                break
            remaining = period - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print()
    finally:
        blank(info)
        print(f"{corners} corner hit{'' if corners == 1 else 's'}")
