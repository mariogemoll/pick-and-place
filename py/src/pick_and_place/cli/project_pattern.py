# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Throw the calibration board onto the workspace floor and hold it there.

Splitting this from the solve is deliberate. Aiming a projector is a physical
loop -- move it, focus it, check the board lands inside the overhead camera's
view and clear of the corner plates -- and that loop wants a command that puts
the board up and leaves it up, not one that also demands a camera and writes a
calibration file.
"""

from __future__ import annotations

import argparse
import time

from pick_and_place.calibration.projector_pattern import (
    DEFAULT_MARGIN_PX,
    DEFAULT_SQUARES_X,
    DEFAULT_SQUARES_Y,
    plan_layout,
    render,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.hardware.projector import blank, read_framebuffer_info, show


def build_parser() -> SuggestingArgumentParser:
    """Flags for ``pap project-pattern``."""
    parser = SuggestingArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--squares-x", type=int, default=DEFAULT_SQUARES_X, help="board squares across"
    )
    parser.add_argument(
        "--squares-y", type=int, default=DEFAULT_SQUARES_Y, help="board squares down"
    )
    parser.add_argument(
        "--margin-px",
        type=int,
        default=DEFAULT_MARGIN_PX,
        help="clearance from the frame edge, where the lens is dimmest",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="hold for this long then blank; 0 holds until interrupted",
    )
    parser.add_argument(
        "--fb-index", type=int, default=0, help="framebuffer index (/dev/fb<N>)"
    )
    parser.add_argument(
        "--save",
        type=argparse.FileType("wb"),
        help="also write the rendered frame to this PNG, for inspection",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Render the board, project it, and hold."""
    import cv2

    info = read_framebuffer_info(args.fb_index)
    layout = plan_layout(
        info.size,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        margin_px=args.margin_px,
    )
    frame = render(cv2, layout)

    if args.save is not None:
        args.save.write(cv2.imencode(".png", frame)[1].tobytes())
        args.save.close()

    print(
        f"{info.device} {info.width}x{info.height}: "
        f"{layout.squares_x}x{layout.squares_y} board, {layout.square_px}px squares, "
        f"origin ({layout.origin_x}, {layout.origin_y})"
    )
    show(frame, info)

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
