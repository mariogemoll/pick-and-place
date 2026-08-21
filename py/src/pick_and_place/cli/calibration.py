# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Flags describing the printed ChArUco board the intrinsics solve is measured against.

One command prints the board and another calibrates against it. **They must
agree on its geometry or the solve is silently wrong**: a board printed at
25 mm squares and solved as 30 mm returns a focal length scaled by 6/5 and no
error anywhere says so. That is the strongest case in the tree for a flag being
declared once, so it is.
"""

from __future__ import annotations

import argparse

from pick_and_place.calibration.charuco_board import (
    DEFAULT_MARKER_MM,
    DEFAULT_SQUARE_MM,
    DEFAULT_SQUARES_X,
    DEFAULT_SQUARES_Y,
)


def add_charuco_board_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the board's square counts and its printed square and marker sizes."""
    parser.add_argument(
        "--squares-x",
        type=int,
        default=DEFAULT_SQUARES_X,
        help=f"board squares along x (default: {DEFAULT_SQUARES_X})",
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=DEFAULT_SQUARES_Y,
        help=f"board squares along y (default: {DEFAULT_SQUARES_Y})",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=DEFAULT_SQUARE_MM,
        help=f"printed square size in millimetres (default: {DEFAULT_SQUARE_MM})",
    )
    parser.add_argument(
        "--marker-mm",
        type=float,
        default=DEFAULT_MARKER_MM,
        help=f"printed marker size in millimetres (default: {DEFAULT_MARKER_MM})",
    )
