# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The printed ChArUco board: its geometry, and the OpenCV object over it.

One command prints the board and another calibrates a camera against it. They
must agree on its geometry or the solve is silently wrong -- a board printed at
25 mm squares and solved as 30 mm returns a focal length scaled by 6/5, and
nothing anywhere reports an error.

This lived in ``scripts/generate_charuco_board.py``, and the calibration command
imported it from there, one script reaching into another. The geometry is a
fact and ``make_board`` is calibration logic, so both belong here.
"""

from __future__ import annotations

from typing import Any

# A4 is 210 x 297 mm. This 180 x 240 mm board leaves a real printable margin
# on every side while still providing 35 ChArUco corners.
DEFAULT_SQUARES_X = 6
DEFAULT_SQUARES_Y = 8
DEFAULT_SQUARE_MM = 30.0
DEFAULT_MARKER_MM = 22.0


def make_board(
    cv2_module: Any, squares_x: int, squares_y: int, square_mm: float, marker_mm: float
) -> Any:
    """Return the OpenCV ChArUco board the printer and the solve both use.

    ``cv2_module`` is passed in rather than imported here so that reading the
    board's geometry costs no OpenCV import.
    """
    if marker_mm >= square_mm:
        raise ValueError("marker size must be smaller than square size")
    dictionary = cv2_module.aruco.getPredefinedDictionary(cv2_module.aruco.DICT_4X4_50)
    return cv2_module.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm / 1000.0, marker_mm / 1000.0, dictionary
    )
