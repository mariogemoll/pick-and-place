# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The ChArUco board the projector throws onto the workspace floor.

Calibrating the projector means fitting a homography from projector pixels to
workspace millimetres. That needs points whose projector-pixel coordinates are
known exactly and whose floor positions can be measured by the overhead camera,
so the board is laid out here in *pixels* and rendered at exactly that layout --
never scaled to fit afterwards, because the layout is the known half of the fit.

**The projected board uses a different ArUco dictionary from the printed one.**
:mod:`pick_and_place.calibration.charuco_board` prints ``DICT_4X4_50`` for
camera intrinsics, and that board can easily be lying on the table inside the
overhead camera's view while this one is being projected onto it. Two boards
sharing a dictionary would cross-detect and the solve would quietly fit to a mix
of both. The workspace corner plates need no such care -- they are AprilTags,
found by a different detector entirely.

The board is defined with ``square_length`` in pixels rather than metres, which
is what makes ``getChessboardCorners`` return projector-pixel offsets directly.
That is a unit convention, not an abuse: OpenCV only ever uses the ratio of
square to marker length, so the scale it is expressed in is the caller's to
choose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Ten by six fills a 16:9 frame better than a squarer board and still leaves
#: 45 interior corners, which is ample for an 8-DOF fit.
DEFAULT_SQUARES_X = 10
DEFAULT_SQUARES_Y = 6

#: Keep the projected board clear of the frame edge, where a projector's lens is
#: dimmest and least in focus.
DEFAULT_MARGIN_PX = 60

#: Marker edge as a fraction of the square. Smaller markers leave more white
#: quiet zone, which matters more for a projected board than a printed one
#: because the surface it lands on is not necessarily white.
MARKER_FRACTION = 0.72


@dataclass(frozen=True)
class BoardLayout:
    """Where the board sits in the projector's frame, in pixels."""

    frame_width: int
    frame_height: int
    squares_x: int
    squares_y: int
    square_px: int
    origin_x: int
    origin_y: int

    @property
    def board_width(self) -> int:
        """Width of the board graphic in pixels."""
        return self.squares_x * self.square_px

    @property
    def board_height(self) -> int:
        """Height of the board graphic in pixels."""
        return self.squares_y * self.square_px

    @property
    def marker_px(self) -> float:
        """Marker edge in pixels."""
        return self.square_px * MARKER_FRACTION


def plan_layout(
    frame_size: tuple[int, int],
    *,
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    margin_px: int = DEFAULT_MARGIN_PX,
) -> BoardLayout:
    """Fit the largest whole-pixel board that clears ``margin_px`` on every side."""
    frame_width, frame_height = frame_size
    usable_x = frame_width - 2 * margin_px
    usable_y = frame_height - 2 * margin_px
    if usable_x <= 0 or usable_y <= 0:
        raise ValueError(f"margin {margin_px}px leaves no room in {frame_width}x{frame_height}")

    square_px = min(usable_x // squares_x, usable_y // squares_y)
    if square_px < 8:
        raise ValueError(
            f"a {squares_x}x{squares_y} board in {frame_width}x{frame_height} gives "
            f"{square_px}px squares, too few for the ArUco bits to survive projection"
        )

    return BoardLayout(
        frame_width=frame_width,
        frame_height=frame_height,
        squares_x=squares_x,
        squares_y=squares_y,
        square_px=square_px,
        origin_x=(frame_width - squares_x * square_px) // 2,
        origin_y=(frame_height - squares_y * square_px) // 2,
    )


def make_board(cv2_module: Any, layout: BoardLayout) -> Any:
    """Return the OpenCV board for ``layout``, in pixel units.

    ``cv2_module`` is passed in rather than imported so that planning a layout
    costs no OpenCV import, matching
    :func:`pick_and_place.calibration.charuco_board.make_board`.
    """
    dictionary = cv2_module.aruco.getPredefinedDictionary(cv2_module.aruco.DICT_5X5_100)
    return cv2_module.aruco.CharucoBoard(
        (layout.squares_x, layout.squares_y),
        float(layout.square_px),
        layout.marker_px,
        dictionary,
    )


def corner_pixels(cv2_module: Any, layout: BoardLayout) -> NDArray[np.float64]:
    """Projector-pixel coordinates of every interior ChArUco corner, by corner id.

    This is the known half of the calibration: row ``i`` is where corner ``i``
    was asked to appear on the panel.
    """
    board = make_board(cv2_module, layout)
    corners = np.asarray(board.getChessboardCorners(), dtype=float)[:, :2]
    return corners + np.array([layout.origin_x, layout.origin_y], dtype=float)


def render(cv2_module: Any, layout: BoardLayout) -> NDArray[np.uint8]:
    """Render the board into a full-frame BGR image, black outside the board.

    The surround is black so the projector throws as little stray light as
    possible onto the rest of the workspace, where it would otherwise disturb
    the AprilTag detection that the same overhead frame is used for.
    """
    board = make_board(cv2_module, layout)
    graphic = board.generateImage((layout.board_width, layout.board_height), marginSize=0)

    frame = np.zeros((layout.frame_height, layout.frame_width, 3), dtype=np.uint8)
    frame[
        layout.origin_y : layout.origin_y + layout.board_height,
        layout.origin_x : layout.origin_x + layout.board_width,
    ] = graphic[:, :, None]
    return frame
