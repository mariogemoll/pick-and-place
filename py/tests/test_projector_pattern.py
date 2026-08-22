# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import cv2
import numpy as np
import pytest

from pick_and_place.calibration import charuco_board
from pick_and_place.calibration.projector_pattern import (
    BoardLayout,
    corner_pixels,
    make_board,
    plan_layout,
    render,
)


def test_plan_layout_centers_the_board_within_the_margin():
    layout = plan_layout((1920, 1080), squares_x=10, squares_y=6, margin_px=60)

    assert layout.square_px == 160
    assert layout.board_width == 1600
    assert layout.board_height == 960
    # Centered: the leftover splits evenly on both sides.
    assert layout.origin_x == (1920 - 1600) // 2
    assert layout.origin_y == (1080 - 960) // 2
    assert layout.origin_x >= 60 and layout.origin_y >= 60


def test_plan_layout_fits_the_narrower_axis():
    # A square frame is limited by whichever axis runs out of squares first.
    layout = plan_layout((1000, 1000), squares_x=10, squares_y=6, margin_px=0)
    assert layout.square_px == 100


def test_plan_layout_rejects_a_margin_that_leaves_no_room():
    with pytest.raises(ValueError, match="no room"):
        plan_layout((100, 100), margin_px=60)


def test_plan_layout_rejects_squares_too_small_to_survive_projection():
    # 70px across 10 squares is 7px each, one under the floor; 80px would pass.
    with pytest.raises(ValueError, match="too few"):
        plan_layout((70, 60), squares_x=10, squares_y=6, margin_px=0)


def test_render_produces_a_full_frame_with_a_black_surround():
    layout = plan_layout((1920, 1080))
    frame = render(cv2, layout)

    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8
    # Outside the board is black, so the projector spills no light on the
    # workspace where the AprilTags are being detected.
    assert frame[: layout.origin_y].max() == 0
    assert frame[:, : layout.origin_x].max() == 0
    # The board itself carries both extremes.
    board_area = frame[
        layout.origin_y : layout.origin_y + layout.board_height,
        layout.origin_x : layout.origin_x + layout.board_width,
    ]
    assert board_area.min() == 0
    assert board_area.max() == 255


def test_rendered_board_detects_back_at_the_predicted_pixels():
    """The known half of the calibration must be exactly where we claim it is.

    A flipped or transposed corner convention would still detect cleanly and
    still fit a homography -- just to a mirrored floor. Round-tripping the
    render is what rules that out.
    """
    layout = plan_layout((1920, 1080))
    frame = render(cv2, layout)
    predicted = corner_pixels(cv2, layout)

    detector = cv2.aruco.CharucoDetector(make_board(cv2, layout))
    found, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    assert ids is not None
    assert len(ids) == (layout.squares_x - 1) * (layout.squares_y - 1) == 45

    error = np.linalg.norm(found.reshape(-1, 2) - predicted[ids.flatten()], axis=1)
    assert error.max() < 0.1


def test_corner_pixels_are_offset_by_the_board_origin():
    layout = plan_layout((1920, 1080))
    corners = corner_pixels(cv2, layout)

    assert corners.shape == (45, 2)
    # The first interior corner sits one square in from the board's top-left.
    assert corners[0] == pytest.approx(
        [layout.origin_x + layout.square_px, layout.origin_y + layout.square_px]
    )
    assert corners[:, 0].min() >= layout.origin_x
    assert corners[:, 1].min() >= layout.origin_y
    assert corners[:, 0].max() <= layout.origin_x + layout.board_width
    assert corners[:, 1].max() <= layout.origin_y + layout.board_height


def test_projected_board_uses_a_different_dictionary_from_the_printed_one():
    """Both boards can be in the overhead view at once; they must not cross-detect."""
    layout = BoardLayout(1920, 1080, 10, 6, 160, 160, 60)
    projected = make_board(cv2, layout)
    printed = charuco_board.make_board(cv2, 6, 8, 30.0, 22.0)

    projected_bits = projected.getDictionary().markerSize
    printed_bits = printed.getDictionary().markerSize
    assert projected_bits != printed_bits

    # Detecting the projected board with the printed board's detector finds nothing.
    frame = render(cv2, layout)
    detector = cv2.aruco.CharucoDetector(printed)
    _, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert ids is None or len(ids) == 0
