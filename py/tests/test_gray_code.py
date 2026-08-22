# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.calibration.gray_code import (
    GrayCodePlan,
    _from_gray,
    _to_gray,
    correspondences,
    decode,
    frames,
)


def test_gray_round_trips_through_binary():
    values = np.arange(1024)
    assert np.array_equal(_from_gray(_to_gray(values), 10), values)


def test_adjacent_gray_codes_differ_in_one_bit():
    """The property that makes a straddling pixel land in a neighbour, not across the field."""
    codes = _to_gray(np.arange(256))
    changed = codes[:-1] ^ codes[1:]
    # A single set bit is a power of two.
    assert np.all((changed & (changed - 1)) == 0)


def test_plan_counts_cells_and_exposures():
    plan = GrayCodePlan(1920, 1080, stripe_px=32)
    assert plan.x_cells == 60
    assert plan.y_cells == 34  # 1080/32 rounds up
    assert plan.x_bits == 6  # 60 cells needs 6 bits
    assert plan.y_bits == 6
    assert plan.frame_count == 2 + 2 * 12 == 26


def test_plan_rejects_nonsense():
    with pytest.raises(ValueError, match="stripe_px"):
        GrayCodePlan(1920, 1080, stripe_px=0)
    with pytest.raises(ValueError, match="positive extent"):
        GrayCodePlan(0, 1080)


def test_frames_yields_the_planned_count_and_shape():
    plan = GrayCodePlan(256, 128, stripe_px=16)
    produced = list(frames(plan))
    assert len(produced) == plan.frame_count
    for frame in produced:
        assert frame.shape == (128, 256, 3)
        assert frame.dtype == np.uint8
        assert set(np.unique(frame)) <= {0, 255}


def test_each_bit_is_followed_by_its_inverse():
    plan = GrayCodePlan(256, 128, stripe_px=16)
    produced = list(frames(plan))
    for i in range(2, len(produced), 2):
        assert np.array_equal(produced[i], 255 - produced[i + 1])


def test_decode_recovers_the_cell_every_pixel_was_lit_by():
    """Project onto an imaginary camera that sees the frame exactly, then decode it."""
    plan = GrayCodePlan(256, 128, stripe_px=16)
    captures = [frame[:, :, 0] for frame in frames(plan)]

    decoded = decode(captures, plan)

    assert decoded.valid.all()
    expected_x = np.arange(256) // 16
    expected_y = np.arange(128) // 16
    assert np.array_equal(decoded.cell_x, np.broadcast_to(expected_x[None, :], (128, 256)))
    assert np.array_equal(decoded.cell_y, np.broadcast_to(expected_y[:, None], (128, 256)))


def test_decode_rejects_pixels_the_projector_never_reached():
    plan = GrayCodePlan(64, 64, stripe_px=16)
    captures = [frame[:, :, 0].copy() for frame in frames(plan)]
    # Blot out a corner in every exposure, as the arm's shadow would.
    for capture in captures:
        capture[:10, :10] = 0

    decoded = decode(captures, plan)

    assert not decoded.valid[:10, :10].any()
    assert decoded.valid[20:, 20:].all()


def test_decode_checks_the_capture_count():
    plan = GrayCodePlan(64, 64, stripe_px=16)
    with pytest.raises(ValueError, match="expected .* captures"):
        decode([np.zeros((64, 64), dtype=np.uint8)] * 3, plan)


def test_correspondences_average_each_cell_to_its_centre():
    plan = GrayCodePlan(64, 64, stripe_px=16)
    captures = [frame[:, :, 0] for frame in frames(plan)]

    matched = correspondences(decode(captures, plan), plan, min_pixels=1)

    assert len(matched) == plan.x_cells * plan.y_cells == 16
    # A camera that sees the frame one-to-one puts each cell's camera centroid
    # at the same place as its projector centre.
    assert np.allclose(matched.camera_xy, matched.projector_xy - 0.5, atol=1e-6)
    assert matched.projector_xy.min() == 8.0  # first cell centre: (0 + 0.5) * 16


def test_correspondences_drop_thinly_covered_cells():
    plan = GrayCodePlan(64, 64, stripe_px=16)
    captures = [frame[:, :, 0].copy() for frame in frames(plan)]
    for capture in captures:
        capture[:16, :14] = 0  # leave only 2 of 16 columns of the first cell lit

    matched = correspondences(decode(captures, plan), plan, min_pixels=100)

    assert len(matched) == 15
    assert not np.any((matched.projector_xy == [8.0, 8.0]).all(axis=1))
