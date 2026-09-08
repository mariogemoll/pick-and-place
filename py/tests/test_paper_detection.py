# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.perception.paper_detection import (
    PaperTracker,
    target_candidates,
    target_rejection,
)
from pick_and_place.spec.drop_zone import PaperTarget


def test_paper_tracker_reset_requires_a_new_detection():
    target = PaperTarget(
        center_px=np.array([100.0, 100.0]),
        corners_px=np.array([[90.0, 90.0], [110.0, 90.0], [110.0, 110.0], [90.0, 110.0]]),
        center_world=np.array([0.1, 0.2, 0.0]),
        corners_world=np.array(
            [[0.05, 0.15, 0.0], [0.15, 0.15, 0.0], [0.15, 0.25, 0.0], [0.05, 0.25, 0.0]]
        ),
        area_px=400.0,
        rectangularity=1.0,
    )
    tracker = PaperTracker()

    assert tracker.update(target) is not None
    assert tracker.update(None) is not None

    tracker.reset()

    assert tracker.update(None) is None


def _mask_with(shape, boxes):
    mask = np.zeros(shape, dtype=np.uint8)
    for top, bottom, left, right in boxes:
        mask[top:bottom, left:right] = 255
    return mask


def test_target_candidates_accepts_a_square_and_names_each_rejection():
    mask = _mask_with(
        (200, 200),
        [
            (20, 80, 20, 80),  # a plausible drop-zone square
            (100, 120, 20, 110),  # a bar, in range by area but far from square
            (150, 160, 150, 160),  # a speckle
        ],
    )

    by_rejection = {
        candidate.rejection: candidate for candidate in target_candidates(mask)
    }

    assert set(by_rejection) == {None, "not square enough", "too small"}
    accepted = by_rejection[None]
    assert accepted.corner_count == 4
    assert accepted.aspect == pytest.approx(1.0, abs=0.01)
    assert accepted.rectangularity == pytest.approx(1.0, abs=0.01)
    assert accepted.area_fraction == pytest.approx(0.086, abs=0.005)


def test_target_rejection_reports_the_first_filter_a_contour_fails():
    measurements = dict(area_fraction=0.05, corner_count=4, convex=True, aspect=1.0, rectangularity=1.0)

    assert target_rejection(**measurements) is None
    assert target_rejection(**{**measurements, "area_fraction": 0.2}) == "too large"
    assert target_rejection(**{**measurements, "area_fraction": 0.001}) == "too small"
    assert "6 corners" in target_rejection(**{**measurements, "corner_count": 6})
    assert target_rejection(**{**measurements, "convex": False}) is not None
    assert target_rejection(**{**measurements, "aspect": 1.4}) == "not square enough"
    assert target_rejection(**{**measurements, "rectangularity": 0.5}) == "too ragged"
