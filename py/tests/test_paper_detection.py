# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.perception.paper_detection import PaperTarget, PaperTracker


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
