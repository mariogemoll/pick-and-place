# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import datetime
from types import SimpleNamespace

import numpy as np

from pick_and_place.runtime.plate_debug import (
    describe_candidate,
    plate_search_debugger,
    write_plate_search_debug,
)

CAMERA_MATRIX = np.array([[100.0, 0.0, 40.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]])
CAMERA_POSITION = np.array([0.0, 0.0, 1.0])
CAMERA_ROTATION = np.eye(3)
WORKSPACE_CORNERS = np.array(
    [[-0.3, -0.25, 0.0], [0.3, -0.25, 0.0], [0.3, 0.25, 0.0], [-0.3, 0.25, 0.0]]
)


def _frame_with_a_dark_patch():
    frame = np.full((60, 80, 3), 220, dtype=np.uint8)
    frame[24:36, 32:48] = 20
    return frame


def _debugger(directory, stamps):
    clock = iter(stamps)
    return plate_search_debugger(
        directory,
        CAMERA_MATRIX,
        CAMERA_POSITION,
        CAMERA_ROTATION,
        target_color="black",
        workspace_corners_world=WORKSPACE_CORNERS,
        now=lambda: next(clock),
    )


def _hunting(hunting):
    return SimpleNamespace(
        phase_name="find_plate" if hunting else "carry", drop_target=None
    )


def test_write_plate_search_debug_writes_frame_and_mask_side_by_side(tmp_path):
    import cv2

    path = tmp_path / "plate.png"

    candidates = write_plate_search_debug(
        path,
        _frame_with_a_dark_patch(),
        CAMERA_MATRIX,
        CAMERA_POSITION,
        CAMERA_ROTATION,
        target_color="black",
        workspace_corners_world=WORKSPACE_CORNERS,
    )

    written = cv2.imread(str(path))
    assert written.shape == (60, 160, 3)
    assert all(candidate.area_px > 0.0 for candidate in candidates)


def test_plate_search_debugger_dumps_once_per_hunt(tmp_path):
    stamps = [
        datetime.datetime(2026, 9, 8, 20, 51, 30),
        datetime.datetime(2026, 9, 8, 20, 52, 10),
    ]
    observe = _debugger(tmp_path, stamps)
    frame = _frame_with_a_dark_patch()

    observe(_hunting(True), frame)
    observe(_hunting(True), frame)
    assert len(list(tmp_path.iterdir())) == 1

    observe(_hunting(False), frame)
    observe(_hunting(True), frame)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "20260908_205130_plate_search.png",
        "20260908_205210_plate_search.png",
    ]


def test_plate_search_debugger_ignores_a_hunt_that_already_has_a_sighting(tmp_path):
    observe = _debugger(tmp_path, [datetime.datetime(2026, 9, 8, 20, 51, 30)])

    observe(SimpleNamespace(phase_name="find_plate", drop_target=object()), _frame_with_a_dark_patch())

    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_describe_candidate_names_the_measurements_and_the_verdict():
    candidate = SimpleNamespace(
        area_fraction=0.0123,
        corner_count=5,
        aspect=1.42,
        rectangularity=0.91,
        rejection="not square enough",
    )

    line = describe_candidate(candidate)

    assert "1.23%" in line
    assert "corners  5" in line
    assert "aspect  1.42" in line
    assert line.endswith("-> not square enough")
