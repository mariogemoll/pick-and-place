# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from types import SimpleNamespace

import numpy as np

from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.overhead_localization import (
    OverheadLocalizer,
    localize_cube,
    localize_drop_target,
)


class StubPaperTracker:
    def __init__(self):
        self.reset_count = 0
        self.updated_with = None

    def reset(self):
        self.reset_count += 1

    def update(self, value):
        self.updated_with = value
        return value


def test_localize_cube_maps_detection_to_world_pose(monkeypatch):
    estimate = object()
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    position = np.array([0.2, -0.1, 0.03])
    monkeypatch.setattr(
        "pick_and_place.overhead_localization.estimate_cube_pose",
        lambda frame, detector, camera_matrix: estimate,
    )
    monkeypatch.setattr(
        "pick_and_place.overhead_localization.cube_pose_to_world",
        lambda detected, camera_position, camera_rotation: (rotation, position),
    )

    pose = localize_cube(
        np.zeros((8, 8, 3), dtype=np.uint8),
        object(),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
    )

    assert pose is not None
    assert pose.x == 0.2
    assert pose.y == -0.1
    assert pose.z == CUBE_HALF_SIZE
    assert pose.roll == 0.0
    assert pose.pitch == 0.0
    np.testing.assert_allclose(pose.yaw, np.pi / 2.0)


def test_localize_cube_returns_none_without_detection(monkeypatch):
    monkeypatch.setattr(
        "pick_and_place.overhead_localization.estimate_cube_pose",
        lambda frame, detector, camera_matrix: None,
    )

    assert (
        localize_cube(
            np.zeros((8, 8, 3), dtype=np.uint8),
            object(),
            np.eye(3),
            np.zeros(3),
            np.eye(3),
        )
        is None
    )


def test_localize_drop_target_updates_tracker(monkeypatch):
    detection = SimpleNamespace(xy=(0.1, 0.2))
    captured = {}

    def detect(*args, **kwargs):
        captured.update(kwargs)
        return detection

    class Tracker:
        def update(self, value):
            assert value is detection
            return value

    monkeypatch.setattr(
        "pick_and_place.overhead_localization.detect_paper_target",
        detect,
    )
    workspace = np.ones((4, 3))

    target = localize_drop_target(
        np.zeros((8, 8, 3), dtype=np.uint8),
        Tracker(),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        target_color="black",
        workspace_corners_world=workspace,
    )

    assert target is detection
    assert captured["target_color"] == "black"
    assert captured["workspace_corners_world"] is workspace


def test_overhead_localizer_owns_state_and_fixed_calibration(monkeypatch):
    detectors = []

    def make_detector():
        detector = object()
        detectors.append(detector)
        return detector

    tracker = StubPaperTracker()
    matrix = np.eye(3)
    position = np.array([0.1, 0.2, 0.3])
    rotation = np.eye(3)
    localizer = OverheadLocalizer(
        matrix,
        position,
        rotation,
        detector_factory=make_detector,
        paper_tracker=tracker,
    )
    matrix[0, 0] = 9.0
    position[0] = 9.0
    rotation[0, 0] = 9.0

    captured = {}

    def capture_cube(frame, detector, camera_matrix, camera_position, camera_rotation, **kwargs):
        captured.update(
            detector=detector,
            camera_matrix=camera_matrix,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
        )
        return None

    monkeypatch.setattr("pick_and_place.overhead_localization.localize_cube", capture_cube)
    localizer.localize_cube(np.zeros((8, 8, 3), dtype=np.uint8))

    assert captured["detector"] is detectors[0]
    np.testing.assert_array_equal(captured["camera_matrix"], np.eye(3))
    np.testing.assert_array_equal(captured["camera_position"], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(captured["camera_rotation"], np.eye(3))
    assert tracker.reset_count == 1

    localizer.reset()

    assert len(detectors) == 2
    assert tracker.reset_count == 2


def test_overhead_localizer_tracks_drop_target(monkeypatch):
    tracker = StubPaperTracker()
    localizer = OverheadLocalizer(
        np.eye(3),
        np.zeros(3),
        np.eye(3),
        detector_factory=lambda: object(),
        paper_tracker=tracker,
    )
    detection = object()
    monkeypatch.setattr(
        "pick_and_place.overhead_localization.detect_paper_target",
        lambda *args, **kwargs: detection,
    )

    result = localizer.localize_drop_target(
        np.zeros((8, 8, 3), dtype=np.uint8),
        target_color="white",
        workspace_corners_world=np.ones((4, 3)),
    )

    assert result is detection
    assert tracker.updated_with is detection
