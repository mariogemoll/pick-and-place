# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from types import SimpleNamespace

import numpy as np

from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.overhead_localization import localize_cube, localize_drop_target


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
