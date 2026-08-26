# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The preview server's rectified view.

The point of serving both views is that they are *not* the same field of view:
rectification recentres the principal point and pushes the periphery outward,
so a marker plainly inside the raw frame can be missing from the rectified one.
These tests pin that the second stream exists, that it is only offered when the
camera is identified and its intrinsics are readable, and that producing it
costs nothing while nobody is watching.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pick_and_place.cli import preview_cameras as pc

WIDTH, HEIGHT = 640, 480

INTRINSICS = {
    "model": "standard",
    "width": 1920,
    "height": 1080,
    "camera_matrix": [[1235.8, 0.0, 907.7], [0.0, 1235.9, 487.5], [0.0, 0.0, 1.0]],
    "dist_coeffs": [-0.4287, 0.2023, -0.0003, -0.0014, -0.0478],
}


class _Node:
    """Just the attributes Camera reads off a discovered node."""

    def __init__(self, role: str) -> None:
        self.index, self.role, self.port, self.tags = 2, role, "0:1.0", "12, 13"


@pytest.fixture
def intrinsics_dir(tmp_path, monkeypatch):
    (tmp_path / "overhead_camera.json").write_text(json.dumps(INTRINSICS))
    monkeypatch.setattr(pc, "LOCAL_CAMERA_INTRINSICS_DIR", tmp_path)
    return tmp_path


def test_an_identified_camera_with_intrinsics_gets_a_map(intrinsics_dir):
    undistort_map, note = pc.undistort_map_for("overhead", WIDTH, HEIGHT)

    assert undistort_map is not None
    assert note == ""


def test_an_unidentified_camera_is_never_rectified(intrinsics_dir):
    """Rectifying with another camera's coefficients would look right and be wrong."""
    undistort_map, note = pc.undistort_map_for("", WIDTH, HEIGHT)

    assert undistort_map is None
    assert "not identified" in note


def test_a_missing_intrinsics_file_is_reported_not_raised(intrinsics_dir):
    undistort_map, note = pc.undistort_map_for("wrist", WIDTH, HEIGHT)

    assert undistort_map is None
    assert "wrist_camera.json" in note


def test_unreadable_intrinsics_are_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "overhead_camera.json").write_text('{"width": 1920}')
    monkeypatch.setattr(pc, "LOCAL_CAMERA_INTRINSICS_DIR", tmp_path)

    undistort_map, note = pc.undistort_map_for("overhead", WIDTH, HEIGHT)

    assert undistort_map is None
    assert "unusable intrinsics" in note


def test_rectification_moves_the_periphery_off_the_frame(intrinsics_dir):
    """The whole reason the second view is worth serving.

    A point near the bottom edge of the raw frame does not survive into the
    rectified one, which is why a preview showing only the raw frame can look
    healthy while a solve fails for want of a marker.
    """
    import cv2

    matrix = np.array(INTRINSICS["camera_matrix"], dtype=float)
    matrix[0, :] *= WIDTH / INTRINSICS["width"]
    matrix[1, :] *= HEIGHT / INTRINSICS["height"]
    rect = np.array(
        [[matrix[1, 1], 0.0, WIDTH / 2.0], [0.0, matrix[1, 1], HEIGHT / 2.0], [0.0, 0.0, 1.0]]
    )
    bottom_corner = np.array([[[WIDTH * 0.15, HEIGHT - 2.0]]])

    mapped = cv2.undistortPoints(
        bottom_corner, matrix, np.array(INTRINSICS["dist_coeffs"], dtype=float), None, rect
    )[0][0]

    assert mapped[1] > HEIGHT, "the raw bottom edge should land past the rectified frame"


def test_the_rectified_stream_is_idle_until_someone_watches(intrinsics_dir):
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)

    assert camera.rectified_viewers == 0
    with camera.viewer(rectified=True):
        assert camera.rectified_viewers == 1
        with camera.viewer(rectified=True):
            assert camera.rectified_viewers == 2
        assert camera.rectified_viewers == 1
    assert camera.rectified_viewers == 0


def test_the_two_streams_count_their_viewers_apart(intrinsics_dir):
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)

    with camera.viewer(rectified=False):
        assert (camera.raw_viewers, camera.rectified_viewers) == (1, 0)
        with camera.viewer(rectified=True):
            assert (camera.raw_viewers, camera.rectified_viewers) == (1, 1)
    assert (camera.raw_viewers, camera.rectified_viewers) == (0, 0)


def test_the_last_viewer_leaving_drops_the_held_frame(intrinsics_dir):
    """Otherwise the next viewer opens on a frame from minutes ago."""
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)

    with camera.viewer(rectified=True):
        camera.rectified = b"stale"
    assert camera.latest(rectified=True) is None


def test_latest_keeps_the_two_views_apart(intrinsics_dir):
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)
    camera.frame, camera.rectified = b"raw", b"rectified"

    assert camera.latest() == b"raw"
    assert camera.latest(rectified=True) == b"rectified"


def test_the_page_offers_a_rectified_pane_only_when_there_is_one(intrinsics_dir):
    with_map = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)
    without_map = pc.Camera(_Node(""), WIDTH, HEIGHT)

    assert with_map.undistort_map is not None
    assert without_map.undistort_map is None
    assert "/rectified/2" in pc.RECTIFIED_PANE.format(index=2)
    assert "not identified" in pc.RECTIFIED_MISSING.format(note=without_map.rectify_note)


# --------------------------------------------------------------------------
# marker overlay


class _FakeDetection:
    def __init__(self, tag_id: int, x: float, y: float, size: float = 30.0) -> None:
        self.tag_id = tag_id
        self.corners = np.array(
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size]], dtype=float
        )


class _FakeDetector:
    """Stands in for pupil_apriltags, and counts how often it was asked."""

    def __init__(self, detections) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, gray):
        self.calls += 1
        return self.detections


def _blank():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def _quad(x: int, y: int, size: int = 30):
    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]])


def _badge_is(image, colour) -> bool:
    """Is the top-left badge painted in ``colour``?"""
    strip = image[0:40, 0:400]
    return bool(np.all(strip == np.array(colour, dtype=np.uint8), axis=-1).any())


def test_detect_markers_returns_plain_arrays_that_outlive_the_detector():
    """They cross a thread boundary, so they must not be detector-owned objects."""
    markers = pc.detect_markers(_blank(), _FakeDetector([_FakeDetection(12, 50, 60)]))

    assert len(markers) == 1
    tag_id, corners = markers[0]
    assert tag_id == 12
    assert isinstance(corners, np.ndarray)
    assert corners.shape == (4, 2)


def test_drawing_a_plate_marks_the_image():
    before = _blank()
    after = pc.draw_markers(before.copy(), [(12, _quad(50, 60))])

    assert after.any(), "the outline should have painted something"
    assert not np.array_equal(before, after)


def test_the_badge_counts_only_corner_plates():
    """A cube tag in view must not be mistaken for progress towards a solve."""
    plates = [
        (tag_id, _quad(60 * k, 120))
        for k, tag_id in enumerate(sorted(pc.WORKSPACE_FRAME_TAG_IDS), start=1)
    ]
    partial = pc.draw_markers(_blank(), plates[:2] + [(3, _quad(400, 300))])
    complete = pc.draw_markers(_blank(), plates)

    assert _badge_is(partial, pc.OTHER_COLOUR)
    assert _badge_is(complete, pc.PLATE_COLOUR)


def test_the_overlay_is_throttled_rather_than_run_per_frame(intrinsics_dir):
    """Detection costs far more than an encode, and the markers do not move."""
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)
    detector = _FakeDetector([_FakeDetection(12, 50, 60)])
    camera._detector = detector

    for _ in range(20):
        camera._annotate(_blank(), False)

    assert detector.calls == 1, "a burst of frames within one period is one detection"


def test_the_overlay_refreshes_once_the_period_has_passed(intrinsics_dir):
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)
    detector = _FakeDetector([_FakeDetection(12, 50, 60)])
    camera._detector = detector

    camera._annotate(_blank(), False)
    camera._detected_at[False] -= 2.0 / pc.OVERLAY_HZ
    camera._annotate(_blank(), False)

    assert detector.calls == 2


def test_the_two_views_keep_separate_markers(intrinsics_dir):
    """The raw and rectified frames disagree about what is visible; that is the point."""
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)
    camera._detector = _FakeDetector([_FakeDetection(12, 50, 60)])

    camera._annotate(_blank(), False)

    assert camera._markers[False] and not camera._markers[True]


def test_the_overlay_never_draws_on_the_frame_it_was_given(intrinsics_dir):
    """The raw JPEG and the detector must not fight over one buffer."""
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT)
    camera._detector = _FakeDetector([_FakeDetection(12, 50, 60)])
    frame = _blank()

    annotated = camera._annotate(frame, False)

    assert annotated is not frame
    assert not frame.any(), "the caller's frame should come back untouched"


def test_no_overlay_leaves_the_frame_untouched(intrinsics_dir):
    camera = pc.Camera(_Node("overhead"), WIDTH, HEIGHT, overlay=False)
    camera._detector = _FakeDetector([_FakeDetection(12, 50, 60)])
    frame = _blank()

    assert camera._annotate(frame, False) is frame


def test_the_flag_turns_the_overlay_off():
    assert pc.build_parser().parse_args([]).overlay is True
    assert pc.build_parser().parse_args(["--no-overlay"]).overlay is False
