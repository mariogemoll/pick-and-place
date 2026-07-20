# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Check that out-of-process detection matches in-process and survives a crash.

The point of :mod:`pick_and_place.detector_process` is that a native segfault
inside ``libapriltag`` costs one detection tick instead of a whole shard
worker, so the crash path is what these tests exercise: a helper killed
mid-run must yield an empty detection list, not an exception, and must be
replaced transparently.
"""

from __future__ import annotations

import os
import signal

import cv2
import numpy as np
import pytest

from pick_and_place.cube_detection import CUBE_TAG_IDS, detect_cube_faces, make_cube_detector
from pick_and_place.detector_process import DetectorProcess
from pick_and_place.environment import APRILTAG_TEXTURE_DIR

_TAG_PX = 260


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    """A mid-grey frame with two real cube-face tags pasted onto it."""
    img = np.full((720, 1280, 3), 140, dtype=np.uint8)
    for tag_id, x in zip(CUBE_TAG_IDS[:2], (200, 700)):
        matches = sorted(APRILTAG_TEXTURE_DIR.glob(f"tagStandard41h12_{tag_id:05d}_*.png"))
        assert matches, f"no texture on disk for tag {tag_id}"
        tag = cv2.imread(str(matches[0]))
        tag = cv2.resize(tag, (_TAG_PX, _TAG_PX), interpolation=cv2.INTER_NEAREST)
        img[230 : 230 + _TAG_PX, x : x + _TAG_PX] = tag
    return img


@pytest.fixture
def detector(tmp_path):
    with DetectorProcess(nthreads=1, crash_dump_dir=tmp_path) as proc:
        yield proc


def test_matches_in_process_detection(frame, detector):
    """The process boundary must not perturb the detection at all."""
    got = sorted(detect_cube_faces(frame, detector), key=lambda d: d.tag_id)
    want = sorted(detect_cube_faces(frame, make_cube_detector(nthreads=1)), key=lambda d: d.tag_id)

    assert len(want) >= 2, "test frame should contain detectable cube tags"
    assert [d.tag_id for d in got] == [d.tag_id for d in want]
    for a, b in zip(got, want):
        assert np.allclose(a.corners, b.corners)
        assert np.allclose(a.center, b.center)


def test_helper_crash_yields_no_detections_and_respawns(frame, detector, tmp_path):
    """A segfaulted helper costs one tick, then detection resumes as before."""
    detect_cube_faces(frame, detector)  # ensure the helper is up and serving
    crashed_pid = detector._proc.pid
    os.kill(crashed_pid, signal.SIGKILL)
    detector._proc.join(timeout=5.0)

    # The tick that lands on the dead helper degrades to "saw no tags", which
    # the servo loop already treats as a routine occlusion.
    assert detect_cube_faces(frame, detector) == []
    assert detector.crash_count == 1

    # The frame that killed it is kept, since that is the repro artefact the
    # root-cause work needs and it cannot be recovered after the fact.
    dumps = list(tmp_path.glob("apriltag_crash_*.npy"))
    assert len(dumps) == 1
    assert np.array_equal(np.load(dumps[0]), cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

    # The next tick runs against a fresh helper and is indistinguishable.
    assert detector._proc.pid != crashed_pid
    assert len(detect_cube_faces(frame, detector)) >= 2


def test_close_is_idempotent(detector):
    detector.close()
    detector.close()
