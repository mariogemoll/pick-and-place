# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The single-slot latest-frame reader, driven by a fake capture."""

import threading
import time

import numpy as np
import pytest

from pick_and_place.runtime.frame_reader import FrameReader


class FakeCapture:
    """A ``VideoCapture`` stand-in whose frames are released one at a time.

    ``read()`` blocks until :meth:`push` hands it a frame, which is what a real
    camera does between exposures, so a test can place the reader in an exact
    state instead of racing it.
    """

    def __init__(self) -> None:
        self._pending: list[np.ndarray | None] = []
        self._condition = threading.Condition()
        self.released = False

    def push(self, frame: np.ndarray | None) -> None:
        with self._condition:
            self._pending.append(frame)
            self._condition.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._condition:
            if not self._condition.wait_for(lambda: bool(self._pending), timeout=2.0):
                return False, None
            frame = self._pending.pop(0)
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


def _image(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("timed out waiting for the reader")


def test_latest_is_none_before_the_first_frame() -> None:
    capture = FakeCapture()
    with FrameReader(capture, "test") as reader:
        assert reader.latest() is None


def test_latest_keeps_only_the_newest_frame() -> None:
    capture = FakeCapture()
    with FrameReader(capture, "test") as reader:
        capture.push(_image(1))
        capture.push(_image(2))
        capture.push(_image(3))
        _wait_until(lambda: reader.latest() is not None and reader.latest().index == 3)
        frame = reader.latest()
        assert frame is not None
        assert int(frame.bgr[0, 0, 0]) == 3


def test_frame_index_counts_from_one_and_carries_a_capture_time() -> None:
    capture = FakeCapture()
    before = time.monotonic()
    with FrameReader(capture, "test") as reader:
        capture.push(_image(7))
        frame = reader.wait_for_frame()
        assert frame.index == 1
        assert before <= frame.captured_at <= time.monotonic()


def test_failed_reads_do_not_advance_the_index() -> None:
    capture = FakeCapture()
    with FrameReader(capture, "test") as reader:
        capture.push(None)
        capture.push(_image(4))
        frame = reader.wait_for_frame()
        assert frame.index == 1
        assert int(frame.bgr[0, 0, 0]) == 4


def test_wait_for_frame_raises_naming_the_camera() -> None:
    capture = FakeCapture()
    with FrameReader(capture, "overhead") as reader:
        with pytest.raises(RuntimeError, match="overhead"):
            reader.wait_for_frame(timeout=0.05)


def test_wait_for_frame_returns_the_latest_once_frames_flow() -> None:
    capture = FakeCapture()
    with FrameReader(capture, "test") as reader:
        capture.push(_image(1))
        reader.wait_for_frame()
        capture.push(_image(2))
        _wait_until(lambda: reader.latest().index == 2)
        assert int(reader.wait_for_frame(timeout=0.0).bgr[0, 0, 0]) == 2


def test_read_returns_each_frame_once() -> None:
    capture = FakeCapture()
    with FrameReader(capture, "test") as reader:
        capture.push(_image(1))
        ok, bgr = reader.read()
        assert ok and int(bgr[0, 0, 0]) == 1
        assert reader.read(timeout=0.05) == (False, None)
        capture.push(_image(2))
        ok, bgr = reader.read()
        assert ok and int(bgr[0, 0, 0]) == 2


def test_read_skips_frames_the_reader_dropped() -> None:
    capture = FakeCapture()
    with FrameReader(capture, "test") as reader:
        capture.push(_image(1))
        capture.push(_image(2))
        _wait_until(lambda: reader.latest() is not None and reader.latest().index == 2)
        ok, bgr = reader.read()
        assert ok and int(bgr[0, 0, 0]) == 2


def test_on_frame_sees_every_capture_and_can_be_set_late() -> None:
    capture = FakeCapture()
    seen: list[tuple[int, float]] = []
    with FrameReader(capture, "test") as reader:
        capture.push(_image(1))
        reader.wait_for_frame()
        reader.on_frame = lambda bgr, t: seen.append((int(bgr[0, 0, 0]), t))
        capture.push(_image(2))
        capture.push(_image(3))
        _wait_until(lambda: len(seen) == 2)
    assert [value for value, _ in seen] == [2, 3]


def test_close_releases_an_owned_capture() -> None:
    capture = FakeCapture()
    FrameReader(capture, "test").close()
    assert capture.released


def test_close_leaves_a_borrowed_capture_alone() -> None:
    capture = FakeCapture()
    FrameReader(capture, "test", owns_capture=False).close()
    assert not capture.released
