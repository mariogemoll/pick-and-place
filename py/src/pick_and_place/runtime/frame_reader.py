# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""One background thread per camera, holding only the newest frame.

Every loop that drives the physical rig wants the same thing from a camera: the
most recent image, right now, without ever blocking on capture. ``VideoCapture``
cannot give that — ``read()`` blocks until the next frame and hands back a
backlog in order, so a loop that reads inline runs at the camera's pace and acts
on stale images.

A :class:`FrameReader` runs ``read()`` on its own thread and keeps a single slot:
each new frame overwrites the last. The loop snapshots that slot once per tick,
so a slow capture costs nothing and the arm always acts on the freshest image.
Frames that never reach the loop are not necessarily lost — ``on_frame`` fires on
the reader thread for *every* capture, which is how a recorder logs the camera's
full native rate while the control loop samples it at ``CONTROL_HZ``.

A published frame is never mutated, by the reader or by anyone else, so
snapshots are handed out without copying.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """One capture: the BGR image, its sequence number, and when it arrived.

    ``index`` counts from 1 and only ever increases, so a consumer running
    independently of the reader can tell a fresh frame from the one it already
    processed. ``captured_at`` is ``time.monotonic()`` taken immediately after
    ``read()`` returned, which is the closest timestamp available to exposure.
    """

    bgr: np.ndarray
    index: int
    captured_at: float


class FrameReader:
    """Keep ``capture``'s newest frame available without blocking the caller.

    Owns a daemon thread from construction until :meth:`close`. ``capture`` must
    already be open; :func:`open_frame_reader` does both in one step. With
    ``owns_capture`` set the capture is released on close, otherwise it is left
    to whoever opened it.

    ``on_frame(bgr, captured_at)`` is called on the reader thread for every
    captured frame. It may be replaced at any time — a recorder that starts
    partway through a run just assigns to the attribute.
    """

    def __init__(
        self,
        capture: Any,
        label: str,
        *,
        on_frame: Callable[[np.ndarray, float], None] | None = None,
        owns_capture: bool = True,
    ) -> None:
        self.label = label
        self.on_frame = on_frame
        self._capture = capture
        self._owns_capture = owns_capture
        self._condition = threading.Condition()
        self._frame: CameraFrame | None = None
        self._last_read_index = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"{label}-reader", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        index = 0
        while self._running:
            ok, bgr = self._capture.read()
            if not ok or bgr is None:
                continue
            captured_at = time.monotonic()
            index += 1
            with self._condition:
                self._frame = CameraFrame(bgr, index, captured_at)
                self._condition.notify_all()
            on_frame = self.on_frame
            if on_frame is not None:
                on_frame(bgr, captured_at)

    def latest(self) -> CameraFrame | None:
        """The newest frame, or ``None`` if the stream has not produced one yet."""
        with self._condition:
            return self._frame

    def wait_for_frame(self, timeout: float = 2.0) -> CameraFrame:
        """The newest frame, waiting up to ``timeout`` for the stream to start.

        Returns immediately once any frame has arrived, so this is also the
        blocking form of :meth:`latest` for a loop that cannot proceed without an
        image. Raises ``RuntimeError`` if none arrives in time.
        """
        with self._condition:
            if self._frame is None:
                self._condition.wait_for(lambda: self._frame is not None, timeout=timeout)
            if self._frame is None:
                raise RuntimeError(f"timed out waiting for the {self.label} camera stream to start")
            return self._frame

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None]:
        """Wait for a frame this method has not returned before, ``cv2``-style.

        The ``VideoCapture.read()`` shape, so a reader can stand in for the
        capture wherever code samples distinct frames over time rather than
        polling the latest one — the calibration solvers average over a series.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: not self._running
                or (self._frame is not None and self._frame.index > self._last_read_index),
                timeout=timeout,
            )
            frame = self._frame
            if frame is None or frame.index <= self._last_read_index:
                return False, None
            self._last_read_index = frame.index
            return True, frame.bgr

    def close(self) -> None:
        """Stop the reader thread and, if this reader opened it, release the capture."""
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        if self._owns_capture:
            self._capture.release()

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def capture_backend(cv2: Any) -> int:
    """Return the ``VideoCapture`` backend to open rig cameras with.

    Every OpenCV build defines every backend constant, whether or not it was
    compiled in, so ``hasattr(cv2, "CAP_AVFOUNDATION")`` is true on Linux too and
    asking for it there opens nothing. Pick by platform instead: AVFoundation on
    macOS, V4L2 on Linux, and let OpenCV choose anywhere else.
    """
    if sys.platform == "darwin":
        return int(cv2.CAP_AVFOUNDATION)
    if sys.platform.startswith("linux"):
        return int(cv2.CAP_V4L2)
    return int(cv2.CAP_ANY)


def open_capture(source: str, width: int, height: int, label: str) -> Any:
    """Open ``source`` as a ``cv2.VideoCapture`` requesting ``width`` x ``height``.

    ``source`` is an OpenCV device index or a device path. Raises ``RuntimeError``
    naming ``label`` if the device will not open, rather than handing back a
    capture whose every ``read()`` fails.
    """
    import cv2

    from pick_and_place.calibration.cam_align_solve import parse_index_or_path

    backend = capture_backend(cv2)
    capture = cv2.VideoCapture(parse_index_or_path(source), backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open {label} camera {source!r}")
    if backend == int(cv2.CAP_V4L2):
        # V4L2 defaults to uncompressed YUYV, and the rig's overhead and wrist
        # streams together do not fit in one USB controller's bandwidth: both
        # cameras open, but the second one's every read() fails. MJPG compresses
        # on the camera, so the two stream side by side. Request it before the
        # resolution -- the driver picks the frame size from the active format.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def open_frame_reader(
    source: str,
    width: int,
    height: int,
    label: str,
    *,
    on_frame: Callable[[np.ndarray, float], None] | None = None,
) -> FrameReader:
    """Open ``source`` and start reading it in the background."""
    return FrameReader(open_capture(source, width, height, label), label, on_frame=on_frame)
