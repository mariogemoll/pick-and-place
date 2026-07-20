# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run cube-tag detection in a disposable child process.

``libapriltag`` segfaults on rare inputs, somewhere in the union-find of its
gradient-clustering pass. The exact trigger is unconfirmed; what is established
is that it is *not* a threading bug, since it reproduces under strictly
single-threaded detection (``nthreads=1``). In a long sharded recording run
that native crash takes down the whole shard worker, and the shard's
episode-metadata parquet is left mid-write and unreadable, so every episode it
had accumulated is lost.

Detecting out-of-process turns that into a non-event. The servo loop already
treats "no tags this tick" as normal — the cube is often occluded by the
gripper during descent — so a crashed helper costs one detection tick, and the
next tick runs against a freshly spawned helper. Nothing needs to unwind, and
no episode is lost.

The frame that killed a helper is also exactly the repro artefact root-causing
the bug needs, and it cannot be recovered after the fact, so ``crash_dump_dir``
saves it on the way past. That turns ordinary recording runs into the
collection mechanism, instead of needing a separate instrumented run to go
hunting for one.
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Generous enough that a slow full-resolution detect() on a loaded machine
# never trips it; short enough that a wedged helper can't stall a run for long.
_DETECT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class TagDetection:
    """The parts of a ``pupil_apriltags`` detection that survive pickling.

    Duck-types the real ``Detection`` for every field this codebase reads off
    one (``tag_id``, ``corners``, ``center``).
    """

    tag_id: int
    corners: NDArray
    center: NDArray


def _detector_main(conn, quad_decimate: float, nthreads: int) -> None:
    """Child entry point: detect on each frame received until the pipe closes."""
    from pick_and_place.cube_detection import make_cube_detector

    detector = make_cube_detector(quad_decimate, nthreads=nthreads)
    while True:
        try:
            gray = conn.recv()
        except EOFError:
            return
        if gray is None:
            return
        conn.send(
            [
                TagDetection(
                    tag_id=int(det.tag_id),
                    corners=np.asarray(det.corners, dtype=float),
                    center=np.asarray(det.center, dtype=float),
                )
                for det in detector.detect(gray)
            ]
        )


class DetectorProcess:
    """AprilTag detection behind a process boundary, respawned when it dies.

    Duck-types ``pupil_apriltags.Detector`` as far as :func:`detect_tags` uses
    it, so it drops into a :class:`~pick_and_place.cube_detection.CubeTracker`
    as its ``detector`` and every existing call site works unchanged. A helper
    that segfaults (or wedges) yields ``[]`` for that frame and is replaced
    before the next one.
    """

    def __init__(
        self,
        quad_decimate: float = 1.0,
        nthreads: int = 1,
        crash_dump_dir: str | Path | None = None,
        timeout: float = _DETECT_TIMEOUT_S,
    ) -> None:
        self._quad_decimate = float(quad_decimate)
        self._nthreads = int(nthreads)
        self._timeout = float(timeout)
        self._crash_dump_dir = Path(crash_dump_dir) if crash_dump_dir is not None else None
        # "spawn" so the child never inherits the parent's MuJoCo/EGL state; it
        # only needs cv2 and the detector, and a forked GL context is a hazard.
        self._ctx = multiprocessing.get_context("spawn")
        self._proc = None
        self._conn = None
        self.crash_count = 0
        self._start()

    def _start(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_detector_main,
            args=(child_conn, self._quad_decimate, self._nthreads),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # only the child holds the far end, so recv() sees EOF
        self._proc, self._conn = proc, parent_conn

    def _dump(self, gray: NDArray) -> None:
        if self._crash_dump_dir is None:
            return
        self._crash_dump_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time():.6f}".replace(".", "_")
        path = self._crash_dump_dir / f"apriltag_crash_{stamp}.npy"
        np.save(path, gray)

    def _restart(self, gray: NDArray) -> None:
        """Replace a helper that crashed or wedged, keeping the frame that did it."""
        self.crash_count += 1
        self._dump(gray)
        # Kill rather than ask politely: a segfaulted child is already gone, and
        # a wedged one would not answer the shutdown sentinel anyway.
        self._teardown(graceful=False)
        self._start()

    def detect(self, gray: NDArray) -> list[TagDetection]:
        """Detect tags, returning ``[]`` if the helper died on this frame."""
        try:
            self._conn.send(gray)
            if not self._conn.poll(self._timeout):
                raise TimeoutError("apriltag helper did not answer in time")
            return self._conn.recv()
        except (EOFError, OSError):
            # A segfault in the child drops the pipe (EOFError / BrokenPipeError);
            # a wedge trips the poll above (TimeoutError). All are OSError bar EOF.
            self._restart(gray)
            return []

    def _teardown(self, graceful: bool) -> None:
        if self._conn is not None:
            if graceful:
                try:
                    self._conn.send(None)
                except OSError:
                    pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            if graceful:
                self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5.0)
            self._proc = None

    def close(self) -> None:
        self._teardown(graceful=True)

    def __enter__(self) -> DetectorProcess:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
