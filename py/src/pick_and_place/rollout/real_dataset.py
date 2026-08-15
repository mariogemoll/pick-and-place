# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turn a running episode into dataset rows, one per control tick.

The dataset's central invariant is that row *i* holds the observation the policy
saw at tick *i* and the action issued from it — so exactly one frame per camera
per tick, in order, no gaps. Capture therefore runs in lockstep with the control
loop rather than at whatever rate the cameras happen to manage: each tick the
loop snapshots every camera's newest frame and queues one record.

The writing itself does not belong on the control loop. Color conversion and the
dataset's ``add_frame`` take milliseconds and would show up directly as missed
ticks, so a writer thread drains the queue behind the loop. Frames are enqueued
in tick order and drained in the same order, so the episode keeps it.

Separately from all of that, ``on_frame`` on each reader feeds the session's
continuous capture at the cameras' full native rate — that is the video footage,
not the dataset, and it is deliberately not tick-sampled.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pick_and_place.data.recording import RecordingSession
from pick_and_place.runtime.frame_reader import FrameReader
from pick_and_place.runtime.wrist_servo import request_camera_fps

#: How long to wait for every camera stream to deliver its first frame. Shared
#: across the readers rather than allowed per camera, because they start together.
FIRST_FRAME_TIMEOUT = 2.0


@dataclass(frozen=True)
class _TickRecord:
    """One control tick's recording payload, queued for the writer thread."""

    state: np.ndarray
    action: np.ndarray
    wrist_bgr: np.ndarray
    overhead_bgr: np.ndarray
    workspace_bgr: np.ndarray | None
    sim_qpos: np.ndarray
    wall_t: float
    servo_active: bool
    servo_source: np.ndarray | None


class TickRecorder:
    """Writes one dataset frame per :meth:`record` call, off the control loop.

    Built by :meth:`open`, which needs every camera to be live already: the
    dataset's image features are declared from the first frame that actually
    arrived, so a camera that ignored the requested resolution still yields a
    self-consistent episode.

    The wrist reader is *borrowed*, not owned — the visual servo opened it and
    closes it. The overhead and workspace captures belong to the caller, who
    keeps them across episodes; only the readers wrapped around them are closed
    here.
    """

    def __init__(
        self,
        recording: RecordingSession,
        wrist_reader: FrameReader,
        overhead_reader: FrameReader,
        workspace_reader: FrameReader | None,
        wall_start: float,
    ) -> None:
        self.recording = recording
        self.wrist_reader = wrist_reader
        self.overhead_reader = overhead_reader
        self.workspace_reader = workspace_reader
        self.wall_start = wall_start
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._write_loop, name="record-writer", daemon=True)
        self._thread.start()

    @classmethod
    def open(
        cls,
        recording: RecordingSession,
        *,
        wrist_reader: FrameReader | None,
        overhead_capture: Any,
        workspace_capture: Any,
        wall_start: float,
    ) -> TickRecorder:
        """Start the overhead and workspace readers and declare the dataset."""
        if wrist_reader is None or overhead_capture is None or not overhead_capture.isOpened():
            raise RuntimeError("Recording requires both the wrist and overhead cameras to be open")

        request_camera_fps(overhead_capture, "overhead")
        overhead_reader = FrameReader(
            overhead_capture,
            "overhead",
            on_frame=lambda bgr, t: recording.record_live_frame("overhead", bgr, t),
            owns_capture=False,
        )
        workspace_reader = None
        if workspace_capture is not None:
            if not workspace_capture.isOpened():
                raise RuntimeError("Workspace camera is not open")
            request_camera_fps(workspace_capture, "workspace")
            workspace_reader = FrameReader(
                workspace_capture,
                "workspace",
                on_frame=lambda bgr, t: recording.record_live_frame("workspace", bgr, t),
                owns_capture=False,
            )

        # The readers run concurrently, so the budget is shared, not per camera.
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT

        def first_frame(reader: FrameReader) -> np.ndarray:
            return reader.wait_for_frame(max(0.0, deadline - time.monotonic())).bgr

        try:
            wrist_shape = first_frame(wrist_reader).shape
            overhead_shape = first_frame(overhead_reader).shape
            workspace_shape = (
                None if workspace_reader is None else first_frame(workspace_reader).shape
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Timed out waiting for the episode camera streams to start"
            ) from exc

        if not recording.initialized:
            recording.create_dataset(wrist_shape, overhead_shape, workspace_shape)
        recording.start_live_capture(time.monotonic() - wall_start)
        return cls(recording, wrist_reader, overhead_reader, workspace_reader, wall_start)

    def record(
        self,
        commanded: np.ndarray,
        actual: np.ndarray,
        sim_qpos: np.ndarray,
        *,
        servo_active: bool = False,
        servo_source: np.ndarray | None = None,
    ) -> None:
        """Queue this tick's frame, or skip it if any camera has yet to deliver one."""
        wrist = self.wrist_reader.latest()
        overhead = self.overhead_reader.latest()
        workspace = None if self.workspace_reader is None else self.workspace_reader.latest()
        if wrist is None or overhead is None or (self.workspace_reader is not None and workspace is None):
            return
        # Start optional audio capture here, alongside the first captured tick,
        # instead of in the asynchronous encoder thread so its clock has the same
        # origin as the MP4 frames.
        self.recording.start_audio_capture()
        self._queue.put(
            _TickRecord(
                state=actual.astype(np.float32),
                action=commanded.astype(np.float32),
                wrist_bgr=wrist.bgr,
                overhead_bgr=overhead.bgr,
                workspace_bgr=None if workspace is None else workspace.bgr,
                sim_qpos=sim_qpos,
                wall_t=time.monotonic() - self.wall_start,
                servo_active=servo_active,
                servo_source=servo_source,
            )
        )

    def close(self) -> None:
        """Stop the readers, then drain everything already queued.

        Called after the visual servo has stopped, so no thread can still be
        producing, and before the episode is committed, so the drop count below
        covers the whole episode.
        """
        for reader in (self.overhead_reader, self.workspace_reader):
            if reader is not None:
                reader.close()
        self.recording.stop_live_capture()
        self._queue.put(None)
        self._thread.join(timeout=30.0)

    def finish(self, status: str, metadata: dict[str, Any] | None, frame_count: int) -> None:
        """Commit a successful episode, or discard one that did not finish."""
        if not self.recording.initialized:
            return
        # The writer thread has drained, so the drop count now covers the whole
        # episode. A dropped frame would desync the video from the recorded rows,
        # so fail before committing rather than save a corrupt episode.
        dropped = self.recording.dropped_frame_count()
        if dropped:
            raise RuntimeError(
                f"Streaming video encoder dropped {dropped} frame(s): the encoder "
                "cannot keep pace with capture, which would desync the video from the "
                "recorded frames. Use a hardware vcodec (auto) or raise the encoder "
                "queue size."
            )
        if status == "success":
            self.recording.save_episode(metadata)
            print(f"Saved episode to dataset ({frame_count} frames).")
        elif self.recording.has_pending_frames():
            self.recording.discard_episode()
            print(f"Discarded {status} episode (not added to dataset).")

    def _write_loop(self) -> None:
        """Drain queued frames into the dataset until the ``None`` sentinel."""
        import cv2

        while True:
            record = self._queue.get()
            if record is None:
                return
            frame = {
                "observation.state": record.state,
                "action": record.action,
                "observation.images.wrist": cv2.cvtColor(record.wrist_bgr, cv2.COLOR_BGR2RGB),
                "observation.images.overhead": cv2.cvtColor(
                    record.overhead_bgr, cv2.COLOR_BGR2RGB
                ),
                "task": self.recording.task,
            }
            if record.workspace_bgr is not None:
                frame["observation.images.workspace"] = cv2.cvtColor(
                    record.workspace_bgr, cv2.COLOR_BGR2RGB
                )
            self.recording.record_frame(
                frame,
                sim_qpos=record.sim_qpos,
                wall_t=record.wall_t,
                servo_active=record.servo_active,
                servo_source=record.servo_source,
            )
