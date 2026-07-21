# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Write self-contained, synchronized videos of real robot episodes.

Unlike a training dataset, an episode directory is meant to be opened directly
by a web viewer.  Every camera gets an MP4 with exactly one frame per control
tick, and ``timeline.npz`` has the matching joint samples and simulated state.
Incomplete episodes stay in a temporary directory and are deleted.
"""

from __future__ import annotations

import datetime
from fractions import Fraction
import json
import queue
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.cube_detection import detect_tags, make_cube_detector
from pick_and_place.camera_compare import load_intrinsics


def _mux_audio_into(video_path: Path, wav_path: Path, duration: float) -> None:
    """Replace ``video_path`` with a copy carrying the WAV as an AAC track.

    The audio is padded/trimmed to exactly ``duration`` seconds so it stays
    aligned with the video timeline it was captured against.
    """
    import imageio_ffmpeg

    muxed_path = video_path.with_stem(f"{video_path.stem}_with_audio")
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path), "-i", str(wav_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-af", "apad", "-t", f"{duration:.9f}", "-movflags", "+faststart", str(muxed_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Could not mux audio into {video_path.name}: {result.stderr.strip()}")
    muxed_path.replace(video_path)


class _WorkspaceAudioCapture:
    """Capture PCM from the selected workspace audio input for one episode."""

    def __init__(self, device: str | int | None) -> None:
        import sounddevice as sd

        info = sd.query_devices(device, "input")
        self.channels = min(2, int(info["max_input_channels"]))
        if self.channels < 1:
            raise RuntimeError(f"Audio input {device!r} has no input channels")
        self.sample_rate = int(round(float(info["default_samplerate"])))
        self._chunks: list[np.ndarray] = []
        self._stream = sd.InputStream(
            device=device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._receive,
        )

    def _receive(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"warning: workspace audio capture: {status}")
        self._chunks.append(indata.copy())

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        """Stop accepting samples while retaining the captured PCM."""
        if self._stream.active:
            self._stream.stop()

    def save_wav(self, path: Path) -> bool:
        self.stop()
        self._stream.close()
        if not self._chunks:
            return False
        with wave.open(str(path), "wb") as output:
            output.setnchannels(self.channels)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(np.concatenate(self._chunks).tobytes())
        return True

    def close(self) -> None:
        self.stop()
        self._stream.close()


class _LiveCameraCapture:
    """Encode native-rate camera frames using one monotonic timestamp origin."""

    _TIME_BASE = Fraction(1, 1_000_000)

    def __init__(
        self,
        directory: Path,
        names: tuple[str, ...],
        origin: float,
        undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]],
        on_tags=None,
    ) -> None:
        self._directory = directory
        self._origin = origin
        self._undistort_maps = undistort_maps
        self._on_tags = on_tags
        self._queues = {name: queue.Queue() for name in names}
        self._threads = {
            name: threading.Thread(target=self._write, args=(name,), daemon=True)
            for name in names
        }
        self._last_pts = {name: 0 for name in names}
        self.duration = 0.0
        for thread in self._threads.values():
            thread.start()

    def submit(self, name: str, bgr: np.ndarray, captured_at: float) -> None:
        if name in self._queues:
            self._queues[name].put((bgr.copy(), captured_at))

    def _write(self, name: str) -> None:
        import av
        import cv2

        container = None
        stream = None
        detector = make_cube_detector(quad_decimate=1.5) if self._on_tags is not None else None
        while True:
            item = self._queues[name].get()
            if item is None:
                break
            bgr, captured_at = item
            bgr = cv2.remap(bgr, *self._undistort_maps[name], cv2.INTER_LINEAR)
            if container is None:
                height, width = bgr.shape[:2]
                container = av.open(str(self._directory / f"{name}_live.mp4"), "w")
                stream = container.add_stream("libx264", rate=30, options={"preset": "ultrafast"})
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.time_base = self._TIME_BASE
                stream.codec_context.time_base = self._TIME_BASE
            pts = max(self._last_pts[name] + 1, round((captured_at - self._origin) * 1_000_000))
            self._last_pts[name] = pts
            self.duration = max(self.duration, pts * float(self._TIME_BASE))
            frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            frame.pts = pts
            frame.time_base = self._TIME_BASE
            for packet in stream.encode(frame):
                container.mux(packet)
            if self._on_tags is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self._on_tags(name, captured_at, detect_tags(rgb, detector))
        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()

    def close(self) -> None:
        for capture_queue in self._queues.values():
            capture_queue.put(None)
        for thread in self._threads.values():
            thread.join(timeout=30.0)


class LiveVideoRecorder:
    """Record continuous, undistorted, native-rate camera MP4s with optional audio.

    The standalone counterpart to :class:`EpisodeVideoSession`'s live capture:
    one recording spans a whole run instead of an episode. Every camera writes
    ``<name>_live.mp4`` at its native resolution and frame rate — undistorted
    with its calibrated intrinsics but never cropped or resized — with frame
    timestamps on one shared monotonic origin, so the videos stay mutually
    synchronized. When audio is enabled the selected input is captured for the
    whole recording and muxed into every video on :meth:`close`.
    """

    def __init__(
        self,
        directory: Path,
        undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        audio: bool = False,
        audio_device: str | int | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._names = tuple(undistort_maps)
        self._capture = _LiveCameraCapture(
            directory,
            self._names,
            origin=time.monotonic(),
            undistort_maps=undistort_maps,
        )
        self._audio = _WorkspaceAudioCapture(audio_device) if audio else None
        if self._audio is not None:
            self._audio.start()

    def submit(self, name: str, bgr: np.ndarray, captured_at: float) -> None:
        """Queue one raw BGR frame captured at monotonic time ``captured_at``."""
        self._capture.submit(name, bgr, captured_at)

    def close(self) -> None:
        """Flush the encoders and mux the captured audio into every video."""
        if self._audio is not None:
            self._audio.stop()
        self._capture.close()
        if self._audio is None:
            return
        audio, self._audio = self._audio, None
        wav_path = self._directory / "audio.wav"
        if not audio.save_wav(wav_path):
            print("warning: audio input produced no samples")
            return
        for name in self._names:
            video_path = self._directory / f"{name}_live.mp4"
            if video_path.exists():
                _mux_audio_into(video_path, wav_path, self._capture.duration)
        wav_path.unlink(missing_ok=True)


@dataclass
class EpisodeVideoSession:
    """Record synchronized MP4 camera views and a simulation-replay timeline.

    Implements the same recording interface as
    :class:`pick_and_place.recording.RecordingSession`, so
    :func:`pick_and_place.executor.execute_episode` drives either
    interchangeably.
    """

    root: Path
    fps: float
    task: str
    camera_intrinsics: dict[str, Path] = field(default_factory=dict)
    workspace_audio: bool = False
    workspace_audio_device: str | int | None = None
    live_videos: bool = False
    record_tag_locations: bool = True
    input_rectified: bool = False
    _shapes: dict[str, tuple[int, int]] = field(default_factory=dict, init=False)
    _writers: dict[str, Any] = field(default_factory=dict, init=False)
    _pending_dir: Path | None = field(default=None, init=False)
    _rows: list[dict[str, np.ndarray | float | bool]] = field(default_factory=list, init=False)
    _episode_index: int = field(default=0, init=False)
    _detector: Any = field(default=None, init=False)
    _undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False
    )
    _camera_metadata: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _audio_capture: _WorkspaceAudioCapture | None = field(default=None, init=False)
    _live_capture: _LiveCameraCapture | None = field(default=None, init=False)
    _live_capture_origin_wall_t: float | None = field(default=None, init=False)
    _live_capture_origin_monotonic: float | None = field(default=None, init=False)
    _live_capture_duration: float | None = field(default=None, init=False)
    _tag_locations: list[dict[str, Any]] = field(default_factory=list, init=False)
    _visual_servo_overlays: list[dict[str, Any]] = field(default_factory=list, init=False)
    _tag_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def create_dataset(
        self,
        wrist_shape: tuple,
        overhead_shape: tuple,
        workspace_shape: tuple | None = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._shapes = {
            "wrist": tuple(wrist_shape[:2]),
            "overhead": tuple(overhead_shape[:2]),
        }
        if workspace_shape is not None:
            self._shapes["workspace"] = tuple(workspace_shape[:2])
        if self.workspace_audio and "workspace" not in self._shapes:
            raise RuntimeError("workspace audio requires a workspace camera")
        import cv2

        for name, (height, width) in self._shapes.items():
            intrinsics = self.camera_intrinsics.get(name)
            if intrinsics is None:
                raise RuntimeError(f"Missing intrinsics for the {name} episode video")
            camera_matrix, undistort_map = load_intrinsics(intrinsics, width, height, cv2)
            if not self.input_rectified:
                self._undistort_maps[name] = undistort_map
            self._camera_metadata[name] = {
                "video": f"{name}.mp4",
                "width": width,
                "height": height,
                "rectified": True,
                "camera_matrix": camera_matrix.tolist(),
                "intrinsics_file": str(intrinsics),
            }

    @property
    def initialized(self) -> bool:
        """Whether :meth:`create_dataset` has run and episodes can be recorded."""
        return bool(self._shapes)

    def _record_tags(self, camera: str, timestamp: float, detections, *, timebase: str) -> None:
        if not self.record_tag_locations:
            return
        entry = {
            "camera": camera,
            "t": timestamp,
            "timebase": timebase,
            "tags": [
                {"id": int(detection.tag_id), "corners": np.asarray(detection.corners).tolist()}
                for detection in detections
            ],
        }
        with self._tag_lock:
            self._tag_locations.append(entry)

    def record_visual_servo_overlay(self, captured_at: float, primitives: dict[str, Any]) -> None:
        """Store wrist-overlay geometry in the native live-video timebase."""
        if self._live_capture_origin_monotonic is None:
            return
        entry = {
            "t": captured_at - self._live_capture_origin_monotonic,
            **primitives,
        }
        with self._tag_lock:
            self._visual_servo_overlays.append(entry)

    def dropped_frame_count(self) -> int:
        return 0

    def _start_episode(self) -> None:
        import imageio_ffmpeg

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._pending_dir = self.root / f".pending_{stamp}"
        self._pending_dir.mkdir()
        for name, (height, width) in self._shapes.items():
            writer = imageio_ffmpeg.write_frames(
                str(self._pending_dir / f"{name}.mp4"),
                (width, height),
                fps=self.fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                macro_block_size=1,
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
            self._writers[name] = writer

    def start_audio_capture(self) -> None:
        """Start audio at the first recorded control tick, not writer dequeue time."""
        if not self.workspace_audio or self._audio_capture is not None:
            return
        self._audio_capture = _WorkspaceAudioCapture(self.workspace_audio_device)
        self._audio_capture.start()

    def start_live_capture(self, wall_t: float) -> None:
        """Start native-rate video capture with a shared clock for every camera."""
        if not self.live_videos or self._live_capture is not None:
            return
        if self._pending_dir is None:
            self._start_episode()
        self._live_capture_origin_wall_t = wall_t
        self._live_capture_origin_monotonic = time.monotonic()
        self._live_capture = _LiveCameraCapture(
            self._pending_dir,
            tuple(self._shapes),
            origin=self._live_capture_origin_monotonic,
            undistort_maps=self._undistort_maps,
            on_tags=lambda name, captured_at, detections: self._record_tags(
                name,
                captured_at
                - self._live_capture_origin_monotonic
                + self._live_capture_origin_wall_t,
                detections,
                timebase="live_video",
            ),
        )
        self.start_audio_capture()
        for name in self._shapes:
            self._camera_metadata[name]["live_video"] = f"{name}_live.mp4"
            self._camera_metadata[name]["live_timebase"] = "shared_monotonic"

    def record_live_frame(self, name: str, bgr: np.ndarray, captured_at: float) -> None:
        if self._live_capture is not None:
            self._live_capture.submit(name, bgr, captured_at)

    def stop_live_capture(self) -> None:
        if self._live_capture is not None:
            self._live_capture.close()
            self._live_capture_duration = self._live_capture.duration
            self._live_capture = None

    def stop_audio_capture(self) -> None:
        """Stop audio at the end of the control timeline before video encoding drains."""
        if self._audio_capture is not None:
            self._audio_capture.stop()

    def record_frame(
        self,
        frame: dict[str, Any],
        *,
        sim_qpos: np.ndarray | None = None,
        wall_t: float | None = None,
        servo_active: bool = False,
        servo_source: np.ndarray | None = None,
    ) -> None:
        if self._pending_dir is None:
            self._start_episode()
        images = {
            "wrist": frame["observation.images.wrist"],
            "overhead": frame["observation.images.overhead"],
        }
        if "observation.images.workspace" in frame:
            images["workspace"] = frame["observation.images.workspace"]
        import cv2

        if self.record_tag_locations and self._detector is None:
            self._detector = make_cube_detector(quad_decimate=1.5)
        for name, rgb in images.items():
            if name in self._undistort_maps:
                rgb = cv2.remap(rgb, *self._undistort_maps[name], cv2.INTER_LINEAR)
            if self.record_tag_locations:
                self._record_tags(
                    name,
                    float(wall_t if wall_t is not None else len(self._rows) / self.fps),
                    detect_tags(rgb, self._detector),
                    timebase="timeline",
                )
            self._writers[name].send(rgb)
        self._rows.append(
            {
                "state": np.asarray(frame["observation.state"], dtype=np.float32),
                "action": np.asarray(frame["action"], dtype=np.float32),
                "sim_qpos": np.asarray(sim_qpos if sim_qpos is not None else [], dtype=np.float64),
                "wall_t": float(wall_t if wall_t is not None else len(self._rows) / self.fps),
                "servo_active": bool(servo_active),
                "servo_source": np.asarray(
                    servo_source if servo_source is not None else (np.nan, np.nan, np.nan, np.nan),
                    dtype=np.float64,
                ),
            }
        )

    def has_pending_frames(self) -> bool:
        return bool(self._rows)

    def _close_writers(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def _mux_workspace_audio(self, video_duration: float) -> None:
        if self._audio_capture is None or self._pending_dir is None:
            return
        audio = self._audio_capture
        self._audio_capture = None
        wav_path = self._pending_dir / "workspace.wav"
        if not audio.save_wav(wav_path):
            print("warning: workspace audio input produced no samples")
            return
        targets = [(self._pending_dir / "workspace.mp4", video_duration)]
        if self.live_videos and self._live_capture_duration is not None:
            targets.extend(
                (self._pending_dir / f"{name}_live.mp4", self._live_capture_duration)
                for name in self._shapes
            )
        for video_path, duration in targets:
            if not video_path.exists():
                continue
            _mux_audio_into(video_path, wav_path, duration)
        wav_path.unlink(missing_ok=True)
        audio_metadata = {
            "codec": "aac",
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
        }
        for name in self._shapes:
            self._camera_metadata[name]["audio"] = audio_metadata

    def save_episode(self, episode_metadata: dict[str, Any] | None = None) -> None:
        if self._pending_dir is None or not self._rows:
            raise RuntimeError("cannot save an empty video episode")
        self._close_writers()
        self._mux_workspace_audio(len(self._rows) / self.fps)
        timeline = {
            key: np.asarray([row[key] for row in self._rows])
            for key in ("state", "action", "sim_qpos", "wall_t", "servo_active", "servo_source")
        }
        np.savez_compressed(self._pending_dir / "timeline.npz", **timeline)
        with (self._pending_dir / "tag_locations.jsonl").open("w") as output:
            for entry in self._tag_locations:
                output.write(json.dumps(entry) + "\n")
        with (self._pending_dir / "visual_servo_overlays.jsonl").open("w") as output:
            for entry in self._visual_servo_overlays:
                output.write(json.dumps(entry) + "\n")
        metadata = {
            "format_version": 1,
            "fps": self.fps,
            "frames": len(self._rows),
            "task": self.task,
            "cameras": self._camera_metadata,
            "live_capture_origin_wall_t": self._live_capture_origin_wall_t,
            "tag_locations": {
                "file": "tag_locations.jsonl",
                "coordinate_space": "rectified_pixel",
                "timebase": "timeline.wall_t",
            },
            "visual_servo_overlays": {
                "file": "visual_servo_overlays.jsonl",
                "coordinate_space": "rectified_pixel",
                "timebase": "live_video",
            },
            "episode_metadata": episode_metadata or {},
        }
        (self._pending_dir / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
        destination = self.root / f"episode_{self._episode_index:04d}"
        self._episode_index += 1
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite existing episode directory {destination}")
        self._pending_dir.rename(destination)
        self._pending_dir = None
        self._rows.clear()
        self._tag_locations.clear()
        self._visual_servo_overlays.clear()

    def discard_episode(self) -> None:
        self._close_writers()
        self.stop_live_capture()
        if self._audio_capture is not None:
            self._audio_capture.close()
            self._audio_capture = None
        if self._pending_dir is not None:
            shutil.rmtree(self._pending_dir, ignore_errors=True)
        self._pending_dir = None
        self._rows.clear()
        self._tag_locations.clear()
        self._visual_servo_overlays.clear()

    def finalize(self) -> None:
        self.discard_episode()
