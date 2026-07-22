# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""LeRobot dataset recording session for real-arm episode collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.follower import JOINT_NAMES


def _silence_ffmpeg_encoder_reports() -> None:
    """Keep routine FFmpeg codec reports off stderr while retaining errors.

    LeRobot's streaming camera thread restores FFmpeg's native callback after
    every video. That callback prints libx264's full statistics block, so a
    per-episode dataset collection produces hundreds of lines that obscure the
    recorder's progress bar. Replacing the restore hook is process-local; pool
    workers each configure their own PyAV instance when their first dataset is
    created.
    """
    import av

    def restore_quiet_callback() -> None:
        av.logging.set_level(av.logging.ERROR)

    restore_quiet_callback()
    av.logging.restore_default_callback = restore_quiet_callback


@dataclass
class RecordingSession:
    """Holds the ``LeRobotDataset`` written across one collection run.

    The dataset is created lazily on the first recorded episode, once the camera
    frame shapes are known, and reused for every later episode. The runner owns
    it and calls :meth:`finalize` when the run ends. Episodes are added straight
    into the dataset during execution (one frame per control tick), so there are
    no intermediate video/motor files and no separate export step.

    This class and :class:`pick_and_place.episode_video.EpisodeVideoSession`
    implement the same recording interface consumed by :func:`pick_and_place.executor.execute_episode`:
    ``create_dataset``/``initialized``, ``record_frame``, ``has_pending_frames``,
    ``save_episode``/``discard_episode``/``finalize``, ``dropped_frame_count``,
    plus the live-capture hooks (``record_live_frame``, ``start_live_capture``/
    ``stop_live_capture``, ``start_audio_capture``/``stop_audio_capture``,
    ``record_visual_servo_overlay``), which only the video session acts on.
    """

    repo_id: str
    root: Path
    task: str
    fps: float
    vcodec: str = "auto"
    streaming_encoding: bool = True
    image_writer_threads: int = 4
    # Frames the streaming encoder may buffer per camera. The default of 30 (one
    # second at 30 Hz) overflows during the descent's visual-servo tick, when
    # AprilTag detection on the control thread briefly starves the encoder and
    # frames get dropped. A deeper buffer rides through that spike.
    encoder_queue_maxsize: int = 300
    dataset: Any = field(default=None, init=False)

    def create_dataset(
        self,
        wrist_shape: tuple,
        overhead_shape: tuple,
        workspace_shape: tuple | None = None,
    ) -> None:
        _silence_ffmpeg_encoder_reports()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        joint_names = list(JOINT_NAMES)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": joint_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": joint_names,
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (wrist_shape[0], wrist_shape[1], 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.overhead": {
                "dtype": "video",
                "shape": (overhead_shape[0], overhead_shape[1], 3),
                "names": ["height", "width", "channels"],
            },
        }
        if workspace_shape is not None:
            features["observation.images.workspace"] = {
                "dtype": "video",
                "shape": (workspace_shape[0], workspace_shape[1], 3),
                "names": ["height", "width", "channels"],
            }
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=int(round(self.fps)),
            features=features,
            root=self.root,
            robot_type="so101",
            use_videos=True,
            image_writer_threads=self.image_writer_threads,
            vcodec=self.vcodec,
            streaming_encoding=self.streaming_encoding,
            encoder_queue_maxsize=self.encoder_queue_maxsize,
            video_backend="pyav",
        )

    @property
    def initialized(self) -> bool:
        """Whether :meth:`create_dataset` has run and episodes can be recorded."""
        return self.dataset is not None

    def record_frame(
        self,
        frame: dict[str, Any],
        *,
        sim_qpos: np.ndarray | None = None,
        wall_t: float | None = None,
        servo_active: bool = False,
        servo_source: np.ndarray | None = None,
    ) -> None:
        """Add one control tick's frame features to the pending episode.

        The keyword extras describe the simulation/servo timeline that a video
        session stores alongside its MP4s; a training dataset keeps only the
        schema-validated frame features, so they are accepted and ignored here.
        """
        self.dataset.add_frame(frame)

    def has_pending_frames(self) -> bool:
        return self.dataset is not None and self.dataset.has_pending_frames()

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()

    def record_live_frame(self, name: str, bgr: np.ndarray, captured_at: float) -> None:
        """Native-rate capture hook; datasets record per-tick frames only."""

    def record_visual_servo_overlay(self, captured_at: float, primitives: dict[str, Any]) -> None:
        """Servo-overlay hook; datasets store no overlay geometry."""

    def start_live_capture(self, wall_t: float) -> None:
        """Live-capture hook; datasets record per-tick frames only."""

    def stop_live_capture(self) -> None:
        """Live-capture hook; datasets record per-tick frames only."""

    def start_audio_capture(self) -> None:
        """Audio hook; datasets record no audio."""

    def stop_audio_capture(self) -> None:
        """Audio hook; datasets record no audio."""

    def dropped_frame_count(self) -> int:
        """Frames the streaming video encoder dropped in the current episode.

        The encoder silently drops a frame when its queue backs up (it can't keep
        pace with capture), which leaves the video shorter than the recorded rows
        and corrupts the episode. Returns 0 in PNG mode (no such queue) or before
        the dataset exists.
        """
        if self.dataset is None:
            return 0
        encoder = getattr(self.dataset.writer, "_streaming_encoder", None)
        if encoder is None:
            return 0
        return sum(encoder._dropped_frames.values())

    def save_episode(self, episode_metadata: dict[str, Any] | None = None) -> None:
        """Commit the pending LeRobot episode, optionally adding episode metadata.

        LeRobot stores frame features and episode metadata through separate paths:
        arbitrary metadata cannot be added to the per-frame buffer because it is
        validated against the dataset feature schema. LeRobot's dataset metadata
        object does accept extra episode metadata internally, so temporarily
        wrap that call and merge our run-specific fields into the episode row.
        """
        if self.dataset is None:
            raise RuntimeError("cannot save episode before the dataset exists")
        if not episode_metadata:
            self.dataset.save_episode()
            return

        meta = self.dataset.meta
        original_save_episode = meta.save_episode

        def save_episode_with_metadata(
            episode_index,
            episode_length,
            episode_tasks,
            episode_stats,
            base_metadata,
        ):
            merged = dict(base_metadata)
            merged.update(episode_metadata)
            return original_save_episode(
                episode_index,
                episode_length,
                episode_tasks,
                episode_stats,
                merged,
            )

        meta.save_episode = save_episode_with_metadata
        try:
            self.dataset.save_episode()
        finally:
            meta.save_episode = original_save_episode

    def finalize(self) -> None:
        if self.dataset is not None:
            self.dataset.finalize()
