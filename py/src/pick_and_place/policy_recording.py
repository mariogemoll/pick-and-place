# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Transactional adapter from physical policy ticks to episode recordings."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import numpy as np

from pick_and_place.policy_controllers import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.policy_real import PhysicalPolicyTick


@dataclass
class PolicyRecordingSession:
    """Expose a video or LeRobot session as a physical episode transaction."""

    session: Any
    task: str
    workspace_rgb: Callable[[], np.ndarray] | None = None
    episode_metadata: Callable[[], dict[str, Any] | None] | None = None

    def record_tick(self, tick: PhysicalPolicyTick) -> None:
        observation = tick.observation
        wrist = observation[WRIST_FEATURE]
        overhead = observation[OVERHEAD_FEATURE]
        workspace = self.workspace_rgb() if self.workspace_rgb is not None else None
        if not self.session.initialized:
            self.session.create_dataset(
                wrist.shape,
                overhead.shape,
                workspace.shape if workspace is not None else None,
            )
        frame = {
            STATE_FEATURE: np.asarray(observation[STATE_FEATURE], dtype=np.float32),
            "action": np.asarray(tick.command, dtype=np.float32),
            WRIST_FEATURE: wrist,
            OVERHEAD_FEATURE: overhead,
            "task": self.task,
        }
        if workspace is not None:
            frame["observation.images.workspace"] = workspace
        self.session.record_frame(
            frame,
            wall_t=(tick.index - 1) / self.session.fps,
        )

    def commit(self) -> None:
        metadata = self.episode_metadata() if self.episode_metadata is not None else None
        self.session.save_episode(metadata)

    def discard(self) -> None:
        if self.session.initialized and self.session.has_pending_frames():
            self.session.discard_episode()

    def finalize(self) -> None:
        self.session.finalize()
