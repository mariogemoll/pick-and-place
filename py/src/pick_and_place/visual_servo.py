# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""State tracking for the wrist-camera descent visual servo.

During the descent phase the executor refines the grasp target from live
wrist-camera cube detections. These classes hold the shared state between the
servo worker thread and the control loop: the latest pose estimate, whether
the target has settled, and the back-up-and-retry budget for when the closing
jaws hide every cube tag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pick_and_place.geometry import CubePose

# Descent PBVS completion gate. The planned descent duration is still the
# minimum time needed to physically move from hover to grasp; these values decide
# whether to wait beyond that for the camera target to settle.
DESCENT_SERVO_MAX_DURATION = 4.5
DESCENT_SERVO_STABLE_FRAMES = 10
DESCENT_SERVO_POSITION_TOLERANCE_M = 0.0015
DESCENT_SERVO_YAW_TOLERANCE_RAD = math.radians(1.5)
DESCENT_SERVO_MAX_NO_DETECTION_RETRIES = 2
DESCENT_SERVO_BACKUP_DURATION = 0.9


def smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def _shortest_angle_delta(start: float, end: float) -> float:
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class DescentServoConvergence:
    """Track whether the wrist-camera servo target has stopped moving."""

    stable_frames: int = 0
    last_x: float | None = None
    last_y: float | None = None
    last_yaw: float | None = None

    def observe(self, source) -> None:
        if self.last_x is None or self.last_y is None or self.last_yaw is None:
            self.stable_frames = 1
        else:
            xy_delta = math.hypot(source.x - self.last_x, source.y - self.last_y)
            yaw_delta = abs(_shortest_angle_delta(self.last_yaw, source.yaw))
            if (
                xy_delta <= DESCENT_SERVO_POSITION_TOLERANCE_M
                and yaw_delta <= DESCENT_SERVO_YAW_TOLERANCE_RAD
            ):
                self.stable_frames += 1
            else:
                self.stable_frames = 1
        self.last_x = source.x
        self.last_y = source.y
        self.last_yaw = source.yaw

    def is_stable(self) -> bool:
        return self.stable_frames >= DESCENT_SERVO_STABLE_FRAMES


@dataclass
class DescentServoRetryState:
    """Reverse to pregrasp and retry when the jaws hide every cube tag."""

    max_retries: int = DESCENT_SERVO_MAX_NO_DETECTION_RETRIES
    backup_duration: float = DESCENT_SERVO_BACKUP_DURATION
    retries_started: int = 0
    backup_start_t: float | None = None

    def is_backing_up(self) -> bool:
        return self.backup_start_t is not None

    def can_retry(self) -> bool:
        return self.retries_started < self.max_retries

    def start_backup(self, phase_t: float) -> None:
        if not self.can_retry():
            raise RuntimeError("descent servo retry budget exhausted")
        self.retries_started += 1
        self.backup_start_t = phase_t

    def command_phase_t(self, phase_t: float, descent_duration: float) -> float:
        if self.backup_start_t is None:
            return phase_t
        alpha = (phase_t - self.backup_start_t) / self.backup_duration
        return descent_duration * (1.0 - smoothstep(alpha))

    def backup_complete(self, phase_t: float) -> bool:
        return (
            self.backup_start_t is not None
            and phase_t - self.backup_start_t >= self.backup_duration - 1e-9
        )

    def finish_backup(self) -> None:
        self.backup_start_t = None


@dataclass(frozen=True)
class WristServoEstimate:
    """Latest wrist-camera cube estimate published by the servo worker."""

    frame_id: int
    source: CubePose


@dataclass(frozen=True)
class WristServoPreview:
    """Annotated wrist frame for optional live display."""

    frame_id: int
    bgr: np.ndarray
