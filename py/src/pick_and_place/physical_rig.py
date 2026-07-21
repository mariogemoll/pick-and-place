# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Lifecycle boundary for the cameras and follower on the physical rig."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pick_and_place.follower import load_joint_zero_offsets


class CameraStream(Protocol):
    def latest(self) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass
class PhysicalRig:
    """Resources and calibrated limits required by a physical policy run."""

    follower: Any
    overhead: CameraStream
    wrist: CameraStream
    clamp_low: np.ndarray
    clamp_high: np.ndarray
    joint_zero_offsets: Mapping[str, float]
    workspace: CameraStream | None = None
    park_action: Callable[[], None] | None = None

    def park_and_release(self) -> None:
        """Park before releasing torque; always close every owned resource."""
        try:
            try:
                if self.park_action is not None:
                    self.park_action()
            finally:
                self.follower.bus.disable_torque()
        finally:
            try:
                self.follower.disconnect()
            finally:
                self.overhead.close()
                self.wrist.close()
                if self.workspace is not None:
                    self.workspace.close()


def require_joint_zero_offsets(
    path: Path,
    *,
    allow_uncalibrated: bool = False,
) -> dict[str, float]:
    """Load joint zeros, refusing raw-servo operation unless explicitly allowed."""
    if allow_uncalibrated:
        return {}
    if not path.is_file():
        raise RuntimeError(
            f"missing required joint-zero calibration at {path}; "
            "use the explicit uncalibrated debug override only for safe bench diagnostics"
        )
    try:
        offsets = load_joint_zero_offsets(path)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid joint-zero calibration at {path}: {exc}") from exc
    if not offsets:
        raise RuntimeError(f"joint-zero calibration at {path} contains no offsets")
    return offsets
