# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Observation-only boundary for the analytic pick-and-place policy."""

from __future__ import annotations

from enum import Enum

import numpy as np

from pick_and_place.geometry import CubePose
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.overhead_localization import OverheadLocalizer
from pick_and_place.paper_detection import PaperTarget
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    ControllerFailure,
    PolicyObservation,
)


class ScriptedPolicyState(str, Enum):
    """Externally inspectable phase of the scripted controller."""

    LOCALIZING = "localizing"
    READY = "ready"
    FAILED = "failed"


class ScriptedPolicy:
    """Localize the task from images while safely holding the observed joints.

    This is the deployable boundary of the analytic controller. It deliberately
    accepts the same observation dictionary as learned policies and never
    accepts simulator poses or task-oracle state. Planning and trajectory
    execution will be added behind this boundary; until then every tick returns
    the reported hardware-frame position as a hold command.
    """

    def __init__(
        self,
        localizer: OverheadLocalizer,
        workspace_corners_world: np.ndarray,
        *,
        target_color: str = "black",
        max_localization_steps: int = 60,
    ) -> None:
        if target_color not in {"black", "white"}:
            raise ValueError("target_color must be 'black' or 'white'")
        if max_localization_steps < 1:
            raise ValueError("max_localization_steps must be at least 1")
        corners = np.asarray(workspace_corners_world, dtype=float)
        if corners.shape != (4, 3) or not np.all(np.isfinite(corners)):
            raise ValueError("workspace_corners_world must have finite shape (4, 3)")

        self.localizer = localizer
        self.workspace_corners_world = corners.copy()
        self.target_color = target_color
        self.max_localization_steps = max_localization_steps
        self.reset()

    def reset(self) -> None:
        """Forget every detection and failure from the previous episode."""
        self.localizer.reset()
        self.state = ScriptedPolicyState.LOCALIZING
        self.cube_pose: CubePose | None = None
        self.drop_target: PaperTarget | None = None
        self.failure: ControllerFailure | None = None
        self._localization_steps = 0

    @property
    def terminal(self) -> bool:
        return self.state is ScriptedPolicyState.FAILED

    @staticmethod
    def _hold_action(observation: PolicyObservation) -> np.ndarray:
        if STATE_FEATURE not in observation:
            raise KeyError(f"observation is missing {STATE_FEATURE!r}; cannot issue a safe hold")
        action = np.asarray(observation[STATE_FEATURE], dtype=np.float32).reshape(-1)
        expected_shape = (len(JOINT_NAMES),)
        if action.shape != expected_shape:
            raise ValueError(
                f"{STATE_FEATURE} must have shape {expected_shape}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{STATE_FEATURE} must contain only finite values")
        return action.copy()

    @staticmethod
    def _image(observation: PolicyObservation, feature: str) -> np.ndarray:
        if feature not in observation:
            raise KeyError(f"observation is missing {feature!r}")
        image = np.asarray(observation[feature])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{feature} must have shape (height, width, 3), got {image.shape}")
        return image

    def _fail(self, code: str, message: str) -> None:
        self.state = ScriptedPolicyState.FAILED
        self.failure = ControllerFailure(code=code, message=message)

    def act(self, observation: PolicyObservation) -> np.ndarray:
        hold = self._hold_action(observation)
        if self.terminal or self.state is ScriptedPolicyState.READY:
            return hold

        try:
            overhead = self._image(observation, OVERHEAD_FEATURE)
            # Wrist RGB is part of the deployable observation contract even
            # though localization in this initial state machine is overhead-only.
            self._image(observation, WRIST_FEATURE)
            if self.cube_pose is None:
                self.cube_pose = self.localizer.localize_cube(overhead)
            if self.drop_target is None:
                self.drop_target = self.localizer.localize_drop_target(
                    overhead,
                    target_color=self.target_color,
                    workspace_corners_world=self.workspace_corners_world,
                )
        except Exception as exc:
            self._fail("localization_error", str(exc))
            return hold

        self._localization_steps += 1
        if self.cube_pose is not None and self.drop_target is not None:
            self.state = ScriptedPolicyState.READY
        elif self._localization_steps >= self.max_localization_steps:
            missing = []
            if self.cube_pose is None:
                missing.append("cube")
            if self.drop_target is None:
                missing.append("drop target")
            self._fail(
                "localization_timeout",
                f"could not localize {' and '.join(missing)} in "
                f"{self.max_localization_steps} control steps",
            )
        return hold
