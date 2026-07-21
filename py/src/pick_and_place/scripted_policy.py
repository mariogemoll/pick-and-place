# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Observation-only boundary for the analytic pick-and-place policy."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np

from pick_and_place.episodes import Episode, prepare_episode, sample_hunt_pose
from pick_and_place.follower import (
    JOINT_NAMES,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.overhead_localization import OverheadLocalizer
from pick_and_place.paper_detection import PaperTarget
from pick_and_place.policy_controllers import (
    OVERHEAD_FEATURE,
    STATE_FEATURE,
    WRIST_FEATURE,
    ControllerFailure,
    PolicyObservation,
)

PlanEpisode = Callable[..., Episode]


class ScriptedPolicyState(str, Enum):
    """Externally inspectable phase of the scripted controller."""

    LOCALIZING = "localizing"
    READY = "ready"
    FAILED = "failed"


class ScriptedPolicy:
    """Localize and plan using only deployable observations and fixed configuration.

    Search commands and the collision-checked initial plan are owned by the
    policy. The environment supplies only RGB images and reported hardware-frame
    joints. Incremental playback of the resulting trajectory is added separately;
    after planning, this controller currently holds the observed position.
    """

    def __init__(
        self,
        localizer: OverheadLocalizer,
        workspace_corners_world: np.ndarray,
        *,
        target_color: str = "black",
        max_localization_steps: int = 60,
        localization_steps_per_search: int = 15,
        rng_seed: int = 0,
        plan_episode: PlanEpisode = prepare_episode,
    ) -> None:
        if target_color not in {"black", "white"}:
            raise ValueError("target_color must be 'black' or 'white'")
        if max_localization_steps < 1:
            raise ValueError("max_localization_steps must be at least 1")
        if localization_steps_per_search < 1:
            raise ValueError("localization_steps_per_search must be at least 1")
        corners = np.asarray(workspace_corners_world, dtype=float)
        if corners.shape != (4, 3) or not np.all(np.isfinite(corners)):
            raise ValueError("workspace_corners_world must have finite shape (4, 3)")

        self.localizer = localizer
        self.workspace_corners_world = corners.copy()
        self.target_color = target_color
        self.max_localization_steps = max_localization_steps
        self.localization_steps_per_search = localization_steps_per_search
        self.rng_seed = rng_seed
        self._plan_episode = plan_episode
        self.reset()

    def reset(self) -> None:
        """Forget detections, planning state, random draws, and failures."""
        self.localizer.reset()
        self.state = ScriptedPolicyState.LOCALIZING
        self.cube_pose: CubePose | None = None
        self.drop_target: PaperTarget | None = None
        self.episode: Episode | None = None
        self.failure: ControllerFailure | None = None
        self._localization_steps = 0
        self._search_target: np.ndarray | None = None
        self._rng = np.random.default_rng(self.rng_seed)

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

    def _search_action(self) -> np.ndarray:
        arm_joints, gripper = sample_hunt_pose(self._rng)
        return sim_frame_to_real(arm_joints, gripper).astype(np.float32)

    def _plan(self, reported_joints: np.ndarray) -> None:
        assert self.cube_pose is not None
        assert self.drop_target is not None
        start_joints, start_gripper = real_frame_to_sim(reported_joints)
        target_xy = np.asarray(self.drop_target.xy, dtype=float).reshape(-1)
        if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
            raise ValueError(
                "localized drop target xy must have finite shape (2,), "
                f"got {target_xy.shape}"
            )
        target = CubePose(
            x=float(target_xy[0]),
            y=float(target_xy[1]),
            z=CUBE_HALF_SIZE,
        )
        self.episode = self._plan_episode(
            self._rng,
            self.cube_pose,
            target,
            start_joints=start_joints,
            start_gripper=start_gripper,
            max_attempts=1,
            include_environment=True,
        )

    def act(self, observation: PolicyObservation) -> np.ndarray:
        hold = self._hold_action(observation)
        if self.terminal or self.state is ScriptedPolicyState.READY:
            return hold

        try:
            overhead = self._image(observation, OVERHEAD_FEATURE)
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
            try:
                self._plan(hold)
            except Exception as exc:
                self._fail("planning_error", str(exc))
                return hold
            self.state = ScriptedPolicyState.READY
            return hold

        if self._localization_steps >= self.max_localization_steps:
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

        if self._localization_steps % self.localization_steps_per_search == 0:
            self._search_target = self._search_action()
        if self._search_target is not None:
            return self._search_target.copy()
        return hold
