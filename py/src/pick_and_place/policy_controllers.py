# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Controller boundary used by closed-loop policy evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from pick_and_place.follower import JOINT_NAMES
from pick_and_place.policy import DEFAULT_INSTRUCTION, make_policy, resolve_checkpoint_cameras

STATE_FEATURE = "observation.state"
OVERHEAD_FEATURE = "observation.images.overhead"
WRIST_FEATURE = "observation.images.wrist"

PolicyObservation = dict[str, np.ndarray]


@runtime_checkable
class PolicyController(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: PolicyObservation) -> np.ndarray: ...


@dataclass(frozen=True)
class ControllerFailure:
    """A terminal failure reported by a controller without unsafe motion."""

    code: str
    message: str


def _action_vector(action: object) -> np.ndarray:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (len(JOINT_NAMES),):
        raise ValueError(f"controller action must have shape ({len(JOINT_NAMES)},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("controller action must contain only finite values")
    return vector


class NoOpPolicyController:
    """Hold the currently observed hardware-frame joint state."""

    def reset(self) -> None:
        pass

    def act(self, observation: PolicyObservation) -> np.ndarray:
        if STATE_FEATURE not in observation:
            raise KeyError(f"observation is missing {STATE_FEATURE!r}")
        return _action_vector(observation[STATE_FEATURE]).copy()


class ScriptedPolicyController:
    """Replay a fixed hardware-frame action sequence, restarting each episode."""

    def __init__(self, actions: Sequence[np.ndarray]) -> None:
        if not actions:
            raise ValueError("scripted controller requires at least one action")
        self._actions = tuple(_action_vector(action).copy() for action in actions)
        self.reset()

    def reset(self) -> None:
        self._index = 0

    def act(self, observation: PolicyObservation) -> np.ndarray:
        del observation
        action = self._actions[min(self._index, len(self._actions) - 1)]
        self._index += 1
        return action.copy()


class LeRobotPolicyController:
    """Adapt a loaded LeRobot policy to the evaluator's observation contract."""

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        device: Any,
        image_keys: tuple[str, str],
        instruction: str = DEFAULT_INSTRUCTION,
        predict_action_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.image_keys = image_keys
        self.instruction = instruction
        self._predict_action_fn = predict_action_fn

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        device: Any,
        image_hw: tuple[int, int] | None = None,
        instruction: str = DEFAULT_INSTRUCTION,
        n_action_steps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ) -> "LeRobotPolicyController":
        resolved_hw, image_keys = resolve_checkpoint_cameras(checkpoint, override_hw=image_hw)
        policy, preprocessor, postprocessor = make_policy(
            checkpoint,
            resolved_hw,
            image_keys,
            device,
            n_action_steps=n_action_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            image_keys=image_keys,
            instruction=instruction,
        )

    def reset(self) -> None:
        self.policy.reset()

    def act(self, observation: PolicyObservation) -> np.ndarray:
        for feature in (STATE_FEATURE, OVERHEAD_FEATURE, WRIST_FEATURE):
            if feature not in observation:
                raise KeyError(f"observation is missing {feature!r}")
        predictor = self._predict_action_fn
        if predictor is None:
            from lerobot.utils.control_utils import predict_action

            predictor = predict_action
        policy_observation = {
            STATE_FEATURE: observation[STATE_FEATURE],
            self.image_keys[0]: observation[OVERHEAD_FEATURE],
            self.image_keys[1]: observation[WRIST_FEATURE],
        }
        action = predictor(
            policy_observation,
            self.policy,
            self.device,
            self.preprocessor,
            self.postprocessor,
            use_amp=False,
            task=self.instruction,
            robot_type="so101",
        )
        if hasattr(action, "detach"):
            action = action.detach()
        if hasattr(action, "cpu"):
            action = action.cpu()
        if hasattr(action, "numpy"):
            action = action.numpy()
        return _action_vector(np.asarray(action).reshape(-1)[: len(JOINT_NAMES)])
