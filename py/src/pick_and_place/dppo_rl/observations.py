# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""What one observation timestep is, per policy family the RL environment serves.

Two families are fine-tuned against the same episode loop and the same reward,
and they disagree about only two things: what the policy is shown, and what a
normalized action means. Both are fixed by the export the policy was trained
against, so each family's bounds are read from that export rather than assumed.

- :class:`CameraObservation` is the visual Diffusion Policy's view -- six
  proprioceptive coordinates and two 96x96 cameras concatenated
  overhead-then-wrist on the channel axis.
- :class:`FlowStateObservation` is the state flow policy's view -- the same six
  coordinates plus the cube's position, the first two columns of its rotation
  matrix, and the target, which the simulator supplies from its believed state.

A specification is a small frozen value so it can be pickled to a worker
process; the arrays it names are loaded there, once, by :meth:`build`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pick_and_place.policies.flow_policy import (
    load_export,
    normalize,
    pack_observation,
    unnormalize,
)
from pick_and_place.spec.action_encoding import (
    ActionEncoding,
    decode_actions,
    read_action_encoding,
)
from pick_and_place.spec.controller import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE


def normalize_state(state: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    """Apply the diffusion exporter's per-dimension min-max map to ``[-1, 1]``."""
    return 2.0 * (state - minimum) / (maximum - minimum + 1e-6) - 1.0


def unnormalize_action(
    action: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    """Invert :func:`normalize_state` for a predicted action."""
    return (action + 1.0) / 2.0 * (maximum - minimum + 1e-6) + minimum


class ObservationCodec(Protocol):
    """Turns a simulator step into what the policy sees, and back again."""

    keys: tuple[str, ...]

    def observe(
        self, observation: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, np.ndarray]: ...

    def command(self, action: np.ndarray, measured_joints: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CameraObservation:
    """The visual Diffusion Policy's observation, from its dataset export."""

    normalization_path: Path
    needs_images: bool = True

    def build(self) -> CameraCodec:
        bounds = np.load(self.normalization_path)
        return CameraCodec(
            observation_min=bounds["obs_min"].astype(np.float32),
            observation_max=bounds["obs_max"].astype(np.float32),
            action_min=bounds["action_min"].astype(np.float32),
            action_max=bounds["action_max"].astype(np.float32),
            # Read, never assumed: the bounds and the encoding are one contract,
            # and decoding a delta as an absolute joint command does not fail, it
            # just commands nonsense.
            action_encoding=read_action_encoding(bounds),
        )


@dataclass(frozen=True)
class CameraCodec:
    observation_min: np.ndarray
    observation_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    action_encoding: ActionEncoding
    keys: tuple[str, ...] = ("state", "rgb")

    def observe(
        self, observation: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        del info
        overhead = np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8)
        wrist = np.asarray(observation[WRIST_FEATURE], dtype=np.uint8)
        return {
            "state": normalize_state(
                np.asarray(observation[STATE_FEATURE], dtype=np.float32),
                self.observation_min,
                self.observation_max,
            ).astype(np.float32),
            # HWC per camera to the CHW pair the encoder expects.
            "rgb": np.concatenate(
                [overhead.transpose(2, 0, 1), wrist.transpose(2, 0, 1)], axis=0
            ),
        }

    def command(self, action: np.ndarray, measured_joints: np.ndarray) -> np.ndarray:
        return decode_actions(
            self.action_encoding,
            unnormalize_action(action, self.action_min, self.action_max),
            measured_joints,
        )


@dataclass(frozen=True)
class FlowStateObservation:
    """The state flow policy's observation, from its own export directory."""

    export_dir: Path
    needs_images: bool = False

    def build(self) -> FlowStateCodec:
        manifest, bounds = load_export(self.export_dir)
        if manifest.get("endpoint_semantics") != "absolute joint command":
            raise ValueError("fine-tuning requires an export of absolute joint commands")
        return FlowStateCodec(
            observation_min=bounds["observation_min"],
            observation_max=bounds["observation_max"],
            endpoint_min=bounds["endpoint_min"],
            endpoint_max=bounds["endpoint_max"],
            observation_dim=int(manifest["observation_dim"]),
        )


@dataclass(frozen=True)
class FlowStateCodec:
    observation_min: np.ndarray
    observation_max: np.ndarray
    endpoint_min: np.ndarray
    endpoint_max: np.ndarray
    observation_dim: int
    keys: tuple[str, ...] = ("state",)

    def observe(
        self, observation: dict[str, np.ndarray], info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        packed = pack_observation(observation, info)
        if packed.shape != (self.observation_dim,):
            raise ValueError(
                f"packed observation must have shape ({self.observation_dim},), "
                f"got {packed.shape}"
            )
        return {"state": normalize(packed, self.observation_min, self.observation_max)}

    def command(self, action: np.ndarray, measured_joints: np.ndarray) -> np.ndarray:
        del measured_joints  # absolute joint commands do not depend on the measurement
        return unnormalize(action, self.endpoint_min, self.endpoint_max)
