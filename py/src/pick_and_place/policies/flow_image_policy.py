# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive an image-conditioned flow policy as a closed-loop controller.

The model predicts a horizon of joint commands from a short history of camera
frames and proprioception; this turns that into the one-action-per-tick contract
the simulator and the hardware runner both speak, by generating a horizon
whenever the queue of pending actions runs dry and handing out ``act_steps`` of
it before generating again.

Sampling is plain Euler integration of the velocity field from Gaussian noise at
``t = 0`` to the sample at ``t = 1``. Ten steps was found to match 100-step
integration on this task at a fraction of the cost.

The model runs in this process: unlike the Diffusion Policy checkpoints, it has
no dependencies beyond the ones the project already imports.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from pick_and_place.policies.dataset_export import (
    load_bounds,
    load_manifest,
    normalize,
    unnormalize,
)
from pick_and_place.policies.flow_image_encoder import FlowImageUnet1D
from pick_and_place.spec.controller import (
    OVERHEAD_FEATURE,
    PolicyObservation,
    STATE_FEATURE,
    WRIST_FEATURE,
)

# ImageNet statistics, which the camera encoder's backbone was trained under.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

CHECKPOINT_MODEL_TYPE = "flow_image_unet1d"


def load_model(checkpoint: str | Path, device: torch.device) -> FlowImageUnet1D:
    """Rebuild the model a training run saved, ready for inference."""
    # Checkpoints written on a CUDA box carry CUDA storage tags, which cannot be
    # mapped straight onto MPS; land them on the CPU and move the built model.
    contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if contents.get("model_type") != CHECKPOINT_MODEL_TYPE:
        raise ValueError(f"{checkpoint} is not an image flow checkpoint")
    model = FlowImageUnet1D(**contents["model_config"]).to(device)
    model.load_state_dict(contents["model"])
    model.eval()
    return model


def stack_cameras(observation: PolicyObservation) -> np.ndarray:
    """Concatenate the two camera views on the channel axis, as the export does.

    The export orders cameras as overhead first, then wrist, each converted from
    HxWx3 to the model's channel-first layout.
    """
    overhead = np.asarray(observation[OVERHEAD_FEATURE], dtype=np.uint8)
    wrist = np.asarray(observation[WRIST_FEATURE], dtype=np.uint8)
    return np.concatenate((np.moveaxis(overhead, -1, 0), np.moveaxis(wrist, -1, 0)), axis=0)


def carry_noise(
    previous: torch.Tensor | None,
    fresh: torch.Tensor,
    independent: torch.Tensor,
    *,
    shift: int,
    correlation: float,
) -> torch.Tensor:
    """Blend the previous horizon's noise into a fresh draw.

    Successive queries predict overlapping stretches of time, offset by
    ``shift`` executed actions, but draw independent noise — so each query can
    land in a different one of several equally valid modes, and the arm lurches
    where the chunks meet. Shifting the previous draw back into alignment and
    mixing it in correlates the latent the model starts from, which keeps
    consecutive chunks in the same mode.

    ``correlation`` is the weight on the carried noise, from 0 (independent
    draws) to 1 (reuse wherever the horizons overlap). The mix is spherical, so
    the result stays a standard normal sample whatever the weight: that is the
    distribution the velocity field was trained to transport.
    """
    if previous is None or correlation <= 0:
        return fresh
    carried = torch.roll(previous, shifts=-shift, dims=1)
    # The steps rolled off the end have no counterpart in the previous horizon.
    carried[:, -shift:] = independent[:, -shift:]
    return correlation * carried + (1 - correlation**2) ** 0.5 * fresh


def generate_horizon(
    model: FlowImageUnet1D,
    images: torch.Tensor,
    states: torch.Tensor,
    *,
    integration_steps: int,
    noise: torch.Tensor,
) -> np.ndarray:
    """Integrate the velocity field from ``noise`` into one action horizon."""
    values = noise
    device = images.device
    time = torch.zeros(1, 1, device=device)
    with torch.no_grad():
        condition = model.encode_observation(images, states)
        for _ in range(integration_steps):
            velocity = model.unet(values, time, condition)
            values = values + velocity / integration_steps
            time = time + 1 / integration_steps
    return values[0].cpu().numpy()


def summarize_smoothness(
    commands: np.ndarray, requeried: np.ndarray, *, joints: int
) -> dict[str, float]:
    """Compare how far the arm is asked to jump within a chunk versus across one.

    ``commands`` is the sequence actually sent, one row per tick, and
    ``requeried`` marks the ticks whose command came from a freshly generated
    horizon. A policy whose chunks disagree at the seam shows a larger step on
    those ticks than on the ticks inside a chunk; equal figures mean the seams
    are invisible and any remaining roughness is not a replanning artifact.
    """
    steps = np.abs(np.diff(commands[:, :joints], axis=0)).max(axis=1)
    # Row i of ``steps`` is the move onto tick i + 1, so the boundary flags line
    # up with the commands from the second tick onward.
    seam = requeried[1:]
    interior = steps[~seam]
    boundary = steps[seam]
    return {
        "mean_step_deg": float(steps.mean()) if steps.size else 0.0,
        "interior_step_deg": float(interior.mean()) if interior.size else 0.0,
        "boundary_step_deg": float(boundary.mean()) if boundary.size else 0.0,
        "max_step_deg": float(steps.max()) if steps.size else 0.0,
    }


class FlowImagePolicyController:
    """Turn observations into a queue of generated joint commands."""

    def __init__(
        self,
        model: FlowImageUnet1D,
        bounds: dict[str, np.ndarray],
        *,
        act_steps: int,
        integration_steps: int,
        device: torch.device,
        seed: int,
        policy_hz: float,
        image_hw: tuple[int, int],
        noise_correlation: float = 0.0,
    ) -> None:
        if act_steps < 1 or act_steps > model.prediction_steps:
            raise ValueError(
                f"act_steps must be between 1 and the model's {model.prediction_steps} "
                f"predicted steps, got {act_steps}"
            )
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        if not 0.0 <= noise_correlation <= 1.0:
            raise ValueError(f"noise_correlation must be in [0, 1], got {noise_correlation}")
        self.model = model
        self.device = device
        self.act_steps = act_steps
        self.integration_steps = integration_steps
        self.seed = seed
        self.policy_hz = policy_hz
        self.image_hw = image_hw
        self.observation_min = bounds["obs_min"]
        self.observation_max = bounds["obs_max"]
        self.action_min = bounds["action_min"]
        self.action_max = bounds["action_max"]
        self.mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
        self.images: deque[np.ndarray] = deque(maxlen=model.observation_steps)
        self.states: deque[np.ndarray] = deque(maxlen=model.observation_steps)
        self.actions: deque[np.ndarray] = deque()
        self.clipped_fraction = 0.0
        self.noise_correlation = noise_correlation
        self.latest_prediction: np.ndarray | None = None
        self.previous_noise: torch.Tensor | None = None
        self.reset()

    @classmethod
    def from_export(
        cls,
        checkpoint: str | Path,
        export: str | Path,
        *,
        act_steps: int,
        integration_steps: int,
        device: torch.device,
        seed: int = 0,
        noise_correlation: float = 0.0,
    ) -> FlowImagePolicyController:
        """Load a checkpoint together with the dataset export it was trained on.

        The export carries the normalization bounds, the control rate and the
        input resolution — all part of the model contract, none of them
        recoverable from the weights alone.
        """
        manifest = load_manifest(export)
        height, width = (int(value) for value in manifest["image_size"])
        return cls(
            load_model(checkpoint, device),
            load_bounds(export),
            act_steps=act_steps,
            integration_steps=integration_steps,
            device=device,
            seed=seed,
            policy_hz=float(manifest["fps"]),
            image_hw=(height, width),
            noise_correlation=noise_correlation,
        )

    @property
    def prediction_steps(self) -> int:
        return int(self.model.prediction_steps)

    @property
    def observation_steps(self) -> int:
        return int(self.model.observation_steps)

    def reset(self) -> None:
        self.images.clear()
        self.states.clear()
        self.actions.clear()
        self.latest_prediction = None
        self.previous_noise = None
        torch.manual_seed(self.seed)

    def _draw_noise(self) -> torch.Tensor:
        """The starting sample for one query, correlated with the last if asked."""
        shape = (1, self.model.prediction_steps, self.model.action_dim)
        noise = carry_noise(
            self.previous_noise,
            torch.randn(*shape, device=self.device),
            torch.randn(*shape, device=self.device),
            shift=self.act_steps,
            correlation=self.noise_correlation,
        )
        self.previous_noise = noise
        return noise

    def _observation_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize the observation history into the model's input tensors."""
        images = torch.from_numpy(np.stack(self.images)[None]).to(self.device)
        steps, channels = images.shape[1], images.shape[2]
        floats = images.float().div_(255.0).reshape(-1, 3, *images.shape[-2:])
        floats = ((floats - self.mean) / self.std).reshape(1, steps, channels, *images.shape[-2:])
        states = torch.from_numpy(np.stack(self.states)[None]).to(self.device)
        return floats, states

    def act(self, observation: PolicyObservation) -> np.ndarray:
        # Only the tick that generates a horizon reports one, so a caller logging
        # predictions records each horizon once rather than on every tick.
        self.latest_prediction = None
        self.images.append(stack_cameras(observation))
        self.states.append(
            normalize(
                np.asarray(observation[STATE_FEATURE], dtype=np.float32),
                self.observation_min,
                self.observation_max,
            )
        )
        # At the start of an episode there is no history yet, so the first
        # observation stands in for the ones that would have preceded it.
        while len(self.images) < self.model.observation_steps:
            self.images.appendleft(self.images[0].copy())
            self.states.appendleft(self.states[0].copy())

        if not self.actions:
            images, states = self._observation_batch()
            generated = generate_horizon(
                self.model,
                images,
                states,
                integration_steps=self.integration_steps,
                noise=self._draw_noise(),
            )
            clipped = np.clip(generated, -1, 1)
            self.clipped_fraction = float(np.mean(generated != clipped))
            commands = unnormalize(clipped, self.action_min, self.action_max)
            self.latest_prediction = commands
            self.actions.extend(commands[: self.act_steps])
        return self.actions.popleft()

    def close(self) -> None:
        """Nothing to release; present so runners can treat controllers alike."""
