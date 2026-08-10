# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Project adapter for Diffusion Policy's unmodified MIT-licensed 1D U-Net."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

DIFFUSION_POLICY_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "diffusion_policy"
_UPSTREAM_MODULE = (
    DIFFUSION_POLICY_ROOT / "diffusion_policy" / "model" / "diffusion" / "conditional_unet1d.py"
)
if not _UPSTREAM_MODULE.is_file():
    raise ImportError(
        "Diffusion Policy submodule is missing; run "
        "`git submodule update --init third_party/diffusion_policy`"
    )

# The upstream package uses absolute imports. Temporarily expose its repository
# root while importing it, then leave the loaded package to resolve through its
# own __path__.
sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))
try:
    from diffusion_policy.model.diffusion.conditional_unet1d import (
        ConditionalUnet1D as UpstreamConditionalUnet1D,
    )
finally:
    sys.path.remove(str(DIFFUSION_POLICY_ROOT))

_loaded_source = Path(sys.modules[UpstreamConditionalUnet1D.__module__].__file__).resolve()
if not _loaded_source.is_relative_to(DIFFUSION_POLICY_ROOT):
    raise ImportError(f"loaded Diffusion Policy from {_loaded_source}, not the pinned submodule")


class FlowConditionalUnet1D(UpstreamConditionalUnet1D):
    """Attach the flow policy's shape contract to the upstream network."""

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        prediction_steps: int,
        time_embedding_dim: int = 32,
        down_dims: tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 5,
        groups: int = 8,
    ) -> None:
        if min(action_dim, observation_dim, prediction_steps, groups) < 1:
            raise ValueError("U-Net dimensions and group count must be positive")
        if len(down_dims) < 2 or any(dimension < 1 for dimension in down_dims):
            raise ValueError("down_dims must contain at least two positive channel dimensions")
        if prediction_steps % (2 ** (len(down_dims) - 1)):
            raise ValueError("prediction_steps must be divisible by the U-Net downsampling factor")
        if kernel_size < 1 or not kernel_size % 2:
            raise ValueError("kernel_size must be a positive odd number")
        if any(dimension % groups for dimension in down_dims):
            raise ValueError("each U-Net channel dimension must be divisible by the group count")
        super().__init__(
            input_dim=action_dim,
            global_cond_dim=observation_dim,
            diffusion_step_embed_dim=time_embedding_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=groups,
            cond_predict_scale=True,
        )
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.prediction_steps = prediction_steps
        self.time_embedding_dim = time_embedding_dim
        self.down_dims = tuple(down_dims)
        self.kernel_size = kernel_size
        self.groups = groups
        self.output_dim = prediction_steps * action_dim
        self.input_dim = self.output_dim + time_embedding_dim + observation_dim

    def forward(
        self, values: torch.Tensor, time: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        """Validate the flow boundary and delegate the model computation upstream."""
        expected_values = (len(values), self.prediction_steps, self.action_dim)
        if values.shape != expected_values:
            raise ValueError(f"values must have shape {expected_values}")
        if time.shape not in ((len(values),), (len(values), 1)):
            raise ValueError("time must have shape (batch,) or (batch, 1)")
        if observations.shape != (len(values), self.observation_dim):
            raise ValueError(
                f"observations must have shape ({len(values)}, {self.observation_dim})"
            )
        return super().forward(
            sample=values,
            timestep=time.reshape(-1),
            global_cond=observations,
        )
