# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The trained flow velocity field, in the calling convention DPPO's models use.

DPPO calls its actor as ``actor(x, t, cond=cond)`` with an action sequence, a
time, and a dictionary of observation histories; this project's flow network
takes a sequence, a flow time, and one flattened observation vector. The
adapter is that translation and nothing else -- the weights, the architecture
and the checkpoint format all stay the ones behavior cloning produced.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pick_and_place.policies.diffusion_policy_unet import FlowConditionalUnet1D
from pick_and_place.policies.flow_matching import load_model
from pick_and_place.policies.flow_policy import load_export


class FlowActor(torch.nn.Module):
    """Wrap a trained velocity field for DPPO's sampler and log-probabilities."""

    def __init__(self, network: FlowConditionalUnet1D) -> None:
        super().__init__()
        self.network = network
        self.action_dim = network.action_dim
        self.horizon_steps = network.prediction_steps
        self.observation_dim = network.observation_dim

    def forward(self, values: torch.Tensor, time: torch.Tensor, cond: dict) -> torch.Tensor:
        """Predict the flow velocity at ``values`` for observation history ``cond``.

        ``values`` is ``(B, horizon_steps, action_dim)``, ``time`` is a flow time
        per batch element in ``[0, 1]``, and ``cond["state"]`` is
        ``(B, cond_steps, observation_dim / cond_steps)``, most recent last --
        the same order and flattening the exported training observations used.
        """
        state = cond["state"]
        return self.network(values, time.reshape(-1), state.reshape(len(state), -1))


def load_flow_actor(
    checkpoint_path: str | Path,
    export_dir: str | Path,
    *,
    device: str = "cpu",
) -> FlowActor:
    """Load a behavior-cloned flow checkpoint as a DPPO actor.

    ``export_dir`` is the export the checkpoint was trained against. Checking
    the pairing here is what the closed-loop runner does too: a checkpoint
    carries its own dimensions, and a mismatched export does not fail, it
    silently feeds the policy the wrong units.
    """
    network = load_model(checkpoint_path, device)
    if not isinstance(network, FlowConditionalUnet1D):
        raise ValueError("reinforcement learning requires a temporal U-Net flow checkpoint")
    manifest, _ = load_export(export_dir)
    expected = (
        int(manifest["prediction_steps"]),
        int(manifest["endpoint_dim"]),
        int(manifest["observation_steps"]) * int(manifest["observation_dim"]),
    )
    actual = (network.prediction_steps, network.action_dim, network.observation_dim)
    if actual != expected:
        raise ValueError(f"checkpoint dimensions {actual} do not match the export's {expected}")
    return FlowActor(network)
