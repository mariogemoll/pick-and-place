# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export the state flow policy as a single ONNX graph the browser can run.

The graph is the whole sampler, not just the velocity field: the Euler
integration is unrolled inside it, so one call takes a normalized observation
history and a noise draw and returns the endpoint at ``t = 1``. Keeping the loop
in the graph means the browser cannot integrate the flow differently from
:func:`pick_and_place.policies.flow_matching.generate` -- there is no loop on
the other side to get wrong.

Noise is an *input* rather than something the graph samples. The caller owns the
randomness, which is what lets a browser rollout be seeded and replayed, and
what lets this export be checked against PyTorch on identical draws.

Normalization stays outside. It is four vectors of bounds, it is exactly
invertible, and leaving it in the caller keeps the graph a pure function of the
model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pick_and_place.policies.flow_matching import VelocityModel, predict_velocity


class FlowSampler(torch.nn.Module):
    """One Euler-integrated sample from the velocity field, as a traceable module."""

    def __init__(self, model: VelocityModel, integration_steps: int) -> None:
        super().__init__()
        if integration_steps < 1:
            raise ValueError("integration_steps must be positive")
        self.model = model
        self.integration_steps = integration_steps

    def forward(self, observations: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Integrate ``noise`` to ``t = 1`` under the observation-conditioned field.

        Mirrors :func:`pick_and_place.policies.flow_matching.generate` step for
        step, with the draw handed in instead of taken.
        """
        values = noise
        time = torch.zeros(noise.shape[0], 1, dtype=noise.dtype, device=noise.device)
        for _ in range(self.integration_steps):
            velocity = predict_velocity(self.model, values, time, observations)
            values = values + velocity / self.integration_steps
            time = time + 1.0 / self.integration_steps
        return values


def export_onnx(
    model: VelocityModel,
    manifest: dict[str, Any],
    destination: Path,
    *,
    integration_steps: int,
    half_precision: bool = False,
) -> None:
    """Trace the sampler at the deployment operating point and write it out.

    The trace fixes the batch size at one. The browser samples one horizon at a
    time and nothing about the page wants a batch, and a fixed batch is what
    lets the upstream network's shape assertions be traced away rather than
    becoming dynamic control flow.

    ``half_precision`` halves the file at the cost of moving the sampled
    endpoint, which is a change in what the policy does rather than only in how
    it is stored. Measure it before shipping it -- see the exporting script.
    """
    observation_dim = int(manifest["observation_steps"]) * int(manifest["observation_dim"])
    output_dim = int(manifest["prediction_steps"]) * int(manifest["endpoint_dim"])
    sampler = FlowSampler(model, integration_steps).eval()
    example = (torch.zeros(1, observation_dim), torch.zeros(1, output_dim))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            sampler,
            example,
            str(destination),
            input_names=["observations", "noise"],
            output_names=["endpoint"],
            opset_version=17,
            dynamo=False,
        )
    if half_precision:
        import onnx
        from onnxconverter_common import float16

        onnx.save(
            float16.convert_float_to_float16(onnx.load(str(destination)), keep_io_types=True),
            str(destination),
        )


def runtime_manifest(
    manifest: dict[str, Any],
    bounds: dict[str, np.ndarray],
    *,
    act_steps: int,
    integration_steps: int,
) -> dict[str, Any]:
    """The subset of the export contract the browser has to agree on.

    The normalization bounds travel with it. ``export.json`` and
    ``normalization.npz`` are part of the model contract, so splitting them from
    the weights is how a mismatched pair gets shipped.
    """
    return {
        "format": "pick-and-place-flow-policy",
        "version": 1,
        "sourceFormat": manifest["format_version"],
        "observationSteps": int(manifest["observation_steps"]),
        "observationDim": int(manifest["observation_dim"]),
        "observationNames": list(manifest["observation_names"]),
        "predictionSteps": int(manifest["prediction_steps"]),
        "endpointDim": int(manifest["endpoint_dim"]),
        "endpointSemantics": manifest["endpoint_semantics"],
        "policyHz": float(manifest["policy_hz"]),
        "actSteps": act_steps,
        "integrationSteps": integration_steps,
        "normalization": {
            name: [float(v) for v in bounds[name]]
            for name in ("observation_min", "observation_max", "endpoint_min", "endpoint_max")
        },
    }


def write_runtime_manifest(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
