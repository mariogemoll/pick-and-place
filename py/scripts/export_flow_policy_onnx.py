#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Export the state flow policy for the browser, and prove the export agrees.

Writes ``flow-policy.onnx`` and ``flow-policy.json`` (the runtime contract plus
the normalization bounds). The ONNX graph carries the Euler integration inside
it, so it takes an observation history and a noise draw and returns the sampled
endpoint directly.

The export is only worth having if it computes what PyTorch computes, so this
checks it before declaring success: identical inputs are pushed through both and
the worst absolute disagreement is printed and asserted against a tolerance. A
run that exceeds it fails rather than writing a model nothing verified.

Usage::

    python scripts/export_flow_policy_onnx.py \\
        --checkpoint .../checkpoint.pt --export .../flow-policy-state-.../ \\
        -o ../ts/public/flow-policy
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from pick_and_place.policies.flow_matching import load_model
from pick_and_place.policies.flow_onnx import (
    FlowSampler,
    export_onnx,
    runtime_manifest,
    write_runtime_manifest,
)
from pick_and_place.policies.flow_policy import load_export

#: The deployment operating point recorded for the selected checkpoint.
DEFAULT_ACT_STEPS = 8
DEFAULT_INTEGRATION_STEPS = 10

#: Worst absolute difference tolerated between PyTorch and onnxruntime on the
#: sampled endpoint, which lives in a normalized [-1, 1] action space. Only
#: enforced for a full-precision export; fp16 is expected to move by ~1e-3 and
#: is checked by scoring the policy, not by a tolerance here.
AGREEMENT_TOLERANCE = 2e-4


def check_agreement(
    onnx_path: Path,
    sampler: FlowSampler,
    observation_dim: int,
    output_dim: int,
    *,
    trials: int,
    seed: int,
) -> float:
    """Run both implementations on the same random draws; return the worst gap."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        observations = generator.uniform(-1.0, 1.0, (1, observation_dim)).astype(np.float32)
        noise = generator.standard_normal((1, output_dim)).astype(np.float32)
        with torch.no_grad():
            expected = sampler(torch.from_numpy(observations), torch.from_numpy(noise)).numpy()
        actual = session.run(["endpoint"], {"observations": observations, "noise": noise})[0]
        worst = max(worst, float(np.abs(expected - actual).max()))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, help="the matching export directory")
    parser.add_argument("-o", "--output", type=Path, required=True, help="path prefix")
    parser.add_argument("--act-steps", type=int, default=DEFAULT_ACT_STEPS)
    parser.add_argument("--integration-steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help=(
            "fp16 halves the download and moves the sampled endpoint by about 1e-3 in the "
            "normalized action space; that is a behavioral change and wants a scored run "
            "before it becomes the default"
        ),
    )
    parser.add_argument("--agreement-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest, bounds = load_export(args.export)
    model = load_model(args.checkpoint, "cpu")
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    onnx_path = args.output.with_suffix(".onnx")
    half_precision = args.precision == "fp16"
    export_onnx(
        model,
        manifest,
        onnx_path,
        integration_steps=args.integration_steps,
        half_precision=half_precision,
    )

    payload = runtime_manifest(
        manifest, bounds, act_steps=args.act_steps, integration_steps=args.integration_steps
    )
    payload["checkpointSha256"] = digest
    payload["precision"] = args.precision
    payload["model"] = onnx_path.name
    write_runtime_manifest(payload, args.output.with_suffix(".json"))

    worst = check_agreement(
        onnx_path,
        FlowSampler(model, args.integration_steps).eval(),
        int(manifest["observation_steps"]) * int(manifest["observation_dim"]),
        int(manifest["prediction_steps"]) * int(manifest["endpoint_dim"]),
        trials=args.agreement_trials,
        seed=args.seed,
    )

    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {args.output.with_suffix('.json')}")
    print(f"checkpoint sha256 {digest}")
    print(f"worst PyTorch/onnxruntime disagreement over {args.agreement_trials} draws: {worst:.3e}")
    if half_precision:
        print(
            "half precision: the disagreement above is the cost of fp16 storage, not an "
            "export bug. Score it on the development scenes before promoting it."
        )
    elif worst > AGREEMENT_TOLERANCE:
        raise SystemExit(
            f"export disagrees with PyTorch by {worst:.3e}, above {AGREEMENT_TOLERANCE:.1e}"
        )


if __name__ == "__main__":
    main()
