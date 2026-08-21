#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Report an image flow policy's open-loop action error in joint units.

The export's actions are already in degrees -- its normalization spans are
147-280 wide, not radians -- so the unnormalized difference is a degree
difference and must not be converted again.

A velocity MSE says nothing legible about behaviour. This runs the sampler the
controller actually uses -- Euler integration from noise -- on held-out windows
and reports how far the generated joint commands land from the expert's, in
degrees, split by position within the chunk.

That separates two very different failures. Accurate open-loop chunks with poor
rollouts mean compounding error under covariate shift, which more training or
better data addresses. Inaccurate chunks mean the mapping itself has not been
learned, and no amount of rollout tuning will help.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pick_and_place.cli.diagnose_flow_image_policy import build_parser
from pick_and_place.data.flow_image_dataset import FlowImageExport
from pick_and_place.policies.flow_image_encoder import FlowImageUnet1D

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def run(args: argparse.Namespace) -> None:
    """Sample held-out windows and report the open-loop joint error."""
    device = torch.device(args.device)
    contents = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FlowImageUnet1D(**contents["model_config"]).to(device)
    model.load_state_dict(contents["model"])
    model.eval()

    export = FlowImageExport(
        args.export,
        observation_steps=model.observation_steps,
        prediction_steps=model.prediction_steps,
        validation_fraction=0.1,
        seed=contents.get("seed", 0),
    )
    with np.load(args.export / "normalization.npz", allow_pickle=False) as archive:
        action_min = np.asarray(archive["action_min"], dtype=np.float32)
        action_max = np.asarray(archive["action_max"], dtype=np.float32)
    span = action_max - action_min

    mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    per_step_errors: list[np.ndarray] = []
    for _ in range(args.batches):
        frames = np.sort(rng.choice(export.validation_frames, size=args.batch_size, replace=False))
        raw_images, raw_states, raw_chunks = export.batch(frames)
        images = torch.from_numpy(np.ascontiguousarray(raw_images)).to(device)
        steps, channels = images.shape[1], images.shape[2]
        floats = images.float().div_(255.0).reshape(-1, 3, *images.shape[-2:])
        floats = ((floats - mean) / std).reshape(len(frames), steps, channels, *images.shape[-2:])
        states = torch.from_numpy(raw_states).to(device)

        values = torch.randn(
            len(frames), model.prediction_steps, model.action_dim, device=device
        )
        time = torch.zeros(len(frames), 1, device=device)
        with torch.no_grad():
            condition = model.encode_observation(floats, states)
            for _ in range(args.integration_steps):
                values = values + model.unet(values, time, condition) / args.integration_steps
                time = time + 1 / args.integration_steps
        generated = np.clip(values.cpu().numpy(), -1, 1)

        # Both are normalized to [-1, 1] over the same span, so a difference
        # scales back to the export's own units -- degrees -- by half the span.
        difference = np.abs(generated - raw_chunks) * span / 2.0
        per_step_errors.append(difference)

    errors = np.concatenate(per_step_errors)
    degrees = errors
    print(f"{len(errors)} held-out windows, Euler-{args.integration_steps}\n")
    print(f"{'chunk step':>11} {'mean deg':>9} {'p90 deg':>9}")
    for step in range(errors.shape[1]):
        arm = degrees[:, step, :5]
        print(f"{step:>11} {arm.mean():>9.3f} {np.percentile(arm, 90):>9.3f}")
    executed = degrees[:, :8, :5]
    print(f"\nexecuted steps 0-7, arm joints: mean {executed.mean():.3f} deg, "
          f"p90 {np.percentile(executed, 90):.3f} deg")
    print(f"gripper: mean {degrees[:, :8, 5].mean():.4f} (normalized units x span/2)")
    print(json.dumps({"mean_deg_executed": float(executed.mean())}))


def main() -> None:
    run(build_parser(description=__doc__).parse_args())


if __name__ == "__main__":
    main()
