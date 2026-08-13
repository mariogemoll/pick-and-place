#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""How far the policy's commands land from the expert's, on *real* episodes.

`diagnose_flow_image_policy.py` answers this on the export the policy was
trained from. This answers it on a dataset recorded by the physical rig, which
is the number that matters for transfer: a policy whose chunks are accurate on
real observations is failing on hardware for some reason other than what it
sees, and one whose chunks are wrong there has no chance closed-loop.

**The two exports do not share a normalization, and that is the whole hazard.**
Every Diffusion Policy export fits its own per-dimension min-max bounds, so the
real export's stored states and actions are scaled by the real data's spread
while the checkpoint only understands the sim export's. Feeding one to the other
silently shifts and stretches every joint. So states are decoded with the real
bounds and re-encoded with the policy's, and predictions are decoded with the
policy's bounds before being compared, in degrees, against expert actions
decoded with the real ones.

Two references make the number readable. Holding the current joint position for
the whole horizon is what a policy that has learned nothing would score, and the
same measurement run on a slice of the *sim* export is the matched in-domain
control -- pass ``--reference-export`` to get it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pick_and_place.data.flow_image_dataset import FlowImageExport
from pick_and_place.policies.dataset_export import load_bounds, normalize, unnormalize
from pick_and_place.policies.flow_image_policy import (
    IMAGE_MEAN,
    IMAGE_STD,
    generate_horizon,
    load_model,
)


def percentiles(signed: np.ndarray) -> dict[str, float]:
    """Magnitude summary of a signed error stack, in degrees."""
    error = np.abs(signed)
    return {
        "mean_deg": float(error.mean()),
        "median_deg": float(np.median(error)),
        "p90_deg": float(np.percentile(error, 90)),
        "max_deg": float(error.max()),
    }


def evaluate(
    export: FlowImageExport,
    export_bounds: dict[str, np.ndarray],
    policy_bounds: dict[str, np.ndarray],
    model,
    frames: np.ndarray,
    *,
    device: torch.device,
    integration_steps: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-example absolute errors for the policy and the hold baseline.

    Both stacks are ``(examples, chunk_steps, joints)`` in degrees.
    """
    mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
    policy_error, hold_error = [], []

    for start in range(0, len(frames), batch_size):
        chosen = np.sort(frames[start : start + batch_size])
        raw_images, raw_states, raw_chunks = export.batch(chosen)

        # Out of this export's scale, into the one the checkpoint was fitted to.
        states_deg = unnormalize(raw_states, export_bounds["obs_min"], export_bounds["obs_max"])
        expert_deg = unnormalize(
            raw_chunks, export_bounds["action_min"], export_bounds["action_max"]
        )
        states_for_policy = normalize(
            states_deg, policy_bounds["obs_min"], policy_bounds["obs_max"]
        )

        images = torch.from_numpy(np.ascontiguousarray(raw_images)).to(device).float().div_(255.0)
        steps, channels = images.shape[1], images.shape[2]
        folded = images.reshape(-1, 3, *images.shape[-2:])
        images = ((folded - mean) / std).reshape(
            len(chosen), steps, channels, *raw_images.shape[-2:]
        )
        states = torch.from_numpy(states_for_policy).to(device).float()

        generator = torch.Generator(device=device).manual_seed(seed * 100_000 + int(chosen[0]))
        noise = torch.randn(
            (len(chosen), model.prediction_steps, model.action_dim),
            generator=generator,
            device=device,
        )
        # generate_horizon samples one observation at a time, so the batch is
        # walked rather than folded -- the model is the same either way.
        predictions = np.stack(
            [
                generate_horizon(
                    model,
                    images[index : index + 1],
                    states[index : index + 1],
                    integration_steps=integration_steps,
                    noise=noise[index : index + 1],
                )
                for index in range(len(chosen))
            ]
        )
        predicted_deg = unnormalize(
            np.clip(predictions, -1, 1), policy_bounds["action_min"], policy_bounds["action_max"]
        )
        # Signed, so the caller can separate a systematic aim from scatter: a
        # policy that is merely imprecise averages to zero per joint, while one
        # fighting a calibration or tracking offset does not.
        policy_error.append(predicted_deg - expert_deg)
        # The newest observation timestep is the arm's current pose; a policy
        # that had learned nothing could still hold it for the whole horizon.
        hold_error.append(states_deg[:, -1][:, None, :] - expert_deg)

    return np.concatenate(policy_error), np.concatenate(hold_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy-export", type=Path, required=True, help="the export the checkpoint was trained on"
    )
    parser.add_argument("--real-export", type=Path, required=True, help="a real-episode export")
    parser.add_argument(
        "--reference-export",
        type=Path,
        default=None,
        help="optional in-domain export scored the same way, as the matched control",
    )
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--integration-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    policy_bounds = load_bounds(args.policy_export)

    report: dict[str, dict] = {}
    for label, export_dir in (("real", args.real_export), ("sim", args.reference_export)):
        if export_dir is None:
            continue
        export = FlowImageExport(
            export_dir,
            observation_steps=model.observation_steps,
            prediction_steps=model.prediction_steps,
            validation_fraction=0.0,
        )
        rng = np.random.default_rng(args.seed)
        available = export.training_frames
        chosen = rng.choice(available, size=min(args.examples, len(available)), replace=False)
        policy_error, hold_error = evaluate(
            export,
            load_bounds(export_dir),
            policy_bounds,
            model,
            chosen,
            device=device,
            integration_steps=args.integration_steps,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        report[label] = {
            "export": str(export_dir),
            "examples": int(len(policy_error)),
            "policy": percentiles(policy_error),
            "hold_baseline": percentiles(hold_error),
            "per_chunk_step": [float(v) for v in np.abs(policy_error).mean(axis=(0, 2))],
            "per_joint": [float(v) for v in np.abs(policy_error).mean(axis=(0, 1))],
            "signed_bias_per_joint": [float(v) for v in policy_error.mean(axis=(0, 1))],
        }

    for label, entry in report.items():
        print(f"\n=== {label} ({entry['examples']} examples) ===")
        print(f"{'':<16}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}   (degrees)")
        for name in ("policy", "hold_baseline"):
            values = entry[name]
            print(
                f"{name:<16}{values['mean_deg']:>9.3f}{values['median_deg']:>9.3f}"
                f"{values['p90_deg']:>9.3f}{values['max_deg']:>9.3f}"
            )
        print("by position in the horizon: " + " ".join(f"{v:.2f}" for v in entry["per_chunk_step"]))
        print("by joint:                   " + " ".join(f"{v:.2f}" for v in entry["per_joint"]))
        print(
            "signed bias by joint:       "
            + " ".join(f"{v:+.2f}" for v in entry["signed_bias_per_joint"])
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump({"checkpoint": str(args.checkpoint), "reports": report}, file, indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
