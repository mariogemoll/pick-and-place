#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train an image-conditioned flow-matching policy on a Diffusion Policy export.

The flow objective is the state policy's, unchanged: a Gaussian conditional
optimal-transport path, a velocity target, a 16-step chunk of which 8 are
executed. Only the conditioning differs -- two camera streams and the robot's
own joints, instead of the simulator's privileged cube and target poses.

The schedule follows the state policy's selected recipe, whose 100,000-update
continuation established that this configuration overfits past roughly 30,000:
warmup, then cosine decay reaching its minimum at the final update.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pick_and_place.data.flow_image_dataset import FlowImageExport
from pick_and_place.policies.flow_image_encoder import FlowImageUnet1D, model_config
from pick_and_place.policies.flow_matching import learning_rate_at_step

# ImageNet statistics: the right normalization when the trunk starts from
# ImageNet weights, and a harmless fixed affine when it does not.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def random_shift(images: torch.Tensor, pad: int, generator: torch.Generator) -> torch.Tensor:
    """Translate each camera stream by a few pixels, replicating at the edge.

    The Diffusion Policy configuration this strand inherits sets ``augment:
    true`` on its vision backbone, and image policies for manipulation rely on
    it heavily: without it the encoder can memorize absolute pixel positions
    rather than learning to locate the objects.

    One shift is drawn per sample and camera and shared across the observation
    timesteps, so the augmentation cannot manufacture apparent motion between
    the two frames the policy differences to infer velocity.
    """
    if pad < 1:
        return images
    batch, steps, channels, height, width = images.shape
    cameras = channels // 3
    folded = images.reshape(batch * steps * cameras, 3, height, width)
    padded = torch.nn.functional.pad(folded, (pad, pad, pad, pad), mode="replicate")
    offsets = torch.randint(
        0, 2 * pad + 1, (batch, 1, cameras, 2), generator=generator, device="cpu"
    ).expand(batch, steps, cameras, 2).reshape(-1, 2)
    rows = torch.arange(height, device=images.device)
    columns = torch.arange(width, device=images.device)
    row_index = (offsets[:, 0:1].to(images.device) + rows[None, :]).reshape(-1, height, 1)
    column_index = (offsets[:, 1:2].to(images.device) + columns[None, :]).reshape(-1, 1, width)
    sample = torch.arange(len(folded), device=images.device).reshape(-1, 1, 1)
    cropped = padded[sample, :, row_index, column_index]
    return cropped.permute(0, 3, 1, 2).reshape(batch, steps, channels, height, width)


def prepare_images(
    raw: np.ndarray, device: torch.device, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Move a uint8 observation window to the device and normalize it."""
    batch = torch.from_numpy(np.ascontiguousarray(raw)).to(device, non_blocking=True)
    steps, channels = batch.shape[1], batch.shape[2]
    images = batch.float().div_(255.0).reshape(-1, 3, *batch.shape[-2:])
    images = (images - mean) / std
    return images.reshape(len(batch), steps, channels, *batch.shape[-2:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="Diffusion Policy image export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--observation-steps", type=int, default=2)
    parser.add_argument("--prediction-steps", type=int, default=16)
    parser.add_argument("--keypoints", type=int, default=32)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--trunk-stages",
        type=int,
        default=3,
        choices=(1, 2, 3, 4),
        help="ResNet18 residual stages to keep; 3 stops after layer3, halving the "
        "model and doubling the keypoint map the spatial softmax localizes over "
        "(default: 3). Pass 4 for the full trunk. Note this is the default for "
        "*new runs* only -- CameraEncoder still defaults to 4, because "
        "checkpoints written before the flag existed carry no trunk_stages in "
        "their model_config and must keep loading as full trunks.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=2_000)
    parser.add_argument("--validation-batches", type=int, default=40)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--random-shift",
        type=int,
        default=0,
        help="pixels of random translation augmentation per camera (0 disables)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "warm-start from a checkpoint's weights and run a fresh schedule. "
            "The optimizer state is deliberately not restored: this is a new "
            "cosine cycle, not a continuation of the old one"
        ),
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    # Matching train_flow_policy.py: logging is opt-in through --wandb-project,
    # so a run without it stays offline rather than half-configured.
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    export = FlowImageExport(
        args.export,
        observation_steps=args.observation_steps,
        prediction_steps=args.prediction_steps,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        f"{len(export.training_frames)} training / {len(export.validation_frames)} validation "
        f"frames, {export.cameras} cameras at {export.image_hw[0]}x{export.image_hw[1]}, "
        f"state {export.state_dim}, action {export.action_dim}",
        flush=True,
    )

    model = FlowImageUnet1D(
        action_dim=export.action_dim,
        state_dim=export.state_dim,
        prediction_steps=args.prediction_steps,
        observation_steps=args.observation_steps,
        cameras=export.cameras,
        keypoints=args.keypoints,
        pretrained_backbone=args.pretrained_backbone,
        trunk_stages=args.trunk_stages,
    ).to(device)
    if args.resume is not None:
        restored = torch.load(args.resume, map_location=device, weights_only=False)
        if restored["model_config"] != model_config(model):
            raise SystemExit("resume checkpoint was built with a different configuration")
        model.load_state_dict(restored["model"])
        print(f"warm-started from {args.resume} at update {restored.get('update')}", flush=True)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model has {parameters / 1e6:.2f}M parameters", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda")

    # A fixed validation draw, so the held-out number moves only when the model
    # does. Interleaving validation with the optimizer is what produced the
    # state policy's periodic training spikes, so it stays infrequent.
    shift_generator = torch.Generator().manual_seed(args.seed + 2)
    validation_rng = np.random.default_rng(args.seed + 1)
    validation_batches = [
        np.sort(validation_rng.choice(export.validation_frames, size=args.batch_size, replace=False))
        for _ in range(args.validation_batches)
    ] if len(export.validation_frames) >= args.batch_size else []

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.output.name,
            config=vars(args)
            | {
                "export": str(args.export.resolve()),
                "output": str(args.output.resolve()),
                "device": str(device),
                "parameters": parameters,
                "image_hw": list(export.image_hw),
                "cameras": export.cameras,
            },
        )

    history: list[dict] = []
    started = time.time()
    try:
        for update in range(args.updates):
            rate = learning_rate_at_step(
                update,
                num_steps=args.updates,
                peak=args.learning_rate,
                minimum=args.min_learning_rate,
                warmup_steps=args.warmup_steps,
            )
            optimizer.param_groups[0]["lr"] = rate
            model.train()

            frames = np.sort(rng.choice(export.training_frames, size=args.batch_size, replace=False))
            raw_images, raw_states, raw_chunks = export.batch(frames)
            images = prepare_images(raw_images, device, mean, std)
            if args.random_shift:
                images = random_shift(images, args.random_shift, shift_generator)
            states = torch.from_numpy(raw_states).to(device)
            endpoints = torch.from_numpy(raw_chunks).to(device)

            time_sample = torch.rand(len(endpoints), 1, 1, device=device)
            noise = torch.randn_like(endpoints)
            values = time_sample * endpoints + (1 - time_sample) * noise
            with autocast:
                prediction = model(values, time_sample.reshape(-1, 1), images, states)
                loss = torch.mean((prediction.float() - (endpoints - noise)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (update + 1) % args.log_interval == 0:
                print(
                    f"update {update + 1:>6}/{args.updates}  loss {loss.item():.5f}  "
                    f"lr {rate:.2e}  {(time.time() - started) / (update + 1) * 1000:.0f} ms/update",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/learning_rate": rate,
                            "train/ms_per_update": (time.time() - started) / (update + 1) * 1000,
                        },
                        step=update + 1,
                    )

            validate = validation_batches and (
                (update + 1) % args.validation_interval == 0 or update == args.updates - 1
            )
            if validate:
                model.eval()
                total = 0.0
                with torch.no_grad():
                    for index, batch_frames in enumerate(validation_batches):
                        v_images, v_states, v_chunks = export.batch(batch_frames)
                        v_endpoints = torch.from_numpy(v_chunks).to(device)
                        generator = torch.Generator(device="cpu").manual_seed(args.seed * 1000 + index)
                        v_time = torch.rand(len(v_endpoints), 1, 1, generator=generator).to(device)
                        v_noise = torch.randn(
                            v_endpoints.shape, generator=generator
                        ).to(device)
                        v_values = v_time * v_endpoints + (1 - v_time) * v_noise
                        with autocast:
                            v_prediction = model(
                                v_values,
                                v_time.reshape(-1, 1),
                                prepare_images(v_images, device, mean, std),
                                torch.from_numpy(v_states).to(device),
                            )
                        total += torch.mean(
                            (v_prediction.float() - (v_endpoints - v_noise)) ** 2
                        ).item()
                validation_loss = total / len(validation_batches)
                row = {"update": update + 1, "train_loss": loss.item(), "validation_loss": validation_loss}
                history.append(row)
                print(f"  validation @ {update + 1}: {validation_loss:.5f}", flush=True)
                if wandb_run is not None:
                    # Logged for the record, not for selection. POLICY_RESULTS.md
                    # has held-out loss plateauing while closed-loop success kept
                    # climbing for ~900 epochs; the score series is the signal.
                    wandb_run.log({"validation/loss": validation_loss}, step=update + 1)
                with (args.output / "history.json").open("w") as file:
                    json.dump(history, file, indent=2)

            if (update + 1) % args.checkpoint_interval == 0 or update == args.updates - 1:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_type": "flow_image_unet1d",
                        "model_config": model_config(model),
                        "update": update + 1,
                        "export": str(args.export),
                        "observation_steps": args.observation_steps,
                        "prediction_steps": args.prediction_steps,
                        "seed": args.seed,
                        "random_shift": args.random_shift,
                        "resumed_from": str(args.resume) if args.resume else None,
                    },
                    args.output / f"checkpoint-{update + 1:06d}.pt",
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_type": "flow_image_unet1d",
                        "model_config": model_config(model),
                        "update": update + 1,
                        "export": str(args.export),
                        "observation_steps": args.observation_steps,
                        "prediction_steps": args.prediction_steps,
                        "seed": args.seed,
                        "random_shift": args.random_shift,
                        "resumed_from": str(args.resume) if args.resume else None,
                    },
                    args.output / "checkpoint.pt",
                )

        print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
