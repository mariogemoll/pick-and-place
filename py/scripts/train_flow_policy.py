#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train the state-conditioned flow policy on an exported dataset.

``--config PATH`` seeds every flag from an earlier run's ``config.json``, so a
recorded run can be repeated with one thing changed rather than reconstructed by
hand. The command line still wins over the file.
"""

from __future__ import annotations

import json

import torch

from pick_and_place.cli.training import (
    StateTrainingRun,
    config_to_json,
    parse_training_config,
)

from pick_and_place.policies.flow_matching import (
    VelocityModel,
    model_checkpoint_config,
    train_model,
)
from pick_and_place.policies.flow_policy import load_cube_symmetry_augmentation


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_training_config(StateTrainingRun, description=__doc__)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.checkpoint_interval is not None and args.checkpoint_interval < 1:
        raise SystemExit("--checkpoint-interval must be positive")
    manifest_path = args.dataset.resolve().parent / "export.json"
    if manifest_path.is_file():
        with manifest_path.open() as file:
            manifest = json.load(file)
        exported_prediction_steps = int(manifest["prediction_steps"])
        if args.prediction_steps != exported_prediction_steps:
            raise SystemExit(
                f"--prediction-steps {args.prediction_steps} does not match "
                f"{manifest_path} ({exported_prediction_steps})"
            )

    device = select_device(args.device)
    augmentation = (
        load_cube_symmetry_augmentation(args.dataset) if args.cube_symmetry_augmentation else None
    )
    args.output.mkdir(parents=True)

    def checkpoint_data(
        step: int,
        model: VelocityModel,
        training_losses: list[float],
        validation_losses: list[float],
    ) -> dict:
        model_type, model_config = model_checkpoint_config(model)
        return {
            "model": model.state_dict(),
            "model_type": model_type,
            "model_config": model_config,
            "step": step,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
        }

    def save_checkpoint(
        step: int,
        model: VelocityModel,
        training_losses: list[float],
        validation_losses: list[float],
    ) -> None:
        if args.checkpoint_interval is not None and step % args.checkpoint_interval == 0:
            torch.save(
                checkpoint_data(step, model, training_losses, validation_losses),
                args.output / f"checkpoint-{step:06d}.pt",
            )

    config = config_to_json(args) | {"dataset": str(args.dataset.resolve()), "device": str(device)}
    config["validation"] = str(args.validation.resolve()) if args.validation else None
    wandb_run = None
    metrics_callback = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.output.name,
            config=config,
        )

        def log_metrics(step: int, metrics: dict[str, float]) -> None:
            wandb_run.log(metrics, step=step)

        metrics_callback = log_metrics
    try:
        model, training_losses, validation_losses = train_model(
            args.dataset,
            args.validation,
            num_updates=args.updates,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            time_embedding_dim=args.time_embedding_dim,
            architecture=args.architecture,
            prediction_steps=args.prediction_steps,
            unet_down_dims=tuple(args.unet_down_dims),
            unet_kernel_size=args.unet_kernel_size,
            unet_groups=args.unet_groups,
            device=device,
            seed=args.seed,
            observation_augmentation=augmentation,
            validation_interval=args.validation_interval,
            metrics_callback=metrics_callback,
            checkpoint_callback=save_checkpoint,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    torch.save(
        checkpoint_data(args.updates, model, training_losses, validation_losses),
        args.output / "checkpoint.pt",
    )
    with (args.output / "config.json").open("w") as file:
        json.dump(config, file, indent=2, default=str)
        file.write("\n")
    print(f"final train_loss={training_losses[-1]:.6f}")
    if validation_losses:
        print(f"final validation_loss={validation_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
