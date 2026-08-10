#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train the state-conditioned flow policy on an exported dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pick_and_place.policies.flow_matching import FlowModel, train_model
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--time-embedding-dim", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--cube-symmetry-augmentation", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.checkpoint_interval is not None and args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")

    device = select_device(args.device)
    augmentation = load_cube_symmetry_augmentation(args.dataset) if args.cube_symmetry_augmentation else None
    args.output.mkdir(parents=True)

    def checkpoint_data(
        step: int, model: FlowModel, training_losses: list[float], validation_losses: list[float]
    ) -> dict:
        return {
            "model": model.state_dict(),
            "model_config": {
                "input_dim": model.input_dim,
                "output_dim": model.output_dim,
                "hidden_dim": model.hidden_dim,
                "hidden_layers": model.hidden_layers,
                "time_embedding_dim": model.time_embedding_dim,
            },
            "step": step,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
        }

    def save_checkpoint(
        step: int, model: FlowModel, training_losses: list[float], validation_losses: list[float]
    ) -> None:
        if args.checkpoint_interval is not None and step % args.checkpoint_interval == 0:
            torch.save(
                checkpoint_data(step, model, training_losses, validation_losses),
                args.output / f"checkpoint-{step:06d}.pt",
            )

    config = vars(args) | {"dataset": str(args.dataset.resolve()), "device": str(device)}
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
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            time_embedding_dim=args.time_embedding_dim,
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
        checkpoint_data(args.epochs, model, training_losses, validation_losses), args.output / "checkpoint.pt"
    )
    with (args.output / "config.json").open("w") as file:
        json.dump(config, file, indent=2, default=str)
        file.write("\n")
    print(f"final train_loss={training_losses[-1]:.6f}")
    if validation_losses:
        print(f"final validation_loss={validation_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
