#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Evaluate a SmolVLA checkpoint with the same supervised loss used for training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(".cache/huggingface/datasets").resolve()))

import torch
from tqdm import tqdm

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


def _default_checkpoint() -> Path:
    return Path("outputs/train/pick-and-place/checkpoints/003000/pretrained_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_checkpoint(),
        help="LeRobot pretrained_model checkpoint directory, unless --all-checkpoints is set.",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every checkpoints/*/pretrained_model directory under --checkpoints-root.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("outputs/train/pick-and-place/checkpoints"),
        help="Root containing step checkpoint directories for --all-checkpoints.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=Path("../datasets-512/combined-success-val"),
        help="Held-out LeRobot dataset root, relative to py/ by default.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cuda | mps | cpu. Auto prefers cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional quick smoke limit. Omit to evaluate the whole validation split.",
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path) -> dict:
    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.dataset.root = str(args.val_root)
    cfg.dataset.repo_id = f"{cfg.dataset.repo_id}-val"
    cfg.dataset.image_transforms.enable = False
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.policy.pretrained_path = checkpoint
    cfg.policy.device = select_device(args.device)

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.policy.device == "cuda",
        drop_last=False,
    )

    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    last_output: dict[str, float] = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            batch = preprocessor(batch)
            loss, output_dict = policy.forward(batch)
            batch_examples = int(next(iter(batch.values())).shape[0])
            total_loss += float(loss.item()) * batch_examples
            total_examples += batch_examples
            total_batches += 1
            last_output = {
                key: float(value)
                for key, value in output_dict.items()
                if isinstance(value, int | float)
            }
            if args.max_batches is not None and total_batches >= args.max_batches:
                break

    return {
        "checkpoint": str(checkpoint),
        "val_root": str(args.val_root),
        "frames_scored": total_examples,
        "batches": total_batches,
        "loss": total_loss / total_examples,
        "last_batch": last_output,
    }


def main() -> None:
    args = parse_args()
    if args.all_checkpoints:
        checkpoints = sorted(args.checkpoints_root.glob("*/pretrained_model"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {args.checkpoints_root}")
    else:
        checkpoints = [args.checkpoint]

    results = [evaluate_checkpoint(args, checkpoint) for checkpoint in checkpoints]
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
