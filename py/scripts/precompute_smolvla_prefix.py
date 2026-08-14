#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a SmolVLA dataset through the frozen vision tower once and store the tokens.

The tower is 59% of a training step and its weights never move, so a run that
covers the dataset N times computes the same tokens N times. This writes them out
once; `train_smolvla_cached.py` reads them back.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pick_and_place.policies.smolvla_prefix_cache import cache_nbytes, write_prefix_cache


def build_dataset(root: Path, augment: bool, episodes: list[int] | None):  # noqa: ANN201 - lerobot type
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig

    transforms = ImageTransforms(ImageTransformsConfig(enable=True)) if augment else None
    # No delta_timestamps: the cache only needs the frames, and asking for the
    # action chunk would make every read fetch fifty more rows for nothing.
    return LeRobotDataset(
        repo_id=root.name, root=root, episodes=episodes, image_transforms=transforms
    )


def build_policy(checkpoint: Path, dataset, device: torch.device):  # noqa: ANN001, ANN201
    """Build the policy the way `--policy.pretrained_path` does.

    Not `PreTrainedConfig.from_pretrained`: that restores the base checkpoint's own
    `camera1`/`camera2`/`camera3` input features, which then fail to match this
    dataset's camera names. A fresh config leaves `input_features` empty, and
    `make_policy` fills it from the dataset -- which is what makes the cache line
    up with training, camera for camera and in the same order.
    """
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig(pretrained_path=str(checkpoint), device=device.type)
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SmolVLA base checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="directory to write the cache into")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="independently augmented passes to store per frame; needs --augment above 1",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="apply lerobot's image transforms, so each variant sees a different draw",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="cache only these episodes; training must then select the same ones",
    )
    args = parser.parse_args()

    if args.variants > 1 and not args.augment:
        parser.error("--variants above 1 stores identical copies unless --augment is set")

    device = torch.device(args.device)
    dataset = build_dataset(args.dataset, args.augment, args.episodes)
    policy = build_policy(args.checkpoint, dataset, device)

    started = time.monotonic()
    last = [started]

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] < 30 and done != total:
            return
        last[0] = now
        rate = done / max(now - started, 1e-9)
        remaining = (total - done) / max(rate, 1e-9)
        print(
            f"{done}/{total} frames  {rate:.0f} frames/s  {remaining / 60:.1f} min left",
            flush=True,
        )

    spec = write_prefix_cache(
        policy,
        dataset,
        args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        variants=args.variants,
        dtype=args.dtype,
        device=device,
        progress=progress,
    )
    elapsed = time.monotonic() - started
    print(f"wrote {spec.shape} {spec.dtype} to {args.output}")
    print(f"{cache_nbytes(spec) / 1e9:.1f} GB in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
