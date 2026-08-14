#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Show that serving cached tower tokens reproduces the stock SmolVLA training loss.

Builds a cache over a handful of episodes, then runs the same batch through the
stock policy and through the patched one with the flow-matching noise and time
held fixed, so any difference is the cache rather than the sampler.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from pick_and_place.policies.smolvla_prefix_cache import (
    CachedPrefixDataset,
    patch_policy_for_cached_prefix,
    write_prefix_cache,
)


def collate(dataset, indices: list[int]) -> dict:  # noqa: ANN001
    from torch.utils.data import default_collate

    return default_collate([dataset[index] for index in indices])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    device = torch.device(args.device)
    config = SmolVLAConfig(pretrained_path=str(args.checkpoint), device=device.type)

    plain = LeRobotDataset(repo_id=args.dataset.name, root=args.dataset, episodes=args.episodes)
    delta = {"action": [index / plain.fps for index in config.action_delta_indices]}
    dataset = LeRobotDataset(
        repo_id=args.dataset.name, root=args.dataset, episodes=args.episodes, delta_timestamps=delta
    )

    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
            },
        },
    )

    indices = list(range(args.batch_size))
    stock_batch = preprocessor(collate(dataset, indices))

    generator = torch.Generator(device="cpu").manual_seed(0)
    actions_shape = (args.batch_size, config.chunk_size, config.max_action_dim)
    noise = torch.randn(actions_shape, generator=generator).to(device)
    time_values = torch.rand(args.batch_size, generator=generator).to(device) * 0.999 + 0.001

    # Both arms run under the precision the training run uses, and the cache is
    # built under it too. Comparing an autocast forward against a cache computed
    # in float32 would measure the precision, not the cache.
    autocast_dtype = None if device.type == "cpu" else torch.bfloat16
    autocast = torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None)
    # Store at the precision the tower produces. On CUDA that is bfloat16 and the
    # cache is bit-exact; on CPU there is no bfloat16 autocast, so a bfloat16
    # cache would be measuring its own rounding rather than the substitution.
    cache_dtype = "float32" if autocast_dtype is None else "bfloat16"

    with torch.no_grad(), autocast:
        stock_loss, _ = policy.forward(dict(stock_batch), noise=noise, time=time_values)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        write_prefix_cache(
            policy,
            plain,
            cache_dir,
            batch_size=args.batch_size,
            num_workers=0,
            dtype=cache_dtype,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        cached_dataset = CachedPrefixDataset(dataset, cache_dir)
        cached_batch = preprocessor(collate(cached_dataset, indices))
        patch_policy_for_cached_prefix(policy)
        with torch.no_grad(), autocast:
            cached_loss, _ = policy.forward(dict(cached_batch), noise=noise, time=time_values)

    difference = abs(cached_loss.item() - stock_loss.item())
    print(f"cache dtype {cache_dtype}, autocast {autocast_dtype}")
    print(f"stock loss  {stock_loss.item():.8f}")
    print(f"cached loss {cached_loss.item():.8f}")
    print(f"difference  {difference:.3e} ({difference / abs(stock_loss.item()):.3e} relative)")


if __name__ == "__main__":
    main()
