#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run `lerobot-train` on SmolVLA with the frozen vision tower served from a cache.

Every argument except `--prefix-cache` is lerobot's own and is forwarded
untouched, so this is the stock trainer with two objects swapped: the dataset
serves precomputed tower tokens instead of decoding video, and the policy reads
them instead of running the tower. Build the cache first with
`precompute_smolvla_prefix.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def install_cached_prefix(cache_dir: Path, seed: int) -> None:
    """Swap the dataset and the policy that `lerobot_train.train` will build.

    Both factories are imported into the training module's namespace at import
    time, so rebinding them there is what the training run actually sees.
    """
    from lerobot.scripts import lerobot_train

    from pick_and_place.policies.smolvla_prefix_cache import (
        CachedPrefixDataset,
        patch_policy_for_cached_prefix,
    )

    make_dataset = lerobot_train.make_dataset
    make_policy = lerobot_train.make_policy

    def cached_make_dataset(cfg):  # noqa: ANN001, ANN202
        return CachedPrefixDataset(make_dataset(cfg), cache_dir, seed=seed)

    def cached_make_policy(**kwargs):  # noqa: ANN003, ANN202
        policy = make_policy(**kwargs)
        if policy.config.type != "smolvla":
            raise ValueError(f"--prefix-cache only applies to smolvla, not {policy.config.type}")
        patch_policy_for_cached_prefix(policy)
        return policy

    lerobot_train.make_dataset = cached_make_dataset
    lerobot_train.make_policy = cached_make_policy


def take_flag(argv: list[str], name: str) -> str | None:
    """Pull one `--name value` or `--name=value` out of argv, leaving the rest for draccus."""
    for index, argument in enumerate(argv):
        if argument == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} needs a value")
            value = argv[index + 1]
            del argv[index : index + 2]
            return value
        if argument.startswith(f"{name}="):
            del argv[index]
            return argument.split("=", 1)[1]
    return None


def main() -> None:
    argv = sys.argv[1:]
    cache = take_flag(argv, "--prefix-cache")
    if cache is None:
        raise SystemExit("--prefix-cache <dir> is required; use lerobot-train for the stock path")
    seed = take_flag(argv, "--prefix-cache-seed")
    sys.argv = [sys.argv[0], *argv]

    from lerobot.scripts.lerobot_train import main as lerobot_main

    install_cached_prefix(Path(cache), int(seed) if seed is not None else 0)
    lerobot_main()


if __name__ == "__main__":
    main()
