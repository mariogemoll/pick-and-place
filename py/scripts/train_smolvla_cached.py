#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run `lerobot-train` on SmolVLA with everything frozen taken out of the step.

Every argument except the three below is lerobot's own and is forwarded
untouched, so this is the stock trainer with three things swapped: the dataset
serves precomputed tower tokens instead of decoding video, the policy reads them
instead of running the tower and backpropagates only the part of the prefix that
can train, and the tokenizer pads to the task string instead of to 48 -- which is
what the policy config asks for anyway. Together they are 4.4x a stock step, and
each has a flag to turn it off for an A/B. Build the cache first with
`precompute_smolvla_prefix.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def install_cached_prefix(
    cache_dir: Path, seed: int, language_padding: str, frozen_prefix: bool
) -> None:
    """Swap the objects that `lerobot_train.train` will build.

    All three factories are imported into the training module's namespace at
    import time, so rebinding them there is what the training run actually sees.
    """
    from lerobot.scripts import lerobot_train

    from pick_and_place.policies.smolvla_frozen_prefix import patch_policy_for_frozen_prefix
    from pick_and_place.policies.smolvla_language_padding import pad_language_to
    from pick_and_place.policies.smolvla_prefix_cache import (
        CachedPrefixDataset,
        patch_policy_for_cached_prefix,
    )

    make_dataset = lerobot_train.make_dataset
    make_policy = lerobot_train.make_policy
    make_pre_post_processors = lerobot_train.make_pre_post_processors

    def cached_make_dataset(cfg):  # noqa: ANN001, ANN202
        return CachedPrefixDataset(make_dataset(cfg), cache_dir, seed=seed)

    def cached_make_policy(**kwargs):  # noqa: ANN003, ANN202
        policy = make_policy(**kwargs)
        if policy.config.type != "smolvla":
            raise ValueError(f"--prefix-cache only applies to smolvla, not {policy.config.type}")
        patch_policy_for_cached_prefix(policy)
        if frozen_prefix:
            patch_policy_for_frozen_prefix(policy)
        return policy

    def repadded_make_pre_post_processors(**kwargs):  # noqa: ANN003, ANN202
        preprocessor, postprocessor = make_pre_post_processors(**kwargs)
        previous, max_length = pad_language_to(preprocessor, language_padding)
        print(
            f"language padding: {previous} ({max_length}) -> {language_padding}",
            flush=True,
        )
        return preprocessor, postprocessor

    lerobot_train.make_dataset = cached_make_dataset
    lerobot_train.make_policy = cached_make_policy
    lerobot_train.make_pre_post_processors = repadded_make_pre_post_processors


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
    # Only to measure what the padding is worth end to end; a run wants "longest".
    padding = take_flag(argv, "--language-padding") or "longest"
    if padding not in ("longest", "max_length"):
        raise SystemExit(f"--language-padding is longest or max_length, not {padding}")
    frozen_prefix = take_flag(argv, "--frozen-prefix") or "on"
    if frozen_prefix not in ("on", "off"):
        raise SystemExit(f"--frozen-prefix is on or off, not {frozen_prefix}")
    sys.argv = [sys.argv[0], *argv]

    from lerobot.scripts.lerobot_train import main as lerobot_main

    install_cached_prefix(
        Path(cache), int(seed) if seed is not None else 0, padding, frozen_prefix == "on"
    )
    lerobot_main()


if __name__ == "__main__":
    main()
