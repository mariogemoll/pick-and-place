#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Find where a live SmolVLA training step spends the time a synthetic one does not.

`benchmark_smolvla_step.py` times the model on a batch that already sits on the
device, and a step through `lerobot-train` used to look 0.08 s more expensive
than that. It is not: a synthetic batch that carries what a real one carries --
the tower's real token count, and the 48 language tokens the checkpoint's
processor pads to -- costs the same, to 3%. This is the harness that showed it,
by running the real loop and changing one thing at a time:

  --arms synthetic  a device-resident synthetic batch, reused every step
  --arms frozen     one real preprocessed batch, reused every step
  --arms live       a fresh batch every step, exactly as training runs

`synthetic` against `frozen` isolates what the batch *contains*; `frozen` against
`live` isolates what producing a fresh one costs. Every arm runs in one process
against one GPU, because marketplace hosts differ by up to 1.68x.

The step itself is a faithful replica of `update_policy`: accelerate's autocast,
backward and gradient clipping, both `.item()` syncs, the `unwrap_model` probe.
Its `live` arm should reproduce lerobot's own `updt_s`, and that is the control
that says the replica is worth trusting.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from pick_and_place.policies.smolvla_prefix_cache import (
    CachedPrefixDataset,
    patch_policy_for_cached_prefix,
)

ARMS = ("synthetic", "frozen", "live")


def build_dataset(dataset_root: Path, policy_config, episodes: int | None):  # noqa: ANN001, ANN201
    """The dataset lerobot's `make_dataset` would have built, minus the config plumbing."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = LeRobotDatasetMetadata(dataset_root.name, root=dataset_root)
    return LeRobotDataset(
        dataset_root.name,
        root=dataset_root,
        episodes=list(range(episodes)) if episodes else None,
        delta_timestamps=resolve_delta_timestamps(policy_config, meta),
    )


def build_preprocessor(policy_config, checkpoint: Path, stats, device: str):  # noqa: ANN001, ANN201
    """lerobot's own preprocessor, built the way `train()` builds it from a checkpoint."""
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": stats,
                "features": {**policy_config.input_features, **policy_config.output_features},
                "norm_map": policy_config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    return preprocessor


def language_padding(preprocessor) -> str:  # noqa: ANN001
    """What the tokenizer step pads to, which is not what the policy config says."""
    from lerobot.processor.tokenizer_processor import TokenizerProcessorStep

    for step in preprocessor.steps:
        if isinstance(step, TokenizerProcessorStep):
            return step.padding
    raise ValueError("this preprocessor has no tokenizer step")


def make_update(accelerator, policy, optimizer, grad_clip_norm: float = 10.0):  # noqa: ANN001, ANN202
    """`lerobot.scripts.lerobot_train.update_policy`, line for line."""
    from lerobot.utils.utils import has_method

    def update(batch: dict[str, Any]) -> None:
        policy.train()
        with accelerator.autocast():
            loss, _ = policy.forward(batch)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
        float(loss.item())
        float(grad_norm.item())

    return update


def timed_loop(
    fetch: Callable[[], dict[str, Any]],
    update: Callable[[dict[str, Any]], None],
    *,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    """Time `fetch` and `update` the way lerobot times `data_s` and `updt_s`.

    No `cuda.synchronize()` anywhere: the point is the wall clock a real run sees,
    and a sync per step would hide exactly the stalls being looked for. Both
    `.item()` calls inside `update` already end the step on the device.
    """
    for _ in range(warmup):
        update(fetch())
    data, updates, cpu = [], [], []
    for _ in range(steps):
        mark = time.perf_counter()
        cpu_mark = time.process_time()
        batch = fetch()
        fetched = time.perf_counter()
        update(batch)
        finished = time.perf_counter()
        data.append(fetched - mark)
        updates.append(finished - fetched)
        cpu.append(time.process_time() - cpu_mark)
    return {
        "data_s": statistics.median(data),
        "update_s": statistics.median(updates),
        "step_s": statistics.median(a + b for a, b in zip(data, updates)),
        "cpu_s": statistics.median(cpu),
        "update_s_min": min(updates),
    }


def split_step(policy, accelerator, optimizer, batch: dict[str, Any], repeats: int) -> dict[str, float]:  # noqa: ANN001
    """Attribute one step to its parts, with a sync between each.

    The syncs perturb the total -- they stop the CPU running ahead -- so the split
    is read as shares of itself, not against the unsynced number above.
    """
    stages = {"h2d": 0.0, "forward": 0.0, "backward": 0.0, "clip": 0.0, "optimizer": 0.0, "item": 0.0}

    def mark() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    for index in range(repeats + 1):
        started = mark()  # everything queued before the step is done here
        policy.train()
        elapsed = {"h2d": mark() - started}
        started = time.perf_counter()
        with accelerator.autocast():
            loss, _ = policy.forward(batch)
        elapsed["forward"] = mark() - started
        started = time.perf_counter()
        accelerator.backward(loss)
        elapsed["backward"] = mark() - started
        started = time.perf_counter()
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), 10.0)
        elapsed["clip"] = mark() - started
        started = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        elapsed["optimizer"] = mark() - started
        started = time.perf_counter()
        float(loss.item())
        float(grad_norm.item())
        elapsed["item"] = time.perf_counter() - started
        if index == 0:
            continue  # the first pass carries allocator growth
        for key, value in elapsed.items():
            stages[key] += value
    return {key: value / repeats for key, value in stages.items()}


def device_busy(
    fetch: Callable[[], dict[str, Any]],
    update: Callable[[dict[str, Any]], None],
    steps: int,
    unprofiled_step_s: float,
) -> dict[str, float]:
    """What fraction of a step the GPU is actually running kernels.

    Device-side entries only: each `aten::` op carries the device time of the
    kernel underneath it as well, so summing every entry counts the same
    microseconds twice and lands at almost exactly 2.0. Compared against an
    unprofiled wall clock, because the profiler's own overhead inflates its own.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            update(fetch())
        torch.cuda.synchronize()
    device_s = (
        sum(
            event.self_device_time_total
            for event in prof.key_averages()
            if event.device_type == DeviceType.CUDA
        )
        / 1e6
    )
    return {
        "device_s_per_step": device_s / steps,
        "busy_fraction": device_s / steps / unprofiled_step_s,
    }


@contextlib.contextmanager
def maybe_cprofile(destination: Path | None) -> Iterator[None]:
    """Profile the main process's Python, which is where a launch-bound step suffers."""
    if destination is None:
        yield
        return
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        with destination.open("w") as handle:
            pstats.Stats(profiler, stream=handle).sort_stats("cumulative").print_stats(40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prefix-cache", type=Path, help="train the cached arm, from this cache")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--pad-longest",
        action="store_true",
        help="tokenize to the longest task string instead of the saved 48",
    )
    parser.add_argument(
        "--frozen-prefix",
        action="store_true",
        help="take the frozen part of the prefix out of the backward, which is "
        "what a training run does now",
    )
    parser.add_argument("--split", action="store_true", help="also attribute a step to its stages")
    parser.add_argument("--busy", action="store_true", help="also measure the GPU busy fraction")
    parser.add_argument("--cprofile", type=Path, help="write a Python profile of the live arm here")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from benchmark_smolvla_step import _tower_token_count, build_policy, synthetic_batch

    device = torch.device("cuda")
    cached = args.prefix_cache is not None

    # Exactly what `train()` sets before the loop, both of which change matmul
    # behaviour and so have to be set here too for the numbers to transfer.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy, meta = build_policy(args.checkpoint, args.dataset, device, compile_model=False)
    # Before the patch below takes the tower away, so the synthetic arm gets the
    # token count the cache actually holds.
    num_tokens = _tower_token_count(policy, device) if cached else None
    if cached:
        patch_policy_for_cached_prefix(policy)
    if args.frozen_prefix:
        from pick_and_place.policies.smolvla_frozen_prefix import patch_policy_for_frozen_prefix

        patch_policy_for_frozen_prefix(policy)

    dataset = build_dataset(args.dataset, policy.config, args.episodes)
    if cached:
        dataset = CachedPrefixDataset(dataset, args.prefix_cache)
    preprocessor = build_preprocessor(policy.config, args.checkpoint, dataset.meta.stats, device.type)
    if args.pad_longest:
        from pick_and_place.policies.smolvla_language_padding import pad_language_to

        pad_language_to(preprocessor)
    # Whatever the preprocessor really does, so the synthetic arm is the same
    # sequence as the live one rather than the one the config asks for.
    padding = language_padding(preprocessor)

    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
    update = make_update(accelerator, policy, optimizer)

    from lerobot.datasets.utils import cycle

    iterator = cycle(loader)

    def fetch_live() -> dict[str, Any]:
        return preprocessor(next(iterator))

    results: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "cached": cached,
        "frozen_prefix": args.frozen_prefix,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "language_padding": padding,
        "mixed_precision": os.environ.get("ACCELERATE_MIXED_PRECISION", "no"),
        "arms": {},
    }

    for arm in args.arms:
        if arm == "synthetic":
            base = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            batch = synthetic_batch(
                base, meta, args.batch_size, device, cached, num_tokens, padding
            )
            fetch = lambda batch=batch: dict(batch)  # noqa: E731 - the loop wants a callable
        elif arm == "frozen":
            batch = fetch_live()
            fetch = lambda batch=batch: dict(batch)  # noqa: E731
        else:
            fetch = fetch_live

        # Per arm, not per process: whether a batch size fits is the other half of
        # a batch-size sweep, and the arms differ in what they hold resident.
        torch.cuda.reset_peak_memory_stats()
        with maybe_cprofile(args.cprofile if arm == "live" else None):
            entry = timed_loop(fetch, update, warmup=args.warmup, steps=args.steps)
        entry["samples_per_s"] = args.batch_size / entry["step_s"]
        entry["peak_vram_mib"] = torch.cuda.max_memory_allocated() // (1024 * 1024)
        if args.split:
            entry["stages_s"] = split_step(policy, accelerator, optimizer, fetch(), max(args.steps // 4, 5))
        if args.busy:
            entry.update(device_busy(fetch, update, max(args.steps // 6, 5), entry["step_s"]))
        results["arms"][arm] = entry
        print(f"{arm}: {json.dumps(entry, indent=2, sort_keys=True)}", flush=True)

    if "synthetic" in results["arms"] and "live" in results["arms"]:
        residual = results["arms"]["live"]["step_s"] - results["arms"]["synthetic"]["step_s"]
        results["live_minus_synthetic_s"] = residual
        print(f"live costs {residual:.4f}s more per step than synthetic")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
