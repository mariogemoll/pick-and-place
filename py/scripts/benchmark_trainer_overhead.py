#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Time what lerobot's training loop adds on top of forward, backward and AdamW.

`update_policy` does more than the three things a step needs. Most of it is
constant per step, so it is invisible next to a 0.35 s stock step and more than
half of a 0.07 s cached one. This adds lerobot's extras to a bare step one at a
time, on the same batch and the same host.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from pick_and_place.policies.smolvla_prefix_cache import patch_policy_for_cached_prefix

# Kept in the same order lerobot applies them, so each row is the cost of adding
# that line to the row above rather than of some arbitrary combination.
LAYERS = ("bare", "train_mode", "clip_grad_norm", "item_sync", "unwrap_model", "accelerate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--cached", action="store_true", help="measure against a cached-prefix step")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Imported here so the module stays importable without the benchmark's deps.
    from benchmark_smolvla_step import build_policy, synthetic_batch

    device = torch.device("cuda")
    policy, meta = build_policy(args.checkpoint, args.dataset, device, compile_model=False)
    if args.cached:
        patch_policy_for_cached_prefix(policy)
    batch = synthetic_batch(policy, meta, args.batch_size, device, args.cached)
    policy.train()

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    def make_step(layer: str):  # noqa: ANN202
        index = LAYERS.index(layer)

        def step() -> None:
            if index >= 1:
                # lerobot calls this every step. SmolVLA overrides train() to put
                # the VLM back in eval, so it walks the module tree each time.
                policy.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(dict(batch))
            loss.backward()
            if index >= 2:
                # Over *all* parameters, not just the trainable ones.
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            if index >= 3:
                # Two device syncs per step, for the metrics tracker.
                float(loss.item())
                if index >= 2:
                    float(grad_norm.item())
            if index >= 4:
                hasattr(policy, "update")

        return step

    def make_accelerate_step():  # noqa: ANN202
        """lerobot's `update_policy` as it actually runs: through an Accelerator.

        `accelerator.prepare` rewraps `forward` for mixed precision and swaps in
        its own `backward` and `clip_grad_norm_`, so the same lines can cost
        something different from the bare ones measured above.
        """
        from accelerate import Accelerator

        accelerator = Accelerator()
        prepared, prepared_optimizer = accelerator.prepare(policy, optimizer)

        def step() -> None:
            prepared.train()
            with accelerator.autocast():
                loss, _ = prepared.forward(dict(batch))
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(prepared.parameters(), 10.0)
            prepared_optimizer.step()
            prepared_optimizer.zero_grad()
            accelerator.unwrap_model(prepared, keep_fp32_wrapper=True)
            float(loss.item())
            float(grad_norm.item())

        return step

    results = {"gpu": torch.cuda.get_device_name(0), "cached": args.cached, "layers": {}}
    previous = None
    for layer in LAYERS:
        step = make_accelerate_step() if layer == "accelerate" else make_step(layer)
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) / 1000.0)
        median = statistics.median(timings)
        entry = {"median_s": median}
        if previous is not None:
            entry["added_s"] = median - previous
        results["layers"][layer] = entry
        previous = median
        added = f"  (+{entry['added_s']:.4f})" if "added_s" in entry else ""
        print(f"{layer:16s} {median:.4f}s{added}", flush=True)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
