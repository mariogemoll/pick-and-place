#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Check that taking the frozen prefix out of the graph changes nothing but the time.

A forward check is not enough here: the whole point of the change is what happens
in the backward, so this compares **gradients**, parameter by parameter, between
lerobot's own layer loop and the split one, on the same batch with the
flow-matching noise and time held fixed. Then it times both arms in the same
process, which is the only way the seconds are comparable.

"Unchanged" cannot mean "bit-identical" here, because the split reduces over
shorter sequences and the model's weights are bfloat16. So the run carries its
own scale for that: a **control** arm that pads the language to 48 tokens
instead of 12, which is a change the attention masks make mathematically
invisible and which therefore moves the gradients by exactly as much as rounding
can. A split that disagrees no more than the control does is as faithful as this
arithmetic allows.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from pick_and_place.policies.smolvla_frozen_prefix import patch_policy_for_frozen_prefix
from pick_and_place.policies.smolvla_prefix_cache import patch_policy_for_cached_prefix


def gradients(policy, batch: dict, noise: torch.Tensor, time: torch.Tensor) -> tuple:  # noqa: ANN001
    """One forward and backward; the loss and a copy of every trainable gradient."""
    policy.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(dict(batch), noise=noise, time=time)
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach().clone(), grads


def compare(reference_grads: dict, other: dict) -> dict:
    """How far apart two sets of gradients are, worst first."""
    differences = []
    for name, reference in reference_grads.items():
        if name not in other:
            continue
        scale = reference.norm().item()
        difference = (other[name].float() - reference.float()).norm().item()
        differences.append((difference / scale if scale else difference, name))
    differences.sort(reverse=True)
    return {
        "parameters_compared": len(differences),
        "parameters_only_in_one_arm": sorted(set(reference_grads) ^ set(other)),
        "worst_relative_difference": differences[0][0] if differences else 0.0,
        "worst": [{"parameter": name, "relative": value} for value, name in differences[:5]],
    }


def time_arm(policy, batch: dict, warmup: int, repeats: int) -> float:  # noqa: ANN001
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(dict(batch))
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        step()
        finished.record()
        torch.cuda.synchronize()
        timings.append(started.elapsed_time(finished) / 1000.0)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="only its metadata is read")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cached", action="store_true", help="serve the tower from a cache too")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from benchmark_smolvla_step import _tower_token_count, build_policy, synthetic_batch

    device = torch.device("cuda")
    policy, meta = build_policy(args.checkpoint, args.dataset, device, compile_model=False)
    num_tokens = _tower_token_count(policy, device) if args.cached else None
    if args.cached:
        patch_policy_for_cached_prefix(policy)
    batch = synthetic_batch(
        policy, meta, args.batch_size, device, args.cached, num_tokens, "longest"
    )
    # The same batch with 36 tokens of padding the masks ignore: a change that
    # cannot alter the result, and does anyway, by however much rounding can.
    padded_batch = synthetic_batch(
        policy, meta, args.batch_size, device, args.cached, num_tokens, "max_length"
    )
    policy.train()

    generator = torch.Generator(device="cpu").manual_seed(0)
    shape = (args.batch_size, policy.config.chunk_size, policy.config.max_action_dim)
    noise = torch.randn(shape, generator=generator).to(device)
    time_values = (torch.rand(args.batch_size, generator=generator) * 0.999 + 0.001).to(device)

    stock_loss, stock_grads = gradients(policy, batch, noise, time_values)
    control_loss, control_grads = gradients(policy, padded_batch, noise, time_values)
    restore = patch_policy_for_frozen_prefix(policy)
    split_loss, split_grads = gradients(policy, batch, noise, time_values)

    result = {
        "batch_size": args.batch_size,
        "cached": args.cached,
        "gpu": torch.cuda.get_device_name(0),
        "stock_loss": stock_loss.item(),
        "split_loss": split_loss.item(),
        "control_loss": control_loss.item(),
        "loss_difference": abs(split_loss.item() - stock_loss.item()),
        "control_loss_difference": abs(control_loss.item() - stock_loss.item()),
        "gradients": compare(stock_grads, split_grads),
        "control_gradients": compare(stock_grads, control_grads),
    }
    result["split_s"] = time_arm(policy, batch, args.warmup, args.repeats)
    restore()
    result["stock_s"] = time_arm(policy, batch, args.warmup, args.repeats)
    result["speedup"] = result["stock_s"] / result["split_s"]

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
