#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Time one SmolVLA training step, with and without the frozen tower in it.

Synthetic batches, so the dataloader, the host's core count and the container's
memory limit cannot confound the comparison. Every arm asked for runs in one
process against one GPU, which is the only way the numbers are comparable --
marketplace hosts of identical advertised specification differ by up to 1.68x.

  --arm stock   the step as it runs today: tower, VLM, expert, backward, AdamW
  --arm cached  the same step fed precomputed tower tokens
  --arm tower   the tower alone, which is what building a cache costs
  --arm lora    stock, with LoRA over the expert instead of dense training
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from pick_and_place.policies.smolvla_prefix_cache import (
    embedding_key,
    patch_policy_for_cached_prefix,
)

ARMS = ("stock", "cached", "tower", "lora")


def build_policy(checkpoint: Path, dataset_root: Path, device: torch.device, compile_model: bool):  # noqa: ANN201
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # Only the metadata is used -- feature names, shapes and stats -- but going
    # through the dataset is the same path training takes, so the policy is built
    # against exactly the features it would see.
    meta = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root).meta
    config = SmolVLAConfig(
        pretrained_path=str(checkpoint),
        device=device.type,
        n_action_steps=10,
        compile_model=compile_model,
    )
    return make_policy(cfg=config, ds_meta=meta), meta


def synthetic_batch(  # noqa: ANN001
    policy,
    meta,
    batch_size: int,
    device: torch.device,
    cached: bool,
    num_tokens: int | None = None,
    language_padding: str = "max_length",
) -> dict:
    """A batch shaped exactly like the trained one, with the content randomized.

    ``num_tokens`` is how many tokens the tower emits per camera, and it has to be
    measured before ``patch_policy_for_cached_prefix`` replaces ``embed_image``
    with the identity -- afterwards the tower cannot be asked any more.

    ``language_padding`` defaults to what a training run actually does rather than
    to ``config.pad_language_to``: the preprocessor comes from the checkpoint, and
    ``smolvla_base``'s pins ``max_length``, so a real batch carries 48 language
    tokens where the config asks for the 12 the task string needs.
    """
    config = policy.config
    camera_keys = [key for key in config.image_features]
    height, width = config.resize_imgs_with_padding
    state_dim = meta.features["observation.state"]["shape"][0]
    action_dim = meta.features["action"]["shape"][0]

    batch: dict[str, torch.Tensor] = {
        "observation.state": torch.randn(batch_size, state_dim, device=device),
        "action": torch.randn(batch_size, config.chunk_size, action_dim, device=device),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool, device=device),
    }
    if cached:
        # The tower's own output shape, taken from the model rather than assumed.
        tokens = policy.model.vlm_with_expert.config.text_config.hidden_size
        if num_tokens is None:
            num_tokens = _tower_token_count(policy, device)
        for key in camera_keys:
            batch[embedding_key(key)] = torch.randn(
                batch_size, num_tokens, tokens, device=device, dtype=torch.bfloat16
            )
    else:
        for key in camera_keys:
            batch[key] = torch.rand(batch_size, 3, height, width, device=device)

    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    encoded = tokenizer(
        ["Pick up the cube and place it at the target.\n"] * batch_size,
        padding=language_padding,
        padding_side="right",
        max_length=config.tokenizer_max_length,
        return_tensors="pt",
    )
    batch["observation.language.tokens"] = encoded["input_ids"].to(device)
    # Boolean, not the tokenizer's int64: make_att_2d_masks multiplies the
    # padding masks together and torch.where refuses a Long condition. lerobot's
    # own tokenizer step converts it, and a synthetic batch has to match.
    batch["observation.language.attention_mask"] = encoded["attention_mask"].to(device).bool()
    return batch


def _tower_token_count(policy, device: torch.device) -> int:  # noqa: ANN001
    """Ask the frozen tower how many tokens it emits for one image.

    Refuses an answer from a policy whose ``embed_image`` is already the cached
    path's identity: that returns the probe unchanged, and reading `shape[1]` off
    a [1, 3, H, W] image yields 3 rather than 64. A synthetic batch built from
    that runs, silently, on a twentieth of the image tokens training uses.
    """
    height, width = policy.config.resize_imgs_with_padding
    probe = torch.zeros(1, 3, height, width, device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        tokens = policy.model.vlm_with_expert.embed_image(probe)
    hidden_size = policy.model.vlm_with_expert.config.text_config.hidden_size
    if tokens.ndim != 3 or tokens.shape[2] != hidden_size:
        raise RuntimeError(
            f"the tower returned {tuple(tokens.shape)}, not [batch, tokens, {hidden_size}]. "
            "Measure the token count before patching the policy for the cached prefix."
        )
    return int(tokens.shape[1])


def freeze_state_projection(policy) -> int:  # noqa: ANN001
    """Stop `state_proj`, the one trainable thing in the prefix, from training.

    Nobody should train this way -- the state would stop reaching the action
    expert through a learned projection. It measures a bound: with nothing
    trainable in the prefix, autograd skips the whole 16-layer prefix backward,
    and the step that results is the floor a split prefix could reach.
    """
    frozen = 0
    for name, parameter in policy.named_parameters():
        if "state_proj" in name and parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    if frozen == 0:
        raise RuntimeError("no trainable state_proj to freeze")
    return frozen


def time_steps(step, warmup: int, repeats: int) -> list[float]:  # noqa: ANN001
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    return timings


def make_train_step(policy, batch: dict, learning_rate: float = 1e-4):  # noqa: ANN001
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(dict(batch))
        loss.backward()
        optimizer.step()

    return step, sum(p.numel() for p in trainable)


def make_tower_step(policy, batch: dict):  # noqa: ANN001
    from pick_and_place.policies.smolvla_prefix_cache import tower_tokens

    images = [batch[key] * 2.0 - 1.0 for key in policy.config.image_features]

    def step() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tower_tokens(policy, images)

    return step


def profile_components(policy, base, batch: dict, repeats: int) -> dict[str, float]:  # noqa: ANN001
    """Split a step into tower, rest-of-forward, backward and optimizer."""
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    tower = {"seconds": 0.0}

    original = base.model.vlm_with_expert.embed_image

    def timed_embed_image(image):  # noqa: ANN001, ANN202
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(image)
        torch.cuda.synchronize()
        tower["seconds"] += time.perf_counter() - started
        return result

    base.model.vlm_with_expert.embed_image = timed_embed_image
    totals = {"tower": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    try:
        for index in range(repeats + 1):
            tower["seconds"] = 0.0
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            mark = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(dict(batch))
            torch.cuda.synchronize()
            forward = time.perf_counter() - mark
            mark = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward = time.perf_counter() - mark
            mark = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter() - mark
            if index == 0:
                continue  # first pass carries autotuning and allocator growth
            totals["tower"] += tower["seconds"]
            totals["forward"] += forward - tower["seconds"]
            totals["backward"] += backward
            totals["optimizer"] += optimizer_seconds
    finally:
        base.model.vlm_with_expert.embed_image = original
    return {key: value / repeats for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="only its metadata is read")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["stock", "cached", "tower"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true", help="also split the step into stages")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--language-padding",
        choices=("max_length", "longest"),
        default="max_length",
        help="what the tokenizer pads to. The default is what the checkpoint's "
        "processor does; 'longest' is what the policy config asks for.",
    )
    parser.add_argument(
        "--frozen-prefix",
        action="store_true",
        help="take the frozen part of the prefix out of the backward, which is "
        "what a training run does now",
    )
    parser.add_argument(
        "--freeze-state-proj",
        action="store_true",
        help="freeze the prefix's one trainable projection, which removes the "
        "prefix backward entirely. A bound, not a trainable configuration.",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        help="tokens per camera in the cached arm, instead of asking the tower. "
        "Only for sweeping sequence length; the cache holds what the tower emits.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    results: dict[str, dict] = {
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "compile": args.compile,
        "language_padding": args.language_padding,
        "frozen_prefix": args.frozen_prefix,
        "state_proj_frozen": args.freeze_state_proj,
        "arms": {},
    }

    for arm in args.arms:
        policy, meta = build_policy(args.checkpoint, args.dataset, device, args.compile)
        if arm == "lora":
            # The policy's own default targets -- the expert's q/v projections and
            # the state and action projections -- so this is LoRA where the dense
            # run trains, not a different set of weights.
            policy = policy.wrap_with_peft(peft_cli_overrides={"r": args.lora_rank})
        base = policy.get_base_model() if arm == "lora" else policy
        cached = arm == "cached"
        # Before the patch, while the tower is still there to answer.
        num_tokens = args.prefix_tokens
        if cached and num_tokens is None:
            num_tokens = _tower_token_count(base, device)
        if cached:
            patch_policy_for_cached_prefix(policy)
        if args.frozen_prefix and arm != "tower":
            from pick_and_place.policies.smolvla_frozen_prefix import (
                patch_policy_for_frozen_prefix,
            )

            patch_policy_for_frozen_prefix(base)
        if args.freeze_state_proj:
            freeze_state_projection(policy)
        batch = synthetic_batch(
            base, meta, args.batch_size, device, cached, num_tokens, args.language_padding
        )
        policy.train()

        if arm == "tower":
            step = make_tower_step(base, batch)
            trainable = 0
        else:
            step, trainable = make_train_step(policy, batch)

        timings = time_steps(step, args.warmup, args.repeats)
        entry = {
            # What the batch really carries, rather than what was asked for:
            # sequence length drives the attention cost, so it is a measurement
            # input and belongs beside the seconds.
            "language_tokens": int(batch["observation.language.tokens"].shape[1]),
            "median_s": statistics.median(timings),
            "min_s": min(timings),
            "samples_per_s": args.batch_size / statistics.median(timings),
            "trainable_parameters": trainable,
            "peak_vram_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
        }
        if cached:
            # The measurement is a function of this, so it belongs in the result.
            entry["prefix_tokens_per_camera"] = num_tokens
        if arm == "tower":
            # Two cameras per sample, and a cache is built one image at a time.
            entry["images_per_s"] = (
                args.batch_size * len(list(base.config.image_features))
            ) / statistics.median(timings)
        if args.profile and arm in ("stock", "cached"):
            entry["stages_s"] = profile_components(policy, base, batch, max(args.repeats // 3, 5))
        results["arms"][arm] = entry
        print(f"{arm}: {json.dumps(entry, indent=2)}", flush=True)

        del policy, base, batch, step
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if "stock" in results["arms"] and "cached" in results["arms"]:
        speedup = results["arms"]["stock"]["median_s"] / results["arms"]["cached"]["median_s"]
        results["cached_speedup"] = speedup
        print(f"cached is {speedup:.3f}x the stock step")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
