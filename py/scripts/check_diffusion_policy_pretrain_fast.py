#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Prove the fast pre-training path is arithmetically the stock one.

``pick_and_place.diffusion_policy_pretrain`` replaces DPPO's per-sample
``DataLoader`` with an
on-device gather and folds the two camera encoders into one batched call. Both
are meant to be reformulations, not approximations, so they are checkable
exactly rather than by eyeballing a loss curve:

* **Batches.** Every sample the fast sampler emits is compared element for
  element against ``StitchedSequenceDataset.__getitem__``, including samples at
  episode starts where the observation history is clamped.
* **Vision encoder.** The batched two-camera forward is compared against the
  stock one at identical weights, in fp32 with augmentation off (augmentation
  draws random shifts, so it is not pointwise reproducible; its per-row
  independence is checked separately).

Runs on CPU against any exported Diffusion Policy artifact — no GPU needed.

    python py/scripts/check_diffusion_policy_pretrain_fast.py \
        --dataset output/blue-cube-no-dr-200-10hz-96x96/train.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

from pick_and_place.diffusion_policy_pretrain import (
    GpuBatchSampler,
    build_gather_indices,
    install_batched_vision_encoder,
)

# The vendored DPPO packages (``agent``, ``model``) are imported lazily inside the
# checks below, so the path only has to be in place before those run.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "dppo"))


def check_batches(dataset_path: Path, max_episodes: int, batch_size: int) -> None:
    from agent.dataset.sequence import StitchedSequenceDataset

    dataset = StitchedSequenceDataset(
        dataset_path=str(dataset_path),
        horizon_steps=16,
        cond_steps=2,
        img_cond_steps=2,
        max_n_episodes=max_episodes,
        use_img=True,
        device="cpu",
    )
    sampler = GpuBatchSampler(dataset, device="cpu", batch_size=batch_size)
    print(f"  dataset: {sampler.num_samples} samples, {sampler.num_batches} batches")

    # Episode starts are where the history clamping happens, so make sure they
    # are in the sample rather than trusting a random draw to include them.
    boundary = [i for i, (_, before) in enumerate(dataset.indices) if before < 2]
    interior = torch.randperm(sampler.num_samples)[:512].tolist()
    sample_ids = sorted(set(boundary[:256] + interior))
    print(f"  comparing {len(sample_ids)} samples ({len(boundary[:256])} at episode starts)")

    ids = torch.tensor(sample_ids, dtype=torch.long)
    actions, conditions = sampler.batch(ids)

    for row, sample_id in enumerate(sample_ids):
        reference = dataset[sample_id]
        assert torch.equal(actions[row], reference.actions), f"actions differ at {sample_id}"
        assert torch.equal(
            conditions["state"][row], reference.conditions["state"]
        ), f"state differs at {sample_id}"
        assert torch.equal(
            conditions["rgb"][row], reference.conditions["rgb"]
        ), f"rgb differs at {sample_id}"

    assert actions.dtype == dataset.actions.dtype
    assert conditions["rgb"].dtype == dataset.images.dtype, "images must stay uint8 until the GPU"
    print("  OK: every sampled batch row is element-identical to __getitem__")

    # A full epoch must visit every sample exactly once (modulo the dropped tail).
    generator = torch.Generator(device="cpu").manual_seed(0)
    perm = sampler.epoch_permutation(generator)
    assert perm.numel() == sampler.num_samples
    assert torch.equal(perm.sort().values, torch.arange(sampler.num_samples))
    covered = sampler.num_batches * batch_size
    print(
        f"  OK: epoch permutation is a bijection; drop_last discards "
        f"{sampler.num_samples - covered} of {sampler.num_samples} samples per epoch"
    )


def check_gather_indices() -> None:
    """Recompute the index arithmetic against a literal transcription."""
    indices = [(0, 0), (1, 1), (2, 2), (3, 3), (10, 0), (11, 1)]
    gathered = build_gather_indices(indices, 3, 2, 4, "cpu")
    for row, (start, before) in enumerate(indices):
        window_base = start - before
        window = list(range(window_base, start + 1))
        expected_state = [window[max(before - t, 0)] for t in reversed(range(3))]
        expected_img = [window[max(before - t, 0)] for t in reversed(range(2))]
        assert gathered["state"][row].tolist() == expected_state, (start, before)
        assert gathered["img"][row].tolist() == expected_img, (start, before)
        assert gathered["action"][row].tolist() == list(range(start, start + 4))
    print("  OK: gather indices match the upstream slice-and-clamp expression")


def build_network(augment: bool) -> torch.nn.Module:
    from model.common.vit import VitEncoder, VitEncoderConfig
    from model.diffusion.unet import VisionUnet1D

    backbone = VitEncoder(
        obs_shape=[6, 96, 96],
        cfg=VitEncoderConfig(
            patch_size=8, depth=1, embed_dim=128, num_heads=4, embed_style="embed2", embed_norm=0
        ),
        num_channel=6,
        img_h=96,
        img_w=96,
    )
    return VisionUnet1D(
        backbone=backbone,
        action_dim=6,
        img_cond_steps=2,
        cond_dim=12,
        diffusion_step_embed_dim=32,
        dim=64,
        dim_mults=[1, 2, 4],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        groupnorm_eps=1e-4,
        spatial_emb=128,
        num_img=2,
        augment=augment,
    )


def check_vision_forward(batch_size: int) -> None:
    torch.manual_seed(0)
    network = build_network(augment=False).eval()
    x = torch.randn(batch_size, 16, 6)
    time = torch.randint(0, 100, (batch_size,))
    cond = {
        "state": torch.randn(batch_size, 2, 6),
        "rgb": torch.randint(0, 256, (batch_size, 2, 6, 96, 96), dtype=torch.uint8),
    }

    with torch.no_grad():
        expected = network(x, time, cond=cond)
        assert install_batched_vision_encoder(network), "patch did not apply"
        actual = network(x, time, cond=cond)

    difference = (expected - actual).abs().max().item()
    scale = expected.abs().max().item()
    print(f"  batched-vision forward: max abs diff {difference:.3e} (output scale {scale:.3e})")
    # Not bit-identical: batching changes matmul reduction order. It must stay
    # at fp32 rounding noise, orders of magnitude below the signal.
    assert difference < 1e-4 * max(scale, 1.0), "batched vision forward diverged"
    print("  OK: batched two-camera encoder matches the stock forward")


def check_augmentation_independence() -> None:
    """Folding cameras into the batch axis must not share a random shift."""
    from model.common.modules import RandomShiftsAug

    torch.manual_seed(0)
    aug = RandomShiftsAug(pad=4)
    # Identical rows: any output difference can only come from distinct shifts.
    rows = torch.arange(96 * 96, dtype=torch.float32).reshape(1, 1, 96, 96)
    stacked = rows.expand(64, 6, 96, 96).contiguous()
    shifted = aug(stacked)
    distinct = len({shifted[i].sum().item() for i in range(64)})
    print(f"  augmentation produced {distinct} distinct shifts across 64 folded rows")
    assert distinct > 4, "RandomShiftsAug appears to share one shift across the batch"
    print("  OK: per-row shifts stay independent when cameras are folded into the batch")


def check_checkpoint_is_server_loadable(tmp_dir: Path) -> None:
    """A checkpoint must open under ``torch.load(weights_only=True)``.

    ``scripts/diffusion_policy_server.py`` loads that way, so anything in the file
    whose pickle needs a global the safe unpickler rejects — numpy's RNG state
    is the easy mistake — produces a checkpoint that trains fine and then
    cannot be served. That failure surfaces only at evaluation time, hours
    after the run that caused it, so it is checked here instead.
    """
    import numpy as np

    path = tmp_dir / "state_probe.pt"
    payload = {
        "epoch": 1,
        "model": {"w": torch.zeros(2)},
        "ema": {"w": torch.zeros(2)},
        "rng": torch.get_rng_state(),
        "rng_generator": torch.Generator().get_state(),
    }
    torch.save(payload, path)
    torch.load(path, map_location="cpu", weights_only=True)
    print("  OK: a checkpoint of this shape loads with weights_only=True")

    # And demonstrate the failure mode the shape is chosen to avoid, so the
    # check is known to have teeth rather than passing vacuously.
    payload["rng"] = {"numpy": np.random.get_state()}
    torch.save(payload, path)
    try:
        torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError):
        print("  OK: numpy RNG state in the checkpoint is correctly rejected")
    else:
        raise AssertionError("weights_only=True accepted numpy RNG state; check is vacuous")
    path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, help="Path to an exported train.npz; skipped when omitted"
    )
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print("Gather-index arithmetic:")
    check_gather_indices()
    print("Vision encoder:")
    check_vision_forward(args.batch_size)
    print("Augmentation:")
    check_augmentation_independence()
    print("Checkpoint format:")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check_checkpoint_is_server_loadable(Path(tmp))
    if args.dataset is not None:
        print(f"Batch equivalence against {args.dataset}:")
        check_batches(args.dataset, args.max_episodes, args.batch_size)
    else:
        print("Batch equivalence: skipped (pass --dataset to run it)")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
