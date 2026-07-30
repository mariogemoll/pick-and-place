#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train and evaluate a supervised cube-localization head.

The "Next step" of docs/DPPO_CLOSED_LOOP_STALL_HANDOFF.md: can the cube be
localized to grasp precision from a 96x96 image at all, independent of the
diffusion policy that failed to grasp it? This trains a small ViT (matching
the policy's visual encoder architecture) plus a two-layer MLP head to
regress the cube's true position directly from the same image tensor the
policy consumes, with no diffusion, no action horizon, and no history.

Held-out evaluation is split by whole episode, not frame, since neighboring
frames within an episode are nearly identical. The decision thresholds from
the handoff doc, against a 3 cm cube: a held-out acquisition-phase median
error above ~2 cm means the auxiliary cube-localization loss proposed
elsewhere is not worth a retrain (the answer is input resolution or a
workspace crop); at or below ~1 cm, the pixels are sufficient and the
diffusion policy's failure to use them becomes the target.

Example:

    python py/scripts/train_cube_localization_head.py \\
      --train-npz output/dp_blue_cube_no_dr/artifact/train.npz \\
      --episodes-root datasets/sim-200_episodes \\
      --source-episodes output/dp_blue_cube_no_dr/artifact/source-episodes.txt \\
      --output outputs/cube_localization_head
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pick_and_place.cube_localization import CubeLocalizationHead
from pick_and_place.cube_localization_dataset import episode_frame_split, load_cube_targets
from pick_and_place.task_phases import PHASES

# The training contract this export was built at; see
# diffusion_policy_dataset.export_diffusion_policy_dataset.
FRAME_STRIDE = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument(
        "--source-episodes",
        type=Path,
        required=True,
        help="One episode id per line, in the export's frame order "
        "(export.json's source_episode_manifest, alongside --train-npz).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dim",
        type=int,
        choices=(2, 3),
        default=3,
        help="Regress (x, y) or (x, y, z); default 3.",
    )
    return parser.parse_args()


def _batches(frame_indices: np.ndarray, batch_size: int, *, shuffle: bool, seed: int | None = None):
    order = np.random.default_rng(seed).permutation(frame_indices) if shuffle else frame_indices
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _summarize(errors_cm: np.ndarray) -> dict:
    return {
        "median_cm": float(np.median(errors_cm)),
        "p95_cm": float(np.percentile(errors_cm, 95)),
        "count": len(errors_cm),
    }


@torch.no_grad()
def _evaluate(
    model: CubeLocalizationHead,
    images: np.ndarray,
    positions: np.ndarray,
    phases: np.ndarray,
    frame_indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    predictions = np.empty((len(frame_indices), positions.shape[1]), dtype=np.float32)
    offset = 0
    for batch in _batches(frame_indices, batch_size, shuffle=False):
        batch_images = torch.from_numpy(np.ascontiguousarray(images[batch])).to(device)
        predictions[offset : offset + len(batch)] = model(batch_images).cpu().numpy()
        offset += len(batch)

    targets = positions[frame_indices]
    errors_cm = np.linalg.norm(predictions - targets, axis=1) * 100.0
    frame_phases = phases[frame_indices]
    result = {"overall": _summarize(errors_cm)}
    for phase in PHASES:
        mask = frame_phases == phase
        if mask.any():
            result[phase] = _summarize(errors_cm[mask])
    return result


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device(args.device)

    episode_ids = args.source_episodes.read_text().split()
    data = np.load(args.train_npz, mmap_mode="r")
    images = data["images"]
    npz_traj_lengths = np.asarray(data["traj_lengths"])
    if len(episode_ids) != len(npz_traj_lengths):
        raise ValueError(
            f"{len(episode_ids)} source episodes but {len(npz_traj_lengths)} traj_lengths"
        )

    positions, phases, traj_lengths = load_cube_targets(
        args.episodes_root, episode_ids, frame_stride=FRAME_STRIDE
    )
    if not np.array_equal(traj_lengths, npz_traj_lengths):
        raise ValueError(
            "rebuilt per-episode frame counts do not match train.npz's traj_lengths; "
            "--source-episodes order must match the export's episode_indices"
        )
    if images.shape[0] != positions.shape[0]:
        raise ValueError(
            f"{images.shape[0]} image frames but {positions.shape[0]} target frames"
        )
    positions = positions[:, : args.output_dim]

    train_frames, held_out_frames = episode_frame_split(
        traj_lengths, args.held_out_fraction, args.seed
    )
    print(
        f"{len(traj_lengths)} episodes, {len(train_frames)} train frames, "
        f"{len(held_out_frames)} held-out frames"
    )

    model = CubeLocalizationHead(
        in_channels=images.shape[1],
        image_size=images.shape[2],
        output_dim=args.output_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_held_out_median = float("inf")
    best_epoch = None
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for batch in _batches(train_frames, args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch_images = torch.from_numpy(np.ascontiguousarray(images[batch])).to(device)
            batch_targets = torch.from_numpy(positions[batch]).to(device)
            optimizer.zero_grad()
            predictions = model(batch_images)
            loss = torch.nn.functional.mse_loss(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch)
        train_loss = running_loss / len(train_frames)

        held_out_metrics = _evaluate(
            model, images, positions, phases, held_out_frames, device=device,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - epoch_start
        held_out_overall = held_out_metrics["overall"]
        acquisition = held_out_metrics.get("acquisition", held_out_overall)
        print(
            f"epoch {epoch}/{args.epochs} mse={train_loss:.6f} "
            f"held-out median={held_out_overall['median_cm']:.2f}cm "
            f"p95={held_out_overall['p95_cm']:.2f}cm | "
            f"acquisition median={acquisition['median_cm']:.2f}cm "
            f"p95={acquisition['p95_cm']:.2f}cm ({elapsed:.1f}s)",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "held_out": held_out_metrics,
                "elapsed_s": elapsed,
            }
        )
        with (args.output / "history.json").open("w") as file:
            json.dump(history, file, indent=2)

        if held_out_overall["median_cm"] < best_held_out_median:
            best_held_out_median = held_out_overall["median_cm"]
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}

    assert best_state is not None and best_epoch is not None
    torch.save(best_state, args.output / "best_model.pt")
    model.load_state_dict(best_state)
    train_metrics_at_best = _evaluate(
        model, images, positions, phases, train_frames, device=device, batch_size=args.batch_size
    )
    held_out_metrics_at_best = _evaluate(
        model, images, positions, phases, held_out_frames, device=device, batch_size=args.batch_size
    )
    report = {
        "output_dim": args.output_dim,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "num_episodes": len(traj_lengths),
        "num_held_out_episodes": max(1, round(len(traj_lengths) * args.held_out_fraction)),
        "train_frames": len(train_frames),
        "held_out_frames": len(held_out_frames),
        "train_at_best": train_metrics_at_best,
        "held_out_at_best": held_out_metrics_at_best,
    }
    with (args.output / "report.json").open("w") as file:
        json.dump(report, file, indent=2)

    acquisition = held_out_metrics_at_best.get("acquisition", held_out_metrics_at_best["overall"])
    print()
    print(f"Best epoch: {best_epoch}")
    print(
        "Held-out acquisition-phase error: "
        f"median {acquisition['median_cm']:.2f} cm, p95 {acquisition['p95_cm']:.2f} cm "
        f"(n={acquisition['count']})"
    )
    print(
        "Held-out overall error: "
        f"median {held_out_metrics_at_best['overall']['median_cm']:.2f} cm, "
        f"p95 {held_out_metrics_at_best['overall']['p95_cm']:.2f} cm"
    )
    if acquisition["median_cm"] > 2.0:
        print("=> median stays above 2 cm: the pixels do not support grasp-precision localization.")
    elif acquisition["median_cm"] <= 1.0:
        print("=> median at or below 1 cm: the pixels are sufficient; the policy is the bottleneck.")
    else:
        print("=> median falls between the 1 cm and 2 cm thresholds; check p95 before concluding.")


if __name__ == "__main__":
    main()
