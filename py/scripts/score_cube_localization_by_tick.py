#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Stratify a trained cube-localization head's error by tick-within-episode.

Tests the occlusion reading in docs/DPPO_CLOSED_LOOP_STALL_HANDOFF.md's
"Result: cube-localization head": the worst held-out acquisition frames come
from the first tick or two of episodes whose randomized start pose hides the
cube from both cameras at once. If that were the whole of the tail, dropping
the first few ticks of every episode should collapse the acquisition p95 back
toward the ~1 cm median.

This re-scores an already-trained checkpoint on its own held-out split — no
retraining — and reports, per split and for the acquisition phase:

- error by tick bucket (0, 1, 2, 3-4, 5+);
- error with the first k ticks of every episode excluded, k in 0,1,2,3,5;
- where the worst 5% of acquisition frames actually sit by tick, and how many
  distinct episodes they come from.

``--seed`` and ``--held-out-fraction`` must match the training run's, or the
split will differ and the numbers will not correspond to that checkpoint's
reported held-out error.

Example:

    python py/scripts/score_cube_localization_by_tick.py \\
      --train-npz output/blue-cube-no-dr-200-10hz-96x96/train.npz \\
      --episodes-root datasets/sim-200_episodes \\
      --source-episodes output/blue-cube-no-dr-200-10hz-96x96/source-episodes.txt \\
      --checkpoint outputs/cube_localization_head/best_model.pt \\
      --output outputs/cube_localization_head/by_tick.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pick_and_place.cube_localization import CubeLocalizationHead
from pick_and_place.cube_localization_dataset import episode_frame_split, load_cube_targets

# Matches the export contract; see diffusion_policy_dataset.export_diffusion_policy_dataset.
FRAME_STRIDE = 3
ACQUISITION = "acquisition"
TICK_BUCKETS = ("tick_0", "tick_1", "tick_2", "tick_3_4", "tick_5plus")
EXCLUDE_K = (0, 1, 2, 3, 5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--source-episodes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0, help="Must match the training run's seed.")
    parser.add_argument("--output-dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _summarize(errors_cm: np.ndarray) -> dict:
    if len(errors_cm) == 0:
        return {"count": 0}
    return {
        "median_cm": round(float(np.median(errors_cm)), 3),
        "p95_cm": round(float(np.percentile(errors_cm, 95)), 3),
        "max_cm": round(float(errors_cm.max()), 3),
        "count": int(len(errors_cm)),
    }


def _tick_masks(ticks: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "tick_0": ticks == 0,
        "tick_1": ticks == 1,
        "tick_2": ticks == 2,
        "tick_3_4": (ticks >= 3) & (ticks <= 4),
        "tick_5plus": ticks >= 5,
    }


@torch.no_grad()
def _predict(
    model: CubeLocalizationHead,
    images: np.ndarray,
    frame_indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    output_dim: int,
) -> np.ndarray:
    model.eval()
    predictions = np.empty((len(frame_indices), output_dim), dtype=np.float32)
    for start in range(0, len(frame_indices), batch_size):
        batch = frame_indices[start : start + batch_size]
        batch_images = torch.from_numpy(np.ascontiguousarray(images[batch])).to(device)
        predictions[start : start + len(batch)] = model(batch_images).cpu().numpy()
    return predictions


def _analyze(
    errors_cm: np.ndarray, phases: np.ndarray, ticks: np.ndarray, episodes: np.ndarray
) -> dict:
    """Acquisition-phase breakdown for one split."""
    acquisition = phases == ACQUISITION
    block: dict = {"acquisition_all": _summarize(errors_cm[acquisition])}

    block["acquisition_by_tick"] = {
        name: _summarize(errors_cm[acquisition & mask])
        for name, mask in _tick_masks(ticks).items()
    }
    block["acquisition_excluding_first_k_ticks"] = {
        f"k={k}": _summarize(errors_cm[acquisition & (ticks >= k)]) for k in EXCLUDE_K
    }

    acquisition_errors = errors_cm[acquisition]
    if len(acquisition_errors):
        threshold = float(np.percentile(acquisition_errors, 95))
        worst = acquisition & (errors_cm >= threshold)
        block["worst_5pct_acquisition"] = {
            "threshold_cm": round(threshold, 3),
            "count": int(worst.sum()),
            "frac_in_ticks_0_1": round(float((ticks[worst] <= 1).mean()), 3),
            "frac_in_ticks_0_2": round(float((ticks[worst] <= 2).mean()), 3),
            "distinct_episodes": int(len(np.unique(episodes[worst]))),
            "tick_histogram": {
                str(int(tick)): int((ticks[worst] == tick).sum())
                for tick in np.unique(ticks[worst])
            },
        }
    return block


def main() -> None:
    args = _parse_args()
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
        raise ValueError(f"{images.shape[0]} image frames but {positions.shape[0]} target frames")
    positions = positions[:, : args.output_dim]

    # Episode-relative tick index and owning episode for every global frame.
    ticks = np.concatenate([np.arange(length) for length in traj_lengths])
    episodes = np.concatenate(
        [np.full(length, index, dtype=np.int64) for index, length in enumerate(traj_lengths)]
    )

    train_frames, held_out_frames = episode_frame_split(
        traj_lengths, args.held_out_fraction, args.seed
    )

    model = CubeLocalizationHead(
        in_channels=images.shape[1], image_size=images.shape[2], output_dim=args.output_dim
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "held_out_fraction": args.held_out_fraction,
    }
    for split_name, frames in (("held_out", held_out_frames), ("train", train_frames)):
        predictions = _predict(
            model,
            images,
            frames,
            device=device,
            batch_size=args.batch_size,
            output_dim=args.output_dim,
        )
        errors_cm = np.linalg.norm(predictions - positions[frames], axis=1) * 100.0
        report[split_name] = _analyze(errors_cm, phases[frames], ticks[frames], episodes[frames])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
