#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Show the worst and most typical held-out frames for a trained cube-
localization head, both cameras, at the 96x96 tensor the model actually sees.

A held-out median/p95 error (docs/DPPO_CLOSED_LOOP_STALL_HANDOFF.md's "Result:
cube-localization head") says how much the tail costs but not what causes it.
This reproduces the same held-out split a training run used
(pick_and_place.cube_localization_dataset.episode_frame_split, seeded), ranks
its frames by error within one task phase, and writes the overhead and wrist
crops plus a static HTML gallery so the worst frames can be inspected by eye.

Example:

    python py/scripts/inspect_cube_localization_errors.py \\
      --train-npz output/dp_blue_cube_no_dr/artifact/train.npz \\
      --episodes-root datasets/sim-200_episodes \\
      --source-episodes output/dp_blue_cube_no_dr/artifact/source-episodes.txt \\
      --checkpoint outputs/cube_localization_head/best_model.pt \\
      --output outputs/cube_localization_head/tail_frames
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pick_and_place.cube_localization import CubeLocalizationHead
from pick_and_place.cube_localization_dataset import episode_frame_split, load_cube_targets

FRAME_STRIDE = 3
THUMBNAIL_SIZE = 288


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--source-episodes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--held-out-fraction",
        type=float,
        default=0.15,
        help="Must match the value the checkpoint was trained with.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Must match the training run's seed.")
    parser.add_argument("--output-dim", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--phase",
        default="acquisition",
        help="Task phase to rank within (see pick_and_place.task_phases.PHASES), "
        "or 'all' to rank across every held-out frame.",
    )
    parser.add_argument("--worst-count", type=int, default=24)
    parser.add_argument("--typical-count", type=int, default=8)
    return parser.parse_args()


def _locate(frame: int, starts: np.ndarray, ends: np.ndarray, episode_ids: list[str]) -> tuple[str, int]:
    episode = int(np.searchsorted(ends, frame, side="right"))
    return episode_ids[episode], int(frame - starts[episode])


def _save_group(
    name: str,
    order: np.ndarray,
    count: int,
    *,
    output: Path,
    images: np.ndarray,
    frames: np.ndarray,
    errors_cm: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    episode_ids: list[str],
) -> list[dict]:
    records = []
    for rank, i in enumerate(order[:count]):
        frame = int(frames[i])
        episode_id, frame_in_episode = _locate(frame, starts, ends, episode_ids)
        image = np.asarray(images[frame])  # (channels, 96, 96) uint8
        overhead = image[0:3].transpose(1, 2, 0)
        wrist = image[3:6].transpose(1, 2, 0)
        Image.fromarray(overhead).resize(
            (THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.NEAREST
        ).save(output / f"{name}_{rank:02d}_overhead.png")
        Image.fromarray(wrist).resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.NEAREST).save(
            output / f"{name}_{rank:02d}_wrist.png"
        )
        records.append(
            {
                "group": name,
                "rank": rank,
                "episode": episode_id,
                "frame_in_episode": frame_in_episode,
                "error_cm": float(errors_cm[i]),
                "true_xyz": [round(float(v), 4) for v in targets[i]],
                "pred_xyz": [round(float(v), 4) for v in predictions[i]],
            }
        )
    return records


def _card(record: dict) -> str:
    name = f"{record['group']}_{record['rank']:02d}"
    true_xyz = ", ".join(f"{v:.3f}" for v in record["true_xyz"])
    pred_xyz = ", ".join(f"{v:.3f}" for v in record["pred_xyz"])
    return f"""
    <div class="card">
      <div class="meta">
        <strong>{html.escape(record['episode'])}</strong> frame {record['frame_in_episode']}
        &mdash; error <strong>{record['error_cm']:.1f} cm</strong>
      </div>
      <div class="images">
        <figure><img src="{name}_overhead.png"><figcaption>overhead</figcaption></figure>
        <figure><img src="{name}_wrist.png"><figcaption>wrist</figcaption></figure>
      </div>
      <div class="coords">
        true&nbsp;(x,y,z): {true_xyz}<br>
        pred&nbsp;(x,y,z): {pred_xyz}
      </div>
    </div>
    """


def _write_gallery(output: Path, records: list[dict], phase: str) -> None:
    worst_cards = "".join(_card(r) for r in records if r["group"] == "worst")
    typical_cards = "".join(_card(r) for r in records if r["group"] == "typical")
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>cube-localization tail frames</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 2rem; }}
  h1 {{ font-size: 1.3rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2.5rem; border-bottom: 1px solid #444; padding-bottom: 0.3rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  .card {{ background: #1c1c1c; border: 1px solid #333; border-radius: 8px; padding: 0.6rem; width: 300px; }}
  .meta {{ font-size: 0.85rem; margin-bottom: 0.4rem; }}
  .images {{ display: flex; gap: 0.4rem; }}
  figure {{ margin: 0; text-align: center; }}
  figure img {{ width: 138px; height: 138px; image-rendering: pixelated; border-radius: 4px; }}
  figcaption {{ font-size: 0.7rem; color: #999; }}
  .coords {{ font-size: 0.72rem; color: #aaa; margin-top: 0.4rem; font-family: monospace; }}
</style>
</head>
<body>
<h1>Cube-localization head: held-out "{html.escape(phase)}" frames</h1>
<h2>Worst by error (the p95 tail)</h2>
<div class="grid">{worst_cards}</div>
<h2>Typical frames (closest to the median error)</h2>
<div class="grid">{typical_cards}</div>
</body>
</html>
"""
    (output / "index.html").write_text(page)


def main() -> None:
    args = _parse_args()

    episode_ids = args.source_episodes.read_text().split()
    data = np.load(args.train_npz, mmap_mode="r")
    images = data["images"]
    npz_traj_lengths = np.asarray(data["traj_lengths"])

    positions, phases, traj_lengths = load_cube_targets(
        args.episodes_root, episode_ids, frame_stride=FRAME_STRIDE
    )
    if not np.array_equal(traj_lengths, npz_traj_lengths):
        raise ValueError(
            "rebuilt per-episode frame counts do not match train.npz's traj_lengths"
        )
    positions = positions[:, : args.output_dim]

    _, held_out_frames = episode_frame_split(traj_lengths, args.held_out_fraction, args.seed)

    model = CubeLocalizationHead(
        in_channels=images.shape[1], image_size=images.shape[2], output_dim=args.output_dim
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    predictions = np.empty((len(held_out_frames), args.output_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(held_out_frames), 256):
            batch = held_out_frames[start : start + 256]
            batch_images = torch.from_numpy(np.ascontiguousarray(images[batch]))
            predictions[start : start + len(batch)] = model(batch_images).numpy()

    targets = positions[held_out_frames]
    errors_cm = np.linalg.norm(predictions - targets, axis=1) * 100.0
    frame_phases = phases[held_out_frames]

    if args.phase != "all":
        mask = frame_phases == args.phase
        if not mask.any():
            raise ValueError(f"no held-out frames have phase {args.phase!r}")
        frames = held_out_frames[mask]
        errors_cm = errors_cm[mask]
        predictions = predictions[mask]
        targets = targets[mask]
    else:
        frames = held_out_frames

    print(
        f"{len(frames)} held-out frames (phase={args.phase}): "
        f"median {np.median(errors_cm):.2f} cm, p95 {np.percentile(errors_cm, 95):.2f} cm"
    )

    ends = np.cumsum(traj_lengths)
    starts = ends - traj_lengths
    worst_order = np.argsort(-errors_cm)
    typical_order = np.argsort(np.abs(errors_cm - np.median(errors_cm)))

    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "output": args.output,
        "images": images,
        "frames": frames,
        "errors_cm": errors_cm,
        "targets": targets,
        "predictions": predictions,
        "starts": starts,
        "ends": ends,
        "episode_ids": episode_ids,
    }
    records = _save_group("worst", worst_order, args.worst_count, **common) + _save_group(
        "typical", typical_order, args.typical_count, **common
    )
    (args.output / "records.json").write_text(json.dumps(records, indent=2))
    _write_gallery(args.output, records, args.phase)
    print(f"wrote {len(records)} frame crops and a gallery to {args.output}")


if __name__ == "__main__":
    main()
