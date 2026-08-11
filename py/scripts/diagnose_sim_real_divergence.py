#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure what the appearance gap alone does to the image flow policy.

Runs the policy over the aligned pairs `export_sim_real_pairs.py` writes, once
conditioned on the rendered frames and once on the real ones, from identical
sampling noise, and reports how far apart the predicted joint commands land.
Two mixed conditions -- real overhead with rendered wrist, and the reverse --
attribute the difference to a camera.

Read the number against the policy's own scale. The export's actions span
147-280 degrees per dimension, and `diagnose_flow_image_policy.py` reports the
open-loop error against the expert in the same units, so the two are directly
comparable: an appearance divergence well under that error is not what is
costing the rig its grasps.

The pairs are already rectified to the calibrated pinhole at the dataset's
recording resolution, so reaching the policy's input needs only the center crop
and resize that `run_policy_real.py` applies live -- the same geometry, so the
frames here are the frames the hardware runner would have fed it.

Joint states come from the source LeRobot dataset that the pairs were exported
from, since `pairs.json` records the cube and the camera but not the arm.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from pick_and_place.analysis.sim_real_divergence import compare_conditions
from pick_and_place.perception.image_rectify import center_crop_and_resize
from pick_and_place.policies.dataset_export import load_bounds, load_manifest, normalize
from pick_and_place.policies.flow_image_policy import (
    IMAGE_MEAN,
    IMAGE_STD,
    generate_horizon,
    load_model,
)

CAMERA_DIRECTORIES = {
    "overhead": ("overhead_sim", "overhead_real"),
    "wrist": ("wrist_sim", "wrist_real"),
}


def read_states(dataset_root: Path, episode_index: int) -> np.ndarray:
    """Read one episode's ``observation.state`` rows from a LeRobot store."""
    with (dataset_root / "meta" / "info.json").open() as file:
        info = json.load(file)
    row = [
        entry
        for parquet in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        for entry in pq.read_table(parquet).to_pylist()
        if int(entry["episode_index"]) == episode_index
    ]
    if not row:
        raise SystemExit(f"episode {episode_index} is not in {dataset_root}")
    data_path = dataset_root / info["data_path"].format(
        chunk_index=int(row[0]["data/chunk_index"]), file_index=int(row[0]["data/file_index"])
    )
    table = pq.read_table(data_path, columns=["episode_index", "observation.state"])
    table = table.filter(pc.equal(table["episode_index"], episode_index))
    return np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)


def load_pair(directory: Path, camera: str, frame: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load one frame's (rendered, real) images as policy-sized RGB."""
    images = []
    for subdirectory in CAMERA_DIRECTORIES[camera]:
        path = directory / subdirectory / f"{frame:06d}.jpg"
        raw = cv2.imread(str(path))
        if raw is None:
            raise SystemExit(f"missing pair frame {path}")
        cropped = center_crop_and_resize(raw, size, size, cv2)
        images.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    return images[0], images[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="one episode_XXXXXX directory")
    parser.add_argument("--dataset", type=Path, required=True, help="source LeRobot dataset root")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True, help="matching image export")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--integration-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--stable-only",
        action="store_true",
        help="skip frames where the cube is occluded or held, whose render is stale",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    bounds = load_bounds(args.export)
    manifest = load_manifest(args.export)
    size = int(manifest["image_size"][0])

    with (args.pairs / "pairs.json").open() as file:
        index = json.load(file)
    frames = [
        frame
        for frame in index["frames"]
        if frame.get("exported", True)
        and not (args.stable_only and frame["cube_tracking"] != "stable")
    ]
    # The policy conditions on two timesteps, so the first frame has no history.
    frames = frames[: args.max_frames]
    if len(frames) < model.observation_steps:
        raise SystemExit(f"only {len(frames)} usable frames; need {model.observation_steps}")

    states = read_states(args.dataset, int(index["episode_index"]))
    mean_tensor = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)

    overhead = [load_pair(args.pairs, "overhead", frame["frame"], size) for frame in frames]
    wrist = [load_pair(args.pairs, "wrist", frame["frame"], size) for frame in frames]

    def predict(real_overhead: bool, real_wrist: bool, position: int) -> np.ndarray:
        """One action chunk, from the history ending at ``position``.

        The whole history is taken from the condition under test: a real episode
        is real at every timestep, and mixing domains inside one query would
        measure something the rig never shows the policy.
        """
        steps = model.observation_steps
        history = [max(0, position - offset) for offset in range(steps - 1, -1, -1)]
        stacked = [
            np.concatenate(
                (
                    np.moveaxis(overhead[step][int(real_overhead)], -1, 0),
                    np.moveaxis(wrist[step][int(real_wrist)], -1, 0),
                ),
                axis=0,
            )
            for step in history
        ]
        images = torch.from_numpy(np.stack(stacked)[None]).to(device).float().div_(255.0)
        folded = images.reshape(-1, 3, size, size)
        images = ((folded - mean_tensor) / std_tensor).reshape(1, steps, -1, size, size)

        state_rows = np.stack(
            [
                normalize(states[min(step, len(states) - 1)], bounds["obs_min"], bounds["obs_max"])
                for step in history
            ]
        )
        state_tensor = torch.from_numpy(state_rows[None]).to(device).float()

        generator = torch.Generator(device=device).manual_seed(args.seed * 100_000 + position)
        noise = torch.randn(
            (1, model.prediction_steps, model.action_dim), generator=generator, device=device
        )
        chunk = generate_horizon(
            model, images, state_tensor, integration_steps=args.integration_steps, noise=noise
        )
        span = bounds["action_max"] - bounds["action_min"]
        # Unnormalize to degrees without re-centering: a difference of two
        # commands only needs the span, and shifting both by the same minimum
        # would cancel anyway.
        return np.asarray(chunk) * span / 2.0

    summaries = compare_conditions(predict, len(frames))

    print(f"{len(frames)} frames from {args.pairs.name}\n")
    print(f"{'condition':<16}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}   (degrees)")
    for name, summary in summaries.items():
        print(
            f"{name:<16}{summary.mean_deg:>9.3f}{summary.median_deg:>9.3f}"
            f"{summary.p90_deg:>9.3f}{summary.max_deg:>9.3f}"
        )
    full = summaries["real"]
    print("\nby position in the predicted horizon:")
    print("  " + "  ".join(f"{value:.2f}" for value in full.per_chunk_step))
    print("by joint:")
    print("  " + "  ".join(f"{value:.2f}" for value in full.per_joint))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump(
                {
                    "pairs": str(args.pairs),
                    "checkpoint": str(args.checkpoint),
                    "frames": len(frames),
                    "integration_steps": args.integration_steps,
                    "seed": args.seed,
                    "conditions": {name: asdict(value) for name, value in summaries.items()},
                },
                file,
                indent=1,
            )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
