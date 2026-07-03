#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fit simple SO-101 joint dynamics from recorded LeRobotDatasets.

This is recording-based system identification: use the dataset's commanded
``action`` stream and measured ``observation.state`` stream to estimate, per
joint, a delayed first-order response:

    state[t + 1] = state[t] + alpha * (action[t - delay] - state[t]) + beta

``delay`` captures command latency, ``alpha`` captures how quickly the physical
joint approaches the delayed command, and ``beta / alpha`` is the fitted
steady-state bias in dataset units. The fitted JSON is intentionally small and
can be consumed by replay/sim tools as a first approximation of the real arm's
control path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pick_and_place.executor import CONTROL_HZ
from pick_and_place.follower import GRIPPER_INDEX, JOINT_NAMES


@dataclass(frozen=True)
class EpisodeSeries:
    dataset_root: Path
    episode_index: int
    states: np.ndarray
    actions: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.dataset_root.name}:{self.episode_index}"


@dataclass(frozen=True)
class JointFit:
    joint: str
    delay_frames: int
    alpha: float
    beta: float
    bias: float
    train_mae: float
    val_mae: float
    baseline_val_mae: float
    train_samples: int
    val_samples: int


def _read_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def _chunked_path(pattern: str, chunk_index: int, file_index: int) -> Path:
    return Path(pattern.format(chunk_index=chunk_index, file_index=file_index))


def _read_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(parquet_path).to_pylist())
    if not rows:
        raise ValueError(f"no episode metadata found in {dataset_root}")
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def _load_dataset(
    dataset_root: Path,
    *,
    max_episodes: int | None,
) -> tuple[float, list[EpisodeSeries]]:
    info = _read_info(dataset_root)
    fps = float(info.get("fps", CONTROL_HZ))
    rows = _read_episode_rows(dataset_root)
    if max_episodes is not None:
        rows = rows[:max_episodes]

    by_data_path: dict[Path, list[dict[str, Any]]] = {}
    for row in rows:
        data_path = dataset_root / _chunked_path(
            info["data_path"],
            int(row["data/chunk_index"]),
            int(row["data/file_index"]),
        )
        by_data_path.setdefault(data_path, []).append(row)

    episodes: list[EpisodeSeries] = []
    for data_path, file_rows in sorted(by_data_path.items()):
        table = pq.read_table(
            data_path,
            columns=["episode_index", "observation.state", "action"],
        )
        for row in file_rows:
            episode_index = int(row["episode_index"])
            episode_table = table.filter(pc.equal(table["episode_index"], episode_index))
            if episode_table.num_rows < 3:
                continue
            states = np.asarray(episode_table["observation.state"].to_pylist(), dtype=float)
            actions = np.asarray(episode_table["action"].to_pylist(), dtype=float)
            if states.shape != actions.shape or states.shape[1] != len(JOINT_NAMES):
                raise ValueError(
                    f"{dataset_root} episode {episode_index} has unexpected "
                    f"state/action shapes {states.shape} / {actions.shape}"
                )
            episodes.append(
                EpisodeSeries(
                    dataset_root=dataset_root,
                    episode_index=episode_index,
                    states=states,
                    actions=actions,
                )
            )
    return fps, episodes


def _split_episodes(
    episodes: list[EpisodeSeries],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[EpisodeSeries], list[EpisodeSeries]]:
    shuffled = episodes.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val = shuffled[:val_count]
    train = shuffled[val_count:] or shuffled
    return train, val


def _samples_for_joint(
    episodes: list[EpisodeSeries],
    joint_index: int,
    *,
    delay_frames: int,
    min_excitation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    baselines = []
    for episode in episodes:
        states = episode.states[:, joint_index]
        actions = episode.actions[:, joint_index]
        if len(states) <= delay_frames + 1:
            continue
        current = states[delay_frames:-1]
        nxt = states[delay_frames + 1 :]
        delayed_action = actions[: len(current)]
        excitation = np.abs(delayed_action - current)
        movement = np.abs(nxt - current)
        keep = (excitation >= min_excitation) | (movement >= min_excitation)
        if not np.any(keep):
            continue
        xs.append((delayed_action - current)[keep])
        ys.append((nxt - current)[keep])
        baselines.append((delayed_action - nxt)[keep])
    if not xs:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(baselines)


def _fit_linear_response(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(alpha), float(beta)


def _mae(alpha: float, beta: float, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return math.nan
    return float(np.mean(np.abs(alpha * x + beta - y)))


def _baseline_mae(baselines: np.ndarray) -> float:
    if len(baselines) == 0:
        return math.nan
    return float(np.mean(np.abs(baselines)))


def _fit_joint(
    train: list[EpisodeSeries],
    val: list[EpisodeSeries],
    joint_index: int,
    *,
    max_delay_frames: int,
    min_excitation: float,
    min_alpha: float,
    max_alpha: float,
) -> JointFit:
    best: JointFit | None = None
    joint = JOINT_NAMES[joint_index]
    for delay in range(max_delay_frames + 1):
        x_train, y_train, baseline_train = _samples_for_joint(
            train,
            joint_index,
            delay_frames=delay,
            min_excitation=min_excitation,
        )
        del baseline_train
        if len(x_train) < 10:
            continue
        alpha, beta = _fit_linear_response(x_train, y_train)
        if not min_alpha <= alpha <= max_alpha:
            continue
        x_val, y_val, baseline_val = _samples_for_joint(
            val or train,
            joint_index,
            delay_frames=delay,
            min_excitation=min_excitation,
        )
        train_mae = _mae(alpha, beta, x_train, y_train)
        val_mae = _mae(alpha, beta, x_val, y_val)
        bias = beta / alpha if abs(alpha) > 1e-9 else math.nan
        fit = JointFit(
            joint=joint,
            delay_frames=delay,
            alpha=alpha,
            beta=beta,
            bias=bias,
            train_mae=train_mae,
            val_mae=val_mae,
            baseline_val_mae=_baseline_mae(baseline_val),
            train_samples=len(x_train),
            val_samples=len(x_val),
        )
        if best is None or fit.val_mae < best.val_mae:
            best = fit
    if best is None:
        raise ValueError(f"not enough samples to fit {joint}")
    return best


def _time_constant(alpha: float, fps: float) -> float | None:
    if not 0.0 < alpha < 1.0:
        return None
    return float(-(1.0 / fps) / math.log(1.0 - alpha))


def _joint_units(joint_index: int) -> str:
    return "position" if joint_index == GRIPPER_INDEX else "degrees"


def _write_config(
    output: Path,
    *,
    dataset_roots: list[Path],
    fps: float,
    train: list[EpisodeSeries],
    val: list[EpisodeSeries],
    fits: list[JointFit],
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "delayed_first_order_joint_response",
        "equation": "state[t+1] = state[t] + alpha * (action[t-delay] - state[t]) + beta",
        "control_hz": fps,
        "source_datasets": [str(root) for root in dataset_roots],
        "train_episodes": len(train),
        "validation_episodes": len(val),
        "fit_options": {
            "max_delay_frames": args.max_delay_frames,
            "min_excitation": args.min_excitation,
            "min_alpha": args.min_alpha,
            "max_alpha": args.max_alpha,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "joints": {
            fit.joint: {
                "delay_frames": fit.delay_frames,
                "delay_s": fit.delay_frames / fps,
                "alpha_per_frame": fit.alpha,
                "time_constant_s": _time_constant(fit.alpha, fps),
                "beta_per_frame": fit.beta,
                "steady_state_bias": fit.bias,
                "unit": _joint_units(i),
                "train_mae_per_frame": fit.train_mae,
                "validation_mae_per_frame": fit.val_mae,
                "validation_baseline_action_mae": fit.baseline_val_mae,
                "train_samples": fit.train_samples,
                "validation_samples": fit.val_samples,
            }
            for i, fit in enumerate(fits)
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_roots", type=Path, nargs="+", help="LeRobotDataset root(s)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/robot_dynamics/so101_follower.json"),
        help="output calibration JSON (default: config/robot_dynamics/so101_follower.json)",
    )
    parser.add_argument(
        "--max-delay-frames",
        type=int,
        default=15,
        help="largest per-joint action delay to test (default: 15)",
    )
    parser.add_argument(
        "--min-excitation",
        type=float,
        default=0.05,
        help="minimum per-frame command/response movement to include a sample (default: 0.05)",
    )
    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.001,
        help="minimum plausible first-order response alpha per frame (default: 0.001)",
    )
    parser.add_argument(
        "--max-alpha",
        type=float,
        default=1.0,
        help="maximum plausible first-order response alpha per frame (default: 1.0)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="episode fraction held out for validation (default: 0.15)",
    )
    parser.add_argument("--seed", type=int, default=0, help="train/val split seed")
    parser.add_argument(
        "--max-episodes-per-dataset",
        type=int,
        default=None,
        help="debug limit: read only the first N episodes from each dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fit without writing the JSON artifact",
    )
    args = parser.parse_args()

    if args.max_delay_frames < 0:
        parser.error("--max-delay-frames must be non-negative")
    if args.min_excitation < 0:
        parser.error("--min-excitation must be non-negative")
    if not 0.0 <= args.min_alpha <= args.max_alpha:
        parser.error("--min-alpha/--max-alpha must satisfy 0 <= min <= max")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val-fraction must be in [0, 1)")
    if args.max_episodes_per_dataset is not None and args.max_episodes_per_dataset < 1:
        parser.error("--max-episodes-per-dataset must be positive")

    fps_values = []
    episodes: list[EpisodeSeries] = []
    for root in args.dataset_roots:
        fps, root_episodes = _load_dataset(root, max_episodes=args.max_episodes_per_dataset)
        fps_values.append(fps)
        episodes.extend(root_episodes)
        print(f"{root}: loaded {len(root_episodes)} episode(s) at {fps:g} Hz")
    if not episodes:
        raise SystemExit("no usable episodes loaded")
    if any(not math.isclose(fps, fps_values[0]) for fps in fps_values):
        raise SystemExit(f"dataset FPS mismatch: {fps_values}")
    fps = fps_values[0]

    train, val = _split_episodes(episodes, val_fraction=args.val_fraction, seed=args.seed)
    print(f"fit split: {len(train)} train episode(s), {len(val)} validation episode(s)")

    fits = [
        _fit_joint(
            train,
            val,
            joint_index,
            max_delay_frames=args.max_delay_frames,
            min_excitation=args.min_excitation,
            min_alpha=args.min_alpha,
            max_alpha=args.max_alpha,
        )
        for joint_index in range(len(JOINT_NAMES))
    ]

    print(
        "\n"
        f"{'joint':<14}{'delay':>7}{'tau(s)':>9}{'alpha':>10}{'bias':>10}"
        f"{'val mae':>11}{'baseline':>11}{'samples':>10}"
    )
    for i, fit in enumerate(fits):
        tau = _time_constant(fit.alpha, fps)
        tau_text = f"{tau:.3f}" if tau is not None else "n/a"
        print(
            f"{fit.joint:<14}{fit.delay_frames:>7}{tau_text:>9}"
            f"{fit.alpha:>10.4f}{fit.bias:>10.3f}"
            f"{fit.val_mae:>11.3f}{fit.baseline_val_mae:>11.3f}"
            f"{fit.val_samples:>10}"
        )
    print("\nUnits: arm joints are degrees; gripper is follower 0-100 position.")

    if args.dry_run:
        print(f"Dry run: would write {args.output}")
        return

    _write_config(
        args.output,
        dataset_roots=args.dataset_roots,
        fps=fps,
        train=train,
        val=val,
        fits=fits,
        args=args,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
