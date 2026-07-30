# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Cube-position targets and phase labels for the cube-localization head.

A diffusion-policy export (:mod:`pick_and_place.diffusion_policy_dataset`)
keeps only ``states``, ``actions``, ``images`` and ``traj_lengths`` in
``train.npz`` — the privileged ``observation.environment_state`` ground truth
is dropped. This module rebuilds it from the staged per-episode source
directories (one LeRobot episode per directory, as produced before
finalization/merging), decimated and concatenated the same way the exporter
built ``train.npz``, so the two line up frame for frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pick_and_place.sim_recorder import CUBE_POSE_STATE_NAMES
from pick_and_place.task_phases import coarse_phase_labels, phase_spans_from_json

ENVIRONMENT_STATE_FEATURE = "observation.environment_state"
# CUBE_POSE_STATE_NAMES is (x, y, z, qw, qx, qy, qz); only position is needed here.
_POSITION_DIMS = 3


def _read_episode_row(dataset_root: Path) -> dict:
    """Read the single episode-metadata row for one staged episode directory."""
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode metadata found under {dataset_root}")
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    rows = table.to_pylist()
    if len(rows) != 1:
        raise ValueError(f"{dataset_root} does not hold exactly one staged episode")
    return rows[0]


def _read_cube_positions(dataset_root: Path, expected_length: int) -> np.ndarray:
    """Per-frame ``(x, y, z)`` cube position, in recorded frame order."""
    paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode data found under {dataset_root}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=["index", ENVIRONMENT_STATE_FEATURE]) for path in paths]
    )
    table = table.sort_by("index")
    if table.num_rows != expected_length:
        raise ValueError(
            f"{dataset_root}: {table.num_rows} data rows, expected {expected_length}"
        )
    states = np.asarray(table[ENVIRONMENT_STATE_FEATURE].to_pylist(), dtype=np.float64)
    if states.shape[1] != len(CUBE_POSE_STATE_NAMES):
        raise ValueError(
            f"{dataset_root}: expected {len(CUBE_POSE_STATE_NAMES)}-dim environment state, "
            f"got {states.shape[1]}"
        )
    return states[:, :_POSITION_DIMS]


def load_cube_targets(
    episodes_root: Path,
    episode_ids: list[str],
    *,
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame cube position and coarse task phase, decimated to match an export.

    ``episode_ids`` must be in the same order used to build the export's
    ``train.npz`` (its ``export.json["episode_indices"]``, equivalently the
    export's ``source-episodes.txt`` manifest). Each id names a directory
    under ``episodes_root`` holding one staged LeRobot episode.

    Returns ``(positions, phases, traj_lengths)``:

    - ``positions``: ``(N, 3)`` float32 cube ``(x, y, z)``.
    - ``phases``: ``(N,)`` array of coarse phase-name strings
      (:data:`pick_and_place.task_phases.PHASES`).
    - ``traj_lengths``: ``(len(episode_ids),)`` int64 decimated per-episode
      frame counts, for an episode-level train/held-out split.

    Decimation keeps episode-relative indices ``0, frame_stride,
    2 * frame_stride, ...``, matching
    :func:`pick_and_place.diffusion_policy_dataset.export_diffusion_policy_dataset`.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if not episode_ids:
        raise ValueError("episode_ids must be nonempty")

    all_positions = []
    all_phases = []
    traj_lengths = []
    for episode_id in episode_ids:
        dataset_root = episodes_root / episode_id
        row = _read_episode_row(dataset_root)
        length = int(row["length"])
        positions = _read_cube_positions(dataset_root, length)
        spans = phase_spans_from_json(row["phase_spans"])
        labels = coarse_phase_labels(spans, length)
        keep = np.arange(0, length, frame_stride)
        all_positions.append(positions[keep])
        all_phases.append(labels[keep])
        traj_lengths.append(len(keep))

    return (
        np.concatenate(all_positions).astype(np.float32),
        np.concatenate(all_phases),
        np.asarray(traj_lengths, dtype=np.int64),
    )


def episode_frame_split(
    traj_lengths: np.ndarray, held_out_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign whole episodes to train/held-out, then expand to frame indices.

    Splitting by episode (rather than by frame) matters because neighboring
    frames within an episode are nearly identical; a frame-level split would
    leak near-duplicates across the split and report an optimistic error.
    Deterministic in ``seed``, so a training run and a later inspection of its
    held-out predictions can reproduce the same split independently.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError("held_out_fraction must be in (0, 1)")
    num_episodes = len(traj_lengths)
    num_held_out = max(1, round(num_episodes * held_out_fraction))
    order = np.random.default_rng(seed).permutation(num_episodes)
    held_out_episodes = set(order[:num_held_out].tolist())

    ends = np.cumsum(traj_lengths)
    starts = ends - traj_lengths
    train_frames = []
    held_out_frames = []
    for episode in range(num_episodes):
        frame_range = np.arange(starts[episode], ends[episode])
        (held_out_frames if episode in held_out_episodes else train_frames).append(frame_range)
    return np.concatenate(train_frames), np.concatenate(held_out_frames)
