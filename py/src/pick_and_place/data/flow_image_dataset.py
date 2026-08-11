# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Window a Diffusion Policy image export into flow-matching training examples.

The stitched export is a flat frame table plus per-episode lengths. A flow
example is an observation history and the action chunk that follows it, so the
only real work is turning a frame index into two index arrays that never reach
across an episode boundary.

The padding rule is the state export's, so the two policies are trained on the
same windows: a history reaching before the episode repeats its first frame, and
a chunk reaching past the end repeats its final action. Every recorded tick
therefore contributes exactly one example.

Images stay ``uint8`` here. Converting and normalizing them is the training
loop's job, on the device, because a float32 copy of this table is four times
the size and would not fit where the uint8 one does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.data.stored_npz import memmap_stored_npz


def window_indices(
    traj_lengths: np.ndarray, *, observation_steps: int, prediction_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame observation and action index arrays that respect episode bounds.

    Returns ``(observation_index, action_index)`` with shapes
    ``(frames, observation_steps)`` and ``(frames, prediction_steps)``, both
    indexing the flat frame table.
    """
    traj_lengths = np.asarray(traj_lengths, dtype=np.int64)
    if traj_lengths.ndim != 1 or not len(traj_lengths) or (traj_lengths < 1).any():
        raise ValueError("traj_lengths must be a nonempty 1-D array of positive lengths")
    if observation_steps < 1 or prediction_steps < 1:
        raise ValueError("observation_steps and prediction_steps must be positive")

    starts = np.concatenate(([0], np.cumsum(traj_lengths)[:-1]))
    ends = starts + traj_lengths
    frame_start = np.repeat(starts, traj_lengths)
    frame_end = np.repeat(ends, traj_lengths)
    absolute = np.arange(int(traj_lengths.sum()), dtype=np.int64)

    history_offsets = np.arange(observation_steps - 1, -1, -1, dtype=np.int64)
    observation_index = np.maximum(absolute[:, None] - history_offsets[None, :], frame_start[:, None])
    prediction_offsets = np.arange(prediction_steps, dtype=np.int64)
    action_index = np.minimum(
        absolute[:, None] + prediction_offsets[None, :], frame_end[:, None] - 1
    )
    return observation_index, action_index


def split_episodes(
    traj_lengths: np.ndarray, *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes into training and validation frame indices."""
    traj_lengths = np.asarray(traj_lengths, dtype=np.int64)
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    count = len(traj_lengths)
    if count < 2 and validation_fraction > 0.0:
        raise ValueError("at least two episodes are required for a validation split")

    episode_of_frame = np.repeat(np.arange(count, dtype=np.int64), traj_lengths)
    if validation_fraction == 0.0:
        return np.arange(len(episode_of_frame)), np.empty(0, dtype=np.int64)

    validation_count = min(count - 1, max(1, round(count * validation_fraction)))
    rng = np.random.default_rng(seed)
    held_out = np.zeros(count, dtype=bool)
    held_out[rng.choice(count, size=validation_count, replace=False)] = True
    is_validation = held_out[episode_of_frame]
    return np.nonzero(~is_validation)[0], np.nonzero(is_validation)[0]


class FlowImageExport:
    """A Diffusion Policy image export, windowed for flow-matching training."""

    def __init__(
        self,
        export_dir: str | Path,
        *,
        observation_steps: int = 2,
        prediction_steps: int = 16,
        validation_fraction: float = 0.1,
        seed: int = 0,
        mmap: bool = True,
    ) -> None:
        export_dir = Path(export_dir)
        with (export_dir / "export.json").open() as file:
            self.manifest: dict[str, Any] = json.load(file)

        # np.load ignores mmap_mode for an NPZ and would materialize the whole
        # image tensor -- tens of gigabytes -- in RAM. The exporter stores its
        # members uncompressed precisely so they can be mapped in place.
        archive = (
            memmap_stored_npz(export_dir / "train.npz")
            if mmap
            else dict(np.load(export_dir / "train.npz", allow_pickle=False))
        )
        self.images = archive["images"]
        self.states = np.asarray(archive["states"], dtype=np.float32)
        self.actions = np.asarray(archive["actions"], dtype=np.float32)
        self.traj_lengths = np.asarray(archive["traj_lengths"], dtype=np.int64)

        frames = int(self.traj_lengths.sum())
        if len(self.images) != frames or len(self.states) != frames:
            raise ValueError("images, states and traj_lengths disagree on the frame count")
        if self.images.ndim != 4:
            raise ValueError("images must be (frames, channels, height, width)")
        if self.images.shape[1] % 3:
            raise ValueError("image channels must be a whole number of RGB cameras")

        self.observation_steps = observation_steps
        self.prediction_steps = prediction_steps
        self.cameras = self.images.shape[1] // 3
        self.image_hw = (int(self.images.shape[2]), int(self.images.shape[3]))
        self.state_dim = int(self.states.shape[1])
        self.action_dim = int(self.actions.shape[1])

        self.observation_index, self.action_index = window_indices(
            self.traj_lengths,
            observation_steps=observation_steps,
            prediction_steps=prediction_steps,
        )
        self.training_frames, self.validation_frames = split_episodes(
            self.traj_lengths, validation_fraction=validation_fraction, seed=seed
        )

    def batch(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gather one batch as ``(images, states, action_chunks)``.

        ``frames`` must be sorted for an efficient memory-mapped read.
        """
        observation = self.observation_index[frames]
        action = self.action_index[frames]
        images = self.images[observation.reshape(-1)].reshape(
            len(frames), self.observation_steps, self.cameras * 3, *self.image_hw
        )
        states = self.states[observation]
        chunks = self.actions[action]
        return images, states, chunks
