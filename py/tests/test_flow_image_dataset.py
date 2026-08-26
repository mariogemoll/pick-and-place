# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Windowing an image export must never reach across an episode boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pick_and_place.data.flow_image_dataset import (
    FlowImageExport,
    split_episodes,
    window_indices,
)
from pick_and_place.spec.robot import JOINT_NAMES


def test_history_repeats_the_first_frame_of_its_own_episode() -> None:
    observation, _ = window_indices(
        np.array([4, 3]), observation_steps=3, prediction_steps=2
    )
    # Episode 0 spans frames 0..3, episode 1 spans 4..6.
    assert observation[0].tolist() == [0, 0, 0]
    assert observation[1].tolist() == [0, 0, 1]
    # The first frame of episode 1 must pad with frame 4, not frames 2 and 3.
    assert observation[4].tolist() == [4, 4, 4]
    assert observation[5].tolist() == [4, 4, 5]


def test_prediction_repeats_the_final_action_of_its_own_episode() -> None:
    _, action = window_indices(np.array([4, 3]), observation_steps=2, prediction_steps=3)
    # Episode 0's last frame may only repeat frame 3, never step into frame 4.
    assert action[3].tolist() == [3, 3, 3]
    assert action[2].tolist() == [2, 3, 3]
    assert action[6].tolist() == [6, 6, 6]


def test_every_frame_contributes_exactly_one_example() -> None:
    lengths = np.array([5, 9, 2])
    observation, action = window_indices(
        lengths, observation_steps=2, prediction_steps=16
    )
    assert len(observation) == len(action) == lengths.sum()


def test_indices_stay_inside_their_episode_for_long_horizons() -> None:
    lengths = np.array([6, 7, 4])
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    ends = starts + lengths
    observation, action = window_indices(
        lengths, observation_steps=4, prediction_steps=20
    )
    episode_of_frame = np.repeat(np.arange(len(lengths)), lengths)
    for frame, episode in enumerate(episode_of_frame):
        low, high = starts[episode], ends[episode] - 1
        assert observation[frame].min() >= low and observation[frame].max() <= high
        assert action[frame].min() >= low and action[frame].max() <= high


def test_split_is_by_episode_so_no_frame_is_shared() -> None:
    lengths = np.array([4, 5, 6, 7])
    training, validation = split_episodes(lengths, validation_fraction=0.5, seed=0)
    assert not set(training.tolist()) & set(validation.tolist())
    assert len(training) + len(validation) == lengths.sum()

    episode_of_frame = np.repeat(np.arange(len(lengths)), lengths)
    training_episodes = set(episode_of_frame[training].tolist())
    validation_episodes = set(episode_of_frame[validation].tolist())
    assert not training_episodes & validation_episodes


def test_zero_validation_fraction_keeps_every_frame_for_training() -> None:
    training, validation = split_episodes(
        np.array([3, 3]), validation_fraction=0.0, seed=0
    )
    assert len(training) == 6 and len(validation) == 0


@pytest.mark.parametrize(
    "lengths, observation_steps, prediction_steps",
    [(np.array([]), 2, 2), (np.array([0, 3]), 2, 2), (np.array([3]), 0, 2), (np.array([3]), 2, 0)],
)
def test_invalid_shapes_are_rejected(
    lengths: np.ndarray, observation_steps: int, prediction_steps: int
) -> None:
    with pytest.raises(ValueError):
        window_indices(
            lengths, observation_steps=observation_steps, prediction_steps=prediction_steps
        )


GOAL_DIM = 2


def write_goal_export(root: Path, *, prediction_steps: int = 2) -> FlowImageExport:
    """Two three-frame episodes whose states are joints plus a goal slot."""
    frames, state_dim = 6, len(JOINT_NAMES) + GOAL_DIM
    states = np.arange(frames * state_dim, dtype=np.float32).reshape(frames, state_dim)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / "train.npz",
        images=np.zeros((frames, 6, 4, 4), dtype=np.uint8),
        states=states,
        actions=np.zeros((frames, len(JOINT_NAMES)), dtype=np.float32),
        traj_lengths=np.array([3, 3], dtype=np.int64),
    )
    (root / "export.json").write_text(
        json.dumps({"goal_dim": GOAL_DIM, "goal_source": "episode_target_xy"})
    )
    return FlowImageExport(
        root, observation_steps=2, prediction_steps=prediction_steps, mmap=False
    )


def test_training_reads_the_wider_state_off_the_export_and_passes_it_through(tmp_path):
    """A goal-conditioned export needs no training change; this is that check."""
    export = write_goal_export(tmp_path / "export")

    assert export.state_dim == len(JOINT_NAMES) + GOAL_DIM
    assert export.action_dim == len(JOINT_NAMES)

    _, states, _ = export.batch(np.array([0, 4]))

    assert states.shape == (2, 2, len(JOINT_NAMES) + GOAL_DIM)
    # Frame 0 has no history, so it repeats itself; frame 4 pairs 3 with 4.
    np.testing.assert_array_equal(states[0, 0], states[0, 1])
    np.testing.assert_array_equal(states[0, 0], export.states[0])
    np.testing.assert_array_equal(states[1], export.states[[3, 4]])


def test_a_goal_conditioned_export_builds_a_model_with_the_matching_state_width(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from pick_and_place.policies.flow_image_encoder import FlowImageUnet1D

    export = write_goal_export(tmp_path / "export", prediction_steps=4)
    # Exactly how cli/train_flow_image_policy.py builds it.
    model = FlowImageUnet1D(
        action_dim=export.action_dim,
        state_dim=export.state_dim,
        prediction_steps=export.prediction_steps,
        observation_steps=export.observation_steps,
        cameras=export.cameras,
        keypoints=8,
    )

    assert model.state_dim == len(JOINT_NAMES) + GOAL_DIM
