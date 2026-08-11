# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Windowing an image export must never reach across an episode boundary."""

from __future__ import annotations

import numpy as np
import pytest

from pick_and_place.data.flow_image_dataset import split_episodes, window_indices


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
