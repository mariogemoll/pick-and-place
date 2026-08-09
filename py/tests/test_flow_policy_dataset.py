# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.data.flow_policy_dataset import (
    EpisodeValues,
    make_examples,
    normalize,
    prepare_splits,
    split_episode_indices,
)


def test_make_examples_pads_only_within_episode() -> None:
    observations = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    endpoints = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)

    conditions, flattened_endpoints = make_examples(
        observations,
        endpoints,
        observation_steps=2,
        prediction_steps=3,
    )

    np.testing.assert_array_equal(conditions, [[1, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(
        flattened_endpoints,
        [[10, 20, 30], [20, 30, 30], [30, 30, 30]],
    )


def test_split_episode_indices_is_deterministic_and_disjoint() -> None:
    first = split_episode_indices(list(range(10)), validation_fraction=0.2, seed=7)
    second = split_episode_indices(list(range(10)), validation_fraction=0.2, seed=7)

    assert first == second
    training, validation = first
    assert len(training) == 8
    assert len(validation) == 2
    assert set(training).isdisjoint(validation)


def test_normalize_maps_constant_columns_to_zero() -> None:
    values = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)

    result = normalize(values, values.min(axis=0), values.max(axis=0))

    np.testing.assert_allclose(result[:, 0], [-1.0, 1.0])
    np.testing.assert_array_equal(result[:, 1], [0.0, 0.0])


def test_prepare_splits_fits_bounds_on_training_episodes_only() -> None:
    episodes = [
        EpisodeValues(
            episode_index=0,
            observations=np.array([[0.0], [2.0]], dtype=np.float32),
            endpoints=np.array([[10.0], [20.0]], dtype=np.float32),
        ),
        EpisodeValues(
            episode_index=1,
            observations=np.array([[100.0], [200.0]], dtype=np.float32),
            endpoints=np.array([[1000.0], [2000.0]], dtype=np.float32),
        ),
    ]
    training, validation, bounds, training_indices, validation_indices = prepare_splits(
        episodes,
        observation_steps=1,
        prediction_steps=1,
        validation_fraction=0.5,
        seed=0,
    )

    source = next(episode for episode in episodes if episode.episode_index == training_indices[0])
    np.testing.assert_array_equal(bounds["observation_min"], source.observations.min(axis=0))
    np.testing.assert_array_equal(bounds["observation_max"], source.observations.max(axis=0))
    np.testing.assert_allclose(training["observations"].min(), -1.0, atol=1e-6)
    np.testing.assert_allclose(training["observations"].max(), 1.0, atol=1e-6)
    assert training_indices != validation_indices
    assert len(validation["observations"]) == 2
