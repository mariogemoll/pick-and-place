# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tests for the action-encoding contract shared by the exporter and every rollout."""

from __future__ import annotations

import numpy as np
import pytest

from pick_and_place.spec.action_encoding import (
    ACTION_ENCODING_KEY,
    ActionEncoding,
    decode_actions,
    encode_actions,
    parse_action_encoding,
    read_action_encoding,
)


def test_delta_encoding_round_trips_through_the_state_it_was_measured_from():
    states = np.array([[10.0, -5.0], [12.0, -4.0]], dtype=np.float32)
    actions = np.array([[11.0, -4.5], [13.5, -4.0]], dtype=np.float32)

    deltas = encode_actions(ActionEncoding.DELTA, actions, states)

    np.testing.assert_allclose(deltas, [[1.0, 0.5], [1.5, 0.0]])
    for index, delta in enumerate(deltas):
        np.testing.assert_allclose(
            decode_actions(ActionEncoding.DELTA, delta, states[index]), actions[index]
        )


def test_absolute_encoding_ignores_the_state_entirely():
    states = np.array([[10.0, -5.0]], dtype=np.float32)
    actions = np.array([[11.0, -4.5]], dtype=np.float32)

    encoded = encode_actions(ActionEncoding.ABSOLUTE, actions, states)

    np.testing.assert_array_equal(encoded, actions)
    np.testing.assert_array_equal(decode_actions(ActionEncoding.ABSOLUTE, encoded, states), actions)


def test_decoding_a_chunk_against_one_state_broadcasts_over_it():
    chunk = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    decoded = decode_actions(ActionEncoding.DELTA, chunk, np.array([10.0, 20.0], dtype=np.float32))

    np.testing.assert_allclose(decoded, [[11.0, 22.0], [13.0, 24.0]])


def test_encoding_rejects_actions_and_states_that_are_not_aligned():
    with pytest.raises(ValueError, match="aligned row for row"):
        encode_actions(ActionEncoding.DELTA, np.zeros((3, 2)), np.zeros((2, 2)))


def test_an_export_without_a_declared_encoding_holds_absolute_commands(tmp_path):
    path = tmp_path / "normalization.npz"
    np.savez_compressed(path, action_min=np.zeros(2), action_max=np.ones(2))

    with np.load(path) as bounds:
        assert read_action_encoding(bounds) is ActionEncoding.ABSOLUTE


def test_a_declared_encoding_survives_the_normalization_archive(tmp_path):
    path = tmp_path / "normalization.npz"
    np.savez_compressed(path, **{ACTION_ENCODING_KEY: ActionEncoding.DELTA.value})

    with np.load(path) as bounds:
        assert read_action_encoding(bounds) is ActionEncoding.DELTA


def test_an_unknown_encoding_names_the_alternatives():
    with pytest.raises(ValueError, match="absolute, delta"):
        parse_action_encoding("relative")
