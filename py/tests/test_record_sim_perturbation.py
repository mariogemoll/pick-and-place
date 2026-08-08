# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Which episodes get a deliberate fumble, and what that must not disturb."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "pick_and_place"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from record_sim import (  # noqa: E402
    PERTURBATION_SEED_SALT,
    _episode_rng,
    _perturbation_rng,
)


def _perturbed_indices(seed: int, count: int, fraction: float) -> list[int]:
    return [
        index
        for index in range(count)
        if _perturbation_rng(seed, index).random() < fraction
    ]


@pytest.mark.parametrize("fraction", [0.2, 0.25, 0.3])
def test_perturbed_fraction_is_approximately_honoured(fraction):
    chosen = _perturbed_indices(20260807, 2000, fraction)
    assert abs(len(chosen) / 2000 - fraction) < 0.03


def test_the_stream_is_a_pure_function_of_seed_and_index():
    # A recorded episode must be reproducible from its index alone, which is what
    # lets a staging area be topped up without changing what is already there.
    assert _perturbation_rng(7, 11).random() == _perturbation_rng(7, 11).random()
    assert _perturbation_rng(7, 11).random() != _perturbation_rng(7, 12).random()
    assert _perturbation_rng(8, 11).random() != _perturbation_rng(7, 11).random()


def test_raising_the_fraction_only_adds_episodes():
    # The decision is a per-episode threshold on one draw, so a higher fraction
    # must be a superset. If it were not, changing the fraction would reshuffle
    # which episodes are perturbed and the arms would stop being comparable.
    low = set(_perturbed_indices(20260807, 500, 0.20))
    high = set(_perturbed_indices(20260807, 500, 0.30))
    assert low < high


def test_perturbation_stream_is_independent_of_the_pose_stream():
    # The whole point of the salt. If these coincided, turning the fraction up
    # would move every other episode's cube and the two dataset arms would differ
    # in their entire pose distribution rather than only in the perturbations.
    for index in range(32):
        assert _perturbation_rng(3, index).random() != _episode_rng(3, index).random()


def test_zero_fraction_perturbs_nothing():
    assert _perturbed_indices(20260807, 500, 0.0) == []


def test_salt_is_pinned():
    # Changing it silently re-rolls which episodes are fumbled in every dataset
    # generated so far, which would break reproducibility of the recorded arms.
    assert PERTURBATION_SEED_SALT == 0x50455254


def test_unseeded_stream_is_nondeterministic():
    # seed=None is the "do not care about reproducibility" path the recorder
    # already allows for pose sampling; mirror it rather than silently seeding 0.
    assert _perturbation_rng(None, 0).random() != _perturbation_rng(None, 0).random()


def test_draws_are_uniform_enough_to_threshold():
    values = np.array([_perturbation_rng(1234, i).random() for i in range(4000)])
    assert 0.47 < values.mean() < 0.53
    assert values.min() < 0.01 and values.max() > 0.99
