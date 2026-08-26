# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A chained run never resets the scene, so every target is also a start pose."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pick_and_place.core.workspace_bounds import (
    is_cube_drop_allowed,
    is_cube_pickup_allowed,
    is_cube_recovery_target_allowed,
)
from pick_and_place.scripted.scenario_sampling import (
    MIN_CHAIN_STEP_M,
    TARGET_INTERIOR_MARGIN_M,
    comfortably_interior,
    sample_target_chain,
)


def test_every_target_in_a_chain_can_also_be_picked_up_from():
    """The whole point: target n is the cube position episode n+1 starts from."""
    chain = sample_target_chain(np.random.default_rng(0), 100)

    assert len(chain) == 100
    for target in chain:
        assert is_cube_pickup_allowed(target.x, target.y)
        assert comfortably_interior(
            target.x, target.y, TARGET_INTERIOR_MARGIN_M, is_cube_recovery_target_allowed
        )


def test_consecutive_targets_are_a_real_transport_apart():
    chain = sample_target_chain(np.random.default_rng(1), 100)

    for previous, following in zip(chain, chain[1:]):
        step = math.hypot(following.x - previous.x, following.y - previous.y)
        assert step >= MIN_CHAIN_STEP_M


def test_a_hundred_long_chain_draws_from_any_seed():
    """Unattended means a seed that cannot finish its chain must not reach the rig."""
    for seed in range(25):
        assert len(sample_target_chain(np.random.default_rng(seed), 100)) == 100


def test_the_chain_is_reproducible_from_its_seed():
    first = sample_target_chain(np.random.default_rng(7), 20)
    second = sample_target_chain(np.random.default_rng(7), 20)

    assert [(t.x, t.y) for t in first] == [(t.x, t.y) for t in second]


def test_the_pickup_zone_is_a_strict_subset_of_the_drop_zone():
    """Why the chain needs its own sampler rather than reusing sample_target.

    If this ever stops holding, a chained target could be pickable but not a
    legal placement, and the screening below would be the wrong way round.
    """
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(20000):
        x = float(rng.uniform(-0.5, 0.5))
        y = float(rng.uniform(-0.5, 0.5))
        if is_cube_pickup_allowed(x, y):
            assert is_cube_drop_allowed(x, y)
            checked += 1
    assert checked > 100


def test_an_impossible_step_is_refused_rather_than_looped_on():
    with pytest.raises(RuntimeError, match="chainable target"):
        sample_target_chain(np.random.default_rng(0), 5, minimum_step_m=10.0)


def test_a_zero_length_chain_is_empty():
    assert sample_target_chain(np.random.default_rng(0), 0) == ()


@pytest.mark.parametrize("count, step", [(-1, 0.1), (5, -0.1)])
def test_invalid_arguments_are_refused(count, step):
    with pytest.raises(ValueError):
        sample_target_chain(np.random.default_rng(0), count, minimum_step_m=step)
