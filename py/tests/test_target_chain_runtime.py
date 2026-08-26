# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""An unattended run must learn its targets are unreachable before the arm moves."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pick_and_place.runtime.target_chain import (
    is_chainable_target,
    pin_episode_target,
    load_target_chain,
    resolve_target_chain,
    write_target_chain,
)
from pick_and_place.scripted.scenario_sampling import MIN_CHAIN_STEP_M, sample_target_chain


def _chain(count=5, seed=0):
    return sample_target_chain(np.random.default_rng(seed), count)


def test_no_chain_flags_means_the_run_localizes_a_plate_as_before():
    assert resolve_target_chain(chain_seed=None, sequence_path=None, episodes=10) is None


def test_a_seeded_chain_is_drawn_to_the_episode_budget():
    chain = resolve_target_chain(chain_seed=3, sequence_path=None, episodes=100)

    assert len(chain) == 100
    assert not chain.endless
    for target in chain.drawn:
        assert is_chainable_target(target.x, target.y)


def test_a_seeded_chain_repeats_so_a_run_can_be_repeated():
    first = resolve_target_chain(chain_seed=11, sequence_path=None, episodes=8)
    second = resolve_target_chain(chain_seed=11, sequence_path=None, episodes=8)

    assert [(t.x, t.y) for t in first.drawn] == [(t.x, t.y) for t in second.drawn]


def test_a_written_chain_round_trips(tmp_path):
    path = tmp_path / "targets.json"
    original = _chain(12)
    write_target_chain(path, original)

    loaded = load_target_chain(path)

    assert len(loaded) == len(original)
    for saved, read in zip(original, loaded):
        assert read.x == pytest.approx(saved.x)
        assert read.y == pytest.approx(saved.y)


def test_a_target_that_cannot_be_picked_up_from_is_refused_by_index(tmp_path):
    """The failure this whole module exists for, found before the arm moves."""
    path = tmp_path / "targets.json"
    good = _chain(3)
    payload = [[t.x, t.y] for t in good]
    payload.insert(2, [0.0, 0.0])  # the pan axis: placeable, never pickable
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry 2 .* is not a chainable target"):
        load_target_chain(path)


def test_a_run_cannot_walk_off_the_end_of_its_own_targets(tmp_path):
    path = tmp_path / "targets.json"
    write_target_chain(path, _chain(5))

    with pytest.raises(ValueError, match="holds 5 targets but the run asks for 20"):
        resolve_target_chain(chain_seed=None, sequence_path=path, episodes=20)


def test_a_continuous_run_draws_its_targets_as_it_goes():
    """The endless recording loop: no budget, so nothing can be drawn up front."""
    chain = resolve_target_chain(chain_seed=1, sequence_path=None, episodes=0)

    assert chain.endless
    assert len(chain) == 0
    seen = [chain.target_for(index) for index in range(1, 51)]
    assert len(chain) == 50
    for target in seen:
        assert is_chainable_target(target.x, target.y)


def test_an_endless_chain_keeps_the_step_constraint_across_its_draws():
    """Drawn one at a time, but indistinguishable from one drawn at once."""
    chain = resolve_target_chain(chain_seed=5, sequence_path=None, episodes=0)
    seen = [chain.target_for(index) for index in range(1, 41)]

    for previous, following in zip(seen, seen[1:]):
        assert math.hypot(following.x - previous.x, following.y - previous.y) >= MIN_CHAIN_STEP_M


def test_an_endless_chain_hands_back_the_same_target_for_the_same_episode():
    chain = resolve_target_chain(chain_seed=9, sequence_path=None, episodes=0)

    third = chain.target_for(3)
    chain.target_for(10)

    assert chain.target_for(3) is third


def test_a_finite_sequence_cannot_drive_an_endless_run(tmp_path):
    path = tmp_path / "targets.json"
    write_target_chain(path, _chain(5))

    with pytest.raises(ValueError, match="no episode budget"):
        resolve_target_chain(chain_seed=None, sequence_path=path, episodes=0)


def test_a_bounded_chain_refuses_to_run_past_its_end():
    chain = resolve_target_chain(chain_seed=2, sequence_path=None, episodes=4)

    with pytest.raises(IndexError):
        chain.target_for(5)


def test_both_chain_sources_at_once_is_refused(tmp_path):
    path = tmp_path / "targets.json"
    write_target_chain(path, _chain(5))

    with pytest.raises(ValueError, match="not both"):
        resolve_target_chain(chain_seed=1, sequence_path=path, episodes=5)


@pytest.mark.parametrize("payload", [[], {}, [[0.1]], [["a", "b"]]])
def test_a_malformed_sequence_file_is_refused(tmp_path, payload):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_target_chain(path)


class _Controller:
    drop_target_xy = None


def test_pinning_points_the_controller_at_the_right_target():
    chain = resolve_target_chain(chain_seed=4, sequence_path=None, episodes=5)
    controller = _Controller()

    line = pin_episode_target(controller, chain, 3)

    assert controller.drop_target_xy == (chain.target_for(3).x, chain.target_for(3).y)
    assert "3/5" in line


def test_pinning_an_endless_chain_says_so_rather_than_naming_a_total():
    chain = resolve_target_chain(chain_seed=4, sequence_path=None, episodes=0)

    assert "7/endless" in pin_episode_target(_Controller(), chain, 7)


@pytest.mark.parametrize("index", [0, 6])
def test_pinning_outside_a_bounded_chain_is_refused(index):
    chain = resolve_target_chain(chain_seed=4, sequence_path=None, episodes=5)

    with pytest.raises(IndexError):
        pin_episode_target(_Controller(), chain, index)
