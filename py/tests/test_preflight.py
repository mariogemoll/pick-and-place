# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Vetting a trajectory under physics, and the record it reports contacts in."""

import dataclasses

import numpy as np
import pytest

from pick_and_place.rollout.scripted import scene_preflight
from pick_and_place.runtime.episodes import prepare_episode
from pick_and_place.runtime.preflight import (
    PreflightCollision,
    PreflightDebug,
    preflight,
    preflight_collision_is_unexpected,
)
from pick_and_place.sim.collisions import is_unexpected


def _event() -> PreflightCollision:
    return PreflightCollision(
        time=0.5,
        phase="descent",
        phase_time=0.1,
        geom1="fixed_jaw_col_0",
        geom2="floor",
        body1="gripper",
        body2="world",
        dist=-0.002,
        position=(0.1, 0.2, 0.0),
    )


def test_a_collision_is_a_frozen_record_built_by_keyword() -> None:
    """Every field is named at the one call site that builds these, inside the
    stepping loop, so the record has to accept them and stay immutable after."""
    event = _event()

    assert event.geom1 == "fixed_jaw_col_0"
    assert dataclasses.is_dataclass(event)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.time = 1.0  # type: ignore[misc]


def test_a_prepared_episode_reports_only_the_grasp_in_both_modes() -> None:
    """Both modes report every contact, the grasp included; ``prepare_episode``
    only returns a trajectory whose contacts are all expected ones. The detailed
    mode reports the same contacts as records rather than tuples."""
    episode = prepare_episode(np.random.default_rng(0), max_attempts=40)
    args = (
        episode.model,
        episode.trajectory,
        episode.actuator_id,
        episode.robot_geom_ids,
        episode.env_geom_ids,
    )

    plain = preflight(*args)
    detailed = preflight(*args, detailed=True)

    assert plain, "a successful pick still touches the cube with both jaws"
    assert len(detailed) == len(plain)
    assert [event for _, *event in plain if is_unexpected(*event)] == []
    assert [event for event in detailed if preflight_collision_is_unexpected(event)] == []


@pytest.mark.parametrize("debug", [PreflightDebug(), PreflightDebug(print_contacts=True)])
def test_the_injected_preflight_accepts_a_prepared_trajectory(debug: PreflightDebug) -> None:
    """The expert's default injection, called the way the controller calls it.

    Every controller test supplies its own stub, so nothing else drives this
    against a real scene — and it is the replan at a checkpoint, not the first
    plan, that reaches it, which puts the only caller mid-episode on hardware.
    Both diagnostic modes are covered because they take different call shapes.
    """
    episode = prepare_episode(np.random.default_rng(0), max_attempts=40)

    assert scene_preflight(debug)(episode, episode.trajectory)
