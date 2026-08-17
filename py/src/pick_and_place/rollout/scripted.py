# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Give the scripted controller the two things it cannot reach for itself.

:class:`~pick_and_place.scripted.policy.ScriptedPolicy` is drivable from images
and reported joints alone, which is what makes it comparable to a learned
policy — but preparing an episode means compiling a scene, and accepting a
candidate trajectory means running it under live physics. Neither belongs to a
controller, so both are injected, and this is where they come from.
"""

from __future__ import annotations

import functools
from typing import Any

from pick_and_place.runtime.episodes import prepare_episode
from pick_and_place.runtime.preflight import (
    PreflightDebug,
    preflight,
    preflight_collision_is_unexpected,
    print_preflight_debug,
)
from pick_and_place.scripted.policy import ScriptedPolicy, TrajectoryPreflight
from pick_and_place.sim.collisions import is_unexpected


def scene_preflight(debug: PreflightDebug = PreflightDebug()) -> TrajectoryPreflight:
    """Accept a candidate only if it runs clean under the episode's own physics."""

    def accepts(episode: Any, trajectory: Any) -> bool:
        args = (
            episode.model,
            trajectory,
            episode.actuator_id,
            episode.robot_geom_ids,
            episode.env_geom_ids,
        )
        if not debug.detailed:
            events = preflight(*args)
            return not any(is_unexpected(name1, name2) for _, name1, name2 in events)
        rejected = [
            event
            for event in preflight(*args, detailed=True)
            if preflight_collision_is_unexpected(event)
        ]
        if rejected and debug.print_contacts:
            print_preflight_debug(1, trajectory, rejected, limit=debug.contact_limit)
        return not rejected

    return accepts


def scripted_policy(
    localizer: Any,
    workspace_corners_world: Any,
    *,
    plan_episode: Any = None,
    trajectory_preflight: TrajectoryPreflight | None = None,
    debug: PreflightDebug = PreflightDebug(),
    **kwargs: Any,
) -> ScriptedPolicy:
    """The expert, wired to a real scene: it plans episodes and preflights them.

    ``debug`` decides what a rejected candidate reports, and is bound into both
    injections rather than passed through the controller: which diagnostics a
    preflight prints is a property of the physics pass, not of the expert.

    Both injections can be overridden, which is how a test drives the controller
    without a scene at all.
    """
    return ScriptedPolicy(
        localizer,
        workspace_corners_world,
        (
            functools.partial(prepare_episode, debug=debug)
            if plan_episode is None
            else plan_episode
        ),
        scene_preflight(debug) if trajectory_preflight is None else trajectory_preflight,
        **kwargs,
    )
