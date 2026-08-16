# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""What one recorded episode is, before it runs.

Two things, and they are both about determinism.

**Which draws an episode index gets.** Every stream is keyed off the global
episode index rather than off a running counter, so which worker records which
episode does not change what gets recorded, and a run can be topped up later
without disturbing what it already has. The streams are also *salted apart*:
turning ``--perturbed-fraction`` or ``--physics-randomization`` up or down must
not move any other episode's cube, or the two arms of a comparison would differ
in their entire pose distribution and stop being paired.

**Whether the planner's belief is looked up or looked at.** With overhead
perception the plate has to be on the table before anything is localized, so
placing it is part of preparing the episode; without it the plate goes down
afterwards and its yaw is drawn last, so an episode index keeps the poses it had
before the plate started rotating.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.grasp_perturbation import GraspPerturbation
from pick_and_place.core.workspace_bounds import (
    PAN_AXIS,
    is_cube_drop_allowed,
    sample_target_plate_yaw,
)
from pick_and_place.rollout.localized_episode import prepare_localized_episode
from pick_and_place.runtime.episodes import Episode, prepare_episode
from pick_and_place.sim.domain_randomization import domain_seed
from pick_and_place.sim.paper_target_marker import place_paper_target_marker
from pick_and_place.spec.workspace import DROP_ZONE_HALF_SIZE

# Salts distinguishing each stream from the pose and domain streams keyed off
# the same (seed, episode) pair. Arbitrary, and must not change: they are part
# of what makes a recorded episode a pure function of its index.
PERTURBATION_SEED_SALT = 0x50455254
PHYSICS_SEED_SALT = 0x50485953
OVERHEAD_SEED_SALT = 0x4F565244


def episode_rng(root_seed: int | None, global_episode: int) -> np.random.Generator:
    """The deterministic pose stream for one globally numbered episode."""
    if root_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([root_seed, global_episode]))


def appearance_seed(root_seed: int | None, global_episode: int) -> int:
    """Stable per-episode seed for domain sampling, independent of pose draws."""
    return domain_seed(root_seed, global_episode)


def _salted_rng(root_seed: int | None, global_episode: int, salt: int) -> np.random.Generator:
    if root_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([root_seed, global_episode, salt]))


def perturbation_rng(root_seed: int | None, global_episode: int) -> np.random.Generator:
    """The stream deciding whether episode ``global_episode`` is deliberately fumbled."""
    return _salted_rng(root_seed, global_episode, PERTURBATION_SEED_SALT)


def overhead_rng(root_seed: int | None, global_episode: int) -> np.random.Generator:
    """The stream deciding how far episode ``global_episode``'s camera calibration is off."""
    return _salted_rng(root_seed, global_episode, OVERHEAD_SEED_SALT)


def physics_rng(root_seed: int | None, global_episode: int) -> np.random.Generator:
    """The stream deciding which arm episode ``global_episode`` is flown by."""
    return _salted_rng(root_seed, global_episode, PHYSICS_SEED_SALT)


def sample_grasp_perturbation(
    root_seed: int | None,
    global_episode: int,
    source: CubePose,
    *,
    fraction: float,
    magnitude_m: float,
    max_source_radius_m: float | None,
) -> GraspPerturbation | None:
    """Sample the episode's fumble, optionally excluding distant cube starts.

    The probability and direction retain the original salted stream and draw
    order. The source-radius gate only suppresses a selected perturbation, so
    every included episode gets exactly the perturbation it did before the gate
    existed and every episode keeps its original pose stream.
    """
    if fraction <= 0.0:
        return None
    perturb_rng = perturbation_rng(root_seed, global_episode)
    selected = perturb_rng.random() < fraction
    source_radius_m = math.hypot(source.x - PAN_AXIS[0], source.y - PAN_AXIS[1])
    within_radius = max_source_radius_m is None or source_radius_m <= max_source_radius_m
    if not selected or not within_radius:
        return None
    return GraspPerturbation.sample(perturb_rng, magnitude_m=magnitude_m)


def prepare_for_recording(
    rng: np.random.Generator,
    model: Any,
    data: Any,
    perception: Any,
    *,
    source: CubePose | None,
    target: CubePose | None,
    miscalibration: Any,
    grasp_perturbation: GraspPerturbation | None,
    max_attempts: int,
) -> tuple[Episode, float]:
    """Prepare one episode, and place the drop plate it will be recorded against."""
    if perception is not None:
        localized = prepare_localized_episode(
            rng,
            model,
            data,
            perception,
            source=source,
            target=target,
            include_environment=True,
            miscalibration=miscalibration,
            grasp_perturbation=grasp_perturbation,
            max_attempts=max_attempts,
        )
        return localized.episode, localized.target_plate_yaw

    episode = prepare_episode(
        rng,
        source,
        target,
        model=model,
        data=data,
        verbose=False,
        include_environment=True,
        miscalibration=miscalibration,
        grasp_perturbation=grasp_perturbation,
        max_attempts=max_attempts,
    )
    plate_yaw = sample_target_plate_yaw(
        rng, episode.target.x, episode.target.y, half_size=DROP_ZONE_HALF_SIZE
    )
    place_paper_target_marker(
        model,
        (episode.target.x, episode.target.y),
        plate_yaw,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=is_cube_drop_allowed(episode.target.x, episode.target.y),
        alpha=1.0,
    )
    return episode, plate_yaw
