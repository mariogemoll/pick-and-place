# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Recording a fumble and its recovery as one episode.

These run real physics with no renderer (``rig=None``), so they need no GL. They
are slow by unit-test standards and deliberately few: the cheap properties of the
perturbation draw are covered in ``test_grasp_perturbation.py``, and what is left
to pin here is the behaviour that only appears once an episode is played --
whether the grasp actually misses, whether the planner re-picks, and whether the
unperturbed path is left alone.
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pytest

from pick_and_place.core.grasp_perturbation import GraspPerturbation
from pick_and_place.scripted.scenario_sampling import sample_scene
from pick_and_place.runtime.episodes import EpisodeSamplingError, prepare_episode
from pick_and_place.rollout.sim import _HELD_MIN_Z_M, record_episode
from pick_and_place.sim.model import get_cube_pose
from pick_and_place.spec.robot import HARDWARE_SIMULATION_HZ

SEED_BASE = 6_000_000
# Measured to miss reliably; see the magnitude sweep in the recovery work. The
# cube's 15 mm half-width is a floor, not the answer, because the planner
# re-selects among grasp candidates when the believed pose moves and absorbs part
# of the offset.
MISSING_MAGNITUDE_M = 0.050


def _play(index: int, magnitude_m: float | None) -> tuple[str, str, object]:
    """Play scene ``index`` with an optional perturbation; return status, log, episode."""
    scene = sample_scene(np.random.default_rng(SEED_BASE + index))
    draw = (
        None
        if magnitude_m is None
        else GraspPerturbation.sample(
            np.random.default_rng(SEED_BASE + index), magnitude_m=magnitude_m
        )
    )
    episode = prepare_episode(
        np.random.default_rng(index),
        scene.source,
        scene.target,
        max_attempts=30,
        grasp_perturbation=draw,
    )
    # The recorder's own model builder does this; prepare_episode leaves MuJoCo's
    # 0.002 s default, which cannot divide the 30 Hz control period.
    episode.model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = record_episode(episode, verbose=True)
    return result.status, buffer.getvalue(), episode


def _placement_error_m(episode) -> float:
    cube = get_cube_pose(episode.model, episode.data)
    return math.hypot(cube.x - episode.target.x, cube.y - episode.target.y)


def test_perturbed_episode_misses_then_recovers_and_places():
    status, log, episode = _play(0, MISSING_MAGNITUDE_M)
    assert "Grasp missed" in log, "the perturbation should have caused a fumble"
    # completed_phase_name=None is the planner's whole-fresh-pick branch; seeing
    # it in the log is what distinguishes a recovery from carrying on blind.
    assert "Replanning remaining trajectory after None" in log
    assert status == "success"
    assert _placement_error_m(episode) < 0.04


def test_recovery_replans_against_the_true_cube_not_the_perturbed_belief():
    # If the retry inherited the belief error it would re-aim at the same empty
    # spot forever, so a successful placement is only possible once the source
    # has been refreshed from ground truth.
    _, log, episode = _play(0, MISSING_MAGNITUDE_M)
    offset = math.hypot(
        episode.believed_source.x - episode.source.x,
        episode.believed_source.y - episode.source.y,
    )
    assert offset == pytest.approx(MISSING_MAGNITUDE_M, abs=1e-6)
    assert "Grasp missed" in log
    assert _placement_error_m(episode) < 0.04


def test_unperturbed_episode_neither_misses_nor_replans():
    # The default path must stay pure feedforward: no perturbation means no
    # checkpoints, which is what keeps recovery episodes the only thing that
    # differs between the two dataset arms.
    status, log, episode = _play(0, None)
    assert status == "success"
    assert "Grasp missed" not in log
    assert "Replanning remaining trajectory" not in log
    assert _placement_error_m(episode) < 0.04


def test_held_threshold_matches_the_oracle_lift_milestone():
    # A local threshold here would drift from the definition the episode is
    # scored against.
    from pick_and_place.policies.policy_evaluation import TaskOracleConfig

    config = TaskOracleConfig()
    assert _HELD_MIN_Z_M == pytest.approx(
        config.resting_height_m + config.lift_clearance_m
    )


def test_unplannable_perturbation_raises_rather_than_recording_a_bad_episode():
    # A belief error can put the cube outside what the planner can reach. That
    # must fail loudly at preparation, not produce an episode to record.
    scene = sample_scene(np.random.default_rng(SEED_BASE))
    absurd = GraspPerturbation(dx_m=0.60, dy_m=0.60)
    with pytest.raises((EpisodeSamplingError, ValueError)):
        prepare_episode(
            np.random.default_rng(0),
            scene.source,
            scene.target,
            max_attempts=2,
            grasp_perturbation=absurd,
        )


def test_a_regrasp_after_a_repick_has_a_grasp_to_fall_back_on():
    """The crash that only a thousand-episode run finds.

    A fumbled pick clears the tracked grasp so the replanner searches fresh
    candidates. If the *next* descent then ends on a pose that admits no
    candidate on its locked face, the fallback used to be that cleared value —
    ``None`` — and rebuilding the grasp phase from it raised ``AttributeError``
    several minutes into an episode. The descent phase's own grasp is always
    there to stand on.
    """
    from pick_and_place.scripted.descent import regrasp_after_descent

    class _Unreachable:
        """A locked face and elbow no candidate can match."""

        face = "no-such-face"
        elbow = "no-such-elbow"
        grasp = "the grasp the jaws came down on"

    _, _, episode = _play(0, None)
    fallback = regrasp_after_descent(
        _Unreachable(),
        get_cube_pose(episode.model, episode.data),
        episode.kinematics,
        free_grasp=False,
        current=None,
    )

    assert fallback == "the grasp the jaws came down on"
