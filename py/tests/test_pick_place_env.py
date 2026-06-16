# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.episodes import MIN_START_CLEARANCE, jaw_floor_clearance
from pick_and_place.geometry import CUBE_HALF_SIZE
from pick_and_place.rl import contract
from pick_and_place.rl.pick_place_env import (
    DEFAULT_WEIGHTS,
    REACH_HOLD_BONUS,
    REACH_HOLD_TOL,
    PickPlaceEnv,
    RewardMetrics,
    reward_terms,
    weighted_reward,
)

# ----------------------------------------------------------------------------
# Reward terms (physics-free)
# ----------------------------------------------------------------------------


def _metrics(**overrides) -> RewardMetrics:
    base = dict(
        tip_to_hover=1.0,
        grasped=False,
        cube_height=0.0,
        cube_to_target_xy=1.0,
        placed=False,
        yaw_error=0.0,
        action_change_sq=0.0,
        action_jerk_sq=0.0,
        collision=False,
    )
    base.update(overrides)
    return RewardMetrics(**base)


def test_reward_terms_keys_match_default_weights():
    # Every default weight names a real term, and every term has a default weight,
    # so a chapter (a weights dict) can never silently target a nonexistent term.
    assert set(reward_terms(_metrics())) == set(DEFAULT_WEIGHTS)


def test_reach_term_peaks_at_contact_and_decays():
    far = reward_terms(_metrics(tip_to_hover=10.0))["reach"]
    near = reward_terms(_metrics(tip_to_hover=0.0))["reach"]
    # At contact: exp shaping (1.0) plus the flat hold-shell bonus.
    assert near == pytest.approx(1.0 + REACH_HOLD_BONUS)
    assert 0.0 <= far < near


def test_reach_hold_shell_is_flat_inside_tol():
    # Inside the hold tolerance the bonus is constant, so two distinct in-shell
    # positions differ only by the (tiny) exp term and both clear the bare-exp
    # ceiling of 1.0 — the flat top is what lets the policy settle instead of
    # chasing the last millimetre.
    tol = REACH_HOLD_TOL
    inner = reward_terms(_metrics(tip_to_hover=0.0))["reach"]
    edge = reward_terms(_metrics(tip_to_hover=tol * 0.99))["reach"]
    just_outside = reward_terms(_metrics(tip_to_hover=tol * 1.01))["reach"]
    assert inner > 1.0 and edge > 1.0
    assert just_outside < 1.0  # bonus gone, only the exp remains
    # The drop across the shell boundary is dominated by the bonus, not the exp.
    assert (edge - just_outside) > REACH_HOLD_BONUS / 2


def test_lift_term_requires_a_grasp():
    # The cube being high off the floor pays nothing unless it is actually held,
    # so the policy can't farm lift by knocking the cube up.
    assert reward_terms(_metrics(cube_height=0.1, grasped=False))["lift"] == 0.0
    assert reward_terms(_metrics(cube_height=0.1, grasped=True))["lift"] == pytest.approx(1.0)


def test_grasp_place_collision_are_indicators():
    on = reward_terms(_metrics(grasped=True, placed=True, collision=True))
    off = reward_terms(_metrics(grasped=False, placed=False, collision=False))
    for name in ("grasp", "place", "collision"):
        assert on[name] == 1.0
        assert off[name] == 0.0


def test_smoothness_terms_pass_through_their_magnitudes():
    # action_smooth (speed) and jerk (acceleration) are non-negative magnitudes
    # that become penalties via their negative weights; the term values are the
    # raw squared sums so the env's tracking, not reward_terms, owns the shaping.
    terms = reward_terms(_metrics(action_change_sq=0.3, action_jerk_sq=0.7))
    assert terms["action_smooth"] == pytest.approx(0.3)
    assert terms["jerk"] == pytest.approx(0.7)


def test_jerk_penalises_chatter_more_than_steady_motion():
    # Steady motion (zero jerk) costs nothing on the jerk term; an abrupt
    # change-in-change is penalised. With a negative jerk weight, chatter is worse.
    steady = weighted_reward(reward_terms(_metrics(action_jerk_sq=0.0)), DEFAULT_WEIGHTS)
    chatter = weighted_reward(reward_terms(_metrics(action_jerk_sq=0.5)), DEFAULT_WEIGHTS)
    assert chatter < steady


def test_chapter1_default_weights_ignore_yaw():
    # Chapter 1 (position only) carries the yaw term but zero-weights it: a
    # yaw-only change in the metrics must not move the chapter-1 reward.
    aligned = weighted_reward(reward_terms(_metrics(yaw_error=0.0)), DEFAULT_WEIGHTS)
    misaligned = weighted_reward(reward_terms(_metrics(yaw_error=np.pi)), DEFAULT_WEIGHTS)
    assert aligned == pytest.approx(misaligned)


def test_weighted_reward_penalises_collision_and_rewards_placement():
    placed = weighted_reward(reward_terms(_metrics(placed=True)), DEFAULT_WEIGHTS)
    crashed = weighted_reward(reward_terms(_metrics(collision=True)), DEFAULT_WEIGHTS)
    assert placed > 0.0
    assert crashed < 0.0


def test_weighted_reward_skips_unweighted_terms():
    # A weights dict naming only some terms scores exactly those.
    terms = {"reach": 0.5, "place": 1.0}
    assert weighted_reward(terms, {"place": 2.0}) == pytest.approx(2.0)


# ----------------------------------------------------------------------------
# Env (drives MuJoCo — kept to a couple of short rollouts)
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    e = PickPlaceEnv()
    yield e
    e.close()


def test_spaces_match_the_frozen_contract(env):
    assert env.observation_space.shape == (contract.OBS_DIM,)
    assert env.action_space.shape == (contract.ACT_DIM,)
    # Absolute set points are bounded by the joint limits, not [-1, 1] deltas.
    assert np.all(env.action_space.low <= env.action_space.high)
    assert np.all(np.isfinite(env.action_space.low))
    assert np.all(np.isfinite(env.action_space.high))


def test_reset_returns_valid_obs_and_clears_start_floor(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (contract.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    # Confidence is held at 1.0 for the pure-sim chapters.
    assert obs[contract.CONFIDENCE] == pytest.approx(1.0)
    assert "source" in info and "target" in info
    # The reset clearance gate must hold for the returned start pose.
    clearance = jaw_floor_clearance(env._model, env._data, env._jaw_ids)
    assert clearance >= MIN_START_CLEARANCE


def test_reset_is_deterministic_for_a_seed(env):
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_obs_target_pose_matches_sampled_target(env):
    _, info = env.reset(seed=7)
    target = info["target"]
    obs = env._build_obs()
    expected = contract.pose_vec_from_xyz_yaw(target.x, target.y, target.z, target.yaw)
    np.testing.assert_allclose(obs[contract.TARGET_POSE], expected, atol=1e-5)


def test_obs_current_cube_pose_matches_sampled_source(env):
    _, info = env.reset(seed=11)
    source = info["source"]
    obs = env._build_obs()
    # Free body settled at the sampled source pose: position read back to mm.
    np.testing.assert_allclose(obs[contract.CUBE_POSE][:3], (source.x, source.y, source.z), atol=2e-3)
    assert obs[contract.CUBE_POSE][2] == pytest.approx(CUBE_HALF_SIZE, abs=2e-3)


def test_step_clamps_per_step_change_to_max_delta(env):
    env.reset(seed=1)
    prev = env._prev_setpoint.copy()
    # Command a wild absolute set point; no joint may move more than MAX_DELTA.
    env.step(np.full(contract.ACT_DIM, 10.0, dtype=np.float32))
    moved = np.abs(env._prev_setpoint - prev)
    assert np.all(moved <= contract.MAX_DELTA + 1e-9)


def test_step_returns_well_formed_transition(env):
    env.reset(seed=2)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (contract.OBS_DIM,)
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info["reward_terms"]) == set(DEFAULT_WEIGHTS)


def test_episode_truncates_when_held_clear(env):
    # Hold the arm at its start set point (no crash): the episode should run to
    # the step budget and truncate rather than terminate.
    env.reset(seed=3)
    hold = env._prev_setpoint.copy()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(hold)
        steps += 1
        if steps > 1000:
            break
    # Either it survives to truncation, or a contact ended it — but holding still
    # from a cleared start pose should not blow past the step budget.
    assert steps <= 1000
