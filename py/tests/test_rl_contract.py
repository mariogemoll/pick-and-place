# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place import transforms as tf
from pick_and_place.geometry import CubePose
from pick_and_place.rl import contract


def test_layout_constants_partition_the_observation():
    assert contract.OBS_DIM == 31
    assert contract.ACT_DIM == 6
    assert contract.POSE_DIM == 9
    # Slices and the confidence index tile [0, OBS_DIM) with no gap or overlap.
    covered = (
        list(range(*contract.JOINT_POS.indices(contract.OBS_DIM)))
        + list(range(*contract.JOINT_VEL.indices(contract.OBS_DIM)))
        + list(range(*contract.CUBE_POSE.indices(contract.OBS_DIM)))
        + list(range(*contract.TARGET_POSE.indices(contract.OBS_DIM)))
        + [contract.CONFIDENCE]
    )
    assert covered == list(range(contract.OBS_DIM))


def _random_rotations(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    angles = rng.uniform(-np.pi, np.pi, size=(n, 3))
    return [tf.rotation_zyx(r, p, y)[:3, :3] for r, p, y in angles]


def test_rotation_6d_round_trips():
    for r in _random_rotations(20):
        recovered = contract.rotation_from_6d(contract.rotation_to_6d(r))
        np.testing.assert_allclose(recovered, r, atol=1e-12)


def test_rotation_from_6d_orthonormalises_arbitrary_input():
    rng = np.random.default_rng(1)
    for _ in range(20):
        rec = contract.rotation_from_6d(rng.normal(size=6))
        np.testing.assert_allclose(rec @ rec.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rec), 1.0)


def test_pose_vec_from_xyz_yaw_encodes_position_and_pure_yaw():
    vec = contract.pose_vec_from_xyz_yaw(0.1, -0.2, 0.015, 0.7)
    np.testing.assert_allclose(vec[0:3], (0.1, -0.2, 0.015))
    np.testing.assert_allclose(
        contract.rotation_from_6d(vec[3:9]), tf.rot_z(0.7)[:3, :3], atol=1e-12
    )


def test_pose_vec_from_cube_pose_round_trips_full_rotation():
    pose = CubePose(x=0.3, y=0.05, z=0.015, roll=0.2, pitch=-0.1, yaw=1.3)
    vec = contract.pose_vec_from_cube_pose(pose)
    np.testing.assert_allclose(vec[0:3], (0.3, 0.05, 0.015))
    expected = tf.rotation_zyx(0.2, -0.1, 1.3)[:3, :3]
    np.testing.assert_allclose(contract.rotation_from_6d(vec[3:9]), expected, atol=1e-12)


def test_build_observation_places_each_field():
    jp = np.arange(6, dtype=np.float64)
    jv = np.arange(6, 12, dtype=np.float64)
    cube = contract.pose_vec_from_xyz_yaw(0.3, 0.0, 0.015, 0.0)
    target = contract.pose_vec_from_xyz_yaw(0.2, 0.1, 0.015, 1.0)

    obs = contract.build_observation(jp, jv, cube, target, confidence=0.4)

    assert obs.shape == (contract.OBS_DIM,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[contract.JOINT_POS], jp)
    np.testing.assert_allclose(obs[contract.JOINT_VEL], jv)
    np.testing.assert_allclose(obs[contract.CUBE_POSE], cube, atol=1e-6)
    np.testing.assert_allclose(obs[contract.TARGET_POSE], target, atol=1e-6)
    assert obs[contract.CONFIDENCE] == pytest.approx(0.4)


def test_build_observation_confidence_defaults_to_one():
    zeros = np.zeros(contract.POSE_DIM)
    obs = contract.build_observation(np.zeros(6), np.zeros(6), zeros, zeros)
    assert obs[contract.CONFIDENCE] == pytest.approx(1.0)


def test_clamp_setpoint_bounds_per_step_change():
    prev = np.zeros(6)
    target = np.array([1.0, -1.0, 0.01, -0.02, 0.5, -0.5])
    clamped = contract.clamp_setpoint(prev, target)
    # Big jumps are pinned to ±MAX_DELTA; small moves pass through untouched.
    expected = np.clip(target, -contract.MAX_DELTA, contract.MAX_DELTA)
    np.testing.assert_allclose(clamped, expected)


def test_clamp_setpoint_is_relative_to_previous():
    prev = np.full(6, 0.5)
    target = np.full(6, 0.52)  # +0.02 < MAX_DELTA, so unclamped
    np.testing.assert_allclose(contract.clamp_setpoint(prev, target), target)
    far = np.full(6, 1.0)  # +0.5 > MAX_DELTA, so clamped to prev + MAX_DELTA
    np.testing.assert_allclose(
        contract.clamp_setpoint(prev, far), prev + contract.MAX_DELTA
    )
