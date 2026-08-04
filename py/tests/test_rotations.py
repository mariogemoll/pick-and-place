# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.core.rotations import (
    average_quaternions_wxyz,
    mat_to_quat_wxyz,
    pose_delta_mm_deg,
    quat_angle_deg,
)


def test_mat_to_quat_wxyz_uses_mujoco_order_and_canonical_sign():
    np.testing.assert_allclose(mat_to_quat_wxyz(np.eye(3)), (1.0, 0.0, 0.0, 0.0))


def test_quat_angle_deg_uses_shortest_arc():
    assert quat_angle_deg(np.array((1.0, 0.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0, 0.0))) == 0.0


def test_average_quaternions_wxyz_normalizes_the_hemisphere():
    quats = [np.array((1.0, 0.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0, 0.0))]
    np.testing.assert_allclose(average_quaternions_wxyz(quats), (1.0, 0.0, 0.0, 0.0))


def test_pose_delta_mm_deg_reports_millimetres_and_degrees():
    identity = np.array((1.0, 0.0, 0.0, 0.0))
    quarter_turn = np.array((np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)))
    mm, deg = pose_delta_mm_deg(np.zeros(3), identity, np.array((0.0, 0.0, 0.01)), quarter_turn)
    assert mm == 10.0
    assert deg == pytest.approx(45.0)
