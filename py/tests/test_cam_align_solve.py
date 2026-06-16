# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.cam_align_solve import (
    NominalDelta,
    SolveResult,
    average_results,
    mat_to_quat_wxyz,
    opencv_camera_pose_to_mujoco_parent_pose,
    quat_angle_deg,
)


def test_mat_to_quat_wxyz_uses_mujoco_order_and_canonical_sign():
    np.testing.assert_allclose(mat_to_quat_wxyz(np.eye(3)), (1.0, 0.0, 0.0, 0.0))


def test_quat_angle_deg_uses_shortest_arc():
    assert quat_angle_deg(np.array((1.0, 0.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0, 0.0))) == 0.0


def test_opencv_camera_pose_to_mujoco_parent_pose_converts_axes():
    camera_center_world = np.array((1.0, 2.0, 3.0))
    rotation_camera_world = np.diag((1.0, -1.0, -1.0))
    translation_camera_world = -rotation_camera_world @ camera_center_world

    pos, quat = opencv_camera_pose_to_mujoco_parent_pose(
        rotation_camera_world,
        translation_camera_world,
        parent_rotation_world=np.eye(3),
        parent_position_world=np.zeros(3),
    )

    np.testing.assert_allclose(pos, camera_center_world)
    np.testing.assert_allclose(quat, (1.0, 0.0, 0.0, 0.0))


def test_average_results_aligns_quaternion_hemispheres():
    results = [
        SolveResult(
            used_tags=(12, 13, 14, 15),
            reprojection_error_px=1.0,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            nominal_delta=NominalDelta(0.0, 0.0),
        ),
        SolveResult(
            used_tags=(12, 13, 14, 15),
            reprojection_error_px=3.0,
            pos=(0.002, 0.004, 0.006),
            quat=(-1.0, 0.0, 0.0, 0.0),
            nominal_delta=NominalDelta(0.0, 0.0),
        ),
    ]

    averaged = average_results(
        results,
        nominal_pos=np.zeros(3),
        nominal_quat=np.array((1.0, 0.0, 0.0, 0.0)),
    )

    assert averaged.reprojection_error_px == 2.0
    np.testing.assert_allclose(averaged.pos, (0.001, 0.002, 0.003))
    np.testing.assert_allclose(averaged.quat, (1.0, 0.0, 0.0, 0.0))
