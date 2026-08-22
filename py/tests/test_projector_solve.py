# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import cv2
import numpy as np
import pytest

from pick_and_place.calibration.projector_solve import (
    apply_homography,
    camera_to_workspace,
    solve_projector_to_workspace,
    workspace_plate_centers,
    workspace_to_projector,
)


def test_plate_centers_are_the_four_declared_corners():
    centers = workspace_plate_centers()
    assert sorted(centers) == [12, 13, 14, 15]
    assert centers[12] == (0.230, 0.230)
    assert centers[14] == (-0.230, -0.230)


def test_camera_to_workspace_maps_the_plates_onto_their_declared_positions():
    # An arbitrary but invertible camera view of the four plates.
    detections = {12: (1400.0, 300.0), 13: (500.0, 320.0), 14: (480.0, 900.0), 15: (1420.0, 880.0)}

    matrix = camera_to_workspace({k: np.array(v) for k, v in detections.items()})

    expected = workspace_plate_centers()
    for tag_id, pixel in detections.items():
        mapped = apply_homography(matrix, np.array([pixel]))[0]
        # 1e-7 m is a ten-thousandth of a millimetre; the solve is float64.
        assert mapped == pytest.approx(expected[tag_id], abs=1e-7)


def test_camera_to_workspace_demands_all_four_plates():
    detections = {12: np.array((1400.0, 300.0)), 13: np.array((500.0, 320.0))}
    with pytest.raises(ValueError, match=r"corner plates \[14, 15\]"):
        camera_to_workspace(detections)


def test_solve_recovers_a_known_projector_homography():
    """The whole chain, end to end, against a homography we chose in advance."""
    rng = np.random.default_rng(0)

    # A camera that sees the workspace through some perspective.
    plates = {12: (1400.0, 300.0), 13: (500.0, 320.0), 14: (480.0, 900.0), 15: (1420.0, 880.0)}
    cam_to_ws = camera_to_workspace({k: np.array(v) for k, v in plates.items()})
    ws_to_cam = np.linalg.inv(cam_to_ws)

    # The truth we are trying to recover: projector pixels onto the floor. This
    # one throws the 1920x1080 frame across roughly the workspace, tilted.
    truth = np.array(
        [
            [2.6e-4, 1.9e-5, -0.25],
            [1.1e-5, 2.4e-4, -0.14],
            [2.0e-6, 9.0e-7, 1.0],
        ]
    )

    projector_xy = np.stack(
        [rng.uniform(0, 1920, 400), rng.uniform(0, 1080, 400)], axis=1
    )
    workspace_xy = apply_homography(truth, projector_xy)
    camera_xy = apply_homography(ws_to_cam, workspace_xy)

    fit = solve_projector_to_workspace(projector_xy, camera_xy, cam_to_ws, cv2_module=cv2)

    assert fit.point_count > 350
    assert fit.rms_mm < 0.01
    # Recovered up to scale, so compare the maps rather than the matrices.
    probe = np.array([[0.0, 0.0], [1920.0, 0.0], [960.0, 540.0], [1920.0, 1080.0]])
    assert apply_homography(fit.matrix, probe) == pytest.approx(
        apply_homography(truth, probe), abs=1e-6
    )


def test_solve_survives_outliers_from_shadow_edges():
    rng = np.random.default_rng(1)
    plates = {12: (1400.0, 300.0), 13: (500.0, 320.0), 14: (480.0, 900.0), 15: (1420.0, 880.0)}
    cam_to_ws = camera_to_workspace({k: np.array(v) for k, v in plates.items()})
    ws_to_cam = np.linalg.inv(cam_to_ws)
    truth = np.array([[2.6e-4, 0.0, -0.25], [0.0, 2.4e-4, -0.13], [0.0, 0.0, 1.0]])

    projector_xy = np.stack([rng.uniform(0, 1920, 300), rng.uniform(0, 1080, 300)], axis=1)
    camera_xy = apply_homography(ws_to_cam, apply_homography(truth, projector_xy))
    # A tenth of the cells decode to a centroid pulled well off the lit part.
    camera_xy[::10] += rng.uniform(-60, 60, size=(len(camera_xy[::10]), 2))

    fit = solve_projector_to_workspace(projector_xy, camera_xy, cam_to_ws, cv2_module=cv2)

    assert fit.rms_mm < 1.0
    assert fit.point_count < 300  # the outliers were rejected, not fitted


def test_solve_needs_four_points():
    cam_to_ws = np.eye(3)
    with pytest.raises(ValueError, match="at least 4"):
        solve_projector_to_workspace(
            np.zeros((3, 2)), np.zeros((3, 2)), cam_to_ws, cv2_module=cv2
        )


def test_workspace_to_projector_inverts_the_fit():
    matrix = np.array([[2.6e-4, 0.0, -0.25], [0.0, 2.4e-4, -0.13], [0.0, 0.0, 1.0]])
    inverse = workspace_to_projector(matrix)

    point = np.array([[900.0, 500.0]])
    there = apply_homography(matrix, point)
    assert apply_homography(inverse, there) == pytest.approx(point, abs=1e-6)
