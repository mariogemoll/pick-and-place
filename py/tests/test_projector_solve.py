# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json

import cv2
import numpy as np
import pytest

from pick_and_place.calibration.projector_solve import (
    apply_homography,
    camera_to_workspace,
    image_to_workspace_square,
    load_projector_calibration,
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


def test_image_bottom_edge_faces_the_arm():
    """The workspace is looked at with the robot at the bottom, so the map is turned."""
    matrix = image_to_workspace_square((400, 400), half_extent_m=0.230)

    corners = apply_homography(matrix, np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    assert corners[0] == pytest.approx([0.230, -0.230], abs=1e-7)  # top-left -> se
    assert corners[1] == pytest.approx([-0.230, -0.230], abs=1e-7)  # top-right -> sw
    assert corners[2] == pytest.approx([-0.230, 0.230], abs=1e-7)  # bottom-right -> nw
    assert corners[3] == pytest.approx([0.230, 0.230], abs=1e-7)  # bottom-left -> ne


def test_the_turn_is_a_rotation_not_a_mirror():
    """A mirrored map fits the square just as well and renders lettering backwards.

    The determinant must be **negative**, which is the giveaway that trips people
    up: an image's rows run downward while workspace y runs up, so the correct
    overhead map already flips one axis. Turning it by half a turn flips both and
    leaves the sign alone. A map that came out positive would be the mirrored
    one -- equally square, equally well fitted, and wrong.
    """
    turned = image_to_workspace_square((400, 300), half_extent_m=0.230)
    assert np.linalg.det(turned[:2, :2]) < 0


def test_image_to_workspace_square_preserves_aspect_by_default():
    """A 2:1 image must not be stretched to fill a 1:1 square."""
    matrix = image_to_workspace_square((800, 400), half_extent_m=0.230)
    corners = apply_homography(matrix, np.array([[0, 0], [800, 0], [800, 400]]))

    # Magnitudes only: the half turn reverses both axes, and which way they
    # run is what test_image_bottom_edge_faces_the_arm pins down.
    width = abs(corners[1][0] - corners[0][0])
    height = abs(corners[1][1] - corners[2][1])
    assert width == pytest.approx(0.460, abs=1e-7)  # fills the square across
    assert height == pytest.approx(0.230, abs=1e-7)  # letterboxed down
    assert width / height == pytest.approx(2.0)


def test_image_to_workspace_square_stretches_on_request():
    matrix = image_to_workspace_square((800, 400), half_extent_m=0.230, stretch=True)
    corners = apply_homography(matrix, np.array([[0, 0], [800, 0], [800, 400]]))

    assert abs(corners[1][1] - corners[2][1]) == pytest.approx(0.460, abs=1e-7)


def test_image_to_workspace_square_rejects_an_empty_image():
    with pytest.raises(ValueError, match="positive extent"):
        image_to_workspace_square((0, 400))


def test_load_projector_calibration_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="pap calibrate-projector"):
        load_projector_calibration(tmp_path / "absent.json")


def test_load_projector_calibration_reads_a_written_fit(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(
        json.dumps(
            {
                "homography": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                "projector_size": [1920, 1080],
                "solved": "2026-08-22",
                "rms_mm": 0.33,
            }
        )
    )

    calibration = load_projector_calibration(path)

    assert calibration.projector_size == (1920, 1080)
    assert calibration.solved == "2026-08-22"
    assert calibration.rms_mm == pytest.approx(0.33)
    assert calibration.projector_to_workspace.shape == (3, 3)


def test_workspace_to_projector_inverts_the_fit():
    matrix = np.array([[2.6e-4, 0.0, -0.25], [0.0, 2.4e-4, -0.13], [0.0, 0.0, 1.0]])
    inverse = workspace_to_projector(matrix)

    point = np.array([[900.0, 500.0]])
    there = apply_homography(matrix, point)
    assert apply_homography(inverse, there) == pytest.approx(point, abs=1e-6)
