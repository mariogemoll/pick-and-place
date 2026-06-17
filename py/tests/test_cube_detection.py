# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Render the AprilTag cube at known poses and check the fusion recovers them."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import pick_and_place.cube_detection as cube_detection
from pick_and_place.cube_detection import (
    CUBE_TAG_IDS,
    CubeTracker,
    OrientationStabilizer,
    PoseEMA,
    cube_pose_to_world,
    detect_cube_faces,
    estimate_cube_pose,
    fuse_cube_faces,
    make_cube_detector,
)
from pick_and_place.environment import APRILTAG_TEXTURE_DIR
from pick_and_place.scene import PICK_CUBE_HALF_SIZE

# OpenCV camera (x right, y down, z forward) <- MuJoCo camera (x right, y up, z back).
_CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])
_RENDER_PX = 1000
_RENDER_FOVY_DEG = 45.0
_CUBE_FACE_PNG = (
    "fileright",
    "fileleft",
    "fileup",
    "filedown",
    "filefront",
    "fileback",
)


def _look_at_camera(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera rotation (columns x, y, z in world) looking at ``target``."""
    z_axis = position - target  # MuJoCo camera looks down -z, so +z points back.
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross((0.0, 0.0, 1.0), z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
    return quat


def _render_cube(
    cube_pos: np.ndarray,
    cube_quat_wxyz: np.ndarray,
    cam_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the textured cube and return ``(rgb_frame, camera_matrix)``."""
    cam_rotation = _look_at_camera(cam_pos, cube_pos)
    cam_quat = _mat_to_quat_wxyz(cam_rotation)
    half = PICK_CUBE_HALF_SIZE
    face_files = " ".join(
        f'{attr}="tagStandard41h12_{tag_id:05d}_30x30mm_tag20mm.png"'
        for attr, tag_id in zip(_CUBE_FACE_PNG, CUBE_TAG_IDS)
    )
    xml = f"""
    <mujoco>
      <compiler texturedir="{APRILTAG_TEXTURE_DIR}"/>
      <visual>
        <headlight diffuse="0.9 0.9 0.9" ambient="0.7 0.7 0.7" specular="0 0 0"/>
        <global offwidth="{_RENDER_PX}" offheight="{_RENDER_PX}" fovy="{_RENDER_FOVY_DEG}"/>
      </visual>
      <asset>
        <texture name="sky" type="skybox" builtin="flat" rgb1="0.3 0.3 0.3" rgb2="0.3 0.3 0.3" width="8" height="8"/>
        <texture name="cube_tags" type="cube" {face_files}/>
        <material name="cube_tags" texture="cube_tags"/>
      </asset>
      <worldbody>
        <camera name="probe" pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}"
                quat="{cam_quat[0]} {cam_quat[1]} {cam_quat[2]} {cam_quat[3]}"/>
        <body pos="{cube_pos[0]} {cube_pos[1]} {cube_pos[2]}"
              quat="{cube_quat_wxyz[0]} {cube_quat_wxyz[1]} {cube_quat_wxyz[2]} {cube_quat_wxyz[3]}">
          <geom name="cube" type="box" size="{half} {half} {half}" material="cube_tags"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=_RENDER_PX, width=_RENDER_PX)
    try:
        renderer.update_scene(data, camera="probe")
        frame = renderer.render()
    finally:
        renderer.close()

    focal = (_RENDER_PX / 2.0) / np.tan(np.radians(_RENDER_FOVY_DEG) / 2.0)
    camera_matrix = np.array(
        [[focal, 0, _RENDER_PX / 2.0], [0, focal, _RENDER_PX / 2.0], [0, 0, 1]],
        dtype=float,
    )
    return frame, camera_matrix


def _expected_camera_frame_pose(
    cube_pos: np.ndarray,
    cube_rotation: np.ndarray,
    cam_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth cube pose in the OpenCV camera frame."""
    cam_rotation = _look_at_camera(cam_pos, cube_pos)
    world_from_cam = cam_rotation.T
    rotation = _CV_FROM_MJ @ world_from_cam @ cube_rotation
    position = _CV_FROM_MJ @ world_from_cam @ (cube_pos - cam_pos)
    return rotation, position


def _geodesic_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos_angle = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation from intrinsic ZYX Euler angles (radians)."""
    quat = np.empty(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), "xyz")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


@pytest.mark.parametrize(
    "cube_rotation",
    [
        np.eye(3),
        _euler_zyx(np.radians(30.0), np.radians(15.0), 0.0),
    ],
)
def test_fused_cube_pose_recovers_known_pose(cube_rotation):
    cube_pos = np.array([0.0, 0.0, 0.0])
    cam_pos = np.array([0.07, 0.05, 0.09])  # oblique so three faces are visible
    cube_quat = _mat_to_quat_wxyz(cube_rotation)

    frame, camera_matrix = _render_cube(cube_pos, cube_quat, cam_pos)
    estimate = estimate_cube_pose(frame, make_cube_detector(), camera_matrix)

    assert estimate is not None
    assert estimate.num_faces_used >= 2  # an oblique view shows several faces
    assert estimate.reproj_px < 1.0

    expected_rotation, expected_position = _expected_camera_frame_pose(
        cube_pos, cube_rotation, cam_pos
    )
    position_error_m = float(np.linalg.norm(estimate.position - expected_position))
    rotation_error_deg = _geodesic_angle_deg(estimate.rotation, expected_rotation)

    assert position_error_m < 1e-3
    assert rotation_error_deg < 1.0

    # Mapping back through the (known) camera world pose recovers the planted
    # world pose: cube at world origin, so position ~ 0 and rotation ~ cube_rotation.
    cam_rotation = _look_at_camera(cam_pos, cube_pos)
    world_rotation, world_position = cube_pose_to_world(estimate, cam_pos, cam_rotation)
    assert float(np.linalg.norm(world_position - cube_pos)) < 1e-3
    assert _geodesic_angle_deg(world_rotation, cube_rotation) < 1.0


def test_single_face_orientation_is_not_flipped():
    # A near-top-down view shows essentially just the top face, so the planar PnP
    # has a two-fold flip ambiguity; the solve must pick the physical solution.
    cube_pos = np.array([0.0, 0.0, 0.0])
    cam_pos = np.array([0.006, 0.0, 0.12])  # ~3 deg off vertical: top face only
    cube_rotation = _euler_zyx(np.radians(25.0), 0.0, 0.0)

    frame, camera_matrix = _render_cube(cube_pos, _mat_to_quat_wxyz(cube_rotation), cam_pos)
    estimate = estimate_cube_pose(frame, make_cube_detector(), camera_matrix)

    assert estimate is not None
    assert estimate.num_faces_used == 1
    expected_rotation, _ = _expected_camera_frame_pose(cube_pos, cube_rotation, cam_pos)
    # A flipped solution would be ~150-180 deg off; the right one is within a degree.
    assert _geodesic_angle_deg(estimate.rotation, expected_rotation) < 2.0


def test_temporal_prior_resolves_single_face_flip():
    import cv2

    # A near-top-down single-face view: the planar PnP genuinely returns two
    # valid solutions, and the reprojection error barely tells them apart -- this
    # is the case that flips frame to frame on the live feed.
    cube_pos = np.array([0.0, 0.0, 0.0])
    cam_pos = np.array([0.006, 0.0, 0.12])
    cube_rotation = _euler_zyx(np.radians(25.0), 0.0, 0.0)
    frame, camera_matrix = _render_cube(cube_pos, _mat_to_quat_wxyz(cube_rotation), cam_pos)
    detections = detect_cube_faces(frame, make_cube_detector())
    assert len({d.tag_id for d in detections}) == 1

    object_points, image_points, ids = cube_detection._tag_correspondences(detections)
    ok, rvecs, _, _ = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, np.zeros(5), flags=cv2.SOLVEPNP_IPPE
    )
    candidates = [cv2.Rodrigues(rvec)[0] @ cube_detection._CUBE_F_TO_BODY for rvec in rvecs]
    assert len(candidates) == 2  # a real two-fold ambiguity to choose between

    # The prior steers the solve to whichever candidate it sits nearer -- so a
    # stable previous pose keeps the correct solution instead of flipping.
    other = {0: 1, 1: 0}
    for index, prior in enumerate(candidates):
        estimate = fuse_cube_faces(detections, camera_matrix, prior_rotation=prior)
        assert _geodesic_angle_deg(estimate.rotation, candidates[index]) < _geodesic_angle_deg(
            estimate.rotation, candidates[other[index]]
        )


def test_orientation_stabilizer_holds_flip_but_follows_motion():
    steady = np.eye(3)
    flipped = _euler_zyx(np.radians(180.0), 0.0, 0.0)  # far from steady
    stab = OrientationStabilizer(window=5, flip_deg=20.0, confirm=3)

    for _ in range(5):
        rotation, held = stab.update(steady)
        assert not held and np.allclose(rotation, steady)

    # One flipped frame is rejected: the held pose stays put.
    rotation, held = stab.update(flipped)
    assert held and np.allclose(rotation, steady)
    # ...and we snap right back when the next good frame arrives.
    rotation, held = stab.update(steady)
    assert not held and np.allclose(rotation, steady)

    # A *sustained* disagreement is real motion, accepted once it's confirmed.
    accepted = False
    for _ in range(stab.confirm):
        rotation, held = stab.update(flipped)
        accepted = not held
    assert accepted and np.allclose(rotation, flipped)
    assert stab.flip_rate > 0.0


def test_authoritative_frame_breaks_wrong_flip_lock():
    # The failure mode: single-face frames seed the consensus onto the mirror
    # (wrong) flip, and the stabilizer then holds the correct orientation off as
    # if *it* were the flip -- so it stays wrong forever on single-face evidence.
    correct = np.eye(3)
    wrong = _euler_zyx(np.radians(180.0), 0.0, 0.0)
    stab = OrientationStabilizer(window=5, flip_deg=20.0, confirm=3)

    for _ in range(5):  # lock onto the wrong flip via single-face frames
        stab.update(wrong)
    assert _geodesic_angle_deg(stab.reference(), wrong) < 1.0
    # A correct single-face frame can't escape: it's held as a "flip".
    held_rotation, held = stab.update(correct)
    assert held and np.allclose(held_rotation, wrong)

    # The first multi-face (authoritative) frame snaps the consensus to the truth.
    out, held = stab.update(correct, authoritative=True)
    assert not held and np.allclose(out, correct)
    assert _geodesic_angle_deg(stab.reference(), correct) < 1.0
    # ...and single-face frames now stay on the correct branch.
    out, held = stab.update(correct)
    assert not held and np.allclose(out, correct)


def test_pose_ema_passthrough_and_damping():
    rotation = np.eye(3)

    # alpha = 0 passes the new pose straight through.
    ema = PoseEMA(0.0)
    out_rotation, out_position = ema.update(rotation, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(out_position, [1.0, 2.0, 3.0])
    assert np.allclose(out_rotation, rotation)

    # A single outlier frame is pulled most of the way back toward the steady value.
    ema = PoseEMA(0.6)
    steady = np.array([0.10, 0.0, 0.015])
    for _ in range(20):
        ema.update(rotation, steady)
    jumped_rotation, jumped_position = ema.update(rotation, steady + np.array([0.05, 0.0, 0.0]))
    # The 50 mm spike is attenuated to well under half.
    assert float(np.linalg.norm(jumped_position - steady)) < 0.025
    assert _geodesic_angle_deg(jumped_rotation, rotation) < 1.0


def test_pose_ema_confidence_scales_pull():
    rotation = np.eye(3)
    steady = np.array([0.10, 0.0, 0.015])
    jump = steady + np.array([0.05, 0.0, 0.0])

    # Two EMAs with identical alpha, settled on the same value; one fed the jump
    # at full confidence, the other at a quarter. The low-confidence frame must
    # move the estimate strictly less.
    full = PoseEMA(0.6)
    weak = PoseEMA(0.6)
    for _ in range(20):
        full.update(rotation, steady)
        weak.update(rotation, steady)
    _, full_pos = full.update(rotation, jump, confidence=1.0)
    _, weak_pos = weak.update(rotation, jump, confidence=0.25)

    full_move = float(np.linalg.norm(full_pos - steady))
    weak_move = float(np.linalg.norm(weak_pos - steady))
    assert weak_move < full_move
    # confidence 0 freezes the estimate entirely.
    _, frozen_pos = PoseEMA(0.6).update(rotation, steady)  # seed
    frozen = PoseEMA(0.6)
    frozen.update(rotation, steady)
    _, held = frozen.update(rotation, jump, confidence=0.0)
    assert np.allclose(held, steady)


def test_cube_tracker_recovers_world_pose():
    cube_pos = np.array([0.0, 0.0, 0.0])
    cam_pos = np.array([0.07, 0.05, 0.09])
    cube_rotation = _euler_zyx(np.radians(30.0), np.radians(15.0), 0.0)
    frame, camera_matrix = _render_cube(cube_pos, _mat_to_quat_wxyz(cube_rotation), cam_pos)
    cam_rotation = _look_at_camera(cam_pos, cube_pos)

    tracker = CubeTracker(smooth=0.0)
    pose = None
    for _ in range(3):
        pose = tracker.update_frame(frame, camera_matrix, cam_pos, cam_rotation)

    assert pose is not None and not pose.held
    assert float(np.linalg.norm(pose.position - cube_pos)) < 1e-3
    assert _geodesic_angle_deg(pose.rotation, cube_rotation) < 1.0


def test_no_cube_tags_returns_none():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    camera_matrix = np.array([[100, 0, 32], [0, 100, 32], [0, 0, 1]], dtype=float)
    assert estimate_cube_pose(frame, make_cube_detector(), camera_matrix) is None
