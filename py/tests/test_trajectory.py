# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import numpy as np

import pytest

import pick_and_place.runtime.episodes as episodes
from pick_and_place.scripted.episode_sampling import sample_recovery_cube
from pick_and_place.runtime.episodes import EpisodeSamplingError, prepare_episode
from pick_and_place.sim.model import build_model, placement_error, set_cube_pose
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.core.geometry import CANONICAL_PREGRASP_DISTANCE, CubePose
from pick_and_place.core.ik import solve_simple_grasp_ik
from pick_and_place.sim.derive_kinematics import derive_kinematics
from pick_and_place.scripted.carry import DROP_CUBE_CENTER_Z, plan_carry_candidates
from pick_and_place.scripted.grasp import grasp_candidates
from pick_and_place.scripted.trajectory import (
    GRASP_CLOSE_DURATION,
    GRASP_SETTLE_DURATION,
    GraspPhase,
)
from pick_and_place.spec.robot import GRIPPER_GRASP, GRIPPER_OPEN
from pick_and_place.core.workspace_bounds import (
    RECOVERY_TARGET_FRAME_BORDER_MARGIN,
    WORKSPACE_FRAME_INNER_HALF_EXTENT,
    world_to_frame_xy,
    is_cube_pickup_allowed,
    is_cube_recovery_target_allowed,
)


def test_free_drop_plans_reachable_joint_carry_to_low_release():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    target = CubePose(x=0.20, y=-0.05, z=CUBE_HALF_SIZE)
    model, _ = build_model(source)
    kinematics = derive_kinematics(model)
    grasp = next(grasp_candidates(kinematics, source))

    carries = list(plan_carry_candidates(kinematics, grasp, target))

    assert carries
    assert all(carry.drop_position[2] == DROP_CUBE_CENTER_Z for carry in carries)
    first = carries[0]
    assert first.mode == "joint"
    assert first.grasp_joints == grasp.lift_joints
    assert set(first.cruise_joints) == set(grasp.lift_joints)
    assert set(first.drop_joints) == set(grasp.lift_joints)
    assert solve_simple_grasp_ik(kinematics, first.drop_matrix)


def test_grasp_choice_exposes_distillation_metadata():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    model, _ = build_model(source)
    kinematics = derive_kinematics(model)

    grasp = next(grasp_candidates(kinematics, source))

    assert grasp.face in {"+x", "-x", "+y", "-y"}
    assert grasp.elbow in {"up", "down"}
    assert 0.0 <= grasp.pitch <= np.pi
    assert abs(grasp.roll_offset) <= np.pi / 4.0
    assert np.isfinite(grasp.closing_azimuth)
    assert np.isfinite(grasp.camera_outward)


def test_canonical_pregrasp_stands_further_back_from_cube():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    model, _ = build_model(source)
    kinematics = derive_kinematics(model)

    grasp = next(grasp_candidates(kinematics, source))

    distance = np.linalg.norm(grasp.hover_matrix[:3, 3] - grasp.grasp_matrix[:3, 3])

    assert CANONICAL_PREGRASP_DISTANCE == 0.045
    assert distance == pytest.approx(CANONICAL_PREGRASP_DISTANCE)


def test_grasp_phase_waits_at_aligned_pose_before_closing():
    joints = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
    }
    phase = GraspPhase(joints)

    assert phase.duration == pytest.approx(GRASP_SETTLE_DURATION + GRASP_CLOSE_DURATION)
    assert phase.evaluate(GRASP_SETTLE_DURATION * 0.5).gripper == pytest.approx(GRIPPER_OPEN)
    assert phase.evaluate(GRASP_SETTLE_DURATION).gripper == pytest.approx(GRIPPER_OPEN)
    assert phase.evaluate(phase.duration).gripper == pytest.approx(GRIPPER_GRASP)


def test_recovery_cube_sampler_stays_away_from_workspace_frame_border():
    rng = np.random.default_rng(0)
    poses = [sample_recovery_cube(rng) for _ in range(100)]
    half_extent = WORKSPACE_FRAME_INNER_HALF_EXTENT - RECOVERY_TARGET_FRAME_BORDER_MARGIN

    assert all(is_cube_pickup_allowed(pose.x, pose.y) for pose in poses)
    assert all(is_cube_recovery_target_allowed(pose.x, pose.y) for pose in poses)
    for pose in poses:
        local_x, local_y = world_to_frame_xy(pose.x, pose.y)
        assert abs(local_x) <= half_extent
        assert abs(local_y) <= half_extent


def test_fixed_target_must_be_in_allowed_drop_zone():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    target_on_apriltag_exclusion = CubePose(x=0.10, y=0.20, z=CUBE_HALF_SIZE)

    with pytest.raises(EpisodeSamplingError, match="outside the allowed drop zone"):
        prepare_episode(
            np.random.default_rng(0),
            source,
            target_on_apriltag_exclusion,
            max_attempts=1,
        )


def test_placement_error_reports_cube_center_offset():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    target = CubePose(x=0.21, y=-0.10, z=CUBE_HALF_SIZE)
    model, data = build_model(source)
    set_cube_pose(model, data, source)

    error = placement_error(model, data, target)

    assert error.dx == pytest.approx(-0.01)
    assert error.dy == pytest.approx(-0.02)
    assert error.dz == pytest.approx(0.0)
    assert error.xy == pytest.approx(np.hypot(0.01, 0.02))
    assert "placement error:" in error.summary()


def test_target_sampler_is_retried_across_attempt_budget(monkeypatch):
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    target = CubePose(x=0.20, y=-0.05, z=CUBE_HALF_SIZE)
    sampled_targets = []

    def target_sampler(_rng):
        sampled_targets.append(target)
        return target

    monkeypatch.setattr(episodes, "grasp_candidates", lambda *args, **kwargs: ())

    with pytest.raises(EpisodeSamplingError, match="within 3 attempts"):
        prepare_episode(
            np.random.default_rng(0),
            source,
            max_attempts=3,
            target_sampler=target_sampler,
        )

    assert sampled_targets == [target, target, target]
