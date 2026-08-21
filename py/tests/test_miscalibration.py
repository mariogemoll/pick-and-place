# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import math

import numpy as np
import pytest

from pick_and_place.core.joint_frames import real_frame_to_sim, sim_frame_to_real
from pick_and_place.core.geometry import CubePose
from pick_and_place.core.miscalibration import (
    DEFAULT_PAN_JITTER_SIGMA_DEG,
    DEFAULT_PAN_JITTER_TAU_S,
    MiscalibrationModel,
    miscalibration_from_payload,
    miscalibration_payload,
)


def test_sample_is_reproducible():
    model = MiscalibrationModel()
    a = model.sample(np.random.default_rng(5))
    b = model.sample(np.random.default_rng(5))
    assert a.base_offsets_deg == b.base_offsets_deg
    assert a.cube_belief_error == b.cube_belief_error
    assert a.target_belief_error == b.target_belief_error


def test_draw_statistics_match_model():
    model = MiscalibrationModel()
    rng = np.random.default_rng(1)
    pans = np.array(
        [model.sample(rng).base_offsets_deg["shoulder_pan"] for _ in range(3000)]
    )
    assert abs(pans.mean()) < 0.1
    assert abs(pans.std() - model.joint_offset_sigma_deg["shoulder_pan"]) < 0.1


def test_mean_offsets_shift_the_draw():
    model = MiscalibrationModel(
        joint_offset_mean_deg={"shoulder_pan": 4.3},
        joint_offset_sigma_deg={},
        pan_jitter_sigma_deg=0.0,
    )
    draw = model.sample(np.random.default_rng(0))
    assert draw.offsets_deg(0.0)["shoulder_pan"] == 4.3


def test_pan_jitter_wanders_but_other_joints_hold():
    draw = MiscalibrationModel().sample(np.random.default_rng(2))
    values = [draw.offsets_deg(t) for t in np.linspace(0.0, 60.0, 200)]
    pans = [v["shoulder_pan"] for v in values]
    assert max(pans) - min(pans) > 0.5
    for name in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        assert len({v[name] for v in values}) == 1


def test_same_time_returns_same_offsets():
    draw = MiscalibrationModel().sample(np.random.default_rng(3))
    assert draw.offsets_deg(1.5) == draw.offsets_deg(1.5)


def test_offsets_follow_the_follower_sign_convention():
    """A commanded believed pose, executed offset away, must read back as the
    same believed pose through ``real_frame_to_sim`` with the same offsets —
    the exact inverse pair the real feed-forward correction relies on."""
    draw = MiscalibrationModel().sample(np.random.default_rng(4))
    offsets_deg = draw.offsets_deg(0.0)
    offsets_rad = draw.offsets_rad(0.0)
    believed = {
        "shoulder_pan": 0.3,
        "shoulder_lift": -0.5,
        "elbow_flex": 0.9,
        "wrist_flex": -0.2,
        "wrist_roll": 0.1,
    }
    # The sim plant executes the believed command at believed + offset...
    true_joints = {
        name: value + offsets_rad.get(name, 0.0) for name, value in believed.items()
    }
    # ...and the servo-style readback (true minus offset) recovers the command.
    readback = sim_frame_to_real(true_joints, 0.0, offsets_deg)
    recovered, _ = real_frame_to_sim(readback)
    for name, value in believed.items():
        assert math.isclose(recovered[name], value, abs_tol=1e-12)


def test_belief_errors_offset_cube_and_target():
    draw = MiscalibrationModel().sample(np.random.default_rng(6))
    true_pose = CubePose(x=0.25, y=-0.1, z=0.015, yaw=0.4)
    believed = draw.believe_cube(true_pose)
    dx, dy, dz, dyaw = draw.cube_belief_error
    assert believed.x == true_pose.x + dx
    assert believed.y == true_pose.y + dy
    assert believed.z == true_pose.z + dz
    assert believed.yaw == true_pose.yaw + dyaw
    target = draw.believe_target(true_pose)
    tx, ty = draw.target_belief_error
    assert (target.x, target.y) == (true_pose.x + tx, true_pose.y + ty)
    assert target.z == true_pose.z


def test_a_drawn_jitter_names_the_seed_it_runs_on():
    """Serializing a draw needs the stream's name, not its current value."""
    draw = MiscalibrationModel().sample(np.random.default_rng(5))
    assert draw.pan_jitter is not None
    assert draw.pan_jitter.seed is not None
    assert draw.pan_jitter.sigma_deg == DEFAULT_PAN_JITTER_SIGMA_DEG
    assert draw.pan_jitter.tau_s == DEFAULT_PAN_JITTER_TAU_S


def test_payload_round_trip_reproduces_the_whole_wander():
    """Not just the shape: the rebuilt draw must wander through the same path.

    The value at ``t`` is drawn lazily as time advances, so a payload that only
    carried sigma and tau would deserialize into a different realization while
    looking exactly right.
    """
    draw = MiscalibrationModel().sample(np.random.default_rng(11))
    rebuilt = miscalibration_from_payload(miscalibration_payload(draw), context="test")
    times = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0]
    assert [rebuilt.offsets_deg(t) for t in times] == [draw.offsets_deg(t) for t in times]


def test_payload_refuses_a_jitter_it_cannot_reproduce():
    """A caller-supplied stream is deliberately not the one the seed names."""
    draw = miscalibration_from_payload(
        miscalibration_payload(MiscalibrationModel().sample(np.random.default_rng(3))),
        context="test",
        jitter_rng=np.random.default_rng(0),
    )
    assert draw.pan_jitter is not None
    assert draw.pan_jitter.seed is None
    with pytest.raises(ValueError, match="caller-supplied stream"):
        miscalibration_payload(draw)


def test_a_draw_without_jitter_round_trips():
    model = MiscalibrationModel(pan_jitter_sigma_deg=0.0)
    draw = model.sample(np.random.default_rng(2))
    assert draw.pan_jitter is None
    rebuilt = miscalibration_from_payload(miscalibration_payload(draw), context="test")
    assert rebuilt.pan_jitter is None
    assert rebuilt.base_offsets_deg == draw.base_offsets_deg
    assert rebuilt.cube_belief_error == draw.cube_belief_error
    assert rebuilt.target_belief_error == draw.target_belief_error
