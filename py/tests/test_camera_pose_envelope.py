# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The overhead camera is only randomized over poses the real rig could solve."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
from pick_and_place.camera_pose_envelope import (
    CAMERA_NAME,
    calibrated_radius_px,
    overhead_pose_filter,
)
from pick_and_place.domain_randomization import DomainRandomizationPreset

PRESET = Path(__file__).parents[2] / "config" / "domain_randomization" / "act_mild_v1.json"
INTRINSICS = LOCAL_CAMERA_INTRINSICS_DIR / f"{CAMERA_NAME}.json"

# The calibration is machine-local and gitignored, and the envelope is a
# statement about the real camera's lens, so there is nothing to assert without
# it -- but its absence must not look like a pass elsewhere.
needs_intrinsics = pytest.mark.skipif(
    not INTRINSICS.exists(), reason=f"no calibrated intrinsics at {INTRINSICS}"
)


@needs_intrinsics
def test_calibrated_radius_is_inside_the_sensor_corner():
    """The distortion fit stops inverting well before the frame corner."""
    intrinsics = json.loads(INTRINSICS.read_text())
    matrix = np.array(intrinsics["camera_matrix"], float)
    dist = np.array(intrinsics["dist_coeffs"], float).ravel()
    radius = calibrated_radius_px(matrix, dist)
    corner = np.hypot(
        max(matrix[0, 2], intrinsics["width"] - matrix[0, 2]),
        max(matrix[1, 2], intrinsics["height"] - matrix[1, 2]),
    )
    assert 0 < radius < corner


@needs_intrinsics
def test_authored_pose_sees_all_four_tags():
    visibility = overhead_pose_filter()
    assert visibility.margin_px(np.zeros(3), np.zeros(3)) > 0.0


@needs_intrinsics
def test_a_wildly_tilted_camera_loses_a_tag():
    visibility = overhead_pose_filter()
    assert not visibility.accepts(np.zeros(3), np.array([15.0, 0.0, 0.0]))


@needs_intrinsics
def test_every_sampled_episode_keeps_the_frame_tags():
    preset = DomainRandomizationPreset.load(PRESET)
    margin = preset.scalars["overhead_camera_frame_tag_margin_px"]
    assert margin > 0.0, "the shipped preset should enforce tag visibility"
    visibility = overhead_pose_filter()
    for seed in range(50):
        sample = preset.sample(seed)
        assert (
            visibility.margin_px(
                np.array(sample.overhead_camera_position_m),
                np.array(sample.overhead_camera_rotation_deg),
                sample.overhead_camera_focal_scale,
            )
            >= margin
        )


@needs_intrinsics
def test_authored_pose_does_not_see_its_own_hardware():
    assert not overhead_pose_filter().sees_own_hardware(np.zeros(3), np.zeros(3))


@needs_intrinsics
def test_camera_hardware_travels_with_the_camera():
    """Lens and board must hold station in the camera's frame under any jitter.

    Regression test for ``geom_sameframe``: the compiler flags a geom whose frame
    coincides with its body's, and ``mj_kinematics`` then copies the body frame
    and ignores ``geom_pos``/``geom_quat`` outright. The board sits at its body's
    origin and so was flagged, meaning it silently stayed behind while the camera
    moved -- writing the fields looked like it worked and changed nothing.
    """
    visibility = overhead_pose_filter()
    geoms = tuple(visibility._geom_base)
    assert geoms, "the overhead camera should carry lens and board geoms"

    def offsets(position, rotation):
        visibility._camera_frame(np.array(position) / 1000.0, np.array(rotation))
        center = visibility._data.cam_xpos[visibility.camera]
        rotation_matrix = visibility._data.cam_xmat[visibility.camera].reshape(3, 3)
        return {
            geom: (visibility._data.geom_xpos[geom] - center) @ rotation_matrix for geom in geoms
        }

    nominal = offsets([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for position, rotation in (
        ([0.0, 0.0, 30.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, -40.0], [0.0, 0.0, 0.0]),
        ([5.8, -5.8, 24.9], [2.89, 1.11, 0.9]),
        ([12.9, 0.6, 21.5], [-2.6, 2.05, -2.6]),
    ):
        moved = offsets(position, rotation)
        for geom in geoms:
            assert np.allclose(moved[geom], nominal[geom], atol=1e-9)


@needs_intrinsics
def test_poses_that_once_showed_the_lens_barrel_are_fine_now():
    """Three draws that rendered a hard black wedge while the assembly came apart."""
    visibility = overhead_pose_filter()
    for position, rotation in (
        ([0.0, 0.0, 30.0], [0.0, 0.0, 0.0]),
        ([5.8, -5.8, 24.9], [2.89, 1.11, 0.9]),
        ([12.9, 0.6, 21.5], [-2.6, 2.05, -2.6]),
    ):
        assert not visibility.sees_own_hardware(np.array(position) / 1000.0, np.array(rotation))


@needs_intrinsics
def test_no_sampled_episode_sees_the_camera_hardware():
    preset = DomainRandomizationPreset.load(PRESET)
    visibility = overhead_pose_filter()
    for seed in range(50):
        sample = preset.sample(seed)
        assert not visibility.sees_own_hardware(
            np.array(sample.overhead_camera_position_m),
            np.array(sample.overhead_camera_rotation_deg),
        )


@needs_intrinsics
def test_rejection_changes_the_draw():
    """Without the filter the same box does produce unsolvable poses."""
    preset = DomainRandomizationPreset.load(PRESET)
    unfiltered = replace(
        preset, scalars={**preset.scalars, "overhead_camera_frame_tag_margin_px": 0.0}
    )
    visibility = overhead_pose_filter()
    margin = preset.scalars["overhead_camera_frame_tag_margin_px"]

    def failures(candidate):
        return sum(
            visibility.margin_px(
                np.array(sample.overhead_camera_position_m),
                np.array(sample.overhead_camera_rotation_deg),
                sample.overhead_camera_focal_scale,
            )
            < margin
            for sample in (candidate.sample(seed) for seed in range(300))
        )

    assert failures(unfiltered) > 0
    assert failures(preset) == 0


@needs_intrinsics
def test_focal_jitter_is_drawn_and_bounded():
    """Focal length is the one intrinsic that survives into a rectified frame."""
    preset = DomainRandomizationPreset.load(PRESET)
    span = preset.scalars["overhead_camera_focal_pct"] / 100.0
    assert span > 0.0, "the shipped preset should randomize focal length"
    scales = np.array([preset.sample(seed).overhead_camera_focal_scale for seed in range(300)])
    assert np.all(np.abs(scales - 1.0) <= span + 1e-12)
    assert np.ptp(scales) > span, "the draw should cover most of the range"


@needs_intrinsics
def test_a_long_enough_lens_pushes_the_tags_out_of_reach():
    """Narrowing the field of view moves tags outward, so it must be judged with the pose."""
    visibility = overhead_pose_filter()
    wide = visibility.margin_px(np.zeros(3), np.zeros(3), 1.0)
    narrow = visibility.margin_px(np.zeros(3), np.zeros(3), 1.5)
    assert narrow < wide
    assert not visibility.accepts(np.zeros(3), np.zeros(3), focal_scale=3.0)


def test_sampling_stays_deterministic():
    """Rejection must not cost reproducibility -- a seed still fixes the pose.

    Compared field by field: ``DomainSample`` carries a ``SlowJitter`` that has
    no ``__eq__``, so whole-sample equality never holds for any two draws.
    """
    preset = DomainRandomizationPreset.load(PRESET)
    first, second = preset.sample(99), preset.sample(99)
    assert first.overhead_camera_position_m == second.overhead_camera_position_m
    assert first.overhead_camera_rotation_deg == second.overhead_camera_rotation_deg
    # A rejected draw must not desync the rest of the episode's randomization.
    assert first.wrist_camera_position_m == second.wrist_camera_position_m
    assert first.appearance_seed == second.appearance_seed
    assert first.cube_orientation_index == second.cube_orientation_index


def test_preset_requires_the_margin_field(tmp_path):
    """A preset predating the check is rejected rather than silently unfiltered."""
    payload = json.loads(PRESET.read_text())
    del payload["overhead_camera_frame_tag_margin_px"]
    path = tmp_path / "stale.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overhead_camera_frame_tag_margin_px"):
        DomainRandomizationPreset.load(path)
