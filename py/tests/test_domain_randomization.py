# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The randomization envelope, and the half of a draw that shapes behavior.

The other half — everything that is only pixels — is applied by
``pick_and_place.variants`` and tested in ``test_variant_scene.py``.
"""

import dataclasses
import json
import colorsys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pick_and_place.core.miscalibration import DEFAULT_JOINT_OFFSET_SIGMA_DEG
from pick_and_place.sim.domain_randomization import (
    DomainRandomizationPreset,
    WristMountRandomizer,
    domain_seed,
    generate_procedural_appearance,
    orient_cube,
)
from pick_and_place.sim.model import build_model
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.core.geometry import CubePose, world_from_cube


PRESET = Path(__file__).parents[2] / "config" / "domain_randomization" / "act_mild_v1.json"


def _procedural_model(preset: DomainRandomizationPreset):
    sample = preset.sample(123)
    appearance = generate_procedural_appearance(sample.appearance())
    return build_model(
        CubePose(0.2, 0.0, CUBE_HALF_SIZE),
        include_environment=True,
        paper_target_marker=True,
        background_panorama=appearance.background_rgb,
        table_texture=appearance.table_rgb,
    )[0]


def test_same_seed_produces_same_serialized_sample():
    preset = DomainRandomizationPreset.load(PRESET)
    assert preset.sample(1234).metadata_json() == preset.sample(1234).metadata_json()


def test_domain_seed_depends_only_on_root_seed_and_episode_index():
    assert domain_seed(17, 3) == domain_seed(17, 3)
    assert domain_seed(17, 3) != domain_seed(17, 4)


def test_different_material_families_receive_independent_draws():
    sample = DomainRandomizationPreset.load(PRESET).sample(1234)
    assert len(set(sample.material_factors.values())) == len(sample.material_factors)


def test_key_light_samples_hard_and_soft_shadow_sources():
    preset = DomainRandomizationPreset.load(PRESET)
    radii = [preset.sample(seed).key_light_bulb_radius for seed in range(100)]
    assert min(radii) < 0.02
    assert max(radii) > 0.3


def test_the_wrist_mount_moves_while_the_controller_keeps_the_nominal_one():
    """The hand-eye error the expert has to servo through.

    The camera and its housing move together — a real sensor cannot slide out of
    its own barrel — while ``believed_wrist_camera_pose`` stays on the authored
    mount, which is what makes every solve inherit the error.
    """
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    canonical = model.cam_pos.copy()
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    overhead = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
    mount = WristMountRandomizer(model)

    mount.apply(preset.sample(1))

    assert not np.array_equal(model.cam_pos[wrist], canonical[wrist])
    np.testing.assert_allclose(mount.believed_wrist_camera_pose[0], canonical[wrist])
    # Only the wrist: the overhead viewpoint is drawn when the episode is rendered.
    np.testing.assert_array_equal(model.cam_pos[overhead], canonical[overhead])

    mount.reset()
    np.testing.assert_allclose(model.cam_pos[wrist], canonical[wrist])


def test_applying_a_second_draw_does_not_compound_onto_the_first():
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    mount = WristMountRandomizer(model)
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")

    mount.apply(preset.sample(1))
    first = model.cam_pos[wrist].copy()
    mount.apply(preset.sample(2))
    mount.apply(preset.sample(1))

    np.testing.assert_allclose(model.cam_pos[wrist], first)


def test_all_24_cube_orientations_are_unique_and_balance_up_faces():
    base = CubePose(0.2, 0.0, CUBE_HALF_SIZE, yaw=0.37)
    matrices = [world_from_cube(orient_cube(base, index))[:3, :3] for index in range(24)]
    rounded = {tuple(np.round(matrix, 8).ravel()) for matrix in matrices}
    assert len(rounded) == 24

    up_faces = []
    for matrix in matrices:
        local_axis = int(np.argmax(np.abs(matrix[2])))
        sign = int(np.sign(matrix[2, local_axis]))
        up_faces.append((local_axis, sign))
    counts = Counter(up_faces)
    assert len(counts) == 6
    assert set(counts.values()) == {4}


def test_procedural_appearance_is_deterministic_and_seeded():
    preset = DomainRandomizationPreset.load(PRESET)
    first = generate_procedural_appearance(preset.sample(5).appearance())
    repeated = generate_procedural_appearance(preset.sample(5).appearance())
    other = generate_procedural_appearance(preset.sample(6).appearance())
    np.testing.assert_array_equal(first.background_rgb, repeated.background_rgb)
    np.testing.assert_array_equal(first.table_rgb, repeated.table_rgb)
    assert not np.array_equal(first.background_rgb, other.background_rgb)
    assert not np.array_equal(first.table_rgb, other.table_rgb)


def test_procedural_appearance_stays_neutral_to_beige():
    preset = DomainRandomizationPreset.load(PRESET)
    checked = 0
    for seed in range(100):
        sample = preset.sample(seed)
        if sample.appearance_mode != "realistic":
            continue
        for rgb in (sample.background_rgb, sample.table_rgb):
            assert rgb[0] >= rgb[2]
            assert max(rgb) - min(rgb) <= 0.13
        checked += 1
    assert checked > 80


def test_colorful_appearance_is_a_deterministic_minority():
    preset = DomainRandomizationPreset.load(PRESET)
    samples = [preset.sample(seed) for seed in range(2_000)]
    colorful = [sample for sample in samples if sample.appearance_mode == "colorful"]
    assert 0.075 <= len(colorful) / len(samples) <= 0.125
    assert any(colorsys.rgb_to_hsv(*sample.background_rgb)[1] >= 0.15 for sample in colorful)
    assert {sample.appearance_mode for sample in samples} == {"realistic", "colorful"}


def test_preset_loading_rejects_unknown_fields(tmp_path):
    payload = json.loads(PRESET.read_text())
    payload["unexpected"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    try:
        DomainRandomizationPreset.load(path)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown preset field was accepted")


def test_omitting_joint_offset_sigma_keeps_the_measured_drift():
    preset = DomainRandomizationPreset.load(PRESET)
    assert preset.joint_offset_sigma_deg == DEFAULT_JOINT_OFFSET_SIGMA_DEG


def test_a_preset_may_widen_the_joint_offset_draws(tmp_path):
    payload = json.loads(PRESET.read_text())
    payload["joint_offset_sigma_deg"] = {"shoulder_pan": 3.0}
    path = tmp_path / "wide.json"
    path.write_text(json.dumps(payload))
    wide = DomainRandomizationPreset.load(path)

    # Named joints are overridden; the rest keep the measured drift.
    assert wide.joint_offset_sigma_deg["shoulder_pan"] == 3.0
    assert wide.joint_offset_sigma_deg["wrist_flex"] == DEFAULT_JOINT_OFFSET_SIGMA_DEG["wrist_flex"]

    narrow = DomainRandomizationPreset.load(PRESET)
    pans = [
        abs(p.sample(s).miscalibration.base_offsets_deg["shoulder_pan"])
        for p in (narrow, wide)
        for s in range(2000)
    ]
    narrow_pans, wide_pans = pans[:2000], pans[2000:]
    assert np.mean(wide_pans) > 1.8 * np.mean(narrow_pans)


def test_widening_one_joint_leaves_the_rest_of_the_draw_alone():
    """The override changes drawn values, not the number or order of draws, so
    a widened preset stays comparable to the narrow one scene for scene."""
    payload = json.loads(PRESET.read_text())
    narrow = DomainRandomizationPreset.load(PRESET)
    wide = dataclasses.replace(
        narrow, joint_offset_sigma_deg={**narrow.joint_offset_sigma_deg, "shoulder_pan": 3.0}
    )
    for seed in (0, 7, 4242):
        a, b = narrow.sample(seed), wide.sample(seed)
        assert a.exposure == b.exposure
        assert a.background_rgb == b.background_rgb
        assert a.miscalibration.cube_belief_error == b.miscalibration.cube_belief_error
        assert a.miscalibration.base_offsets_deg["shoulder_pan"] != pytest.approx(
            b.miscalibration.base_offsets_deg["shoulder_pan"]
        )
    assert payload["name"] == "act_mild_v1"


def test_preset_loading_rejects_unfitted_joints(tmp_path):
    payload = json.loads(PRESET.read_text())
    payload["joint_offset_sigma_deg"] = {"wrist_roll": 1.0}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    try:
        DomainRandomizationPreset.load(path)
    except ValueError as exc:
        assert "wrist_roll" in str(exc)
    else:
        raise AssertionError("an unfitted joint was accepted")


def test_the_shipped_wide_preset_is_the_narrow_one_with_wider_offsets():
    wide_path = PRESET.parent / "act_mild_wide_offsets_v1.json"
    narrow, wide = json.loads(PRESET.read_text()), json.loads(wide_path.read_text())
    assert set(wide) - set(narrow) == {"joint_offset_sigma_deg"}
    for key in narrow:
        if key != "name":
            assert wide[key] == narrow[key], f"{key} differs beyond the offsets"
    for joint, sigma in wide["joint_offset_sigma_deg"].items():
        assert sigma == pytest.approx(2.0 * DEFAULT_JOINT_OFFSET_SIGMA_DEG[joint])
