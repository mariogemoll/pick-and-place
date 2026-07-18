# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
import colorsys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.domain_randomization import (
    DomainRandomizationPreset,
    DomainRandomizer,
    domain_seed,
    generate_procedural_appearance,
    orient_cube,
)
from pick_and_place.episodes import _build_model
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose, world_from_cube
from pick_and_place.paper_detection import (
    DROP_ZONE_HALF_SIZE,
    place_paper_target_marker,
)


PRESET = Path(__file__).parents[2] / "config" / "domain_randomization" / "act_mild_v1.json"


def _procedural_model(preset: DomainRandomizationPreset):
    sample = preset.sample(123)
    appearance = generate_procedural_appearance(sample)
    return _build_model(
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


def test_apply_restores_canonical_visual_model_before_next_sample():
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    randomizer = DomainRandomizer(model)
    canonical_camera = model.cam_pos.copy()
    canonical_collision = model.geom_size.copy()
    first = preset.sample(1)
    second = preset.sample(2)
    randomizer.apply(first)
    first_camera = model.cam_pos.copy()
    first_light = model.light_pos.copy()
    first_texture = model.tex_data.copy()
    randomizer.apply(second)
    randomizer.apply(first)
    np.testing.assert_allclose(model.cam_pos, first_camera)
    np.testing.assert_allclose(model.light_pos, first_light)
    np.testing.assert_array_equal(model.tex_data, first_texture)
    assert not np.array_equal(first_camera, canonical_camera)
    nominal_wrist_pos, _ = randomizer.believed_wrist_camera_pose
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    np.testing.assert_allclose(nominal_wrist_pos, canonical_camera[wrist])
    assert not np.array_equal(model.cam_pos[wrist], nominal_wrist_pos)
    np.testing.assert_array_equal(model.geom_size, canonical_collision)


def test_marker_tint_preserves_the_placed_target_color():
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    randomizer = DomainRandomizer(model)
    sample = preset.sample(8)
    randomizer.apply(sample)
    place_paper_target_marker(
        model,
        (0.2, 0.0),
        0.0,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=True,
        alpha=1.0,
    )
    randomizer.tint_episode_markers()
    target = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "paper_target_marker_geom"
    )
    expected = 0.12 * np.asarray(sample.material_factors["target"])
    np.testing.assert_allclose(model.geom_rgba[target, :3], expected)
    assert model.geom_rgba[target, 3] == 1.0


def test_key_light_casts_shadow_from_sampled_direction():
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    sample = preset.sample(9)
    DomainRandomizer(model).apply(sample)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "warm_spotlight")
    expected = np.asarray(sample.key_light_target) - np.asarray(sample.key_light_position)
    expected /= np.linalg.norm(expected)
    assert model.light_castshadow[key]
    np.testing.assert_allclose(model.light_dir[key], expected)
    assert model.light_diffuse[key].max() / model.light_diffuse[key].min() <= 1.25


def test_key_light_samples_hard_and_soft_shadow_sources():
    preset = DomainRandomizationPreset.load(PRESET)
    radii = [preset.sample(seed).key_light_bulb_radius for seed in range(100)]
    assert min(radii) < 0.02
    assert max(radii) > 0.3


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
    first = generate_procedural_appearance(preset.sample(5))
    repeated = generate_procedural_appearance(preset.sample(5))
    other = generate_procedural_appearance(preset.sample(6))
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
