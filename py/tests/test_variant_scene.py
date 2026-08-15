# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Restyling a scene: the draws that move pixels and nothing else.

The other half of a domain sample — the wrist mount error the expert has to
servo through — is applied when the trajectory is generated and tested in
``test_domain_randomization.py``.
"""

from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.sim.domain_randomization import (
    DomainRandomizationPreset,
    generate_procedural_appearance,
)
from pick_and_place.variants.scene import AppearanceRandomizer
from pick_and_place.core.geometry import CubePose
from pick_and_place.sim.model import build_model
from pick_and_place.sim.paper_target_marker import place_paper_target_marker
from pick_and_place.spec.workspace import CUBE_HALF_SIZE, DROP_ZONE_HALF_SIZE

from pick_and_place.sim.camera_pose_envelope import (
    CameraJitter,
    set_camera_jitter,
    snapshot_camera,
)
from pick_and_place.variants.draw import BackgroundRandomization, CameraRandomization
from pick_and_place.variants.scene import scene_texture_ids, set_scene_texture
from pick_and_place.rollout.sim import OVERHEAD_CAMERA, build_recording_scene

PRESET = Path(__file__).parents[2] / "config" / "domain_randomization" / "act_mild_v1.json"


def _scene(**kwargs):
    return build_recording_scene(render_width=64, render_height=64, **kwargs)[0]


def test_camera_jitter_moves_the_camera_and_its_hardware_and_restores_them():
    model = _scene()
    base = snapshot_camera(model, OVERHEAD_CAMERA)
    assert base.geoms, "the overhead camera should carry lens and board geoms"
    geom = next(iter(base.geoms))

    set_camera_jitter(
        model, base, CameraJitter((0.01, -0.02, 0.005), (1.0, -2.0, 0.5), focal_scale=1.05)
    )
    assert not np.allclose(model.cam_pos[base.camera], base.pos)
    assert not np.allclose(model.geom_pos[geom], base.geoms[geom][0])
    # A longer focal length is a narrower field of view.
    assert model.cam_fovy[base.camera] < base.fovy

    set_camera_jitter(model, base, None)
    assert np.allclose(model.cam_pos[base.camera], base.pos)
    assert np.allclose(model.cam_quat[base.camera], base.quat)
    assert np.allclose(model.geom_pos[geom], base.geoms[geom][0])
    assert model.cam_fovy[base.camera] == base.fovy


def test_successive_jitters_do_not_compound():
    """Every draw is measured from the authored pose, not the previous episode's."""
    model = _scene()
    base = snapshot_camera(model, OVERHEAD_CAMERA)
    jitter = CameraJitter((0.01, 0.0, 0.0), (0.0, 0.0, 0.0))

    set_camera_jitter(model, base, jitter)
    once = model.cam_pos[base.camera].copy()
    set_camera_jitter(model, base, jitter)
    assert np.allclose(model.cam_pos[base.camera], once)


def test_camera_randomization_is_seeded_per_episode():
    randomization = CameraRandomization.from_preset(PRESET, seed=7)
    assert randomization.position_mm == 25.0
    assert randomization.rotation_deg == 3.0
    assert randomization.focal_pct == 2.5

    assert randomization.draw(3) == randomization.draw(3)
    assert randomization.draw(3) != randomization.draw(4)
    assert CameraRandomization.from_preset(PRESET, seed=8).draw(3) != randomization.draw(3)

    jitter = randomization.draw(0)
    assert max(abs(value) for value in jitter.position_m) <= 0.025
    assert max(abs(value) for value in jitter.rotation_deg) <= 3.0
    assert abs(jitter.focal_scale - 1.0) <= 0.025


def test_background_randomization_is_seeded_per_episode():
    randomization = BackgroundRandomization.from_preset(PRESET, seed=11)

    first = randomization.draw(0)
    assert np.array_equal(first.background_rgb, randomization.draw(0).background_rgb)
    assert not np.array_equal(first.background_rgb, randomization.draw(1).background_rgb)
    assert not np.array_equal(
        first.table_rgb, BackgroundRandomization.from_preset(PRESET, seed=12).draw(0).table_rgb
    )


def test_the_groundplane_scene_has_no_scene_textures_to_randomize():
    model = _scene()
    assert scene_texture_ids(model) == ()
    # Nothing to write into, so the model must come out untouched.
    before = model.tex_data.copy()
    set_scene_texture(
        model, (), BackgroundRandomization.from_preset(PRESET, seed=1).draw(0), before
    )
    assert np.array_equal(model.tex_data, before)


def test_the_finite_floor_scene_textures_are_written_and_restored():
    table = np.full((32, 32, 3), 128, dtype=np.uint8)
    background = np.full((32, 64, 3), 64, dtype=np.uint8)
    model = _scene(table_texture=table, background_panorama=background)
    texture_ids = scene_texture_ids(model)
    assert len(texture_ids) == 2

    compiled = model.tex_data.copy()
    appearance = BackgroundRandomization.from_preset(PRESET, seed=11).draw(0)
    set_scene_texture(model, texture_ids, appearance, compiled)
    assert not np.array_equal(model.tex_data, compiled)

    set_scene_texture(model, texture_ids, None, compiled)
    assert np.array_equal(model.tex_data, compiled)


def _procedural_model(preset: DomainRandomizationPreset):
    """The finite-floor scene, whose skybox and table an appearance draw repaints."""
    sample = preset.sample(123)
    appearance = generate_procedural_appearance(sample.appearance())
    return build_model(
        CubePose(0.2, 0.0, CUBE_HALF_SIZE),
        include_environment=True,
        paper_target_marker=True,
        background_panorama=appearance.background_rgb,
        table_texture=appearance.table_rgb,
    )[0]


def test_apply_restores_the_canonical_scene_before_the_next_draw():
    """A reused scene must not let one episode's draw compound onto the next."""
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    randomizer = AppearanceRandomizer(model)
    canonical_camera = model.cam_pos.copy()
    canonical_collision = model.geom_size.copy()

    randomizer.apply(preset.sample(1).appearance())
    first_camera = model.cam_pos.copy()
    first_light = model.light_pos.copy()
    first_texture = model.tex_data.copy()
    randomizer.apply(preset.sample(2).appearance())
    randomizer.apply(preset.sample(1).appearance())

    np.testing.assert_allclose(model.cam_pos, first_camera)
    np.testing.assert_allclose(model.light_pos, first_light)
    np.testing.assert_array_equal(model.tex_data, first_texture)
    assert not np.array_equal(first_camera, canonical_camera)
    # Appearance is appearance: no draw here may move a collision geom.
    np.testing.assert_array_equal(model.geom_size, canonical_collision)


def test_the_wrist_camera_is_left_alone():
    """It is the one camera whose displacement changes the correct action."""
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    canonical = model.cam_pos[wrist].copy()

    AppearanceRandomizer(model).apply(preset.sample(1).appearance())

    np.testing.assert_array_equal(model.cam_pos[wrist], canonical)


def test_reset_restores_the_canonical_scene_without_applying_another_draw():
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    randomizer = AppearanceRandomizer(model)
    canonical_camera = model.cam_pos.copy()
    canonical_texture = model.tex_data.copy()

    randomizer.apply(preset.sample(1).appearance())
    randomizer.reset()

    np.testing.assert_array_equal(model.cam_pos, canonical_camera)
    np.testing.assert_array_equal(model.tex_data, canonical_texture)


def test_marker_tint_preserves_the_placed_target_color():
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    randomizer = AppearanceRandomizer(model)
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

    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paper_target_marker_geom")
    expected = 0.12 * np.asarray(sample.material_factors["target"])
    np.testing.assert_allclose(model.geom_rgba[target, :3], expected)
    assert model.geom_rgba[target, 3] == 1.0


def test_key_light_casts_shadow_from_the_sampled_direction():
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    sample = preset.sample(9)

    AppearanceRandomizer(model).apply(sample)

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "warm_spotlight")
    expected = np.asarray(sample.key_light_target) - np.asarray(sample.key_light_position)
    expected /= np.linalg.norm(expected)
    assert model.light_castshadow[key]
    np.testing.assert_allclose(model.light_dir[key], expected)
    assert model.light_diffuse[key].max() / model.light_diffuse[key].min() <= 1.25


def test_the_camera_response_is_a_deterministic_function_of_the_frame():
    """Exposure, gamma, white balance, blur and noise, seeded per frame."""
    preset = DomainRandomizationPreset.load(PRESET)
    model = _procedural_model(preset)
    image = np.full((16, 16, 3), 128, dtype=np.uint8)

    first = AppearanceRandomizer(model)
    first.apply(preset.sample(3).appearance())
    repeated = AppearanceRandomizer(model)
    repeated.apply(preset.sample(3).appearance())

    np.testing.assert_array_equal(first.postprocess(image), repeated.postprocess(image))
    # Without a draw applied the image is passed through untouched.
    np.testing.assert_array_equal(AppearanceRandomizer(model).postprocess(image), image)
