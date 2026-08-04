# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pathlib import Path

import numpy as np

from pick_and_place.sim.render_randomization import (
    BackgroundRandomization,
    CameraJitter,
    CameraRandomization,
    scene_texture_ids,
    set_camera_jitter,
    set_scene_texture,
    snapshot_overhead_camera,
)
from pick_and_place.runtime.sim_recorder import OVERHEAD_CAMERA, build_recording_scene

PRESET = Path(__file__).parents[2] / "config" / "domain_randomization" / "act_mild_v1.json"


def _scene(**kwargs):
    return build_recording_scene(render_width=64, render_height=64, **kwargs)[0]


def test_camera_jitter_moves_the_camera_and_its_hardware_and_restores_them():
    model = _scene()
    base = snapshot_overhead_camera(model, OVERHEAD_CAMERA)
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
    base = snapshot_overhead_camera(model, OVERHEAD_CAMERA)
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
