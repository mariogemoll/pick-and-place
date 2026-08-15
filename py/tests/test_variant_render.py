# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Replaying an artifact under a different look.

What is asserted here is the part a re-render can get wrong silently: which look
the scene ends up in, and whether the arm lands where physics actually held it.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from pick_and_place.core.geometry import CubePose
from pick_and_place.data.trajectory_artifact import (
    ARTIFACT_FILENAME,
    EpisodeFacts,
    TrajectoryArtifact,
    TrajectoryWriter,
    WristCameraMount,
    save_trajectory,
)
from pick_and_place.sim.domain_randomization import DomainRandomizationPreset
from pick_and_place.variants.appearance import AS_RECORDED, SceneAppearance
from pick_and_place.variants.draw import CameraRandomization
from pick_and_place.variants.render import Variant, scene_textures_for
from pick_and_place.variants.renderer import VariantRenderer
from pick_and_place.spec.workspace import CUBE_HALF_SIZE

PRESET = Path(__file__).parents[2] / "config" / "domain_randomization" / "act_mild_v1.json"


@pytest.fixture(scope="module")
def renderer():
    """One small renderer, shared: compiling the scene is the expensive part."""
    built = VariantRenderer(render_hw=(64, 64), image_hw=(64, 64))
    yield built
    built.close()


def _artifact(**overrides) -> TrajectoryArtifact:
    writer = TrajectoryWriter()
    for index in range(2):
        writer.record(
            phase_name="approach",
            true_state=np.array([10.0 + index, 5.0, -5.0, 0.0, 0.0, 50.0]),
            believed_state=np.array([8.0 + index, 5.0, -5.0, 0.0, 0.0, 50.0]),
            action=np.zeros(6),
            true_cube_pose=np.array([0.30, 0.02, CUBE_HALF_SIZE, 1.0, 0.0, 0.0, 0.0]),
            believed_cube_pose=CubePose(x=0.30, y=0.02, z=CUBE_HALF_SIZE),
            wrist_sighting=None,
        )
    defaults = {
        "target_xy": (0.22, 0.04),
        "target_plate_yaw": 0.3,
        "verdict": "success",
        "phase_spans": writer.spans,
        "fingerprint": {},
        "episode_index": 4,
    }
    return TrajectoryArtifact(
        frames=writer.frames(), facts=EpisodeFacts(**{**defaults, **overrides})
    )


def _wrist_pos(renderer) -> np.ndarray:
    camera = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    return renderer.model.cam_pos[camera].copy()


def _overhead_pos(renderer) -> np.ndarray:
    camera = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_camera")
    return renderer.model.cam_pos[camera].copy()


def test_the_arm_is_posed_where_physics_held_it_not_where_the_servos_reported(renderer):
    """The whole point of replaying the artifact rather than the dataset."""
    artifact = _artifact()
    renderer.set_episode(artifact.facts)

    renderer.set_frame(artifact.frames.true_state[0], artifact.frames.true_cube_pose[0])
    true_qpos = renderer.data.qpos.copy()
    renderer.set_frame(artifact.frames.believed_state[0], artifact.frames.true_cube_pose[0])

    assert not np.allclose(true_qpos, renderer.data.qpos)


def test_the_wrist_camera_goes_back_on_the_mount_the_episode_was_recorded_with(renderer):
    nominal = _wrist_pos(renderer)
    mount = WristCameraMount(position_m=(0.002, -0.001, 0.0005), rotation_deg=(0.5, -0.9, 0.4))

    renderer.set_episode(_artifact(wrist_camera_mount=mount).facts)
    displaced = _wrist_pos(renderer)

    np.testing.assert_allclose(displaced, nominal + np.asarray(mount.position_m), atol=1e-9)

    # An episode recorded on the nominal mount puts it back.
    renderer.set_episode(_artifact().facts)
    np.testing.assert_allclose(_wrist_pos(renderer), nominal, atol=1e-9)


def test_an_episodes_recorded_look_is_restored_when_no_variant_overrides_it(renderer):
    """Rendering must not change anything it was not asked to change."""
    recorded = DomainRandomizationPreset.load(PRESET).sample(11).appearance()
    artifact = _artifact(recorded_appearance=recorded)

    Variant("as-recorded").setup(renderer, artifact)
    lights = renderer.model.light_pos.copy()

    renderer.set_episode(_artifact().facts)
    assert not np.allclose(renderer.model.light_pos, lights)

    Variant("as-recorded").setup(renderer, artifact)
    np.testing.assert_allclose(renderer.model.light_pos, lights)


def test_a_variants_own_draw_replaces_the_recorded_one(renderer):
    recorded = DomainRandomizationPreset.load(PRESET).sample(11).appearance()
    artifact = _artifact(recorded_appearance=recorded)

    Variant("as-recorded").setup(renderer, artifact)
    as_recorded = renderer.model.light_pos.copy()
    Variant("randomized", domain=_appearance_randomization()).setup(renderer, artifact)

    assert not np.allclose(renderer.model.light_pos, as_recorded)


def test_a_camera_jitter_layers_on_top_of_an_appearance_draw(renderer):
    """Both are viewpoint at render time, so the narrower one has to win."""
    artifact = _artifact()
    jitter = CameraRandomization(
        position_mm=10.0, rotation_deg=1.0, focal_pct=1.0, margin_px=0.0, seed=3
    )

    Variant("plain", domain=_appearance_randomization()).setup(renderer, artifact)
    without = _overhead_pos(renderer)
    Variant("jittered", domain=_appearance_randomization(), camera=jitter).setup(
        renderer, artifact
    )

    assert not np.allclose(_overhead_pos(renderer), without)


def test_the_named_appearance_applies_over_whatever_the_draw_painted(renderer):
    artifact = _artifact(recorded_appearance=DomainRandomizationPreset.load(PRESET).sample(2).appearance())
    renderer.set_episode(artifact.facts)
    cube = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube")

    renderer.capture(SceneAppearance(cube="blue"))
    blue = renderer.model.geom_rgba[cube, :3].copy()
    renderer.capture(AS_RECORDED)

    assert blue[2] > blue[0]
    assert not np.allclose(renderer.model.geom_rgba[cube, :3], blue)


def test_only_a_randomized_recording_asks_for_the_finite_floor_scene(tmp_path):
    plain = tmp_path / "ep000000"
    save_trajectory(plain / ARTIFACT_FILENAME, _artifact())
    assert scene_textures_for(plain) is None

    randomized = tmp_path / "ep000001"
    save_trajectory(
        randomized / ARTIFACT_FILENAME,
        _artifact(recorded_appearance=DomainRandomizationPreset.load(PRESET).sample(7).appearance()),
    )
    background, table = scene_textures_for(randomized)
    assert background.ndim == 3 and table.ndim == 3


def _appearance_randomization():
    from pick_and_place.variants.draw import AppearanceRandomization

    return AppearanceRandomization.from_preset(PRESET, seed=5)
