# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Restyle a compiled scene without changing anything the expert reacted to.

Everything applied here answers "no" to the organizing question: change it and
the correct action is unchanged. Lighting, materials, the background and table
surface, the overhead viewpoint, exposure and sensor noise move pixels only, so
they may be drawn freshly whenever an episode is rendered, regardless of how (or
whether) it was recorded with them.

The wrist camera is deliberately absent. Its mount error is the one camera
displacement the expert has to servo through, so it is drawn when the trajectory
is generated and lives with the simulator's own randomization instead. The
overhead camera is here because at render time it is pure viewpoint nuisance —
the arm really did execute the recorded actions, and the workspace-frame tags in
the view are what let a policy tell "the camera moved" from "the cube moved".

Each randomizer snapshots only the model slices it writes, so applying one never
undoes the other's draw.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

from pick_and_place.sim.camera_pose_envelope import (
    CameraBase,
    CameraJitter,
    set_camera_jitter,
    snapshot_camera,
)
from pick_and_place.core.appearance import MATERIAL_FAMILIES, AppearanceDraw
from pick_and_place.sim.domain_randomization import (
    ProceduralAppearance,
    generate_procedural_appearance,
    write_procedural_textures,
)

#: Textures of the finite-floor scene, absent from the infinite-groundplane one.
SCENE_TEXTURE_NAMES = ("table_texture", "background_panorama")

OVERHEAD_CAMERA = "overhead_camera"

#: The episode's own markers, tinted after they are placed rather than with the
#: rest of the scene.
MARKER_FAMILIES = (("pick_cube", "cube"), ("paper_target_marker_geom", "target"))

#: The shared materials a draw tints, which is every family except the two
#: markers above.
SCENE_MATERIAL_FAMILIES = tuple(
    name for name in MATERIAL_FAMILIES if name not in {"cube", "target"}
)


def scene_texture_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    """Ids of the background/table textures, empty on the groundplane scene."""
    return tuple(
        ident
        for name in SCENE_TEXTURE_NAMES
        if (ident := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, name)) >= 0
    )


def set_scene_texture(
    model: mujoco.MjModel,
    texture_ids: tuple[int, ...],
    appearance: ProceduralAppearance | None,
    base_tex_data: np.ndarray,
) -> None:
    """Write a procedural background/table appearance, or restore the compiled one.

    A no-op without any scene textures (the groundplane scene). The caller still
    has to push the change into a live GL context, e.g. via
    :func:`~pick_and_place.sim.domain_randomization.reload_renderer_textures`.
    """
    if not texture_ids:
        return
    if appearance is None:
        model.tex_data[:] = base_tex_data
    else:
        write_procedural_textures(model, texture_ids, appearance)


class AppearanceRandomizer:
    """Applies a domain sample's pixel-only half, and restores the scene between draws.

    Restoring matters more than applying: a scene is reused across episodes, so a
    draw that is not undone compounds onto the next one and the envelope silently
    widens. Everything written below is snapshotted at construction and written
    back in full by :meth:`reset`.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._light_pos = model.light_pos.copy()
        self._light_dir = model.light_dir.copy()
        self._light_diffuse = model.light_diffuse.copy()
        self._light_ambient = model.light_ambient.copy()
        self._light_specular = model.light_specular.copy()
        self._light_castshadow = model.light_castshadow.copy()
        self._light_bulbradius = model.light_bulbradius.copy()
        self._headlight_diffuse = np.array(model.vis.headlight.diffuse)
        self._headlight_ambient = np.array(model.vis.headlight.ambient)
        self._headlight_specular = np.array(model.vis.headlight.specular)
        self._mat_rgba = model.mat_rgba.copy()
        self._geom_rgba = model.geom_rgba.copy()
        self._tex_data = model.tex_data.copy()
        self._texture_ids = scene_texture_ids(model)
        # Snapshotted lazily: a camera rig overrides cam_fovy from the calibrated
        # intrinsics *after* this is constructed, so capturing it now would bank
        # the authored value and reset() would quietly undo the calibration.
        self._overhead: CameraBase | None = None
        self._sample: AppearanceDraw | None = None
        self._frame = 0
        self._image_rng_seed = 0

    @property
    def texture_ids(self) -> tuple[int, ...]:
        return self._texture_ids

    def overhead_base(self) -> CameraBase:
        """The authored overhead camera, captured on first use."""
        if self._overhead is None:
            self._overhead = snapshot_camera(self.model, OVERHEAD_CAMERA)
        return self._overhead

    def set_camera_jitter(self, jitter: CameraJitter | None) -> None:
        """Move the overhead camera, or put it back where the scene authored it."""
        set_camera_jitter(self.model, self.overhead_base(), jitter)

    def set_scene_texture(self, appearance: ProceduralAppearance | None) -> None:
        """Repaint the background and table, or restore the compiled textures."""
        set_scene_texture(self.model, self._texture_ids, appearance, self._tex_data)

    def reset(self) -> None:
        """Restore the canonical compiled scene after a randomized episode."""
        model = self.model
        model.light_pos[:] = self._light_pos
        model.light_dir[:] = self._light_dir
        model.light_diffuse[:] = self._light_diffuse
        model.light_ambient[:] = self._light_ambient
        model.light_specular[:] = self._light_specular
        model.light_castshadow[:] = self._light_castshadow
        model.light_bulbradius[:] = self._light_bulbradius
        model.vis.headlight.diffuse = self._headlight_diffuse
        model.vis.headlight.ambient = self._headlight_ambient
        model.vis.headlight.specular = self._headlight_specular
        model.mat_rgba[:] = self._mat_rgba
        model.geom_rgba[:] = self._geom_rgba
        model.tex_data[:] = self._tex_data
        if self._overhead is not None:
            set_camera_jitter(model, self._overhead, None)
        self._sample = None
        self._frame = 0
        self._image_rng_seed = 0

    def apply(self, sample: AppearanceDraw) -> None:
        self.reset()
        model = self.model

        cool = np.array((1.0 / sample.light_warm_cool, 1.0, sample.light_warm_cool))
        fill = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "scene_light")
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "warm_spotlight")
        if fill >= 0:
            model.light_diffuse[fill] *= sample.fill_light_intensity
            model.light_ambient[fill] *= sample.fill_light_intensity
            model.light_specular[fill] *= sample.fill_light_intensity
            model.light_castshadow[fill] = False
        if key >= 0:
            model.light_pos[key] = sample.key_light_position
            direction = np.asarray(sample.key_light_target) - np.asarray(sample.key_light_position)
            model.light_dir[key] = direction / np.linalg.norm(direction)
            model.light_diffuse[key] = (
                np.mean(self._light_diffuse[key]) * sample.light_intensity * cool
            )
            model.light_ambient[key] = (
                np.mean(self._light_ambient[key]) * sample.light_intensity * cool
            )
            model.light_specular[key] = (
                np.mean(self._light_specular[key]) * sample.light_intensity * cool
            )
            model.light_castshadow[key] = True
            model.light_bulbradius[key] = sample.key_light_bulb_radius
            model.light_cutoff[key] = 80.0
            model.light_exponent[key] = 2.0
        model.vis.headlight.diffuse = self._headlight_diffuse * sample.fill_light_intensity
        model.vis.headlight.ambient = self._headlight_ambient * sample.fill_light_intensity
        model.vis.headlight.specular = self._headlight_specular * sample.fill_light_intensity

        for name in SCENE_MATERIAL_FAMILIES:
            ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if ident >= 0:
                model.mat_rgba[ident, :3] = np.clip(
                    self._mat_rgba[ident, :3] * sample.material_factors[name], 0.0, 1.0
                )

        self.set_camera_jitter(
            CameraJitter(
                position_m=sample.overhead_camera_position_m,
                rotation_deg=sample.overhead_camera_rotation_deg,
                focal_scale=sample.overhead_camera_focal_scale,
            )
        )
        self.set_scene_texture(generate_procedural_appearance(sample))
        self._sample = sample
        self._frame = 0
        self._image_rng_seed = sample.seed

    def tint_episode_markers(self) -> None:
        """Tint the cube and drop plate, which are placed per episode.

        Separate from :meth:`apply` because both are moved and recoloured after
        the scene is set up, so a tint applied with the rest would be overwritten.
        """
        if self._sample is None:
            return
        for name, family in MARKER_FAMILIES:
            ident = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if ident >= 0:
                self.model.geom_rgba[ident, :3] = np.clip(
                    self.model.geom_rgba[ident, :3] * self._sample.material_factors[family],
                    0.0,
                    1.0,
                )

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply the sample's camera response: exposure, gamma, white balance, blur, noise."""
        rng = np.random.default_rng(
            np.random.SeedSequence([self._image_rng_seed, self._frame, 0x1A6E])
        )
        self._frame += 1
        sample = self._sample
        if sample is None:
            return image
        result = image.astype(np.float32) * sample.exposure
        result = np.clip(result / 255.0, 0.0, 1.0) ** (1.0 / sample.gamma)
        result *= np.asarray(sample.white_balance)
        result = np.clip(result * 255.0, 0.0, 255.0)
        if sample.blur_sigma > 0:
            result = cv2.GaussianBlur(result, (0, 0), sample.blur_sigma)
        if sample.noise_sigma > 0:
            result += rng.normal(0.0, sample.noise_sigma, result.shape)
        return np.clip(result, 0.0, 255.0).astype(np.uint8)
