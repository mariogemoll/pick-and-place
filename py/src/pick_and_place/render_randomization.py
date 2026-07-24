# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Per-episode rendering randomization: overhead camera pose and scene textures.

These are the two randomization axes that touch pixels only. Nothing reads the
overhead image back into control and no controller sees the background or table
surface, so both may be drawn freshly at render time regardless of how (or
whether) an episode was recorded with them — which is what lets a re-render add
them to plain recordings, and what lets a closed-loop run reproduce the
appearance a policy was trained on without also taking on the lighting,
material, image-postprocess and miscalibration axes of a full domain sample.

Both draws reuse a :class:`~pick_and_place.domain_randomization.DomainRandomizationPreset`
for its scalars, so the envelope is described in exactly one place, and both are
seeded per episode through :func:`~pick_and_place.domain_randomization.domain_seed`,
so a given root seed reproduces the same sequence of scenes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.camera_pose_envelope import (
    apply_camera_jitter,
    camera_module_geoms,
    draw_overhead_camera_jitter,
)
from pick_and_place.domain_randomization import (
    DomainRandomizationPreset,
    ProceduralAppearance,
    domain_seed,
    generate_procedural_appearance,
    write_procedural_textures,
)

#: Textures of the finite-floor scene, absent from the infinite-groundplane one.
SCENE_TEXTURE_NAMES = ("table_texture", "background_panorama")


@dataclass(frozen=True)
class CameraJitter:
    """One overhead-camera pose+focal draw, as a rigid transform from the authored pose."""

    position_m: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    focal_scale: float = 1.0


@dataclass(frozen=True)
class OverheadCameraBase:
    """The authored overhead camera pose, focal length and hardware geom poses.

    Snapshotted once so a jittered scene can be restored exactly, and so the
    jitter is always applied to the authored pose rather than compounding onto
    the previous episode's.
    """

    camera: int
    pos: np.ndarray
    quat: np.ndarray
    fovy: float
    geoms: dict[int, tuple[np.ndarray, np.ndarray]]


def snapshot_overhead_camera(model: mujoco.MjModel, camera_name: str) -> OverheadCameraBase:
    """Capture a camera's authored pose, focal length and rigidly attached geoms."""
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera < 0:
        raise ValueError(f"scene is missing the {camera_name!r} camera")
    return OverheadCameraBase(
        camera=camera,
        pos=model.cam_pos[camera].copy(),
        quat=model.cam_quat[camera].copy(),
        fovy=float(model.cam_fovy[camera]),
        geoms={
            geom: (model.geom_pos[geom].copy(), model.geom_quat[geom].copy())
            for geom in camera_module_geoms(model, camera)
        },
    )


def set_camera_jitter(
    model: mujoco.MjModel, base: OverheadCameraBase, jitter: CameraJitter | None
) -> None:
    """Apply a pose+focal jitter to the snapshotted camera, or restore its authored pose."""
    if jitter is None:
        model.cam_pos[base.camera] = base.pos
        model.cam_quat[base.camera] = base.quat
        model.cam_fovy[base.camera] = base.fovy
        for geom, (geom_pos, geom_quat) in base.geoms.items():
            model.geom_pos[geom] = geom_pos
            model.geom_quat[geom] = geom_quat
        return
    apply_camera_jitter(
        model,
        base.camera,
        base.pos,
        base.quat,
        base.geoms,
        np.asarray(jitter.position_m, float),
        np.asarray(jitter.rotation_deg, float),
    )
    half = math.radians(base.fovy) / 2.0
    model.cam_fovy[base.camera] = math.degrees(2.0 * math.atan(math.tan(half) / jitter.focal_scale))


def scene_texture_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    """Ids of the background/table textures, empty on the groundplane scene."""
    return tuple(
        ident
        for name in SCENE_TEXTURE_NAMES
        if (ident := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, name)) >= 0
    )


@dataclass(frozen=True)
class CameraRandomization:
    """Overhead-camera jitter parameters and root seed for a rendering pass.

    Draws a fresh pose+focal jitter per episode. Reuses a domain-randomization
    preset file just for its overhead-camera scalars; the rest of the preset
    (lighting, materials, miscalibration, ...) is ignored.
    """

    position_mm: float
    rotation_deg: float
    focal_pct: float
    margin_px: float
    seed: int

    @classmethod
    def from_preset(cls, path: Path, *, seed: int) -> CameraRandomization:
        preset = DomainRandomizationPreset.load(path)
        return cls(
            position_mm=preset.scalars["overhead_camera_position_mm"],
            rotation_deg=preset.scalars["overhead_camera_rotation_deg"],
            focal_pct=preset.scalars["overhead_camera_focal_pct"],
            margin_px=preset.scalars["overhead_camera_frame_tag_margin_px"],
            seed=seed,
        )

    def draw(self, episode_idx: int) -> CameraJitter:
        rng = np.random.default_rng(domain_seed(self.seed, episode_idx))
        position, rotation, focal_scale = draw_overhead_camera_jitter(
            rng,
            position_mm=self.position_mm,
            rotation_deg=self.rotation_deg,
            focal_pct=self.focal_pct,
            margin_px=self.margin_px,
        )
        return CameraJitter(position, rotation, focal_scale)


@dataclass(frozen=True)
class BackgroundRandomization:
    """A domain-randomization preset's background/table draw, applied per episode.

    Only meaningful on the finite-floor scene (built with a background panorama
    or a table texture), since that is what puts a separate skybox and table
    surface into the model to vary. Reuses
    :meth:`~pick_and_place.domain_randomization.DomainRandomizationPreset.sample`
    and :func:`~pick_and_place.domain_randomization.generate_procedural_appearance`
    wholesale rather than re-deriving the colour/blur/blob-count draw: the unused
    fields of the sample (lighting, camera jitter, cube orientation,
    miscalibration) cost nothing to compute and ignoring them is simpler than a
    second, narrower sampler.
    """

    preset_path: Path
    preset: DomainRandomizationPreset
    seed: int

    @classmethod
    def from_preset(cls, path: Path, *, seed: int) -> BackgroundRandomization:
        return cls(preset_path=path, preset=DomainRandomizationPreset.load(path), seed=seed)

    def draw(self, episode_idx: int) -> ProceduralAppearance:
        sample = self.preset.sample(domain_seed(self.seed, episode_idx))
        return generate_procedural_appearance(sample)


def set_scene_texture(
    model: mujoco.MjModel,
    texture_ids: tuple[int, ...],
    appearance: ProceduralAppearance | None,
    base_tex_data: np.ndarray,
) -> None:
    """Write a procedural background/table appearance, or restore the compiled one.

    A no-op without any scene textures (the groundplane scene). The caller still
    has to push the change into a live GL context, e.g. via
    :func:`~pick_and_place.domain_randomization.reload_renderer_textures`.
    """
    if not texture_ids:
        return
    if appearance is None:
        model.tex_data[:] = base_tex_data
    else:
        write_procedural_textures(model, texture_ids, appearance)
