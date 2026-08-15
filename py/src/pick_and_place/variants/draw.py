# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Which appearance a variant gets, and where that choice comes from.

Two ways to decide, and they compose. A :class:`~pick_and_place.variants.appearance.SceneAppearance`
is a *named* look — the blue cube, the black floor — chosen deliberately because
some experiment wants exactly it. The randomizations here are the other kind: an
envelope, sampled per episode, so a training set covers a range of scenes rather
than a list of them.

Every draw below is keyed off the episode index rather than off a running
counter, so which worker renders which episode does not change what it looks
like, and re-running a pass reproduces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pick_and_place.core.appearance import AppearanceDraw
from pick_and_place.sim.camera_pose_envelope import CameraJitter, draw_overhead_camera_jitter
from pick_and_place.sim.domain_randomization import (
    DomainRandomizationPreset,
    ProceduralAppearance,
    domain_seed,
    generate_procedural_appearance,
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
    :meth:`~pick_and_place.sim.domain_randomization.DomainRandomizationPreset.sample`
    and :func:`~pick_and_place.sim.domain_randomization.generate_procedural_appearance`
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
        return generate_procedural_appearance(sample.appearance())


@dataclass(frozen=True)
class AppearanceRandomization:
    """A whole domain-randomization preset, drawn per episode at render time.

    The widest of the three: lighting, materials, overhead viewpoint, background
    and table, and the camera response. This is what makes a
    domain-randomization experiment cheap — the trajectories are already
    recorded, so a new envelope costs a render pass rather than a fresh
    collection run.

    The sample's own L1 fields are drawn too and then ignored, because they
    belong to a generation that already happened: an episode's wrist mount error
    and miscalibration are baked into the trajectory it produced, and its cube
    orientation into where the cube physically is.
    """

    preset_path: Path
    preset: DomainRandomizationPreset
    seed: int

    @classmethod
    def from_preset(cls, path: Path, *, seed: int) -> AppearanceRandomization:
        return cls(preset_path=path, preset=DomainRandomizationPreset.load(path), seed=seed)

    def draw(self, episode_idx: int) -> AppearanceDraw:
        return self.preset.sample(domain_seed(self.seed, episode_idx)).appearance()
