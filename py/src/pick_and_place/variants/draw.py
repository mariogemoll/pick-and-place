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


from pick_and_place.core.appearance import AppearanceDraw
from pick_and_place.sim.domain_randomization import (
    DomainRandomizationPreset,
    domain_seed,
)


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
