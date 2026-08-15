# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turning one artifact into N rendered episodes.

**The variant is the outer loop and the frame is the inner one.** Restyle the
scene once, then replay the whole episode through it. The other order — hold a
frame and cycle the looks — pays a texture upload per variant per frame, which
is most of the cost of a wide appearance draw. Replaying an artifact in frame
order is cheap, so the order that costs nothing is the one to take.

The variants of an episode stay trajectory-identical for a stronger reason than
loop order: they are all replays of the same stored trajectory, so the
appearance factor is separable by construction rather than by seed.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pick_and_place.data.trajectory_artifact import (
    ARTIFACT_FILENAME,
    TrajectoryArtifact,
    load_trajectory,
)
from pick_and_place.variants.appearance import AS_RECORDED, SceneAppearance
from pick_and_place.variants.draw import (
    AppearanceRandomization,
    BackgroundRandomization,
    CameraRandomization,
)
from pick_and_place.sim.domain_randomization import generate_procedural_appearance
from pick_and_place.variants.renderer import VariantRenderer
from pick_and_place.variants.video import (
    ImageStatsAccumulator,
    VideoWriter,
    copy_episode_scaffold,
    episode_video_paths,
    rewrite_image_stats,
    video_frame_count,
    x264_settings,
)


@dataclass(frozen=True)
class Variant:
    """One named look, and everything that decides it.

    A :class:`~pick_and_place.variants.appearance.SceneAppearance` is a chosen
    look — the blue cube, the black floor — while the randomizations are drawn
    per episode from an envelope. They compose: a draw paints the scene, and the
    named appearance is applied on top, so ``cube=blue`` still means a blue cube
    however the lighting came out.
    """

    name: str
    appearance: SceneAppearance = field(default_factory=lambda: AS_RECORDED)
    camera: CameraRandomization | None = None
    background: BackgroundRandomization | None = None
    domain: AppearanceRandomization | None = None

    def setup(self, renderer: VariantRenderer, artifact: TrajectoryArtifact) -> None:
        """Put the scene into this variant's look for one episode."""
        index = artifact.facts.episode_index or 0
        renderer.set_episode(
            artifact.facts,
            camera_jitter=None if self.camera is None else self.camera.draw(index),
            scene_texture=None if self.background is None else self.background.draw(index),
            appearance_draw=None if self.domain is None else self.domain.draw(index),
        )


def scene_textures_for(episode: Path) -> tuple[Any, Any] | None:
    """The floor and skybox an episode's scene needs, or ``None`` for the groundplane.

    A randomized recording is made in the finite-floor scene — a workspace floor
    with a skybox beyond it, both procedural textures — while a plain one sits on
    the infinite groundplane. Re-rendering the first through the second changes
    the largest thing in both views, so the scene has to be built the way the
    recording built it.

    Only the texture *slots* matter here: their contents are rewritten per
    episode from that episode's own draw. So one episode's appearance is enough
    to compile the right scene for all of them.
    """
    facts = load_trajectory(episode / ARTIFACT_FILENAME).facts
    if facts.recorded_appearance is None:
        return None
    appearance = generate_procedural_appearance(facts.recorded_appearance)
    return appearance.background_rgb, appearance.table_rgb


def assert_rerenderable(episode_root: Path) -> None:
    """Raise unless the episode's pixels can be rebuilt from what it stores.

    One condition, and it is the one the trajectory artifact exists to satisfy:
    the episode must carry it. ``observation.state`` is the servo-style readback,
    so under a miscalibration draw it is not where physics held the arm, and the
    pan jitter separating the two is a random walk stored nowhere else. The
    artifact holds the true pose per frame, plus the wrist camera's physical
    mount and the drop plate's placement.
    """
    if not (episode_root / ARTIFACT_FILENAME).is_file():
        raise ValueError(
            f"{episode_root.name} carries no {ARTIFACT_FILENAME}, so the arm pose physics "
            "actually used is unrecoverable and its pixels cannot be reproduced"
        )
    metadata_paths = sorted((episode_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if len(metadata_paths) != 1:
        raise ValueError(f"{episode_root.name} must contain one episode metadata parquet")
    # Read only to fail early on a truncated or half-written episode; nothing in
    # the metadata decides the pixels any more.
    pq.read_schema(metadata_paths[0])


def render_episode(
    renderer: VariantRenderer,
    source: Path,
    variants: list[Variant],
    output: Path,
    *,
    fps: int,
    staging_root,
) -> int:
    """Write every variant of one episode; return the frames rendered per variant."""
    assert_rerenderable(source)
    artifact = load_trajectory(source / ARTIFACT_FILENAME)
    frames = len(artifact.frames)
    source_videos = episode_video_paths(source)
    settings = {feature: x264_settings(path) for feature, path in source_videos.items()}
    for feature, path in source_videos.items():
        recorded = video_frame_count(path)
        if recorded != frames:
            raise ValueError(
                f"{source.name} {feature} holds {recorded} frames against "
                f"{frames} frames in its trajectory artifact"
            )

    for variant in variants:
        partial = staging_root(variant, output) / f".{source.name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        copy_episode_scaffold(source, partial)

        writers: dict[str, Any] = {}
        stats: dict[str, ImageStatsAccumulator] = {}
        for feature, video_path in source_videos.items():
            destination = partial / video_path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            writers[feature] = VideoWriter(
                destination,
                width=renderer.rig.width,
                height=renderer.rig.height,
                fps=fps,
                settings=settings[feature],
            )
            stats[feature] = ImageStatsAccumulator()

        try:
            variant.setup(renderer, artifact)
            for index in range(frames):
                renderer.set_frame(
                    artifact.frames.true_state[index], artifact.frames.true_cube_pose[index]
                )
                for feature, image in renderer.capture(variant.appearance).items():
                    writers[feature].write(image)
                    stats[feature].add(image)
        finally:
            for writer in writers.values():
                writer.close()

        for feature, video_path in source_videos.items():
            written = video_frame_count(partial / video_path.relative_to(source))
            if written != frames:
                raise RuntimeError(
                    f"{source.name} {variant.name} {feature} wrote {written} frames "
                    f"against {frames} recorded rows"
                )
        rewrite_image_stats(
            partial / "meta" / "stats.json",
            {feature: accumulator.result() for feature, accumulator in stats.items()},
        )
        final = partial.with_name(source.name)
        if final.exists():
            shutil.rmtree(final)
        os.replace(partial, final)
    return frames
