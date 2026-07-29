#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Re-render staged ground-truth episodes under a different scene appearance.

The pick cube must be rendered blue to be legible at 96x96 (see
``docs/SCENE_APPEARANCE_SWEEP.md``) but cannot be *recorded* that way: with
``--miscalibration`` the descent visual servo detects the cube's AprilTags in
the wrist image, so a solid-colour cube fails every episode. Datasets are
therefore produced in two passes — record with the tags, then replay each
episode from its stored ground truth with the cube blue.

This command is the second pass. For every staged episode it replays the
recorded per-frame scene (arm joints from ``observation.state``, true cube pose
from ``observation.environment_state``, drop-zone marker at the episode's
target), re-encodes both camera videos through the recording pipeline, and
copies the parquet data and metadata through untouched. Only pixels change;
states, actions and phase spans stay exactly as recorded. Each variant is
written as its own staged-episode root, ready for ``finalize_sim_dataset.py``.

Because several appearances are rendered from one replay of the same frame, the
variants of an episode are trajectory-identical by construction rather than by
seed — which is what keeps the appearance factor separable in the retrain.

Verify before trusting a re-render::

    python scripts/rerender_episodes.py \\
      --episodes-root ../datasets/sim_plain_gt_1000_episodes \\
      --output ../datasets/sim_gt_rerender --verify --max-episodes 2

``--verify`` re-renders with the *recorded* appearance and diffs every sampled
frame against the originally recorded video, reporting the difference next to
the codec's own noise floor (the recorded video is lossy H.264, so a perfect
re-render still differs from it). It writes ``verification.json`` into the
output directory, and the render pass refuses to run without a passing report
from the same environment — the camera calibrations are machine-local files and
the OpenGL backend decides the shading, so a verification is only evidence for
re-renders produced beside it.

Then produce the passes::

    python scripts/rerender_episodes.py \\
      --episodes-root ../datasets/sim_plain_gt_1000_episodes \\
      --output ../datasets/sim_gt_rerender \\
      --variant as-recorded blue-cube black-floor --workers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.camera_pose_envelope import draw_overhead_camera_jitter
from pick_and_place.domain_randomization import (
    DomainRandomizationPreset,
    domain_seed,
    generate_procedural_appearance,
)
from pick_and_place.episode_rerender import (
    CAMERA_FEATURES,
    CameraJitter,
    EpisodeRenderer,
    FrameDiff,
    ImageStatsAccumulator,
    VideoWriter,
    assert_rerenderable,
    copy_episode_scaffold,
    decode_video,
    directory_size,
    encode_decode,
    episode_video_fps,
    episode_video_paths,
    evenly_spaced,
    mean_of,
    render_environment_fingerprint,
    rewrite_image_stats,
    video_frame_count,
    x264_settings,
)
from pick_and_place.scene_appearance import SceneAppearance
from pick_and_place.scene_visibility import load_episode_truth, video_render_hw
from pick_and_place.sim_dataset_staging import (
    COLLECTION_CONFIG_FILENAME,
    episode_index,
    episode_staging_root,
    find_episode_datasets,
)

#: Named appearances. ``as-recorded`` changes nothing and is what ``--verify``
#: measures. The rest come from ``docs/SCENE_APPEARANCE_SWEEP.md``: recolouring
#: the cube, or darkening the floor instead. A dark floor needs the white
#: target — the black one is found by local darkness and all but vanishes
#: against it (plate contrast 0.24 on black, against 0.64 with a white target).
#:
#: Overhead cube contrast during acquisition, medians over 3 episodes:
#: as-recorded 0.159, blue-cube 0.528, gray-floor 0.379, black-floor 0.598.
#: The floor variants keep the tagged cube, so they inherit its poor
#: separability from the white arm (~49 against the blue cube's ~196).
VARIANT_PRESETS: dict[str, SceneAppearance] = {
    "as-recorded": SceneAppearance(),
    "blue-cube": SceneAppearance(cube="blue"),
    "black-floor": SceneAppearance(floor="black", target="white"),
    "gray-floor": SceneAppearance(floor="dark-gray", target="white"),
}

VERIFICATION_FILENAME = "verification.json"
MANIFEST_FILENAME = "rerender.json"

# A perfect re-render still differs from the recorded video by the H.264 loss in
# the recording itself, which measures ~1 grey level mean at the recorded CRF.
# The gate is therefore both absolute and relative to that measured floor.
DEFAULT_MAX_MEAN_ABS_DIFF = 3.0
DEFAULT_MAX_NOISE_RATIO = 2.0
# Re-rendering writes a second copy of every video; keep this much room spare.
DISK_HEADROOM_BYTES = 2 * 1024**3

# Fallback source render size for a staging area with no collection config.
FALLBACK_RENDER_HW = (1080, 1920)


@dataclass(frozen=True)
class Variant:
    """One named appearance and the staged root it is written to."""

    name: str
    appearance: SceneAppearance

    def staging_root(self, output: Path) -> Path:
        return episode_staging_root(output / self.name)


@dataclass(frozen=True)
class CameraRandomization:
    """Overhead-camera jitter parameters and root seed for a re-render pass.

    Draws a fresh pose+focal jitter per episode rather than replaying one from
    the recording: nothing reads the overhead image back into control, so the
    jitter is purely a rendering matter and does not need to have existed when
    the episode was recorded. Reuses a domain-randomization preset file just for
    its overhead-camera scalars; the rest of the preset (lighting, materials,
    miscalibration, ...) is ignored.
    """

    position_mm: float
    rotation_deg: float
    focal_pct: float
    margin_px: float
    seed: int

    @classmethod
    def from_preset(cls, path: Path, *, seed: int) -> "CameraRandomization":
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

    Only meaningful together with ``--background-panorama``/``--table-texture``,
    since those are what put the scene into the finite-floor mode with a
    separate skybox to vary. Reuses ``DomainRandomizationPreset.sample`` and
    ``generate_procedural_appearance`` wholesale rather than re-deriving the
    colour/blur/blob-count draw: the unused fields of the sample (lighting,
    camera jitter, cube orientation, miscalibration) cost nothing to compute
    and ignoring them is simpler than a second, narrower sampler.
    """

    preset_path: Path
    preset: DomainRandomizationPreset
    seed: int

    @classmethod
    def from_preset(cls, path: Path, *, seed: int) -> "BackgroundRandomization":
        return cls(preset_path=path, preset=DomainRandomizationPreset.load(path), seed=seed)

    def draw(self, episode_idx: int):  # -> ProceduralAppearance
        sample = self.preset.sample(domain_seed(self.seed, episode_idx))
        return generate_procedural_appearance(sample)


def parse_variant(token: str) -> Variant:
    """Parse ``NAME`` from the preset table, or an ad-hoc ``key=value,...`` spec."""
    if "=" not in token:
        if token not in VARIANT_PRESETS:
            raise argparse.ArgumentTypeError(
                f"unknown variant {token!r}; expected one of {sorted(VARIANT_PRESETS)} "
                "or a spec like 'cube=blue,floor=dark-gray'"
            )
        return Variant(token, VARIANT_PRESETS[token])

    fields: dict[str, Any] = {}
    for item in token.split(","):
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key == "frame_tags":
            fields[key] = value.lower() in ("1", "true", "on", "yes")
        elif key in ("floor", "target", "cube"):
            fields[key] = value
        else:
            raise argparse.ArgumentTypeError(
                f"unknown appearance field {key!r} in {token!r}; "
                "expected floor, target, cube or frame_tags"
            )
    try:
        appearance = SceneAppearance(**fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    name = "_".join(f"{key}-{fields[key]}" for key in sorted(fields))
    return Variant(name, appearance)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--episodes-root",
        type=Path,
        required=True,
        help="staged episodes directory holding ep*/ with per-frame ground truth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "output directory; each variant is staged at <output>/<variant>_episodes, "
            "so finalize_sim_dataset.py takes --dataset-root <output>/<variant>"
        ),
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=["blue-cube"],
        metavar="NAME",
        help=(
            "appearances to render, either a preset "
            f"({', '.join(sorted(VARIANT_PRESETS))}) or an ad-hoc spec such as "
            "'cube=blue,floor=dark-gray'. Rendering several in one run replays each "
            "frame once, so the variants are trajectory-identical (default: blue-cube)"
        ),
    )
    parser.add_argument(
        "--background-panorama",
        type=Path,
        default=None,
        help=(
            "equirectangular image; switches the scene from the infinite groundplane "
            "to a finite workspace floor (flush with the frame) plus this skybox beyond "
            "it, so a floor-colour variant stays scoped to the workspace interior. "
            "Omit for the groundplane scene every recording so far has used (default)."
        ),
    )
    parser.add_argument(
        "--table-texture",
        type=Path,
        default=None,
        help=(
            "top-down table-surface image for the finite workspace floor. Implies the "
            "same finite-floor scene as --background-panorama even without one."
        ),
    )
    parser.add_argument(
        "--camera-randomization",
        type=Path,
        default=None,
        help=(
            "domain-randomization preset (e.g. config/domain_randomization/act_mild_v1.json) "
            "whose overhead_camera_* scalars are used to draw a fresh per-episode overhead "
            "camera pose+focal jitter at render time, independent of how the source episode "
            "was recorded. Omit for the authored, unjittered camera (default)."
        ),
    )
    parser.add_argument(
        "--camera-seed",
        type=int,
        default=0,
        help="root seed for --camera-randomization's per-episode draw (default: 0)",
    )
    parser.add_argument(
        "--background-randomization",
        type=Path,
        default=None,
        help=(
            "domain-randomization preset whose background/table colour, blur and blob-count "
            "ranges draw a fresh procedural background+table texture per episode. Only takes "
            "effect together with --background-panorama/--table-texture (the finite-floor "
            "scene); without it there is no separate skybox to vary. Omit to keep the same "
            "static image for every episode (default)."
        ),
    )
    parser.add_argument(
        "--background-seed",
        type=int,
        default=0,
        help="root seed for --background-randomization's per-episode draw (default: 0)",
    )
    parser.add_argument(
        "--first-episode",
        type=int,
        default=None,
        help="lowest staged episode index to include (default: the first one staged)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="episodes to process (default: all of them)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="render across N processes, each with its own scene and GL context",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "re-render with the recorded appearance and diff against the recorded "
            "video instead of writing datasets; writes verification.json"
        ),
    )
    parser.add_argument(
        "--verify-frames",
        type=int,
        default=24,
        help=(
            "frames per episode and camera to compare in --verify mode, evenly "
            "spaced over the episode (0 compares every frame; default: 24)"
        ),
    )
    parser.add_argument(
        "--max-mean-abs-diff",
        type=float,
        default=DEFAULT_MAX_MEAN_ABS_DIFF,
        help=(
            "verification gate: largest allowed mean absolute per-pixel difference "
            f"from the recorded video, in grey levels (default: {DEFAULT_MAX_MEAN_ABS_DIFF})"
        ),
    )
    parser.add_argument(
        "--max-noise-ratio",
        type=float,
        default=DEFAULT_MAX_NOISE_RATIO,
        help=(
            "verification gate: largest allowed ratio of that difference to the "
            f"codec's own noise floor (default: {DEFAULT_MAX_NOISE_RATIO})"
        ),
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=None,
        help=(
            "verification report to gate this render on (default: "
            f"<output>/{VERIFICATION_FILENAME}). Verifying against a couple of "
            "freshly recorded probe episodes and pointing every render at that "
            "one report is the intended workflow: what it establishes is that "
            "this environment reproduces a recording, not anything about the "
            "dataset being re-rendered"
        ),
    )
    parser.add_argument(
        "--skip-verification-check",
        action="store_true",
        help="render without a passing verification report for this environment",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="proceed even when the projected output does not fit on the volume",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be at least 1")
    if args.verify_frames < 0:
        parser.error("--verify-frames must not be negative")
    return args


def select_episodes(
    episodes_root: Path, *, first_episode: int | None, max_episodes: int | None
) -> list[Path]:
    """Complete staged episodes in global-index order, filtered by the CLI range."""
    episodes = find_episode_datasets(episodes_root)
    if first_episode is not None:
        episodes = [path for path in episodes if episode_index(path) >= first_episode]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise FileNotFoundError(f"no complete staged episodes selected under {episodes_root}")
    return episodes


def collection_config(episodes_root: Path) -> dict[str, Any]:
    path = episodes_root / COLLECTION_CONFIG_FILENAME
    if not path.exists():
        return {}
    with path.open() as file:
        return json.load(file)


def render_sizes(episode: Path, config: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(render_hw, image_hw)`` for a staged episode.

    The saved image size is whatever the episode's videos carry; the source
    render size it was downsampled from is only recorded in the staging area's
    collection config, because it leaves no trace in the dataset itself.
    """
    image_hw = video_render_hw(episode)
    if config:
        configured = (int(config["image_height"]), int(config["image_width"]))
        if configured != image_hw:
            raise ValueError(
                f"{episode.name} videos are {image_hw}, but the collection config "
                f"records {configured}"
            )
        render_hw = (int(config["render_height"]), int(config["render_width"]))
    else:
        render_hw = FALLBACK_RENDER_HW
        print(
            f"warning: no {COLLECTION_CONFIG_FILENAME}; assuming the recording rendered at "
            f"{render_hw[1]}x{render_hw[0]} before downsampling",
            file=sys.stderr,
        )
    return render_hw, image_hw


def rerender_episode(
    renderer: EpisodeRenderer,
    source: Path,
    variants: list[Variant],
    output: Path,
    *,
    fps: int,
    camera_randomization: CameraRandomization | None = None,
    background_randomization: BackgroundRandomization | None = None,
) -> int:
    """Write every variant of one episode; return the frames rendered per variant."""
    assert_rerenderable(source)
    truth = load_episode_truth(source)
    source_videos = episode_video_paths(source)
    settings = {feature: x264_settings(path) for feature, path in source_videos.items()}
    frames = len(truth.states)
    for feature, path in source_videos.items():
        recorded = video_frame_count(path)
        if recorded != frames:
            raise ValueError(
                f"{source.name} {feature} holds {recorded} frames against "
                f"{frames} recorded rows"
            )

    staging: dict[str, Path] = {}
    writers: dict[tuple[str, str], Any] = {}
    stats: dict[tuple[str, str], ImageStatsAccumulator] = {}
    for variant in variants:
        partial = variant.staging_root(output) / f".{source.name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        copy_episode_scaffold(source, partial)
        staging[variant.name] = partial
        for feature, video_path in source_videos.items():
            destination = partial / video_path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            writers[(variant.name, feature)] = VideoWriter(
                destination,
                width=renderer.rig.width,
                height=renderer.rig.height,
                fps=fps,
                settings=settings[feature],
            )
            stats[(variant.name, feature)] = ImageStatsAccumulator()

    camera_jitter = (
        camera_randomization.draw(episode_index(source))
        if camera_randomization is not None
        else None
    )
    scene_texture = (
        background_randomization.draw(episode_index(source))
        if background_randomization is not None
        else None
    )
    try:
        renderer.set_episode(
            truth.target_xy,
            truth.target_plate_yaw,
            camera_jitter=camera_jitter,
            scene_texture=scene_texture,
        )
        for index in range(frames):
            renderer.set_frame(truth.states[index], truth.cube_poses[index])
            for variant in variants:
                images = renderer.capture(variant.appearance)
                for feature, image in images.items():
                    writers[(variant.name, feature)].write(image)
                    stats[(variant.name, feature)].add(image)
    finally:
        for writer in writers.values():
            writer.close()

    for variant in variants:
        partial = staging[variant.name]
        for feature, video_path in source_videos.items():
            written = video_frame_count(partial / video_path.relative_to(source))
            if written != frames:
                raise RuntimeError(
                    f"{source.name} {variant.name} {feature} wrote {written} frames "
                    f"against {frames} recorded rows"
                )
        rewrite_image_stats(
            partial / "meta" / "stats.json",
            {feature: stats[(variant.name, feature)].result() for feature in source_videos},
        )
        final = partial.with_name(source.name)
        if final.exists():
            shutil.rmtree(final)
        os.replace(partial, final)
    return frames


def render_episodes(
    episodes: list[Path],
    variants: list[Variant],
    output: Path,
    *,
    render_hw: tuple[int, int],
    image_hw: tuple[int, int],
    fps: int,
    camera_randomization: CameraRandomization | None = None,
    background_randomization: BackgroundRandomization | None = None,
    background_panorama: Path | None = None,
    table_texture: Path | None = None,
    label: str = "",
) -> tuple[int, int]:
    """Re-render a list of episodes; return ``(written, skipped)`` counts."""
    pending = [
        episode
        for episode in episodes
        if not all(
            (variant.staging_root(output) / episode.name / "meta" / "info.json").is_file()
            for variant in variants
        )
    ]
    skipped = len(episodes) - len(pending)
    if skipped:
        print(f"{label}{skipped} episode(s) already re-rendered; skipping them.", flush=True)
    if not pending:
        return 0, skipped

    renderer = EpisodeRenderer(
        render_hw=render_hw,
        image_hw=image_hw,
        background_panorama=background_panorama,
        table_texture=table_texture,
    )
    written = 0
    try:
        for position, episode in enumerate(pending, start=1):
            started = time.monotonic()
            frames = rerender_episode(
                renderer,
                episode,
                variants,
                output,
                fps=fps,
                camera_randomization=camera_randomization,
                background_randomization=background_randomization,
            )
            written += 1
            elapsed = time.monotonic() - started
            print(
                f"{label}{episode.name}: {frames} frames x {len(variants)} variant(s) "
                f"in {elapsed:.1f}s ({position}/{len(pending)})",
                flush=True,
            )
    finally:
        renderer.close()
    return written, skipped


def _worker(payload: dict[str, Any]) -> None:
    """multiprocessing entry point: re-render one shard of the episode list."""
    camera_randomization = (
        CameraRandomization(**payload["camera_randomization"])
        if payload["camera_randomization"] is not None
        else None
    )
    background_randomization = (
        BackgroundRandomization.from_preset(
            Path(payload["background_randomization"]["preset_path"]),
            seed=payload["background_randomization"]["seed"],
        )
        if payload["background_randomization"] is not None
        else None
    )
    render_episodes(
        [Path(path) for path in payload["episodes"]],
        [Variant(name, SceneAppearance(**fields)) for name, fields in payload["variants"]],
        Path(payload["output"]),
        render_hw=tuple(payload["render_hw"]),
        image_hw=tuple(payload["image_hw"]),
        fps=payload["fps"],
        camera_randomization=camera_randomization,
        background_randomization=background_randomization,
        background_panorama=(
            Path(payload["background_panorama"]) if payload["background_panorama"] else None
        ),
        table_texture=Path(payload["table_texture"]) if payload["table_texture"] else None,
        label=payload["label"],
    )


def render_in_pool(
    episodes: list[Path],
    variants: list[Variant],
    output: Path,
    *,
    render_hw: tuple[int, int],
    image_hw: tuple[int, int],
    fps: int,
    workers: int,
    camera_randomization: CameraRandomization | None = None,
    background_randomization: BackgroundRandomization | None = None,
    background_panorama: Path | None = None,
    table_texture: Path | None = None,
) -> None:
    """Shard the episodes across worker processes, each with its own GL context."""
    # Spawn rather than fork: a MuJoCo GL context does not survive a fork.
    ctx = multiprocessing.get_context("spawn")
    shards = [episodes[index::workers] for index in range(workers)]
    serialized_variants = [
        (variant.name, dataclasses.asdict(variant.appearance)) for variant in variants
    ]
    serialized_camera_randomization = (
        dataclasses.asdict(camera_randomization) if camera_randomization is not None else None
    )
    serialized_background_randomization = (
        {"preset_path": str(background_randomization.preset_path), "seed": background_randomization.seed}
        if background_randomization is not None
        else None
    )
    processes = []
    for worker_id, shard in enumerate(shards):
        if not shard:
            continue
        payload = {
            "episodes": [str(path) for path in shard],
            "variants": serialized_variants,
            "output": str(output),
            "render_hw": list(render_hw),
            "image_hw": list(image_hw),
            "fps": fps,
            "camera_randomization": serialized_camera_randomization,
            "background_randomization": serialized_background_randomization,
            "background_panorama": str(background_panorama) if background_panorama else None,
            "table_texture": str(table_texture) if table_texture else None,
            "label": f"[w{worker_id}] ",
        }
        process = ctx.Process(target=_worker, args=(payload,))
        process.start()
        processes.append(process)
    failed = []
    for worker_id, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed.append((worker_id, process.exitcode))
    if failed:
        raise RuntimeError(f"re-render worker(s) failed: {failed}")


def verify_episodes(
    episodes: list[Path],
    *,
    render_hw: tuple[int, int],
    image_hw: tuple[int, int],
    fps: int,
    verify_frames: int,
    panel_dir: Path,
    background_panorama: Path | None = None,
    table_texture: Path | None = None,
) -> dict[str, Any]:
    """Re-render the recorded appearance and diff it against the recorded videos.

    Three quantities per camera, all in grey levels:

    ``mean_abs_diff``
        re-rendered frame against the decoded recorded frame — the measure that
        matters, and the one that includes every source of mismatch.
    ``codec_noise_floor``
        the same re-rendered frames encoded with the episode's own H.264
        settings and decoded back, against themselves. This is what an exact
        re-render would still score, because the recording is lossy.
    ``end_to_end``
        the re-encoded re-render against the recorded video: what a consumer
        that reads both datasets would actually see.
    """
    renderer = EpisodeRenderer(
        render_hw=render_hw,
        image_hw=image_hw,
        background_panorama=background_panorama,
        table_texture=table_texture,
    )
    per_camera: dict[str, dict[str, Any]] = {
        feature: {"diffs": [], "floors": [], "end_to_end": [], "worst": None}
        for feature in CAMERA_FEATURES
    }
    compared = 0
    try:
        for episode in episodes:
            assert_rerenderable(episode)
            truth = load_episode_truth(episode)
            videos = episode_video_paths(episode)
            indices = evenly_spaced(len(truth.states), verify_frames)
            wanted = set(indices)
            recorded = {
                feature: {
                    index: frame
                    for index, frame in enumerate(decode_video(path))
                    if index in wanted
                }
                for feature, path in videos.items()
            }
            renderer.set_episode(truth.target_xy, truth.target_plate_yaw)
            rendered: dict[str, list[np.ndarray]] = {feature: [] for feature in videos}
            for index in indices:
                renderer.set_frame(truth.states[index], truth.cube_poses[index])
                for feature, image in renderer.capture(SceneAppearance()).items():
                    rendered[feature].append(image)
            compared += len(indices)

            for feature, path in videos.items():
                settings = x264_settings(path)
                decoded = encode_decode(rendered[feature], fps=fps, settings=settings)
                camera = per_camera[feature]
                for position, index in enumerate(indices):
                    original = recorded[feature][index]
                    render = rendered[feature][position]
                    diff = FrameDiff.between(render, original)
                    camera["diffs"].append(diff)
                    camera["floors"].append(FrameDiff.between(render, decoded[position]).mean)
                    camera["end_to_end"].append(
                        FrameDiff.between(decoded[position], original).mean
                    )
                    if camera["worst"] is None or diff.mean > camera["worst"][0].mean:
                        camera["worst"] = (diff, episode.name, index, original, render)
    finally:
        renderer.close()

    summary: dict[str, Any] = {}
    for feature, camera in per_camera.items():
        diffs = camera["diffs"]
        floor = mean_of(camera["floors"])
        mean_abs = mean_of(diff.mean for diff in diffs)
        worst_diff, worst_episode, worst_index, original, render = camera["worst"]
        summary[feature] = {
            "frames_compared": len(diffs),
            "mean_abs_diff": mean_abs,
            "codec_noise_floor": floor,
            "noise_ratio": mean_abs / floor if floor > 0 else float("inf"),
            "end_to_end_mean_abs_diff": mean_of(camera["end_to_end"]),
            "p99_abs_diff": mean_of(diff.p99 for diff in diffs),
            "max_abs_diff": max(diff.max for diff in diffs),
            "worst_frame": {
                "episode": worst_episode,
                "frame": worst_index,
                "mean_abs_diff": worst_diff.mean,
                "panel": str(
                    _write_panel(panel_dir, feature, worst_episode, worst_index, original, render)
                ),
            },
        }
    summary["frames_compared"] = compared
    return summary


def _write_panel(
    panel_dir: Path,
    feature: str,
    episode: str,
    index: int,
    original: np.ndarray,
    render: np.ndarray,
) -> Path:
    """Save recorded | re-rendered | amplified difference, side by side."""
    import cv2

    panel_dir.mkdir(parents=True, exist_ok=True)
    difference = np.abs(original.astype(np.int16) - render.astype(np.int16))
    amplified = np.clip(difference * 4, 0, 255).astype(np.uint8)
    panel = np.concatenate([original, render, amplified], axis=1)
    path = panel_dir / f"{episode}-{feature.rsplit('.', 1)[-1]}-f{index:04d}.png"
    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return path


def check_disk_space(
    episodes: list[Path], variants: list[Variant], output: Path, *, force: bool
) -> None:
    """Refuse to start a run whose output would not fit, before rendering anything."""
    sample = episodes[: min(5, len(episodes))]
    per_episode = int(np.mean([directory_size(episode) for episode in sample]))
    projected = per_episode * len(episodes) * len(variants)
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    print(
        f"Projected output: {projected / 1024**2:,.0f} MiB "
        f"({len(episodes)} episodes x {len(variants)} variant(s), "
        f"{per_episode / 1024**2:.1f} MiB each); {free / 1024**3:.1f} GiB free."
    )
    if projected + DISK_HEADROOM_BYTES > free and not force:
        raise SystemExit(
            f"refusing to start: {projected / 1024**3:.1f} GiB projected against "
            f"{free / 1024**3:.1f} GiB free (keeping {DISK_HEADROOM_BYTES / 1024**3:.0f} GiB "
            "spare). Re-run with fewer --max-episodes, fewer variants, or --force."
        )


def require_verification(
    path: Path, episodes_root: Path, fingerprint: dict[str, Any]
) -> dict[str, Any]:
    """Refuse to render unless this environment has been verified against a recording.

    Returns the report, which the per-variant manifest records so a staged
    dataset carries the evidence it was produced under.
    """
    hint = (
        f"Run --verify (against {episodes_root} or a couple of freshly recorded probe "
        "episodes) and pass --verification <report>, or --skip-verification-check to "
        "render anyway (the result is then not known to reproduce a recording)."
    )
    if not path.exists():
        raise SystemExit(f"no verification report at {path}. {hint}")
    with path.open() as file:
        report = json.load(file)
    if report.get("status") != "pass":
        raise SystemExit(f"{path} records status {report.get('status')!r}. {hint}")
    if report.get("environment") != fingerprint:
        differing = sorted(
            key
            for key in set(report.get("environment", {})) | set(fingerprint)
            if report.get("environment", {}).get(key) != fingerprint.get(key)
        )
        raise SystemExit(
            f"{path} was produced in a different render environment (differs in "
            f"{differing}). {hint}"
        )
    return report


def write_manifest(
    staging_root: Path,
    *,
    variant: Variant,
    episodes_root: Path,
    episodes: list[Path],
    fingerprint: dict[str, Any],
    config: dict[str, Any],
    verification: dict[str, Any] | None,
    camera_randomization: CameraRandomization | None = None,
    background_randomization: BackgroundRandomization | None = None,
) -> None:
    """Record what this staged root is, next to the collection config it inherits."""
    staging_root.mkdir(parents=True, exist_ok=True)
    if config:
        with (staging_root / COLLECTION_CONFIG_FILENAME).open("w") as file:
            json.dump(config, file, indent=2, sort_keys=True)
            file.write("\n")
    manifest = {
        "rerendered_from": str(episodes_root.resolve()),
        "variant": variant.name,
        "appearance": variant.appearance.describe(),
        "appearance_label": variant.appearance.name,
        "episodes": [episode.name for episode in episodes],
        "environment": fingerprint,
        "verified_against": (
            {
                "source": verification.get("source"),
                "episodes": verification.get("episodes"),
                "cameras": {
                    feature: {
                        key: camera[key]
                        for key in ("mean_abs_diff", "codec_noise_floor", "noise_ratio")
                    }
                    for feature, camera in verification.get("cameras", {}).items()
                },
            }
            if verification is not None
            else None
        ),
        "camera_randomization": (
            dataclasses.asdict(camera_randomization) if camera_randomization is not None else None
        ),
        "background_randomization": (
            {
                "preset_path": str(background_randomization.preset_path),
                "preset_name": background_randomization.preset.name,
                "seed": background_randomization.seed,
            }
            if background_randomization is not None
            else None
        ),
        "notes": (
            "Videos were re-rendered from the recorded per-frame ground truth; parquet "
            "data and metadata are copies of the recording. meta/stats.json keeps every "
            "recorded entry except the camera features, which describe the new pixels."
        ),
    }
    with (staging_root / MANIFEST_FILENAME).open("w") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> None:
    args = _parse_args()
    episodes_root = args.episodes_root.resolve()
    output = args.output.resolve()
    episodes = select_episodes(
        episodes_root, first_episode=args.first_episode, max_episodes=args.max_episodes
    )
    config = collection_config(episodes_root)
    render_hw, image_hw = render_sizes(episodes[0], config)
    fps = episode_video_fps(episodes[0])
    fingerprint = render_environment_fingerprint(render_hw=render_hw, image_hw=image_hw)
    print(
        f"{len(episodes)} episode(s) from {episodes_root}: "
        f"{image_hw[1]}x{image_hw[0]} at {fps} fps, rendered from "
        f"{render_hw[1]}x{render_hw[0]}."
    )

    if args.verify:
        output.mkdir(parents=True, exist_ok=True)
        summary = verify_episodes(
            episodes,
            render_hw=render_hw,
            image_hw=image_hw,
            fps=fps,
            verify_frames=args.verify_frames,
            panel_dir=output / "verify_panels",
            background_panorama=args.background_panorama,
            table_texture=args.table_texture,
        )
        failures = []
        for feature in CAMERA_FEATURES:
            camera = summary[feature]
            if camera["mean_abs_diff"] > args.max_mean_abs_diff:
                failures.append(
                    f"{feature}: mean abs diff {camera['mean_abs_diff']:.2f} exceeds "
                    f"{args.max_mean_abs_diff:.2f} grey levels"
                )
            if camera["noise_ratio"] > args.max_noise_ratio:
                failures.append(
                    f"{feature}: mean abs diff is {camera['noise_ratio']:.1f}x the codec "
                    f"noise floor, over the {args.max_noise_ratio:.1f}x limit"
                )
        report = {
            "source": str(episodes_root),
            "episodes": [episode.name for episode in episodes],
            "frames_per_episode": args.verify_frames or "all",
            "environment": fingerprint,
            "thresholds": {
                "max_mean_abs_diff": args.max_mean_abs_diff,
                "max_noise_ratio": args.max_noise_ratio,
            },
            "cameras": {feature: summary[feature] for feature in CAMERA_FEATURES},
            "status": "fail" if failures else "pass",
            "failures": failures,
        }
        with (output / VERIFICATION_FILENAME).open("w") as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")
        for feature in CAMERA_FEATURES:
            camera = summary[feature]
            print(
                f"{feature}: mean {camera['mean_abs_diff']:.2f} grey levels vs recorded "
                f"(codec floor {camera['codec_noise_floor']:.2f}, "
                f"{camera['noise_ratio']:.1f}x), p99 {camera['p99_abs_diff']:.1f}, "
                f"max {camera['max_abs_diff']:.0f}; worst frame "
                f"{camera['worst_frame']['episode']} f{camera['worst_frame']['frame']}"
            )
        print(f"\nVerification {report['status'].upper()} -> {output / VERIFICATION_FILENAME}")
        for failure in failures:
            print(f"  {failure}")
        if failures:
            raise SystemExit(1)
        return

    variants = [parse_variant(token) for token in args.variant]
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate variant name(s) in {names}")
    camera_randomization = (
        CameraRandomization.from_preset(args.camera_randomization, seed=args.camera_seed)
        if args.camera_randomization is not None
        else None
    )
    if camera_randomization is not None:
        print(
            f"Camera randomization: {args.camera_randomization} (seed {args.camera_seed}) -> "
            f"±{camera_randomization.position_mm:g} mm / ±{camera_randomization.rotation_deg:g}° "
            f"/ ±{camera_randomization.focal_pct:g}% focal, "
            f"margin {camera_randomization.margin_px:g} px."
        )
    background_randomization = (
        BackgroundRandomization.from_preset(args.background_randomization, seed=args.background_seed)
        if args.background_randomization is not None
        else None
    )
    if background_randomization is not None:
        if args.background_panorama is None and args.table_texture is None:
            raise SystemExit(
                "--background-randomization needs --background-panorama and/or "
                "--table-texture to put the scene in finite-floor mode; without one, "
                "there is no separate skybox/table texture to vary."
            )
        print(
            f"Background randomization: {args.background_randomization} "
            f"(seed {args.background_seed}) -> fresh background/table texture per episode."
        )
    verification = None
    if not args.skip_verification_check:
        report_path = args.verification or (output / VERIFICATION_FILENAME)
        verification = require_verification(report_path.resolve(), episodes_root, fingerprint)
        print(f"Verified by {report_path} (source {verification.get('source')}).")
    check_disk_space(episodes, variants, output, force=args.force)

    for variant in variants:
        write_manifest(
            variant.staging_root(output),
            variant=variant,
            episodes_root=episodes_root,
            episodes=episodes,
            fingerprint=fingerprint,
            config=config,
            verification=verification,
            camera_randomization=camera_randomization,
            background_randomization=background_randomization,
        )
        print(f"{variant.name}: {variant.appearance.name} -> {variant.staging_root(output)}")

    started = time.monotonic()
    if args.workers > 1:
        render_in_pool(
            episodes,
            variants,
            output,
            render_hw=render_hw,
            image_hw=image_hw,
            fps=fps,
            workers=args.workers,
            camera_randomization=camera_randomization,
            background_randomization=background_randomization,
            background_panorama=args.background_panorama,
            table_texture=args.table_texture,
        )
    else:
        render_episodes(
            episodes,
            variants,
            output,
            render_hw=render_hw,
            image_hw=image_hw,
            fps=fps,
            camera_randomization=camera_randomization,
            background_randomization=background_randomization,
            background_panorama=args.background_panorama,
            table_texture=args.table_texture,
        )
    elapsed = time.monotonic() - started
    print(f"\nFinished {len(episodes)} selected episode(s) in {elapsed / 60:.1f} min.")
    for variant in variants:
        staged = find_episode_datasets(variant.staging_root(output))
        print(f"  {variant.name}: {len(staged)} staged -> {variant.staging_root(output)}")
    print(
        "\nFinalize a variant with:\n"
        f"  python scripts/pick_and_place/finalize_sim_dataset.py "
        f"--dataset-root {output / variants[0].name} --episodes N --write"
    )


if __name__ == "__main__":
    main()
