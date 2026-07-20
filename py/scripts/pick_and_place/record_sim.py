#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record pick-and-place LeRobotDatasets from the sim, mirroring ``real.py``.

Each run samples episodes, plays their trajectories under the model's
position-servo physics, and writes them straight into one LeRobotDataset
(``datasets/<timestamp>/`` by default) with the same schema the real arm
produces: per control tick, the measured joints as ``observation.state``, the
commanded set point as ``action``, and a 960x720 wrist and overhead image by
default. Cameras are rendered at 1920x1080 before downsampling so silhouettes
and shadow edges are antialiased. No hardware is involved.

Camera fields of view come from the calibrated intrinsics in
``config/camera_intrinsics``, so a sim frame matches the calibrated real camera
resolution by default.

The episode rollout is sequential within a process (stateful physics, one
persistent scene), so ``--workers N`` shards the run across N processes, each
writing its own ``<root>_shard<i>`` dataset that can be merged afterwards. Pose
sampling and rendering are pure CPU/GL — no training GPU is involved.

This is sim-only. To collect on the physical SO-101 follower, use ``real.py``.
"""

from __future__ import annotations

import argparse
import datetime
import multiprocessing
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm

from pick_and_place.camera_extrinsics import (
    apply_camera_extrinsics_to_model,
    load_local_camera_extrinsics,
)
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.dataset_metadata import cube_pose_metadata, placement_error_metadata
from pick_and_place.domain_randomization import (
    DomainRandomizationPreset,
    DomainRandomizer,
    domain_seed,
    generate_procedural_appearance,
    orient_cube,
)
from pick_and_place.episodes import (
    EpisodeSamplingError,
    _build_model,
    placement_error,
    prepare_episode,
    sample_cube,
)
from pick_and_place.executor import CONTROL_HZ, HARDWARE_SIMULATION_HZ
from pick_and_place.miscalibration import MiscalibrationDraw, MiscalibrationModel
from pick_and_place.recording import RecordingSession
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.paper_detection import DROP_ZONE_HALF_SIZE, place_paper_target_marker
from pick_and_place.sim_recorder import SimCameraRig, record_episode
from pick_and_place.workspace_overlays import PAN_AXIS, is_cube_drop_allowed


SAVED_IMAGE_WIDTH = 960
SAVED_IMAGE_HEIGHT = 720
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
SHADOW_MAP_SIZE = 8192
OFFSCREEN_SAMPLES = 8
SHADOW_CONE_SCALE = 0.4


class _MockViewer:
    """Stand-in for a passive viewer when running headless."""

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


def _to_cube(xy: tuple[float, float] | None) -> CubePose | None:
    return CubePose(x=xy[0], y=xy[1], z=CUBE_HALF_SIZE) if xy is not None else None


def _miscalibration_metadata(draw: MiscalibrationDraw) -> dict[str, float]:
    """Episode metadata recording the injected draw (believed-vs-true errors)."""
    metadata = {
        f"injected_offset_{name}_deg": float(value) for name, value in draw.base_offsets_deg.items()
    }
    dx, dy, dz, dyaw = draw.cube_belief_error
    tx, ty = draw.target_belief_error
    metadata.update(
        {
            "injected_cube_belief_dx": float(dx),
            "injected_cube_belief_dy": float(dy),
            "injected_cube_belief_dz": float(dz),
            "injected_cube_belief_dyaw": float(dyaw),
            "injected_target_belief_dx": float(tx),
            "injected_target_belief_dy": float(ty),
        }
    )
    return metadata


def run_recording(
    *,
    episodes: int,
    seed: int | None,
    dataset_root: Path,
    repo_id: str,
    task: str,
    source_xy: tuple[float, float] | None = None,
    target_xy: tuple[float, float] | None = None,
    background_panorama: Path | None = None,
    table_texture: Path | None = None,
    speed: float = 1.0,
    vcodec: str = "auto",
    streaming_encoding: bool = True,
    image_writer_threads: int = 4,
    image_width: int = SAVED_IMAGE_WIDTH,
    image_height: int = SAVED_IMAGE_HEIGHT,
    render_width: int = RENDER_WIDTH,
    render_height: int = RENDER_HEIGHT,
    use_viewer: bool = False,
    miscalibration: bool = False,
    domain_randomization: Path | None = None,
    label: str = "",
    first_episode: int = 0,
    max_attempts: int = 50,
    show_progress: bool = True,
    detector_crash_dump_dir: str | None = None,
) -> int:
    """Record ``episodes`` episodes into one LeRobotDataset; return the count saved.

    Builds a single persistent scene (the cube freejoint is repositioned and the
    arm reset each episode), renders the wrist/overhead cameras offscreen, and
    plays each sampled trajectory under physics. ``label`` prefixes log lines so
    parallel shards stay legible. ``first_episode`` is the global index of this
    shard's first episode, ensuring each episode's RNG stream is independent of
    the worker count. ``use_viewer`` opens the 3D viewer (single process only);
    shard workers always run headless.
    """
    preset = DomainRandomizationPreset.load(domain_randomization) if domain_randomization else None
    domain_seeds = (
        [_domain_seed(seed, first_episode + index) for index in range(episodes)]
        if preset is not None
        else []
    )
    initial_sample = preset.sample(domain_seeds[0]) if preset is not None else None
    if preset is not None:
        initial_appearance = generate_procedural_appearance(initial_sample)
        table_texture = initial_appearance.table_rgb
        background_panorama = initial_appearance.background_rgb
    source = _to_cube(source_xy)
    target = _to_cube(target_xy)

    # One persistent scene reused across episodes. The environment is required for
    # the overhead camera; calibrated extrinsics place it where the real one sits.
    dummy_source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(
        dummy_source,
        include_environment=True,
        paper_target_marker=True,
        background_panorama=background_panorama,
        table_texture=table_texture,
        offwidth=render_width,
        offheight=render_height,
    )
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    _configure_render_quality(model)
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)

    randomizer = DomainRandomizer(model) if preset is not None else None
    rig = SimCameraRig(
        model,
        load_local_camera_intrinsics(),
        width=image_width,
        height=image_height,
        render_width=render_width,
        render_height=render_height,
        postprocess=randomizer.postprocess if randomizer is not None else None,
    )

    recording = RecordingSession(
        repo_id=repo_id,
        root=dataset_root,
        task=task,
        fps=CONTROL_HZ,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
        image_writer_threads=image_writer_threads,
    )
    # LeRobot/Hugging Face otherwise emits a separate Map bar while finalizing
    # every episode, obscuring the one useful recording-level progress bar.
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()

    miscalibration_model = MiscalibrationModel() if miscalibration and preset is None else None
    viewer_cm = mujoco.viewer.launch_passive(model, data) if use_viewer else None
    viewer = viewer_cm.__enter__() if viewer_cm is not None else _MockViewer()

    recorded = 0
    try:
        progress = tqdm(
            range(episodes),
            desc=label.strip() or "recording",
            unit="ep",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for index in progress:
            if not viewer.is_running():
                if show_progress:
                    tqdm.write(f"{label}Viewer closed; stopping.")
                break
            global_episode = first_episode + index
            rng = _episode_rng(seed, global_episode)
            domain_seed = domain_seeds[index] if preset is not None else None
            sample = preset.sample(domain_seed) if preset is not None else None
            draw = (
                sample.miscalibration
                if sample is not None
                else (
                    miscalibration_model.sample(rng) if miscalibration_model is not None else None
                )
            )
            if sample is not None:
                randomizer.apply(sample)
                rig.reload_textures(randomizer.texture_ids)
            try:
                episode_source = source
                if sample is not None:
                    episode_source = orient_cube(
                        source if source is not None else sample_cube(rng),
                        sample.cube_orientation_index,
                    )
                episode = prepare_episode(
                    rng,
                    episode_source,
                    target,
                    model=model,
                    data=data,
                    verbose=False,
                    include_environment=True,
                    miscalibration=draw,
                    max_attempts=max_attempts,
                )
            except EpisodeSamplingError as exc:
                tqdm.write(f"{label}Skipping episode {index + 1}: {exc}")
                progress.set_postfix(saved=recorded, skipped=index + 1 - recorded)
                continue

            # Render the black drop-zone square at the episode's target so the
            # frames match a real recording, where a physical paper square sits on
            # the table marking where the cube must be placed.
            ep_target = episode.target
            place_paper_target_marker(
                model,
                (ep_target.x, ep_target.y),
                0.0,
                (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
                usable=is_cube_drop_allowed(ep_target.x, ep_target.y),
                alpha=1.0,
            )
            if randomizer is not None:
                randomizer.tint_episode_markers()

            status = record_episode(
                episode,
                recording=recording,
                rig=rig,
                viewer=viewer if use_viewer else None,
                speed=speed,
                believed_wrist_camera_pose=(
                    randomizer.believed_wrist_camera_pose if randomizer is not None else None
                ),
                detector_crash_dump_dir=detector_crash_dump_dir,
                verbose=False,
            )
            if status != "success":
                if recording.has_pending_frames():
                    recording.discard_episode()
                progress.set_postfix(saved=recorded, skipped=index + 1 - recorded)
                continue
            error = placement_error(model, data, episode.target)
            metadata = cube_pose_metadata(episode.source, episode.target)
            metadata.update(placement_error_metadata(error, detected=True))
            if draw is not None:
                metadata.update(
                    {
                        "believed_cube_start_x": float(episode.believed_source.x),
                        "believed_cube_start_y": float(episode.believed_source.y),
                        "believed_cube_start_yaw": float(episode.believed_source.yaw),
                        "believed_target_x": float(episode.believed_target.x),
                        "believed_target_y": float(episode.believed_target.y),
                    }
                )
                metadata.update(_miscalibration_metadata(draw))
            if sample is not None:
                metadata.update(
                    {
                        "source_domain": "sim",
                        "domain_preset": preset.name,
                        "domain_seed": domain_seed,
                        "domain_sample_json": sample.metadata_json(),
                        "cube_start_roll": float(episode.source.roll),
                        "cube_start_pitch": float(episode.source.pitch),
                        "cube_orientation_index": sample.cube_orientation_index,
                    }
                )
            recording.save_episode(metadata)
            recorded += 1
            progress.set_postfix(saved=recorded, skipped=index + 1 - recorded)
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        rig.close()
        if recording.dataset is not None:
            recording.finalize()
    return recorded


def _configure_render_quality(model: mujoco.MjModel) -> None:
    """Use a dense, tightly focused shadow map for supersampled recordings."""
    model.vis.quality.shadowsize = SHADOW_MAP_SIZE
    model.vis.quality.offsamples = OFFSCREEN_SAMPLES
    model.vis.map.shadowscale = SHADOW_CONE_SCALE


def _worker(kwargs: dict) -> None:
    """multiprocessing entry point: record one shard, headless."""
    run_recording(**kwargs)


def _split(total: int, workers: int) -> list[int]:
    """Spread ``total`` episodes as evenly as possible over ``workers`` shards."""
    base, extra = divmod(total, workers)
    return [base + (1 if i < extra else 0) for i in range(workers)]


def _episode_rng(root_seed: int | None, global_episode: int) -> np.random.Generator:
    """Return the deterministic RNG stream for one globally numbered episode."""
    if root_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([root_seed, global_episode]))


def _domain_seed(root_seed: int | None, global_episode: int) -> int:
    """Stable per-episode seed for domain sampling, independent of pose draws."""
    return domain_seed(root_seed, global_episode)


def merge_shards(
    jobs: list[dict],
    *,
    output_root: Path,
    output_repo_id: str,
    keep_shards: bool,
) -> None:
    """Merge the non-empty shard datasets into one dataset at ``output_root``.

    Shards that recorded no episodes (all samples infeasible) are skipped. Unless
    ``keep_shards`` is set, the merged shard directories are removed afterwards.
    """
    import shutil

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = []
    used_roots: list[Path] = []
    for job in jobs:
        root = job["dataset_root"]
        if not (root / "meta" / "info.json").exists():
            continue
        dataset = LeRobotDataset(repo_id=job["repo_id"], root=root)
        if dataset.meta.total_episodes == 0:
            continue
        datasets.append(dataset)
        used_roots.append(root)

    if not datasets:
        print("No non-empty shards to merge.")
        return

    print(f"Merging {len(datasets)} shard(s) into {output_root}...")
    merged = merge_datasets(datasets, output_repo_id=output_repo_id, output_dir=output_root)
    print(
        f"Merged dataset: {merged.meta.total_episodes} episodes, "
        f"{merged.meta.total_frames} frames -> {output_root}"
    )
    if not keep_shards:
        for root in used_roots:
            shutil.rmtree(root, ignore_errors=True)
        print(f"Removed {len(used_roots)} shard dir(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to record")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="shard the run across N processes (each writes its own <root>_shard<i> dataset)",
    )
    parser.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after a sharded run, merge the shards into one dataset at <root> (default: on)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="keep the per-shard datasets after merging (default: remove them)",
    )
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the source cube (x, y); omit to resample each episode",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the target (x, y); omit to resample each episode",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier of the nominal trajectory pace (1.0 = nominal)",
    )
    parser.add_argument("--viewer", action="store_true", help="open the 3D MuJoCo viewer")
    parser.add_argument(
        "--miscalibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "inject per-episode draws of the measured real-robot miscalibration "
            "(joint-zero offsets, believed cube/target pose error): the plan runs "
            "in the believed frame, physics in the true frame, and the descent "
            "runs the wrist-camera visual servo like the real arm"
        ),
    )
    parser.add_argument(
        "--domain-randomization",
        type=Path,
        default=None,
        help="strict visual domain-randomization preset; includes measured miscalibration",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for pose sampling")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="trajectory resamples allowed per episode before skipping it (default: 50)",
    )
    parser.add_argument(
        "--detector-crash-dump-dir",
        default=None,
        help=(
            "save the wrist frame that crashes the AprilTag helper process here, "
            "for diagnosing the crash; the run itself continues either way"
        ),
    )
    parser.add_argument(
        "--background-panorama",
        type=Path,
        default=None,
        help="equirectangular room panorama to render as a skybox behind the scene",
    )
    parser.add_argument(
        "--table-texture",
        type=Path,
        default=None,
        help="top-down table texture (from reconstruct_table_texture.py) for the floor",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="output dir for the LeRobotDataset (default: datasets/<timestamp>)",
    )
    parser.add_argument(
        "--repo-id",
        default="local/pick-and-place-so101-sim",
        help="dataset repo id stored in metadata",
    )
    parser.add_argument(
        "--task",
        default="Pick up the cube and place it at the target.",
        help="natural-language task instruction saved with every frame",
    )
    parser.add_argument(
        "--vcodec",
        default="auto",
        help="LeRobot video codec (default: auto = best available HW encoder)",
    )
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode video during capture; --no-streaming-encoding falls back to PNG-then-encode",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="background image-writer threads for PNG-then-encode mode",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=SAVED_IMAGE_WIDTH,
        help=f"saved camera image width (default: {SAVED_IMAGE_WIDTH})",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=SAVED_IMAGE_HEIGHT,
        help=f"saved camera image height (default: {SAVED_IMAGE_HEIGHT})",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=RENDER_WIDTH,
        help=f"MuJoCo source render width before downsampling/cropping (default: {RENDER_WIDTH})",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=RENDER_HEIGHT,
        help=f"MuJoCo source render height before downsampling/cropping (default: {RENDER_HEIGHT})",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.image_width < 1 or args.image_height < 1:
        parser.error("--image-width and --image-height must be positive")
    if args.render_width < args.image_width or args.render_height < args.image_height:
        parser.error("--render-width and --render-height must be at least the image dimensions")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.viewer and args.workers > 1:
        parser.error("--viewer requires --workers 1")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(__file__).resolve().parents[2] / "datasets" / timestamp
    )

    common = dict(
        source_xy=tuple(args.source) if args.source is not None else None,
        target_xy=tuple(args.target) if args.target is not None else None,
        background_panorama=args.background_panorama,
        table_texture=args.table_texture,
        speed=args.speed,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        image_writer_threads=args.image_writer_threads,
        image_width=args.image_width,
        image_height=args.image_height,
        render_width=args.render_width,
        render_height=args.render_height,
        miscalibration=args.miscalibration,
        domain_randomization=args.domain_randomization,
        max_attempts=args.max_attempts,
        detector_crash_dump_dir=args.detector_crash_dump_dir,
    )

    if args.workers == 1:
        run_recording(
            episodes=args.episodes,
            seed=args.seed,
            dataset_root=base_root,
            repo_id=args.repo_id,
            task=args.task,
            use_viewer=args.viewer,
            **common,
        )
        return

    counts = _split(args.episodes, args.workers)
    jobs = []
    first_episode = 0
    for i, count in enumerate(counts):
        if count == 0:
            continue
        jobs.append(
            dict(
                episodes=count,
                seed=args.seed,
                first_episode=first_episode,
                dataset_root=base_root.with_name(f"{base_root.name}_shard{i}"),
                repo_id=f"{args.repo_id}-shard{i}",
                task=args.task,
                label=f"[shard {i}] ",
                show_progress=i == 0,
                **common,
            )
        )
        first_episode += count

    print(f"Sharding {args.episodes} episodes across {len(jobs)} workers (spawn).")
    # Spawn rather than fork: each worker needs its own MuJoCo GL context, which
    # does not survive a fork. Spawn is the default on macOS and safe on Linux.
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(job,)) for job in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    print("\nShard datasets:")
    for job in jobs:
        print(f"  {job['dataset_root']}")
    if failed:
        raise SystemExit(f"{len(failed)} shard worker(s) exited with an error: {failed}")

    if args.merge:
        merge_shards(
            jobs,
            output_root=base_root,
            output_repo_id=args.repo_id,
            keep_shards=args.keep_shards,
        )
    else:
        print("Skipping merge (--no-merge); combine later with merge_datasets.sh.")


if __name__ == "__main__":
    main()
