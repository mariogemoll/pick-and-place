#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record pick-and-place LeRobotDatasets from the sim, mirroring ``real.py``.

Each run samples episodes, plays their trajectories under the model's
position-servo physics, and stages each completed trajectory as an independent
LeRobotDataset. They use the same schema the real arm produces: per control
tick, the measured joints as ``observation.state``, the commanded set point as
``action``, and a 960x720 wrist and overhead image by default. Cameras are
rendered at 1920x1080 before downsampling so silhouettes and shadow edges are
antialiased. No hardware is involved.

Camera fields of view come from the calibrated intrinsics in
``config/camera_intrinsics``, so a sim frame matches the calibrated real camera
resolution by default.

The episode rollout is sequential within a process (stateful physics, one
persistent scene), so ``--workers N`` runs a pool of N processes pulling
episode indices off a shared queue. Each episode is written as its own
single-episode dataset under ``<root>_episodes/`` and finalized immediately.
Repeated runs against the same root append new global episode indices, making
it possible to top up the staging area until it contains enough successful
placements. Run ``finalize_sim_dataset.py`` afterward to select exactly the
desired number and merge them into ``<root>`` without re-encoding video.

That granularity is also what bounds a failure: an episode that wedges or dies
costs only itself, never the episodes a worker had already banked. The parent
kills and replaces a worker whose episode exceeds ``--episode-timeout``. Pose
sampling and rendering are pure CPU/GL — no training GPU is involved.

This is sim-only. To collect on the physical SO-101 follower, use ``real.py``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import multiprocessing
import queue as queue_module
import shutil
import time
from collections.abc import Callable
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
from pick_and_place.sim_dataset_staging import (
    episode_index,
    episode_staging_root,
    ensure_collection_config,
    find_episode_datasets,
    next_episode_index,
    successful_episode_datasets,
)
from pick_and_place.workspace_overlays import (
    PAN_AXIS,
    is_cube_drop_allowed,
    sample_target_plate_yaw,
)


SAVED_IMAGE_WIDTH = 960
SAVED_IMAGE_HEIGHT = 720
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
SHADOW_MAP_SIZE = 8192
OFFSCREEN_SAMPLES = 8
SHADOW_CONE_SCALE = 0.4
# ~8.5x the ~35 s nominal episode under libx264, so an episode that burns many
# trajectory resamples (up to --max-attempts) is not mistaken for a wedge.
DEFAULT_EPISODE_TIMEOUT = 300.0


class _MockViewer:
    """Stand-in for a passive viewer when running headless."""

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


def _to_cube(xy: tuple[float, float] | None) -> CubePose | None:
    return CubePose(x=xy[0], y=xy[1], z=CUBE_HALF_SIZE) if xy is not None else None


def _configured_file(path: Path | None) -> dict[str, str] | None:
    """Identify a collection input by stable path and content hash."""
    if path is None:
        return None
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


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
    index_source: Callable[[], int | None],
    seed: int | None,
    dataset_root: Path,
    repo_id: str,
    task: str,
    heartbeat: Callable[[int | None], None] | None = None,
    source_xy: tuple[float, float] | None = None,
    target_xy: tuple[float, float] | None = None,
    background_panorama: Path | None = None,
    table_texture: Path | None = None,
    speed: float = 1.0,
    vcodec: str = "h264",
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
    max_attempts: int = 50,
    show_progress: bool = True,
    detector_crash_dump_dir: str | None = None,
) -> int:
    """Record episodes pulled from ``index_source``; return the count saved.

    Builds a single persistent scene (the cube freejoint is repositioned and the
    arm reset each episode), renders the wrist/overhead cameras offscreen, and
    plays each sampled trajectory under physics. ``label`` prefixes log lines so
    parallel workers stay legible.

    ``index_source`` yields the next global episode index, or ``None`` when the
    run is done. Pulling rather than owning a contiguous block means a worker
    that finishes early takes more work instead of idling, and a worker that
    dies costs only its in-flight episode. Every per-episode RNG stream is keyed
    off the global index, so which worker records which episode does not change
    what gets recorded.

    Each episode is written as its own single-episode LeRobotDataset under
    ``dataset_root`` and finalized immediately, then merged afterwards. A
    dataset is only readable once its parquet writers are closed, so finalizing
    per episode is what makes a killed worker cost one episode rather than
    every episode it had banked.

    ``heartbeat`` reports the in-flight global index (``None`` between
    episodes) so the parent's watchdog can tell a wedged worker from a slow one.
    ``use_viewer`` opens the 3D viewer (single process only); pool workers
    always run headless.
    """
    preset = DomainRandomizationPreset.load(domain_randomization) if domain_randomization else None
    # Appearance is re-applied per episode from that episode's own domain seed,
    # so this initial sample only seeds the textures the scene is built with.
    initial_sample = preset.sample(_domain_seed(seed, 0)) if preset is not None else None
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

    # LeRobot/Hugging Face otherwise emits a separate Map bar while finalizing
    # every episode, obscuring the one useful recording-level progress bar.
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()

    miscalibration_model = MiscalibrationModel() if miscalibration and preset is None else None
    viewer_cm = mujoco.viewer.launch_passive(model, data) if use_viewer else None
    viewer = viewer_cm.__enter__() if viewer_cm is not None else _MockViewer()

    recorded = 0
    attempted = 0
    recording: RecordingSession | None = None
    try:
        progress = tqdm(
            desc=label.strip() or "recording",
            unit="ep",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        while True:
            if not viewer.is_running():
                if show_progress:
                    tqdm.write(f"{label}Viewer closed; stopping.")
                break
            global_episode = index_source()
            if global_episode is None:
                break
            if heartbeat is not None:
                heartbeat(global_episode)
            attempted += 1
            progress.update(1)

            # One dataset per episode, finalized below. `record_episode` creates
            # it lazily on the first frame, once the camera shapes are known.
            episode_root = dataset_root / f"ep{global_episode:06d}"
            if episode_root.exists():
                if (episode_root / "meta" / "info.json").is_file():
                    raise FileExistsError(
                        f"refusing to overwrite complete staged episode {episode_root}"
                    )
                # A killed worker may leave an incomplete directory before the
                # watchdog retries the same deterministic global index.
                shutil.rmtree(episode_root, ignore_errors=True)
            recording = RecordingSession(
                repo_id=f"{repo_id}-ep{global_episode:06d}",
                root=episode_root,
                task=task,
                fps=CONTROL_HZ,
                vcodec=vcodec,
                streaming_encoding=streaming_encoding,
                image_writer_threads=image_writer_threads,
            )

            rng = _episode_rng(seed, global_episode)
            domain_seed = _domain_seed(seed, global_episode) if preset is not None else None
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
                tqdm.write(f"{label}Skipping episode {global_episode}: {exc}")
                progress.set_postfix(saved=recorded, skipped=attempted - recorded)
                continue

            # Render the black drop-zone square at the episode's target so the
            # frames match a real recording, where a physical paper square sits on
            # the table marking where the cube must be placed. The yaw is drawn
            # after `prepare_episode` so it does not perturb the pose stream: an
            # episode index keeps the poses it had before the plate rotated.
            ep_target = episode.target
            target_plate_yaw = sample_target_plate_yaw(
                rng, ep_target.x, ep_target.y, half_size=DROP_ZONE_HALF_SIZE
            )
            place_paper_target_marker(
                model,
                (ep_target.x, ep_target.y),
                target_plate_yaw,
                (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
                usable=is_cube_drop_allowed(ep_target.x, ep_target.y),
                alpha=1.0,
            )
            if randomizer is not None:
                randomizer.tint_episode_markers()
            # `place_paper_target_marker` writes `model.body_pos`/`body_quat`,
            # but rendering reads the derived `data.xpos`. Without this the
            # episode's first frame is rendered from the previous episode's
            # kinematics, showing the plate at its old target for one frame
            # before it snaps into place.
            mujoco.mj_forward(model, data)

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
                # An aborted episode leaves a dataset dir holding no episode;
                # drop it so the merge sees only banked episodes.
                recording.finalize()
                shutil.rmtree(episode_root, ignore_errors=True)
                recording = None
                progress.set_postfix(saved=recorded, skipped=attempted - recorded)
                continue
            error = placement_error(model, data, episode.target)
            metadata = cube_pose_metadata(episode.source, episode.target)
            metadata.update(placement_error_metadata(error, detected=True))
            metadata["target_plate_yaw"] = float(target_plate_yaw)
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
            # Close the parquet writers now: until finalize runs the files carry
            # no footer and are unreadable, which is exactly how a killed worker
            # used to lose every episode it had banked.
            recording.finalize()
            recording = None
            recorded += 1
            progress.set_postfix(saved=recorded, skipped=attempted - recorded)
            if heartbeat is not None:
                heartbeat(None)
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        rig.close()
        if recording is not None and recording.dataset is not None:
            recording.finalize()
    return recorded


def _configure_render_quality(model: mujoco.MjModel) -> None:
    """Use a dense, tightly focused shadow map for supersampled recordings."""
    model.vis.quality.shadowsize = SHADOW_MAP_SIZE
    model.vis.quality.offsamples = OFFSCREEN_SAMPLES
    model.vis.map.shadowscale = SHADOW_CONE_SCALE


def _worker(kwargs: dict, index_queue, status, worker_id: int) -> None:
    """multiprocessing entry point: pull episodes off the queue, headless.

    ``status[worker_id]`` is ``(global_episode, started_at)`` while an episode is
    in flight and ``(None, time)`` between episodes, which is what lets the
    parent's watchdog distinguish a wedged worker from an idle one.
    """

    def next_index() -> int | None:
        try:
            return index_queue.get_nowait()
        except queue_module.Empty:
            return None

    def report(global_episode: int | None) -> None:
        status[worker_id] = (global_episode, time.time())

    report(None)
    run_recording(index_source=next_index, heartbeat=report, **kwargs)


def _episode_rng(root_seed: int | None, global_episode: int) -> np.random.Generator:
    """Return the deterministic RNG stream for one globally numbered episode."""
    if root_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([root_seed, global_episode]))


def _domain_seed(root_seed: int | None, global_episode: int) -> int:
    """Stable per-episode seed for domain sampling, independent of pose draws."""
    return domain_seed(root_seed, global_episode)


def find_wedged_workers(
    status: dict,
    worker_ids,
    *,
    now: float,
    episode_timeout: float,
) -> list[tuple[int, int, float]]:
    """Return ``(worker_id, episode, age)`` for each worker past its deadline.

    A worker is only judged while an episode is in flight. Between episodes it
    reports ``None``, and an idle worker with an empty queue would otherwise
    look indistinguishable from a wedged one and be killed forever.
    """
    wedged = []
    for worker_id in worker_ids:
        episode, since = status.get(worker_id, (None, now))
        if episode is None:
            continue
        age = now - since
        if age > episode_timeout:
            wedged.append((worker_id, episode, age))
    return wedged


def claim_retry(attempts: dict[int, int], episode: int, episode_retries: int) -> bool:
    """Record a wedge against ``episode``; return whether to requeue it.

    Bounding this matters: requeuing unconditionally would spin forever on an
    index that wedges every time it is attempted.
    """
    attempts[episode] = attempts.get(episode, 0) + 1
    return attempts[episode] <= episode_retries


def run_pool(
    job: dict,
    *,
    indices: list[int],
    workers: int,
    episode_timeout: float,
    episode_retries: int = 1,
    poll_interval: float = 5.0,
) -> None:
    """Run ``indices`` across a pool of workers, replacing any that wedge.

    Workers pull from a shared queue, so a worker that finishes early takes more
    work rather than idling on a pre-assigned block. The parent watches each
    worker's in-flight episode: one that exceeds ``episode_timeout`` is killed
    and a replacement started. Workers have been observed to spin at 100% CPU
    indefinitely, both before recording anything and partway through a run; the
    previous ``join()`` on every worker meant one such worker hung the entire
    run silently. The timeout has to be enforced from out here because a wedged
    worker cannot time itself out -- it is not running Python that would notice.

    A killed episode is requeued at most ``episode_retries`` times and then
    abandoned. Unbounded requeuing would spin forever if an index wedges
    deterministically; abandoning costs one episode, and the loop already treats
    episodes as attempts rather than guaranteed successes.
    """
    # Spawn rather than fork: each worker needs its own MuJoCo GL context, which
    # does not survive a fork. Spawn is the default on macOS and safe on Linux.
    ctx = multiprocessing.get_context("spawn")
    index_queue = ctx.Queue()
    for index in indices:
        index_queue.put(index)
    status = ctx.Manager().dict()

    def start(worker_id: int):
        status[worker_id] = (None, time.time())
        proc = ctx.Process(
            target=_worker,
            args=(
                {**job, "label": f"[w{worker_id}] ", "show_progress": worker_id == 0},
                index_queue,
                status,
                worker_id,
            ),
        )
        proc.start()
        return proc

    procs = {worker_id: start(worker_id) for worker_id in range(workers)}
    killed = 0
    abandoned: list[int] = []
    attempts: dict[int, int] = {}
    try:
        while True:
            alive = {wid: p for wid, p in procs.items() if p.is_alive()}
            if not alive:
                break
            now = time.time()
            wedged = find_wedged_workers(
                status, list(alive), now=now, episode_timeout=episode_timeout
            )
            for worker_id, episode, age in wedged:
                # Kill it and replace it. The partial dataset dir has no
                # info.json, so the merge skips it.
                retry = claim_retry(attempts, episode, episode_retries)
                print(
                    f"\n[watchdog] worker {worker_id} stuck on episode {episode} "
                    f"for {age:.0f}s (limit {episode_timeout:.0f}s); killing and "
                    + ("requeuing" if retry else "abandoning it (retry limit reached)")
                )
                alive[worker_id].kill()
                alive[worker_id].join(timeout=30)
                if retry:
                    index_queue.put(episode)
                else:
                    abandoned.append(episode)
                killed += 1
                procs[worker_id] = start(worker_id)
            time.sleep(poll_interval)
    finally:
        for proc in procs.values():
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=30)

    failed = [wid for wid, p in procs.items() if p.exitcode not in (0, -9)]
    if killed:
        print(f"[watchdog] replaced {killed} wedged worker(s) during the run")
    if abandoned:
        print(f"[watchdog] abandoned episode(s) after repeated wedges: {sorted(abandoned)}")
    if failed:
        # Loud, not silent: the old code could not distinguish this from success.
        print(f"WARNING: worker(s) exited with an error: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="number of additional global episode indices to attempt",
    )
    parser.add_argument(
        "--first-episode",
        type=int,
        default=None,
        help=(
            "global index of the first new episode (default: one past every complete "
            "or partial episode already under <root>_episodes, otherwise 0). Each "
            "episode's pose and domain-randomization seeds are derived from --seed "
            "and this index; reuse the same seed for every top-up run"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "record across N processes pulling from a shared episode queue "
            "(each completed episode is independently finalized in the staging area)"
        ),
    )
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=DEFAULT_EPISODE_TIMEOUT,
        help=(
            "seconds a single episode may take before its worker is treated as "
            f"wedged, killed and replaced, and the episode requeued (default: "
            f"{DEFAULT_EPISODE_TIMEOUT:.0f}). Workers have been observed to spin "
            "at 100%% CPU forever; without this the run never returns"
        ),
    )
    parser.add_argument(
        "--episode-retries",
        type=int,
        default=1,
        help=(
            "times a wedged episode may be requeued before it is abandoned "
            "(default: 1). 0 marks it failed immediately. Unbounded retries "
            "would spin forever on an index that wedges every time"
        ),
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
        help=(
            "eventual finalized dataset path; collection writes to the sibling "
            "<root>_episodes directory (default base: datasets/<timestamp>)"
        ),
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
        default="h264",
        help=(
            "LeRobot video codec (default: h264 = software libx264). Measured "
            "~35 s/episode against ~122-167 s for h264_nvenc on a single-GPU "
            "machine: MuJoCo renders offscreen through EGL on that same GPU, so "
            "hardware encoding contends with rendering while software encoding "
            "runs on otherwise-idle cores. Prefer an explicit codec over 'auto', "
            "which probes for a hardware encoder and silently picks the slow "
            "path; pinning it also keeps one encoding profile across a dataset"
        ),
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
    if args.first_episode is not None and args.first_episode < 0:
        parser.error("--first-episode must not be negative")

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

    # Episodes are staged as siblings of the eventual aggregate so collection
    # can be topped up and resumed without touching an already-finalized root.
    episodes_root = episode_staging_root(base_root)
    episodes_root.mkdir(parents=True, exist_ok=True)
    collection_config = {
        "format_version": 1,
        "seed": args.seed,
        "repo_id": args.repo_id,
        "task": args.task,
        "source_xy": common["source_xy"],
        "target_xy": common["target_xy"],
        "background_panorama": _configured_file(args.background_panorama),
        "table_texture": _configured_file(args.table_texture),
        "speed": args.speed,
        "vcodec": args.vcodec,
        "streaming_encoding": args.streaming_encoding,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "render_width": args.render_width,
        "render_height": args.render_height,
        "miscalibration": args.miscalibration,
        "domain_randomization": _configured_file(args.domain_randomization),
        "max_attempts": args.max_attempts,
    }
    first_episode = (
        next_episode_index(episodes_root)
        if args.first_episode is None
        else args.first_episode
    )
    if first_episode > 0 and args.seed is None:
        parser.error("a top-up run requires --seed; reuse the staging area's original seed")
    complete_indices = {episode_index(path) for path in find_episode_datasets(episodes_root)}
    indices = list(range(first_episode, first_episode + args.episodes))
    conflicts = sorted(complete_indices.intersection(indices))
    if conflicts:
        parser.error(
            f"requested range overlaps complete staged episode(s): {conflicts[:10]}"
        )
    try:
        ensure_collection_config(episodes_root, collection_config)
    except ValueError as exc:
        parser.error(str(exc))

    job = dict(
        seed=args.seed,
        dataset_root=episodes_root,
        repo_id=args.repo_id,
        task=args.task,
        **common,
    )

    print(
        f"Recording {args.episodes} episodes "
        f"[{first_episode}, {first_episode + args.episodes}) "
        f"across {args.workers} worker(s) -> {episodes_root}"
    )

    if args.workers == 1 and args.viewer:
        # The viewer needs the main process, so skip the pool entirely.
        remaining = list(indices)
        run_recording(
            index_source=lambda: remaining.pop(0) if remaining else None,
            use_viewer=True,
            **job,
        )
    else:
        run_pool(
            job,
            indices=indices,
            workers=args.workers,
            episode_timeout=args.episode_timeout,
            episode_retries=args.episode_retries,
        )

    banked = find_episode_datasets(episodes_root)
    successful = successful_episode_datasets(banked)
    print(
        f"\nStaged totals in {episodes_root}: {len(banked)} complete, "
        f"{len(successful)} successful."
    )
    print(
        "Top up by running this recorder again with the same --dataset-root and --seed. "
        "Finalize with finalize_sim_dataset.py once enough successful episodes are staged."
    )


if __name__ == "__main__":
    main()
