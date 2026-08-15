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
import math
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

from pick_and_place.core.camera_calibration import load_local_camera_intrinsics
from pick_and_place.sim.domain_randomization import (
    DomainRandomizationPreset,
    WristMountRandomizer,
    domain_seed,
    generate_procedural_appearance,
    orient_cube,
)
from pick_and_place.scripted.episode_sampling import sample_cube
from pick_and_place.plant.overhead import SimOverheadPerception
from pick_and_place.rollout.localized_episode import prepare_localized_episode
from pick_and_place.runtime.episodes import EpisodeSamplingError, prepare_episode
from pick_and_place.sim.model import placement_error
from pick_and_place.cli.dataset import add_dataset_arguments
from pick_and_place.data.recording_config import (
    DatasetOutput,
    FrameSizes,
    SAVED_IMAGE_HEIGHT,
    SAVED_IMAGE_WIDTH,
    SceneDraw,
)
from pick_and_place.cli.scene import (
    add_cube_pose_arguments,
    add_render_size_arguments,
    add_scene_texture_arguments,
)
from pick_and_place.spec.robot import CONTROL_HZ
from pick_and_place.core.miscalibration import MiscalibrationModel
from pick_and_place.core.grasp_perturbation import (
    DEFAULT_MAGNITUDE_M,
    GraspPerturbation,
)
from pick_and_place.data.recording import RecordingSession
from pick_and_place.variants.scene import AppearanceRandomizer
from pick_and_place.data.trajectory_artifact import render_environment_fingerprint
from pick_and_place.spec.workspace import CUBE_HALF_SIZE, DROP_ZONE_HALF_SIZE
from pick_and_place.core.geometry import CubePose
from pick_and_place.sim.paper_target_marker import place_paper_target_marker
from pick_and_place.core.paths import datasets_root
from pick_and_place.rollout.records import episode_metadata, save_episode_artifact
from pick_and_place.rollout.sim import SimCameraRig, build_recording_scene, record_episode
from pick_and_place.data.sim_dataset_staging import (
    episode_index,
    episode_staging_root,
    ensure_collection_config,
    find_episode_datasets,
    next_episode_index,
    successful_episode_datasets,
)
from pick_and_place.core.workspace_bounds import (
    PAN_AXIS,
    is_cube_drop_allowed,
    sample_target_plate_yaw,
)


# ~8.5x the ~35 s nominal episode under libx264, so an episode that burns many
# trajectory resamples (up to --max-attempts) is not mistaken for a wedge.
DEFAULT_EPISODE_TIMEOUT = 300.0

# Salt distinguishing the fumble-or-not stream from the pose and domain streams
# keyed off the same (seed, episode) pair. Arbitrary, and must not change: it is
# part of what makes a recorded episode a pure function of its index.
PERTURBATION_SEED_SALT = 0x50455254


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


def _prepare(
    rng,
    model,
    data,
    perception,
    *,
    source,
    target,
    miscalibration,
    grasp_perturbation,
    max_attempts: int,
):
    """Prepare one episode, and place the drop plate it will be recorded against.

    With ``perception``, the planner's belief comes from rendering the overhead
    camera and running the detector, and the plate has to be down before that
    happens — so the plate is placed as part of preparing the episode. Without
    it, the belief is the injected draw and the plate is placed afterwards, its
    yaw drawn last so an episode index keeps the poses it had before the plate
    started rotating.
    """
    if perception is not None:
        localized = prepare_localized_episode(
            rng,
            model,
            data,
            perception,
            source=source,
            target=target,
            include_environment=True,
            miscalibration=miscalibration,
            grasp_perturbation=grasp_perturbation,
            max_attempts=max_attempts,
        )
        return localized.episode, localized.target_plate_yaw

    episode = prepare_episode(
        rng,
        source,
        target,
        model=model,
        data=data,
        verbose=False,
        include_environment=True,
        miscalibration=miscalibration,
        grasp_perturbation=grasp_perturbation,
        max_attempts=max_attempts,
    )
    plate_yaw = sample_target_plate_yaw(
        rng, episode.target.x, episode.target.y, half_size=DROP_ZONE_HALF_SIZE
    )
    place_paper_target_marker(
        model,
        (episode.target.x, episode.target.y),
        plate_yaw,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=is_cube_drop_allowed(episode.target.x, episode.target.y),
        alpha=1.0,
    )
    return episode, plate_yaw


def run_recording(
    *,
    index_source: Callable[[], int | None],
    seed: int | None,
    output: DatasetOutput,
    scene: SceneDraw = SceneDraw(),
    frames: FrameSizes = FrameSizes(),
    heartbeat: Callable[[int | None], None] | None = None,
    speed: float = 1.0,
    use_viewer: bool = False,
    label: str = "",
    max_attempts: int = 50,
    show_progress: bool = True,
    detector_crash_dump_dir: str | None = None,
    perturbed_fraction: float = 0.0,
    perturbation_magnitude_m: float = DEFAULT_MAGNITUDE_M,
    perturbation_max_source_radius_m: float | None = None,
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
    preset = (
        DomainRandomizationPreset.load(scene.domain_randomization)
        if scene.domain_randomization
        else None
    )
    # Appearance is re-applied per episode from that episode's own domain seed,
    # so this initial sample only seeds the textures the scene is built with.
    initial_sample = preset.sample(_domain_seed(seed, 0)) if preset is not None else None
    table_texture = scene.table_texture
    background_panorama = scene.background_panorama
    if preset is not None:
        initial_appearance = generate_procedural_appearance(initial_sample.appearance())
        table_texture = initial_appearance.table_rgb
        background_panorama = initial_appearance.background_rgb
    source = _to_cube(scene.source_xy)
    target = _to_cube(scene.target_xy)
    # Constant for the run: it identifies this machine's renderer and camera
    # calibration, not anything an episode draws.
    fingerprint = render_environment_fingerprint(
        render_hw=(frames.render_height, frames.render_width),
        image_hw=(frames.image_height, frames.image_width),
    )

    # One persistent scene reused across episodes. The environment is required for
    # the overhead camera; calibrated extrinsics place it where the real one sits.
    # `rerender_episodes.py` replays recordings through this same builder.
    model, data = build_recording_scene(
        render_width=frames.render_width,
        render_height=frames.render_height,
        background_panorama=background_panorama,
        table_texture=table_texture,
    )

    # The preset's two halves are applied by different owners: the wrist mount
    # while the trajectory is generated, the appearance while it is rendered.
    wrist_mount = WristMountRandomizer(model) if preset is not None else None
    randomizer = AppearanceRandomizer(model) if preset is not None else None
    rig = SimCameraRig(
        model,
        load_local_camera_intrinsics(),
        width=frames.image_width,
        height=frames.image_height,
        render_width=frames.render_width,
        render_height=frames.render_height,
        postprocess=randomizer.postprocess if randomizer is not None else None,
    )

    # LeRobot/Hugging Face otherwise emits a separate Map bar while finalizing
    # every episode, obscuring the one useful recording-level progress bar.
    from datasets.utils.logging import disable_progress_bar

    disable_progress_bar()

    miscalibration_model = (
        MiscalibrationModel() if scene.miscalibration and preset is None else None
    )
    # Rendered at detection resolution, separately from the recorded
    # observations: the detector needs more pixels than a dataset frame carries.
    perception = (
        SimOverheadPerception(model, data, detector=None)
        if scene.overhead_perception
        else None
    )
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
            episode_root = output.root / f"ep{global_episode:06d}"
            if episode_root.exists():
                if (episode_root / "meta" / "info.json").is_file():
                    raise FileExistsError(
                        f"refusing to overwrite complete staged episode {episode_root}"
                    )
                # A killed worker may leave an incomplete directory before the
                # watchdog retries the same deterministic global index.
                shutil.rmtree(episode_root, ignore_errors=True)
            recording = RecordingSession(
                repo_id=f"{output.repo_id}-ep{global_episode:06d}",
                root=episode_root,
                task=output.task,
                fps=CONTROL_HZ,
                vcodec=output.vcodec,
                streaming_encoding=output.streaming_encoding,
                image_writer_threads=output.image_writer_threads,
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
                randomizer.apply(sample.appearance())
                wrist_mount.apply(sample)
                rig.reload_textures(randomizer.texture_ids)
            # Every recording draws one of the cube's 24 rotational orientations,
            # not just its yaw -- a DR preset's own draw when one is active,
            # otherwise a fresh one from this episode's RNG, so a plain recording
            # still trains on cubes resting on any face rather than always the
            # same one.
            orientation_index = (
                sample.cube_orientation_index if sample is not None else int(rng.integers(24))
            )
            episode_source = orient_cube(
                source if source is not None else sample_cube(rng), orientation_index
            )
            # Deliberate fumbles, on a minority of episodes. Drawn from a stream
            # keyed off the global episode index rather than from `rng`, so
            # turning the fraction or source-radius gate up or down leaves every
            # episode's poses untouched. This keeps dataset arms paired: only the
            # deliberate perturbation changes.
            perturbation = _sample_grasp_perturbation(
                seed,
                global_episode,
                episode_source,
                fraction=perturbed_fraction,
                magnitude_m=perturbation_magnitude_m,
                max_source_radius_m=perturbation_max_source_radius_m,
            )
            if perception is not None:
                perception.set_error(draw.overhead_camera_error)
            try:
                episode, target_plate_yaw = _prepare(
                    rng,
                    model,
                    data,
                    perception,
                    source=episode_source,
                    target=target,
                    miscalibration=draw,
                    grasp_perturbation=perturbation,
                    max_attempts=max_attempts,
                )
            except EpisodeSamplingError as exc:
                tqdm.write(f"{label}Skipping episode {global_episode}: {exc}")
                progress.set_postfix(saved=recorded, skipped=attempted - recorded)
                continue

            if randomizer is not None:
                randomizer.tint_episode_markers()
            # `place_paper_target_marker` writes `model.body_pos`/`body_quat`,
            # but rendering reads the derived `data.xpos`. Without this the
            # episode's first frame is rendered from the previous episode's
            # kinematics, showing the plate at its old target for one frame
            # before it snaps into place.
            mujoco.mj_forward(model, data)

            result = record_episode(
                episode,
                recording=recording,
                rig=rig,
                viewer=viewer if use_viewer else None,
                speed=speed,
                believed_wrist_camera_pose=(
                    wrist_mount.believed_wrist_camera_pose
                    if wrist_mount is not None
                    else None
                ),
                detector_crash_dump_dir=detector_crash_dump_dir,
                verbose=False,
            )
            if result.status != "success":
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
            metadata = episode_metadata(
                episode,
                result,
                error,
                target_plate_yaw=target_plate_yaw,
                orientation_index=orientation_index,
                perturbation=perturbation,
                draw=draw,
                sample=sample,
                preset_name=preset.name if preset is not None else None,
                domain_seed=domain_seed,
            )
            save_episode_artifact(
                episode_root,
                episode,
                result,
                target_plate_yaw=target_plate_yaw,
                draw=draw,
                sample=sample,
                seed=seed,
                episode_index=global_episode,
                fingerprint=fingerprint,
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
        if perception is not None:
            perception.close()
        rig.close()
        if recording is not None and recording.dataset is not None:
            recording.finalize()
    return recorded


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


def _perturbation_rng(root_seed: int | None, global_episode: int) -> np.random.Generator:
    """Deterministic stream deciding whether episode ``global_episode`` is fumbled.

    Salted so it is independent of the pose and domain streams. That independence
    is the point: changing ``--perturbed-fraction`` must not move any other
    episode's cube, or the perturbed and unperturbed dataset arms would differ in
    their entire pose distribution and the comparison would stop being paired.
    """
    if root_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(
        np.random.SeedSequence([root_seed, global_episode, PERTURBATION_SEED_SALT])
    )


def _sample_grasp_perturbation(
    root_seed: int | None,
    global_episode: int,
    source: CubePose,
    *,
    fraction: float,
    magnitude_m: float,
    max_source_radius_m: float | None,
) -> GraspPerturbation | None:
    """Sample the episode's fumble, optionally excluding distant cube starts.

    The probability and direction retain the original salted stream and draw
    order. The source-radius gate only suppresses a selected perturbation, so
    every included episode gets exactly the perturbation it did before the gate
    existed and every episode keeps its original pose stream.
    """
    if fraction <= 0.0:
        return None
    perturb_rng = _perturbation_rng(root_seed, global_episode)
    selected = perturb_rng.random() < fraction
    source_radius_m = math.hypot(source.x - PAN_AXIS[0], source.y - PAN_AXIS[1])
    within_radius = (
        max_source_radius_m is None or source_radius_m <= max_source_radius_m
    )
    if not selected or not within_radius:
        return None
    return GraspPerturbation.sample(perturb_rng, magnitude_m=magnitude_m)


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
    add_cube_pose_arguments(parser, source_yaw=False)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier of the nominal trajectory pace (1.0 = nominal)",
    )
    parser.add_argument("--viewer", action="store_true", help="open the 3D MuJoCo viewer")
    parser.add_argument(
        "--perturbed-fraction",
        type=float,
        default=0.0,
        help=(
            "fraction of episodes given a deliberate grasp fumble to recover from "
            "(default: 0.0, off). The planner aims at a displaced *believed* cube, "
            "misses, shoves the cube, and re-picks from where it actually ended up "
            "-- recorded as one episode. Keep it a minority (0.2-0.3): too many and "
            "the policy may learn that botching is survivable and get sloppier on "
            "the first attempt, which is the main risk here and is worth sweeping "
            "rather than guessing. The choice is recorded per episode as "
            "grasp_perturbation_kind, so a generated dataset can be re-filtered to "
            "a lower fraction without regenerating it"
        ),
    )
    parser.add_argument(
        "--perturbation-magnitude",
        type=float,
        default=DEFAULT_MAGNITUDE_M,
        help=(
            "planar magnitude of the injected belief error, metres (default: "
            f"{DEFAULT_MAGNITUDE_M}). Measured to miss reliably from 0.022 upward; "
            "the cube's 15 mm half-width is a floor, not the answer, because the "
            "planner re-selects among grasp candidates and absorbs part of it"
        ),
    )
    parser.add_argument(
        "--perturbation-max-source-radius",
        type=float,
        default=None,
        help=(
            "only perturb episodes whose cube starts at most this many metres "
            "from the shoulder-pan axis (default: no radius limit). The pose and "
            "salted perturbation draws remain unchanged for paired dataset arms"
        ),
    )
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
        "--overhead-perception",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "localize the cube and drop plate by rendering the overhead camera and "
            "running the detector, instead of taking the true poses and adding the "
            "draw's noise to them. The planner's belief error becomes an outcome of "
            "a calibration that is slightly wrong, and the arm blocking its own view "
            "becomes a real failure mode -- so the rig's hunt behavior runs here too. "
            "Costs one 1920x1080 render per search pose, once per episode"
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
    add_scene_texture_arguments(parser)
    add_dataset_arguments(
        parser,
        repo_id="local/pick-and-place-so101-sim",
        vcodec="h264",
        vcodec_help=(
            "LeRobot video codec (default: h264 = software libx264). Measured ~35 s/episode "
            "against ~122-167 s for h264_nvenc on a single-GPU machine: MuJoCo renders "
            "offscreen through EGL on that same GPU, so hardware encoding contends with "
            "rendering while software encoding runs on otherwise-idle cores. Prefer an "
            "explicit codec over 'auto', which probes for a hardware encoder and silently "
            "picks the slow path; pinning it also keeps one encoding profile across a dataset"
        ),
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
    add_render_size_arguments(parser)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.viewer and args.workers > 1:
        parser.error("--viewer requires --workers 1")
    if args.first_episode is not None and args.first_episode < 0:
        parser.error("--first-episode must not be negative")
    if (
        args.perturbation_max_source_radius is not None
        and args.perturbation_max_source_radius <= 0.0
    ):
        parser.error("--perturbation-max-source-radius must be positive")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = (
        args.dataset_root if args.dataset_root is not None else datasets_root() / timestamp
    )

    scene = SceneDraw(
        source_xy=tuple(args.source) if args.source is not None else None,
        target_xy=tuple(args.target) if args.target is not None else None,
        background_panorama=args.background_panorama,
        table_texture=args.table_texture,
        miscalibration=args.miscalibration,
        overhead_perception=args.overhead_perception,
        domain_randomization=args.domain_randomization,
    )
    try:
        frames = FrameSizes(
            render_width=args.render_width,
            render_height=args.render_height,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Episodes are staged as siblings of the eventual aggregate so collection
    # can be topped up and resumed without touching an already-finalized root.
    episodes_root = episode_staging_root(base_root)
    episodes_root.mkdir(parents=True, exist_ok=True)
    collection_config = {
        "format_version": 1,
        "seed": args.seed,
        "repo_id": args.repo_id,
        "task": args.task,
        "source_xy": scene.source_xy,
        "target_xy": scene.target_xy,
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
        "overhead_perception": args.overhead_perception,
        "domain_randomization": _configured_file(args.domain_randomization),
        "max_attempts": args.max_attempts,
        "perturbed_fraction": args.perturbed_fraction,
        "perturbation_magnitude_m": args.perturbation_magnitude,
        "perturbation_max_source_radius_m": args.perturbation_max_source_radius,
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
        output=DatasetOutput(
            root=episodes_root,
            repo_id=args.repo_id,
            task=args.task,
            vcodec=args.vcodec,
            streaming_encoding=args.streaming_encoding,
            image_writer_threads=args.image_writer_threads,
        ),
        scene=scene,
        frames=frames,
        speed=args.speed,
        max_attempts=args.max_attempts,
        detector_crash_dump_dir=args.detector_crash_dump_dir,
        perturbed_fraction=args.perturbed_fraction,
        perturbation_magnitude_m=args.perturbation_magnitude,
        perturbation_max_source_radius_m=args.perturbation_max_source_radius,
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
