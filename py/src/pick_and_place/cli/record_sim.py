# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record pick-and-place LeRobotDatasets from the sim, mirroring ``pap run-scripted-real``.

Each run samples episodes, plays their trajectories under the model's
position-servo physics, and stages each completed trajectory as an independent
LeRobotDataset. They use the same schema the real arm produces: per control
tick, the measured joints as ``observation.state``, the commanded set point as
``action``, and a 960x720 wrist and overhead image by default. Cameras are
rendered at 1920x1080 before downsampling so silhouettes and shadow edges are
antialiased. No hardware is involved.

Camera fields of view come from the authored optics in
:mod:`pick_and_place.spec.camera`, not from a measured rig, so a recording made
from a fresh clone is the same recording this machine makes. Domain
randomization perturbs the cameras around those authored poses, and its
envelope is what covers the deviation any particular rig has from them.

The episode rollout is sequential within a process (stateful physics, one
persistent scene), so ``--workers N`` runs a pool of N processes pulling
episode indices off a shared queue. Each episode is written as its own
single-episode dataset under ``<root>_episodes/`` and finalized immediately.
Repeated runs against the same root append new global episode indices, making
it possible to top up the staging area until it contains enough successful
placements. Run ``pap finalize-sim-dataset`` afterward to select exactly the
desired number and merge them into ``<root>`` without re-encoding video.

That granularity is also what bounds a failure: an episode that wedges or dies
costs only itself, never the episodes a worker had already banked. The parent
kills and replaces a worker whose episode exceeds ``--episode-timeout``. Pose
sampling and rendering are pure CPU/GL — no training GPU is involved.

This is sim-only. To collect on the physical SO-101 follower, use ``pap run-scripted-real``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import queue as queue_module
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import mujoco.viewer
from tqdm import tqdm

from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.spec.camera import CAMERA_INTRINSICS_BY_NAME
from pick_and_place.sim.domain_randomization import (
    DomainRandomizationPreset,
    WristMountRandomizer,
    generate_procedural_appearance,
    orient_cube,
)
from pick_and_place.sim.frozen_rig import FrozenRig
from pick_and_place.scripted.episode_sampling import sample_cube
from pick_and_place.core.physics import PhysicsModel
from pick_and_place.core.robot_dynamics import (
    load_robot_dynamics_config,
    tracking_bias_rad,
)
from pick_and_place.plant.overhead import SimOverheadPerception
from pick_and_place.plant.geometric_overhead import SimGeometricOverheadLocalizer
from pick_and_place.sim.physics import PhysicsRandomizer, tracking_bias_offsets
from pick_and_place.rollout.episode_setup import (
    appearance_seed,
    episode_rng,
    overhead_rng,
    pan_jitter_rng,
    physics_rng,
    prepare_for_recording,
    sample_grasp_perturbation,
)
from pick_and_place.runtime.episodes import EpisodeSamplingError
from pick_and_place.sim.model import placement_error
from pick_and_place.cli.dataset import add_dataset_arguments
from pick_and_place.data.recording_config import (
    DatasetOutput,
    FrameSizes,
    SAVED_IMAGE_HEIGHT,
    SAVED_IMAGE_WIDTH,
    SceneDraw,
)
from pick_and_place.cli.common import add_seed_argument
from pick_and_place.cli.scene import (
    add_cube_pose_arguments,
    add_domain_randomization_argument,
    add_miscalibration_argument,
    add_physics_randomization_argument,
    add_render_size_arguments,
    add_scene_texture_arguments,
    add_speed_argument,
    add_viewer_argument,
)
from pick_and_place.spec.robot import CONTROL_HZ
from pick_and_place.core.miscalibration import MiscalibrationModel, OverheadCameraModel
from pick_and_place.core.grasp_perturbation import (
    DEFAULT_MAGNITUDE_M,
)
from pick_and_place.data.recording import RecordingSession
from pick_and_place.variants.scene import AppearanceRandomizer
from pick_and_place.data.trajectory_artifact import render_environment_fingerprint
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.core.geometry import CubePose
from pick_and_place.core.paths import datasets_root
from pick_and_place.rollout.records import episode_metadata, save_episode_artifact
from pick_and_place.rollout.sim import SimCameraRig, build_recording_scene, record_episode
from pick_and_place.rollout.worker_pool import run_pool
from pick_and_place.data.sim_dataset_staging import (
    episode_index,
    is_complete_episode,
    episode_staging_root,
    ensure_collection_config,
    find_episode_datasets,
    next_episode_index,
    successful_episode_datasets,
)


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
    wrist_servo_mode: str = "geometric",
    perturbed_fraction: float = 0.0,
    perturbation_magnitude_m: float = DEFAULT_MAGNITUDE_M,
    perturbation_max_source_radius_m: float | None = None,
    ground_truth_drop_target: bool = False,
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
    frozen_rig = FrozenRig.load(scene.frozen_rig) if scene.frozen_rig is not None else None
    if frozen_rig is not None and preset is None:
        raise ValueError(
            "a frozen rig still needs a domain-randomization preset: it names the "
            "fields that keep varying rather than carrying an envelope to draw them from"
        )
    # Appearance is re-applied per episode from that episode's own domain seed,
    # so this initial sample only seeds the textures the scene is built with.
    initial_sample = preset.sample(appearance_seed(seed, 0)) if preset is not None else None
    if frozen_rig is not None and initial_sample is not None:
        initial_sample = frozen_rig.session(initial_sample, pan_jitter_rng(seed, 0))
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
    # Recordings are replayed through this same builder.
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
        CAMERA_INTRINSICS_BY_NAME,
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
    detector_perception = (
        SimOverheadPerception(model, data, detector=None)
        if scene.overhead_perception == "detector"
        else None
    )
    geometric_perception = (
        SimGeometricOverheadLocalizer(
            model,
            data,
            width=min(320, frames.render_width),
            height=min(240, frames.render_height),
        )
        if scene.overhead_perception == "geometric"
        else None
    )
    # A run is miscalibrated either because --miscalibration asked for it or
    # because a domain-randomization preset draws one per episode; the overhead
    # camera has to be off by the same token, or a randomized run would localize
    # perfectly while every other axis of its calibration was wrong. Without
    # either, the camera sits exactly where its calibration says and localizes
    # far better than the rig -- honest, and the reason --sim-perception is
    # worth little on its own.
    overhead_model = (
        OverheadCameraModel()
        if scene.miscalibration or preset is not None
        else OverheadCameraModel(0.0, 0.0, 0.0)
    )
    physics_model = PhysicsModel(amount=scene.physics_amount)
    # A frozen rig always needs the randomizer, whatever the dial says: its arm
    # is a draw the envelope already made, so there is something to apply even
    # when this run would not have drawn one.
    physics = (
        PhysicsRandomizer(model) if scene.physics_amount or frozen_rig is not None else None
    )
    fitted_bias = tracking_bias_rad(load_robot_dynamics_config()) if physics else {}
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
                if is_complete_episode(episode_root):
                    # Already recorded, so there is nothing to do and nothing to
                    # overwrite. This is the watchdog requeuing an index whose
                    # episode had in fact finished writing -- it kills on a
                    # deadline, without knowing that. Raising here was safe for
                    # the data and fatal for the worker, which then went
                    # unreplaced; skipping keeps the episode and the worker.
                    tqdm.write(f"{label}Episode {global_episode} already recorded; skipping.")
                    progress.set_postfix(saved=recorded, skipped=attempted - recorded)
                    continue
                # A killed worker may leave an incomplete directory before the
                # watchdog retries the same deterministic global index. That
                # directory can already hold meta/info.json -- LeRobot writes it
                # when the dataset is created, not when an episode is saved --
                # so completeness is judged by the episode metadata parquet, or
                # the retry would refuse a corpse and take the worker with it.
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

            rng = episode_rng(seed, global_episode)
            domain_seed = appearance_seed(seed, global_episode) if preset is not None else None
            sample = preset.sample(domain_seed) if preset is not None else None
            if frozen_rig is not None and sample is not None:
                sample = frozen_rig.session(sample, pan_jitter_rng(seed, global_episode))
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
            perturbation = sample_grasp_perturbation(
                seed,
                global_episode,
                episode_source,
                fraction=perturbed_fraction,
                magnitude_m=perturbation_magnitude_m,
                max_source_radius_m=perturbation_max_source_radius_m,
            )
            # Applied before the episode is planned, because planning ends in a
            # preflight and preflight runs live physics: vetting a candidate
            # against the nominal arm when a drawn one will fly it is checking a
            # different world than the one that follows.
            physics_draw = (
                frozen_rig.physics
                if frozen_rig is not None
                else physics_model.sample(physics_rng(seed, global_episode))
            )
            if physics is not None:
                physics.apply(physics_draw)
            if detector_perception is not None:
                # Not pinned by a frozen rig: the sidecar records where the
                # overhead camera *is*, which it does pin, but nothing about how
                # far its solved extrinsics are from that, so there is no rig
                # value to hold still. Only --sim-perception detector reads it.
                detector_perception.set_error(
                    overhead_model.sample(overhead_rng(seed, global_episode))
                )
            if geometric_perception is not None:
                geometric_perception.set_cube_belief_error(
                    (0.0, 0.0, 0.0, 0.0)
                    if draw is None
                    else draw.cube_belief_error
                )
            try:
                episode, target_plate_yaw = prepare_for_recording(
                    rng,
                    model,
                    data,
                    None,
                    source=episode_source,
                    target=target,
                    miscalibration=draw,
                    grasp_perturbation=perturbation,
                    max_attempts=max_attempts,
                    ground_truth_drop_target=(
                        ground_truth_drop_target
                        or scene.overhead_perception == "geometric"
                    ),
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
                tracking_bias_rad=tracking_bias_offsets(fitted_bias, physics_draw),
                detector_crash_dump_dir=detector_crash_dump_dir,
                wrist_servo_mode=wrist_servo_mode,
                overhead_observer=(
                    geometric_perception
                    if geometric_perception is not None
                    else detector_perception
                ),
                search_rng=rng,
                ground_truth_drop_target=(
                    ground_truth_drop_target
                    or scene.overhead_perception == "geometric"
                ),
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
                physics=physics_draw,
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
                physics=physics_draw,
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
        if detector_perception is not None:
            detector_perception.close()
        if geometric_perception is not None:
            geometric_perception.close()
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


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the sim recorder."""
    parser = SuggestingArgumentParser(description=__doc__)
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
    add_speed_argument(parser)
    add_viewer_argument(parser, help="open the 3D MuJoCo viewer")
    parser.add_argument(
        "--ground-truth-drop-target",
        action="store_true",
        help=(
            "plan the drop against the plate's true pose instead of the overhead "
            "estimate. The plate is still localized, so a scene that cannot be seen "
            "is still rejected and the cube belief stays honest -- the descent servo "
            "corrects the pickup, but nothing corrects the drop, so the overhead "
            "estimate's error lands directly in placement error and caps how "
            "accurately a cloned policy can ever place. Measured on "
            "randomized_selection_200_v1: pinning the plate moves the scripted "
            "expert from 80/200 to 121/200 settled placements and its median "
            "placement error among successes from 23.2 mm to 13.1 mm"
        ),
    )
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
    add_miscalibration_argument(
        parser,
        extra_help=", and the descent runs the wrist-camera visual servo like the real arm",
    )
    parser.add_argument(
        "--sim-perception",
        choices=("geometric", "detector"),
        default="geometric",
        help=(
            "overhead mode: geometric uses an 80%% segmentation visibility gate, "
            "truth-plus-error cube poses and exact plate poses; detector runs the "
            "real optical pipeline for robustness testing (default: geometric)"
        ),
    )
    add_physics_randomization_argument(parser)
    add_domain_randomization_argument(parser)
    parser.add_argument(
        "--frozen-rig",
        type=Path,
        default=None,
        help=(
            "record on one fixed rig: the *.frozen_rig.json sidecar written by "
            "pap freeze-scenario-rig beside a frozen evaluation suite. Every episode "
            "then faces the same robot, cameras, room and physics that suite scores "
            "on, while the light, the camera response, the sensor noise and the "
            "cube's resting orientation keep varying -- one installation over many "
            "sessions, rather than a fresh robot per episode. Requires "
            "--domain-randomization, which supplies the envelope the still-varying "
            "fields are drawn from, and overrides --physics-randomization, since the "
            "sidecar already names the arm"
        ),
    )
    add_seed_argument(parser, default=None, help="RNG seed for pose sampling")
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
        "--wrist-servo",
        choices=("geometric", "detector"),
        default="geometric",
        help=(
            "simulated descent localization: geometric maps the true cube through the "
            "true and believed wrist-camera frames without another render; detector runs "
            "the AprilTag pipeline for validation (default: geometric)"
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
    return parser


def validate(parser: SuggestingArgumentParser, args: argparse.Namespace) -> None:
    """Reject what the parser's own types cannot check, before anything is recorded.

    Only what is decidable from the arguments themselves. Whether a staging area
    already holds the episodes being asked for is a fact about the disk, so it
    is a runtime failure in :func:`run` rather than a usage error here.
    """
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
    if args.frozen_rig is not None and args.domain_randomization is None:
        parser.error("--frozen-rig requires --domain-randomization")
    if (
        args.perturbation_max_source_radius is not None
        and args.perturbation_max_source_radius <= 0.0
    ):
        parser.error("--perturbation-max-source-radius must be positive")
    try:
        FrameSizes(
            render_width=args.render_width,
            render_height=args.render_height,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    except ValueError as exc:
        parser.error(str(exc))


def run(args: argparse.Namespace) -> None:
    """Record the episodes into the staging area."""
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
        overhead_perception=args.sim_perception,
        physics_amount=args.physics_randomization,
        domain_randomization=args.domain_randomization,
        frozen_rig=args.frozen_rig,
    )
    frames = FrameSizes(
        render_width=args.render_width,
        render_height=args.render_height,
        image_width=args.image_width,
        image_height=args.image_height,
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
        "sim_perception": args.sim_perception,
        "wrist_servo": args.wrist_servo,
        "physics_randomization": args.physics_randomization,
        "domain_randomization": _configured_file(args.domain_randomization),
        # Content-affecting in the strongest sense available: it decides which
        # robot every episode was recorded on, so topping up a rig-99 staging
        # area with rig-42 episodes has to fail rather than mix two robots into
        # one dataset.
        "frozen_rig": _configured_file(args.frozen_rig),
        "max_attempts": args.max_attempts,
        "perturbed_fraction": args.perturbed_fraction,
        "perturbation_magnitude_m": args.perturbation_magnitude,
        "perturbation_max_source_radius_m": args.perturbation_max_source_radius,
        # Content-affecting: episodes planned against the true plate are a
        # different demonstration distribution, so a top-up across this flag is
        # rejected rather than silently mixed.
        "ground_truth_drop_target": (
            args.ground_truth_drop_target or args.sim_perception == "geometric"
        ),
    }
    first_episode = (
        next_episode_index(episodes_root)
        if args.first_episode is None
        else args.first_episode
    )
    if first_episode > 0 and args.seed is None:
        raise SystemExit(
            "a top-up run requires --seed; reuse the staging area's original seed"
        )
    complete_indices = {episode_index(path) for path in find_episode_datasets(episodes_root)}
    indices = list(range(first_episode, first_episode + args.episodes))
    conflicts = sorted(complete_indices.intersection(indices))
    if conflicts:
        raise SystemExit(
            f"requested range overlaps complete staged episode(s): {conflicts[:10]}"
        )
    try:
        ensure_collection_config(episodes_root, collection_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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
        wrist_servo_mode=args.wrist_servo,
        perturbed_fraction=args.perturbed_fraction,
        perturbation_magnitude_m=args.perturbation_magnitude,
        perturbation_max_source_radius_m=args.perturbation_max_source_radius,
        ground_truth_drop_target=args.ground_truth_drop_target,
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
            worker=_worker,
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
        "Finalize with pap finalize-sim-dataset once enough successful episodes are staged."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    run(args)


if __name__ == "__main__":
    main()
