# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record pick-and-place LeRobotDatasets from the simulation.

Plays a prepared episode's trajectory under the same position-servo physics as
``sim.py`` and writes it straight into a LeRobotDataset with the same schema as
the real recordings (``pap run-scripted-real``): one frame per control tick, holding the
measured joints as ``observation.state``, the commanded set point as ``action``,
and a wrist and overhead camera image. State and images are captured before the
tick's command is applied, so each frame pairs the observation at time t with
the action issued at time t — the same observe-then-act ordering as a real
recording. The two cameras are rendered offscreen from the named MuJoCo
cameras, so no hardware is involved.

Each camera's vertical field of view is set from the authored optics in
:mod:`pick_and_place.spec.camera`. The source render and saved output
resolutions are independently configurable.
(Undistortion is a no-op in sim: the sim camera is an ideal pinhole, so there is
nothing to fake-then-invert — matching the field of view is all that is needed.)

When the episode carries a miscalibration draw (see
:mod:`pick_and_place.core.miscalibration`), playback mirrors the real hardware
executor instead of open-loop feedforward: commands live in the *believed*
frame while physics runs in the *true* frame (each arm ctrl gets the drawn
joint-zero offset added; the recorded ``observation.state`` is the servo-style
readback, true joints minus the offset), the descent runs the same wrist-camera
AprilTag visual servo as the real arm — detecting the cube in a wrist render of
the true world but mapping it to world coordinates through the *believed*
camera pose, so the estimate inherits the hand-eye error exactly as on hardware
— and phase checkpoints replan the remainder from the believed readback. A
descent that never sees the cube (or fails to settle) aborts with ``"restart"``
just like the real executor, and the caller discards the episode.

Alongside the dataset, every run produces a trajectory artifact (see
:mod:`pick_and_place.data.trajectory_artifact`): the same episode with no images,
holding the true joints and true cube pose next to the believed ones. The dataset
keeps only the believed state, because that is the training label; the artifact
keeps both, because reproducing the episode's pixels means putting the arm back
where it physically was, and a miscalibration draw makes those two different
places.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.image_ops import resize_and_center_crop
from pick_and_place.core.task_phases import PhaseSpan
from pick_and_place.core.workspace_bounds import PAN_AXIS
from pick_and_place.data.recording import RecordingSession
from pick_and_place.data.trajectory_artifact import TrajectoryFrames, TrajectoryWriter
from pick_and_place.perception.image_rectify import SQUARE_SIZE
from pick_and_place.runtime.believed_frame import BelievedFrame
from pick_and_place.rollout.checkpoint import replan_from_checkpoint
from pick_and_place.scripted.checkpoint import fuses_into_next
from pick_and_place.scripted.descent import regrasp_after_descent
from pick_and_place.scripted.episode_sampling import sample_hunt_pose
from pick_and_place.scripted.trajectory import SearchPhase
from pick_and_place.runtime.episodes import Episode
from pick_and_place.plant.sim import SimPlant
from pick_and_place.rollout.phase import Run, play_phase
from pick_and_place.rollout.sim_dataset import SimTickRecorder
from pick_and_place.runtime.sim_wrist_servo import SimWristServo
from pick_and_place.runtime.wrist_mixed_view import blend_mixed, close_mixed, show_mixed
from pick_and_place.sim.domain_randomization import reload_renderer_textures
from pick_and_place.policies.policy_evaluation import TaskOracleConfig
from pick_and_place.sim.model import build_model, get_cube_pose, placement_error
from pick_and_place.spec.robot import CONTROL_HZ, HARDWARE_SIMULATION_HZ
from pick_and_place.spec.workspace import CUBE_HALF_SIZE

WRIST_CAMERA = "wrist_camera"
OVERHEAD_CAMERA = "overhead_camera"

# Render quality of a recorded dataset: a dense, tightly focused shadow map and
# multisampled offscreen buffers, so supersampled frames have clean silhouettes
# and shadow edges.
SHADOW_MAP_SIZE = 8192
OFFSCREEN_SAMPLES = 8
SHADOW_CONE_SCALE = 0.4

# Cube height above which the grasp is judged to have taken, used only to decide
# whether a fumbled pick needs re-picking. Taken from the oracle's own lift
# milestone rather than chosen here, so "held" during recording means the same
# thing as "lifted" when the episode is scored; a local threshold would drift
# from it silently.
_ORACLE = TaskOracleConfig()
_HELD_MIN_Z_M = _ORACLE.resting_height_m + _ORACLE.lift_clearance_m

@dataclasses.dataclass(frozen=True)
class RecordEpisodeResult:
    """Outcome of one episode playback, plus the trajectory it produced.

    ``frames`` is the imageless artifact: the true world and the believed one,
    tick by tick. It is produced whether or not anything captured images, since
    it costs a few dozen floats per tick and is the only record from which the
    episode's pixels can be made again. The caller pairs it with the facts only
    it knows — the seed, the drop plate's yaw — to write an artifact file.
    """

    status: str  # "success", "restart", or "stopped"
    phase_spans: tuple[PhaseSpan, ...]
    frames: TrajectoryFrames


def fovy_from_intrinsics(intrinsics: dict[str, Any]) -> float:
    """Vertical field of view (degrees) implied by a camera's intrinsics.

    Derived from the calibrated focal length, ``2*atan((h/2)/fy)``, which is the
    vertical FOV of the rectified (undistorted) pinhole image. A center-square
    crop keeps the full image height, so the square's vertical — and, being
    square, horizontal — FOV is this same angle. The result is independent of the
    render resolution, so the same value drives a 512x512 sim render.
    """
    height = float(intrinsics["height"])
    fy = float(intrinsics["camera_matrix"][1][1])
    return math.degrees(2.0 * math.atan((height / 2.0) / fy))


class SimCameraRig:
    """Offscreen renderers for the wrist and overhead cameras.

    Each camera's ``cam_fovy`` is overridden from the intrinsics it is given,
    falling back to the model's built-in fovy for any camera absent from them.
    Sim callers pass the authored optics; the real-rig paths pass that rig's
    measured ones, which is where a solved calibration belongs.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        intrinsics_by_name: dict[str, dict[str, Any]] | None = None,
        width: int = SQUARE_SIZE,
        height: int = SQUARE_SIZE,
        render_width: int | None = None,
        render_height: int | None = None,
        postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("camera width and height must be positive")
        render_width = width if render_width is None else render_width
        render_height = height if render_height is None else render_height
        if render_width < 1 or render_height < 1:
            raise ValueError("render width and height must be positive")
        if render_width < width or render_height < height:
            raise ValueError("render dimensions must be at least the output dimensions")
        self.width = width
        self.height = height
        self.render_width = render_width
        self.render_height = render_height
        self.postprocess = postprocess
        intrinsics_by_name = intrinsics_by_name or {}
        self._cameras = []
        for name in (WRIST_CAMERA, OVERHEAD_CAMERA):
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id < 0:
                raise ValueError(f"model has no camera named {name!r}")
            intrinsics = intrinsics_by_name.get(name)
            if intrinsics is not None:
                model.cam_fovy[cam_id] = fovy_from_intrinsics(intrinsics)
            self._cameras.append(name)
        self._renderer = mujoco.Renderer(
            model, width=render_width, height=render_height
        )

    def render(self, data: mujoco.MjData, camera: str) -> np.ndarray:
        """Render one camera to an ``(height, width, 3)`` uint8 RGB array."""
        self._renderer.update_scene(data, camera=camera)
        image = resize_and_center_crop(
            self._renderer.render(), self.height, self.width
        )
        return self.postprocess(image) if self.postprocess is not None else image

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Render both cameras, returning ``(wrist_rgb, overhead_rgb)``."""
        return self.render(data, WRIST_CAMERA), self.render(data, OVERHEAD_CAMERA)

    def reload_textures(self, texture_ids: tuple[int, ...]) -> None:
        """Upload textures changed in ``model.tex_data`` to this rig's GL context."""
        reload_renderer_textures(self._renderer, texture_ids)

    def close(self) -> None:
        self._renderer.close()


def configure_render_quality(model: mujoco.MjModel) -> None:
    """Use a dense, tightly focused shadow map for supersampled recordings."""
    model.vis.quality.shadowsize = SHADOW_MAP_SIZE
    model.vis.quality.offsamples = OFFSCREEN_SAMPLES
    model.vis.map.shadowscale = SHADOW_CONE_SCALE


def build_recording_scene(
    *,
    render_width: int,
    render_height: int,
    background_panorama: Path | str | np.ndarray | None = None,
    table_texture: Path | str | np.ndarray | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the scene a sim recording is rendered from.

    One persistent scene is reused across a run's episodes: the cube is a
    freejoint body repositioned per episode, so where it compiles is irrelevant
    and a fixed placeholder pose keeps the compiled model independent of which
    episodes a worker happens to draw.

    Anything that moves a pixel lives here rather than in the caller —
    the environment and drop-zone marker, the offscreen buffer size, the
    physics rate, and the render-quality settings. Recording and re-rendering
    both build through this function so a recorded frame and a re-rendered one
    differ only where they are meant to.

    The cameras sit at their authored poses. That is the origin domain
    randomization perturbs around, so a recording made from a fresh clone draws
    from the same distribution as one made here, and a published dataset is one
    the code can regenerate.
    """
    placeholder = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = build_model(
        placeholder,
        include_environment=True,
        paper_target_marker=True,
        background_panorama=background_panorama,
        table_texture=table_texture,
        offwidth=render_width,
        offheight=render_height,
    )
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    configure_render_quality(model)
    mujoco.mj_forward(model, data)
    return model, data



def _grasp_and_lift(kinematics, grasp, remaining_lift) -> tuple[Any, Any]:
    """The close-and-lift pair, rebuilt from the grasp the descent settled on.

    Which lift it is comes from the plan being replaced rather than from the
    grasp mode: the planner picks the low recovery lift for its own reasons, and
    the descent has no say in that.
    """
    from pick_and_place.scripted.trajectory import GraspPhase, LiftPhase, RecoveryLiftPhase
    from pick_and_place.spec.robot import GRIPPER_OPEN

    lift_cls = RecoveryLiftPhase if isinstance(remaining_lift, RecoveryLiftPhase) else LiftPhase
    return (
        GraspPhase(grasp.grasp_joints, start_gripper=GRIPPER_OPEN),
        lift_cls(kinematics, grasp.grasp_joints, grasp.lift_joints),
    )


def record_episode(
    episode: Episode,
    *,
    recording: RecordingSession | None = None,
    rig: SimCameraRig | None = None,
    viewer: Any = None,
    speed: float = 1.0,
    realtime: bool = False,
    should_stop: Callable[[], bool] | None = None,
    show_wrist_mixed: bool = False,
    believed_wrist_camera_pose: tuple[np.ndarray, np.ndarray] | None = None,
    tracking_bias_rad: dict[str, float] | None = None,
    detector_crash_dump_dir: str | None = None,
    wrist_servo_mode: str = "geometric",
    overhead_observer: Any = None,
    search_rng: np.random.Generator | None = None,
    ground_truth_drop_target: bool = True,
    max_search_poses: int = 8,
    verbose: bool = True,
) -> RecordEpisodeResult:
    """Play ``episode``'s trajectory under physics and record every control tick.

    The arm is driven through the position-servo actuators exactly as in
    ``sim.py``: each tick captures the frame (measured joints as
    ``observation.state``, cameras) first, then writes the trajectory set point
    — the ``action`` — to ``data.ctrl`` and advances the sim by a batch of
    physics substeps. Both streams are expressed in the real joint frame
    (degrees / 0-100 gripper), so the dataset is unit-for-unit comparable to a
    real recording. ``viewer`` (a launched passive viewer, or ``None``) is synced
    once per tick if given.

    Without a miscalibration draw on the episode this is pure feedforward
    playback of a vetted plan. With one, the run mirrors the hardware executor:
    physics holds the true joints while commands and the recorded streams stay in
    the believed frame, the descent runs the wrist-camera visual servo, and
    completed phases replan the remainder from the believed readback. Returns
    ``"success"`` when the trajectory ran to completion, ``"stopped"`` when
    ``should_stop`` asked for it, or ``"restart"`` when the descent servo or a
    checkpoint replan failed — the caller should then discard the recording.

    The result also carries the exact phase spans of the recorded frames, taken
    from the controller itself rather than reconstructed afterwards.

    ``recording``/``rig`` may be omitted together to play the episode without
    capturing anything — the closed-loop path still runs, which is how the sim
    viewer inspects a miscalibrated episode. ``realtime`` paces the loop to the
    control rate for live viewing (recording runs unpaced). ``should_stop`` is
    polled every tick.

    ``show_wrist_mixed`` (closed-loop episodes only) opens an OpenCV window
    blending the true-world wrist render with a render of the believed world; see
    :mod:`pick_and_place.runtime.wrist_mixed_view`. Needs no MuJoCo viewer, so it
    works under plain ``python``.

    The dataset is created lazily on the first episode once the fixed output
    frame shape is known. The caller commits the episode with ``save_episode``
    and ends the run with :meth:`RecordingSession.finalize`.
    """
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if (recording is None) != (rig is None):
        raise ValueError("recording and rig must be provided together")
    if show_wrist_mixed and episode.miscalibration is None:
        raise ValueError("show_wrist_mixed requires a miscalibration draw (closed-loop playback)")
    if show_wrist_mixed and wrist_servo_mode != "detector":
        raise ValueError("show_wrist_mixed requires wrist_servo_mode='detector'")

    model = episode.model
    data = episode.data
    kinematics = episode.kinematics

    substeps_per_tick = round(HARDWARE_SIMULATION_HZ / CONTROL_HZ)
    if not math.isclose(model.opt.timestep * substeps_per_tick, 1.0 / CONTROL_HZ):
        raise ValueError(
            f"MuJoCo timestep {model.opt.timestep:g}s cannot produce {CONTROL_HZ:g} Hz exactly"
        )

    belief = BelievedFrame(model, data, episode.miscalibration, data.time)
    artifact = TrajectoryWriter()
    recorder = None if recording is None else SimTickRecorder(recording, rig, data)
    servo = (
        SimWristServo(
            model,
            data,
            belief,
            WRIST_CAMERA,
            mode=wrist_servo_mode,
            believed_camera_pose=believed_wrist_camera_pose,
            detector_crash_dump_dir=detector_crash_dump_dir,
        )
        if belief.closed_loop
        else None
    )
    # Checkpoints exist because the plan may be wrong, and there are two ways for
    # that to be true. A miscalibration draw is the original one: the belief error
    # is real, so the descent servos and every phase boundary replans from
    # readback. A deliberate grasp perturbation is the other: the plan is wrong
    # *by construction*, and without checkpoints the arm would fly the rest of it
    # blind -- carrying nothing, dropping nothing -- and the episode would be
    # unusable rather than a recovery demonstration.
    #
    # These were one flag until now, which coupled replanning to servoing. They
    # are not the same capability, and this task wants replanning *without* a
    # servo: the servo follows a fresh sighting every tick and would steer the
    # jaws back onto the true cube, correcting away the very fumble being staged.
    run_checkpoints = (
        belief.closed_loop
        or episode.grasp_perturbation is not None
        or overhead_observer is not None
    )
    plant = SimPlant(
        model,
        data,
        belief=belief,
        actuator_id=episode.actuator_id,
        robot_geom_ids=episode.robot_geom_ids,
        env_geom_ids=episode.env_geom_ids,
        kinematics=kinematics,
        substeps_per_tick=substeps_per_tick,
        servo=servo,
        tracking_bias_rad=tracking_bias_rad,
        speed=speed,
        realtime=realtime,
    )
    def on_tick(tick) -> None:
        if recorder is not None:
            recorder.record(
                believed_state=tick.observation.state,
                action=tick.action,
                true_cube_pose=tick.observation.true_cube_pose,
            )

    def on_look(plant, sighting, tracked, is_descent) -> None:
        if not show_wrist_mixed or plant.servo.mode != "detector":
            return
        true_rgb, camera_position, camera_rotation = plant.last_look
        show_mixed(
            blend_mixed(
                true_rgb,
                plant.servo.render_believed(tracked),
                sighting,
                plant.servo.camera_matrix,
                camera_position,
                camera_rotation,
            )
        )

    run = Run(
        on_tick=on_tick,
        on_look=on_look,
        sync=(lambda: None) if viewer is None else viewer.sync,
        should_stop=(lambda: False) if should_stop is None else should_stop,
        verbose=verbose,
    )

    def outcome(status: str) -> RecordEpisodeResult:
        """Snapshot the artifact as it stands. Reads the latest bindings when called."""
        return RecordEpisodeResult(status, artifact.spans, artifact.frames())

    current_traj = episode.trajectory
    tracked_source = episode.believed_source
    tracked_grasp = current_traj.grasp
    contacts: set[tuple[str, str]] = set()
    status = "incomplete"
    search_rng = np.random.default_rng(0) if search_rng is None else search_rng

    def overhead_reading(object_name: str):
        if overhead_observer is None:
            return None
        look = getattr(overhead_observer, "look", None)
        if look is not None:
            reading = look()
            return reading.cube if object_name in ("cube", "placement") else reading.target
        placeholder = np.empty((0, 0, 3), dtype=np.uint8)
        if object_name in ("cube", "placement"):
            return overhead_observer.localize_cube(placeholder)
        return overhead_observer.localize_drop_target(
            placeholder,
            target_color="black",
            workspace_corners_world=np.empty((4, 3)),
        )

    def search_for(
        object_name: str,
        *,
        carrying: bool,
        tracked: CubePose,
        current_contacts: set[tuple[str, str]],
    ) -> tuple[str, CubePose, Any, set[tuple[str, str]]]:
        found = overhead_reading(object_name)
        if found is not None:
            return "completed", tracked, found, current_contacts
        for _ in range(max_search_poses):
            measured_joints, measured_gripper = plant.measured()
            if carrying:
                target_joints = dict(measured_joints)
                target_joints["shoulder_pan"] = search_rng.uniform(-0.35, 0.35)
                target_gripper = measured_gripper
            else:
                target_joints, target_gripper = sample_hunt_pose(search_rng)
            phase = SearchPhase(
                kinematics,
                measured_joints,
                target_joints,
                measured_gripper,
                target_gripper,
                object_name,
            )
            played = play_phase(
                plant,
                phase,
                run=run,
                artifact=artifact,
                tracked_source=tracked,
                contacts=current_contacts,
                watch=False,
            )
            tracked = played.tracked_source
            current_contacts = played.contacts
            if played.outcome != "completed":
                return played.outcome, tracked, None, current_contacts
            found = overhead_reading(object_name)
            if found is not None:
                return "completed", tracked, found, current_contacts
        return "restart", tracked, None, current_contacts

    try:
        if overhead_observer is not None:
            search_status, tracked_source, cube_reading, contacts = search_for(
                "cube",
                carrying=False,
                tracked=tracked_source,
                current_contacts=contacts,
            )
            if search_status != "completed":
                return outcome(search_status)
            tracked_source = cube_reading
            measured_joints, measured_gripper = plant.measured()
            current_traj = replan_from_checkpoint(
                model,
                kinematics=kinematics,
                actuator_id=episode.actuator_id,
                robot_geom_ids=episode.robot_geom_ids,
                env_geom_ids=episode.env_geom_ids,
                measured_joints=measured_joints,
                measured_gripper=measured_gripper,
                completed_phase_name=None,
                source=tracked_source,
                target=episode.believed_target,
                grasp=None,
                end_joints=episode.end_joints,
                end_gripper=episode.end_gripper,
                free_grasp=False,
                verbose=verbose,
            )
            if current_traj is None:
                return outcome("restart")
            tracked_grasp = current_traj.grasp
        while current_traj is not None and current_traj.phases:
            played = play_phase(
                plant,
                current_traj.phases[0],
                run=run,
                artifact=artifact,
                tracked_source=tracked_source,
                contacts=contacts,
                watch=show_wrist_mixed,
            )
            phase = played.phase
            tracked_source = played.tracked_source
            contacts = played.contacts
            if played.outcome != "completed":
                return outcome(played.outcome)

            completed = phase.name
            remaining_phases = current_traj.phases[1:]

            if not run_checkpoints:
                # Pure feedforward playback: the vetted plan needs no
                # checkpoints, so just advance phase by phase.
                current_traj = dataclasses.replace(current_traj, phases=remaining_phases)
                if not current_traj.phases:
                    status = "success"
                continue

            # From here the transitions mirror the hardware executor exactly:
            # the same fused pairs, the same rebuilt grasp and lift after the
            # descent, and the same replan from the measured (believed) state
            # everywhere else.
            if remaining_phases and fuses_into_next(completed, remaining_phases[0].name):
                current_traj = dataclasses.replace(current_traj, phases=remaining_phases)
                continue

            if completed == "descent":
                tracked_grasp = regrasp_after_descent(
                    phase,
                    tracked_source,
                    kinematics,
                    free_grasp=phase.grasp.face == "free",
                    current=tracked_grasp,
                )
                current_traj = dataclasses.replace(
                    current_traj,
                    phases=(
                        *_grasp_and_lift(kinematics, tracked_grasp, current_traj.phases[2]),
                        *current_traj.phases[3:],
                    ),
                    grasp=tracked_grasp,
                )
                continue

            if not remaining_phases:
                if completed == "retreat" and overhead_observer is not None:
                    search_status, tracked_source, _, contacts = search_for(
                        "placement",
                        carrying=False,
                        tracked=tracked_source,
                        current_contacts=contacts,
                    )
                    if search_status != "completed":
                        return outcome(search_status)
                status = "success"
                break

            # Did the grasp actually take? Every branch below this point assumes
            # the cube is in the jaws and continues to carry and drop, so a
            # fumbled grasp otherwise produces an episode that carries nothing,
            # drops nothing, and is recorded as a success -- which is how a
            # perturbed episode would quietly poison the dataset.
            #
            # Judged from ground truth rather than the gripper encoder. The real
            # rig has only the encoder, and `_pickup_report` in the hardware
            # executor deliberately refuses to branch on it because the jaws read
            # open when they jam; sim knows where the cube is, so it can decide
            # this exactly. The threshold is the oracle's own lift criterion, so
            # "held" here means the same thing it means when the episode is
            # scored.
            if run_checkpoints and completed in ("lift", "recovery_lift"):
                cube_now = get_cube_pose(model, data)
                held = cube_now.z >= _HELD_MIN_Z_M
                if not held:
                    if verbose:
                        print(
                            f"Grasp missed (cube at z={cube_now.z * 1000:.1f} mm, "
                            f"needs {_HELD_MIN_Z_M * 1000:.1f}); "
                            "re-picking from where the cube actually is."
                        )
                    # The retry has to re-localize. `tracked_source` still holds
                    # the perturbed belief that caused the miss, and re-planning
                    # against it would aim at the same empty spot forever. This
                    # is where the injected error expires -- the recovery is
                    # planned against the cube's real pose, which is also what
                    # makes the fumble transient rather than a permanent bias.
                    tracked_source = cube_now
                    tracked_grasp = None
                    # completed_phase_name=None is the planner's "plan a whole
                    # fresh pick from the current arm state" branch: approach,
                    # descent, grasp, lift, carry, drop, retreat, with grasp
                    # candidates regenerated from the pose above and every
                    # candidate preflighted. No new planning code is needed.
                    completed = None
                    if overhead_observer is not None:
                        search_status, tracked_source, cube_reading, contacts = search_for(
                            "cube",
                            carrying=False,
                            tracked=tracked_source,
                            current_contacts=contacts,
                        )
                        if search_status != "completed":
                            return outcome(search_status)
                        tracked_source = cube_reading
            if (
                overhead_observer is not None
                and completed in ("lift", "recovery_lift")
            ):
                search_status, tracked_source, plate_reading, contacts = search_for(
                    "plate",
                    carrying=True,
                    tracked=tracked_source,
                    current_contacts=contacts,
                )
                if search_status != "completed":
                    return outcome(search_status)
                target_xy = (
                    (episode.target.x, episode.target.y)
                    if ground_truth_drop_target
                    else plate_reading.xy
                )
                episode.believed_target = CubePose(*target_xy, CUBE_HALF_SIZE)
            if verbose:
                print(f"Replanning remaining trajectory after {completed}...")
            measured_joints, measured_gripper = plant.measured()
            candidate = replan_from_checkpoint(
                model,
                kinematics=kinematics,
                actuator_id=episode.actuator_id,
                robot_geom_ids=episode.robot_geom_ids,
                env_geom_ids=episode.env_geom_ids,
                measured_joints=measured_joints,
                measured_gripper=measured_gripper,
                completed_phase_name=completed,
                source=tracked_source,
                target=episode.believed_target,
                grasp=tracked_grasp,
                end_joints=episode.end_joints,
                end_gripper=episode.end_gripper,
                free_grasp=False,
                verbose=verbose,
            )
            if candidate is None:
                if verbose:
                    print(f"No clean replan after {completed}; aborting episode.")
                return outcome("restart")
            current_traj = candidate
    finally:
        plant.close()
        if show_wrist_mixed:
            close_mixed()

    if verbose:
        print(placement_error(model, data, episode.target).summary())
    return outcome("success" if status == "success" else "restart")
