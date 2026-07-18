# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Record pick-and-place LeRobotDatasets from the simulation.

Plays a prepared episode's trajectory under the same position-servo physics as
``sim.py`` and writes it straight into a LeRobotDataset with the same schema as
the real recordings (``real.py``): one frame per control tick, holding the
measured joints as ``observation.state``, the commanded set point as ``action``,
and a wrist and overhead camera image. State and images are captured before the
tick's command is applied, so each frame pairs the observation at time t with
the action issued at time t — the same observe-then-act ordering as a real
recording. The two cameras are rendered offscreen from the named MuJoCo
cameras, so no hardware is involved.

Each camera's vertical field of view is set from its calibrated intrinsics. The
source render and saved output resolutions are independently configurable.
(Undistortion is a no-op in sim: the sim camera is an ideal pinhole, so there is
nothing to fake-then-invert — matching the field of view is all that is needed.)

When the episode carries a miscalibration draw (see
:mod:`pick_and_place.miscalibration`), playback mirrors the real hardware
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
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable

import cv2
import mujoco
import numpy as np

from pick_and_place.episodes import (
    Episode,
    _preflight,
    get_joint,
    is_unexpected,
    placement_error,
    scan_contacts,
    set_cube_pose,
    set_joint,
)
from pick_and_place.domain_randomization import reload_renderer_textures
from pick_and_place.executor import (
    CONTROL_HZ,
    HARDWARE_SIMULATION_HZ,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.recording import RecordingSession
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    sim_frame_to_real,
)
from pick_and_place.image_rectify import SQUARE_SIZE
from pick_and_place.trajectory import (
    DescentPhase,
    GRIPPER_OPEN,
    GraspPhase,
    LiftPhase,
    RecoveryLiftPhase,
    _shortest_delta,
    fold_cube_yaw,
    grasp_candidates,
    replan_remaining_candidates,
)
from pick_and_place.visual_servo import (
    DESCENT_SERVO_MAX_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
)

WRIST_CAMERA = "wrist_camera"
OVERHEAD_CAMERA = "overhead_camera"


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

    Each camera's ``cam_fovy`` is overridden from its calibrated intrinsics (when
    available) so the sim render matches the real undistorted feed. Falls back
    to the model's built-in fovy for any camera without local intrinsics.
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


def resize_and_center_crop(
    image: np.ndarray, output_height: int, output_width: int
) -> np.ndarray:
    """Area-downsample an image to cover the output, then center-crop it."""
    if output_width < 1 or output_height < 1:
        raise ValueError("output width and height must be positive")
    height, width = image.shape[:2]
    scale = max(output_width / width, output_height / height)
    resized_width = max(output_width, round(width * scale))
    resized_height = max(output_height, round(height * scale))
    if (resized_width, resized_height) != (width, height):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
    left = (resized_width - output_width) // 2
    top = (resized_height - output_height) // 2
    return image[top : top + output_height, left : left + output_width].copy()


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
    verbose: bool = True,
) -> str:
    """Play ``episode``'s trajectory under physics and record every control tick.

    The arm is driven through the position-servo actuators exactly as in
    ``sim.py``: each tick captures the frame (measured joints as
    ``observation.state``, cameras) first, then writes the trajectory set point
    — the ``action`` — to ``data.ctrl`` and advances the sim by a batch of
    physics substeps. Capturing before stepping pairs each observation with the
    command issued from it, matching a real recording, where the state is read
    before the servos have tracked the new set point. Both streams are
    expressed in the real joint frame (degrees / 0-100 gripper), so the dataset
    is unit-for-unit comparable to a real recording. ``viewer`` (a launched
    passive viewer, or ``None``) is synced once per tick if given.

    Without a miscalibration draw on the episode this is pure feedforward
    playback. With one, the run mirrors the hardware executor: ctrl gets the
    drawn joint-zero offsets added (physics runs the *true* joints while
    commands and the recorded streams stay in the believed/servo frame), the
    descent phase runs the wrist-camera AprilTag visual servo against renders
    of the true world, and completed phases replan the remainder from the
    believed readback. Returns ``"success"`` when the trajectory ran to
    completion or ``"restart"`` when the descent servo or a checkpoint replan
    failed — the caller should then discard the recorded episode.

    ``recording``/``rig`` may be omitted together to play the episode without
    capturing anything — the closed-loop path (offsets, wrist servo, replans)
    still runs, which is how the sim viewer inspects a miscalibrated episode.
    ``realtime`` paces the loop to the control rate for live viewing (recording
    runs unpaced). ``should_stop`` is polled every tick; returning ``True``
    aborts the episode with ``"stopped"``.

    ``show_wrist_mixed`` (closed-loop episodes only) opens an OpenCV window
    blending the true-world wrist render (detected cube tags outlined) with a
    render of the *believed* world — the arm at the believed joints, the cube
    at the planner's current believed pose — the sim analog of the hardware
    ``--show-wrist-mixed`` overlay: the visual offset between the two layers is
    the injected miscalibration, and the descent shows the servo pulling them
    into register. Needs no MuJoCo viewer, so it works under plain ``python``.

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

    model = episode.model
    data = episode.data
    actuator_id = episode.actuator_id
    kinematics = episode.kinematics
    draw = episode.miscalibration

    simulation_steps_per_tick = round(HARDWARE_SIMULATION_HZ / CONTROL_HZ)
    control_period = 1.0 / CONTROL_HZ
    if not math.isclose(model.opt.timestep * simulation_steps_per_tick, control_period):
        raise ValueError(
            f"MuJoCo timestep {model.opt.timestep:g}s cannot produce {CONTROL_HZ:g} Hz exactly"
        )

    if recording is not None and recording.dataset is None:
        image_shape = (rig.height, rig.width, 3)
        recording.create_dataset(image_shape, image_shape)

    time_origin = data.time

    def offsets_deg_now() -> dict[str, float] | None:
        return None if draw is None else draw.offsets_deg(data.time - time_origin)

    def offsets_rad_now() -> dict[str, float]:
        return {} if draw is None else draw.offsets_rad(data.time - time_origin)

    def believed_arm_joints() -> dict[str, float]:
        """The servo-style readback: true joints minus the offsets in effect."""
        offsets = offsets_rad_now()
        return {
            name: get_joint(model, data, name) - offsets.get(name, 0.0) for name in ARM_JOINT_NAMES
        }

    # --- wrist visual servo (miscalibrated episodes only) -------------------
    servo_enabled = draw is not None
    servo_renderer = None
    tracker = None
    servo_camera_matrix = None
    believed_shadow = None
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
    if servo_enabled:
        from pick_and_place.cube_detection import CubeTracker, detect_cube_faces
        from scipy.spatial.transform import Rotation

        if show_wrist_mixed:
            import cv2

        # Match the real servo's working resolution as far as the model's
        # offscreen buffer allows (the executor detects on 1280x720 frames).
        render_w = min(1280, int(model.vis.global_.offwidth))
        render_h = min(720, int(model.vis.global_.offheight))
        servo_renderer = mujoco.Renderer(model, width=render_w, height=render_h)
        fovy = math.radians(float(model.cam_fovy[wrist_cam_id]))
        fy = (render_h / 2.0) / math.tan(fovy / 2.0)
        servo_camera_matrix = np.array(
            [[fy, 0.0, render_w / 2.0], [0.0, fy, render_h / 2.0], [0.0, 0.0, 1.0]]
        )
        tracker = CubeTracker(smooth=0.95)
        # Kinematics-only mirror holding the believed joints, whose wrist camera
        # pose is what the detection is mapped to world through — the believed
        # pose, not the true one, exactly as the hardware servo uses the sim
        # mirror's camera. The world-frame estimate therefore inherits the
        # injected hand-eye error.
        believed_shadow = mujoco.MjData(model)

    def believed_camera_pose(believed_cube: CubePose) -> tuple[np.ndarray, np.ndarray]:
        """Pose the believed-world shadow and return its wrist camera pose."""
        for name, value in believed_arm_joints().items():
            set_joint(model, believed_shadow, name, value)
        set_joint(model, believed_shadow, "gripper", get_joint(model, data, "gripper"))
        set_cube_pose(model, believed_shadow, believed_cube)
        mujoco.mj_kinematics(model, believed_shadow)
        # Body kinematics alone leaves camera poses unset; this fills cam_xpos/xmat.
        if believed_wrist_camera_pose is None:
            mujoco.mj_camlight(model, believed_shadow)
            return (
                believed_shadow.cam_xpos[wrist_cam_id].copy(),
                believed_shadow.cam_xmat[wrist_cam_id].reshape(3, 3).copy(),
            )

        # Rendering uses the perturbed physical camera stored in the model. The
        # controller maps detections through the nominal mount calibration, so
        # temporarily restore that local pose only while updating the believed
        # shadow's camera transform.
        true_pos = model.cam_pos[wrist_cam_id].copy()
        true_quat = model.cam_quat[wrist_cam_id].copy()
        try:
            model.cam_pos[wrist_cam_id] = believed_wrist_camera_pose[0]
            model.cam_quat[wrist_cam_id] = believed_wrist_camera_pose[1]
            mujoco.mj_camlight(model, believed_shadow)
            return (
                believed_shadow.cam_xpos[wrist_cam_id].copy(),
                believed_shadow.cam_xmat[wrist_cam_id].reshape(3, 3).copy(),
            )
        finally:
            model.cam_pos[wrist_cam_id] = true_pos
            model.cam_quat[wrist_cam_id] = true_quat

    def record_tick(frame) -> None:
        if recording is None:
            return
        measured_arm = {name: get_joint(model, data, name) for name in ARM_JOINT_NAMES}
        measured_gripper = get_joint(model, data, "gripper")
        state = sim_frame_to_real(measured_arm, measured_gripper, offsets_deg_now())
        action = sim_frame_to_real(frame.joints, frame.gripper)

        wrist_rgb, overhead_rgb = rig.capture(data)
        recording.dataset.add_frame(
            {
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "observation.images.wrist": wrist_rgb,
                "observation.images.overhead": overhead_rgb,
                "task": recording.task,
            }
        )

        # A dropped encoder frame would leave the video shorter than the recorded
        # rows; rather than write a corrupt episode, fail the moment it happens.
        dropped = recording.dropped_frame_count()
        if dropped:
            raise RuntimeError(
                f"Streaming video encoder dropped {dropped} frame(s): the encoder "
                "cannot keep pace with capture, which would desync the video from "
                "the recorded frames. Use a hardware vcodec (auto) or raise the "
                "encoder queue size."
            )

    prev_contacts: set[tuple[str, str]] = set()
    current_traj = episode.trajectory
    dynamic_source = episode.believed_source
    dynamic_grasp = current_traj.grasp
    status = "incomplete"

    try:
        while current_traj is not None and current_traj.phases:
            phase = current_traj.phases[0]
            playback_start = data.time

            is_descent = servo_enabled and isinstance(phase, DescentPhase)
            convergence = DescentServoConvergence() if is_descent else None
            retry = DescentServoRetryState() if is_descent else None
            saw_detection = False
            max_duration = (
                max(phase.duration, DESCENT_SERVO_MAX_DURATION) if is_descent else phase.duration
            )

            while True:
                if should_stop is not None and should_stop():
                    return "stopped"
                tick_start = time.monotonic()
                raw_phase_t = (data.time - playback_start) * speed
                phase_t = (
                    retry.command_phase_t(raw_phase_t, phase.duration)
                    if retry is not None
                    else raw_phase_t
                )

                true_rgb = None
                detections: list = []
                estimate = None
                if is_descent or show_wrist_mixed:
                    cam_pos, cam_rot = believed_camera_pose(dynamic_source)
                    servo_renderer.update_scene(data, camera=WRIST_CAMERA)
                    true_rgb = servo_renderer.render()
                if is_descent:
                    detections = detect_cube_faces(true_rgb, tracker.detector)
                    estimate = tracker.update(
                        detections, servo_camera_matrix, cam_pos, cam_rot, dist=None
                    )
                    if estimate is not None:
                        _, _, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
                        # A cube grasp repeats every 90 deg, so fold the detected
                        # yaw onto the quarter-turn nearest the current target;
                        # the single-tag planar-pose ambiguity otherwise flips it
                        # 90/180 deg and spins the re-solved wrist roll around.
                        folded_yaw = fold_cube_yaw(dynamic_source.yaw, float(yaw))
                        new_source = CubePose(
                            x=float(estimate.position[0]),
                            y=float(estimate.position[1]),
                            z=CUBE_HALF_SIZE,
                            yaw=folded_yaw,
                        )
                        # Smoothly interpolate the target to avoid arm jumps.
                        alpha = 0.1
                        dynamic_source = dataclasses.replace(
                            new_source,
                            x=dynamic_source.x * (1 - alpha) + new_source.x * alpha,
                            y=dynamic_source.y * (1 - alpha) + new_source.y * alpha,
                            yaw=dynamic_source.yaw
                            + _shortest_delta(dynamic_source.yaw, new_source.yaw) * alpha,
                        )
                        if phase.grasp.face != "free":
                            updated_grasp = next(
                                (
                                    g
                                    for g in grasp_candidates(kinematics, dynamic_source)
                                    if g.face == phase.grasp.face and g.elbow == phase.grasp.elbow
                                ),
                                None,
                            )
                            if updated_grasp is not None:
                                phase = dataclasses.replace(phase, grasp=updated_grasp)
                        saw_detection = True
                        convergence.observe(dynamic_source)

                if show_wrist_mixed:
                    # Re-pose the shadow so the underlay shows the believed cube
                    # as updated by this tick's servo estimate, then blend it
                    # under the true render (detections outlined): the visual
                    # offset between the layers is the live believed-true gap.
                    believed_camera_pose(dynamic_source)
                    servo_renderer.update_scene(believed_shadow, camera=WRIST_CAMERA)
                    believed_rgb = servo_renderer.render()
                    bgr = cv2.cvtColor(true_rgb, cv2.COLOR_RGB2BGR)
                    for det in detections:
                        corners = np.array(det.corners, dtype=np.int32)
                        cv2.polylines(bgr, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(
                            bgr,
                            str(det.tag_id),
                            tuple(corners[0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    if estimate is not None:
                        # The tracker's world estimate, projected back through
                        # the believed camera it was solved with: pose axes and
                        # the orange cube wireframe, as on the hardware overlay.
                        cv_to_mj = np.diag([1.0, -1.0, -1.0])
                        pos_mj_cam = cam_rot.T @ (estimate.position - cam_pos)
                        rot_mj_cam = cam_rot.T @ estimate.rotation
                        tvec = cv_to_mj @ pos_mj_cam
                        rvec, _ = cv2.Rodrigues(cv_to_mj @ rot_mj_cam)
                        cv2.drawFrameAxes(
                            bgr, servo_camera_matrix, np.zeros(5), rvec, tvec, 0.03, 2
                        )
                        s = CUBE_HALF_SIZE
                        pts_3d = np.float32(
                            [
                                [-s, -s, -s],
                                [s, -s, -s],
                                [s, s, -s],
                                [-s, s, -s],
                                [-s, -s, s],
                                [s, -s, s],
                                [s, s, s],
                                [-s, s, s],
                            ]
                        )
                        pts_img, _ = cv2.projectPoints(
                            pts_3d, rvec, tvec, servo_camera_matrix, np.zeros(5)
                        )
                        pts_img = pts_img.reshape(-1, 2).astype(int)
                        edges = [
                            (0, 1),
                            (1, 2),
                            (2, 3),
                            (3, 0),
                            (4, 5),
                            (5, 6),
                            (6, 7),
                            (7, 4),
                            (0, 4),
                            (1, 5),
                            (2, 6),
                            (3, 7),
                        ]
                        for i, j in edges:
                            cv2.line(
                                bgr,
                                tuple(pts_img[i]),
                                tuple(pts_img[j]),
                                (0, 165, 255),
                                2,
                                cv2.LINE_AA,
                            )
                    mixed = cv2.addWeighted(
                        bgr, 0.6, cv2.cvtColor(believed_rgb, cv2.COLOR_RGB2BGR), 0.4, 0.0
                    )
                    cv2.imshow("Wrist Mixed (true + believed)", mixed)
                    cv2.waitKey(1)

                frame = phase.evaluate(min(phase_t, phase.duration))
                record_tick(frame)

                if is_descent:
                    if retry.is_backing_up():
                        if retry.backup_complete(raw_phase_t):
                            retry.finish_backup()
                            convergence = DescentServoConvergence()
                            saw_detection = False
                            playback_start = data.time
                    elif not saw_detection and raw_phase_t >= phase.duration and retry.can_retry():
                        retry.start_backup(raw_phase_t)
                        if verbose:
                            print(
                                "warning: descent saw no cube tags; backing up to "
                                "pregrasp and retrying "
                                f"({retry.retries_started}/{retry.max_retries})"
                            )
                    elif phase_t >= max_duration:
                        if verbose:
                            if saw_detection:
                                print(
                                    "warning: descent visual servo hit "
                                    f"{max_duration:.1f}s cap before settling "
                                    f"({convergence.stable_frames}/"
                                    f"{DESCENT_SERVO_STABLE_FRAMES} stable frames)"
                                )
                            else:
                                print(
                                    "warning: descent visual servo hit "
                                    f"{max_duration:.1f}s cap without a cube detection"
                                )
                        return "restart"
                    elif phase_t >= phase.duration and convergence.is_stable():
                        break
                elif phase_t >= phase.duration:
                    break

                for name, value in frame.joints.items():
                    data.ctrl[actuator_id[name]] = value
                offsets = offsets_rad_now()
                for name, offset in offsets.items():
                    if name in frame.joints:
                        data.ctrl[actuator_id[name]] += offset
                data.ctrl[actuator_id["gripper"]] = frame.gripper
                mujoco.mj_step(model, data, nstep=simulation_steps_per_tick)

                curr_contacts = {
                    (min(n1, n2), max(n1, n2))
                    for n1, n2 in scan_contacts(
                        model, data, episode.robot_geom_ids, episode.env_geom_ids
                    )
                    if is_unexpected(n1, n2)
                }
                if verbose:
                    for pair in curr_contacts - prev_contacts:
                        print(f"collision t={raw_phase_t:.3f}s  {pair[0]} ↔ {pair[1]}")
                prev_contacts = curr_contacts

                if viewer is not None:
                    viewer.sync()

                if realtime:
                    remaining = control_period - (time.monotonic() - tick_start)
                    if remaining > 0:
                        time.sleep(remaining)

            completed = phase.name

            if not servo_enabled:
                # Pure feedforward playback: the vetted plan needs no
                # checkpoints, so just advance phase by phase.
                current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                if not current_traj.phases:
                    status = "success"
                continue

            # The transitions below mirror the hardware executor: approach flows
            # straight into the servo descent; the descent's converged grasp
            # rebuilds grasp+lift as one contact-critical section; grasp+lift,
            # carry+drop_descent and drop_descent+release likewise run from the
            # locked plan; everything else replans from the believed readback.
            if completed == "approach" and (
                len(current_traj.phases) > 1 and current_traj.phases[1].name == "descent"
            ):
                current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                continue

            if completed == "descent" and isinstance(phase, DescentPhase):
                if phase.grasp.face == "free":
                    dynamic_grasp = phase.grasp
                else:
                    for g in grasp_candidates(kinematics, dynamic_source):
                        if g.face == phase.face and g.elbow == phase.elbow:
                            dynamic_grasp = g
                            break
                lift_cls = (
                    RecoveryLiftPhase
                    if isinstance(current_traj.phases[2], RecoveryLiftPhase)
                    else LiftPhase
                )
                grasp_phase = GraspPhase(dynamic_grasp.grasp_joints, start_gripper=GRIPPER_OPEN)
                lift_phase = lift_cls(
                    kinematics, dynamic_grasp.grasp_joints, dynamic_grasp.lift_joints
                )
                current_traj = dataclasses.replace(
                    current_traj,
                    phases=(grasp_phase, lift_phase, *current_traj.phases[3:]),
                    grasp=dynamic_grasp,
                )
                continue

            if completed == "grasp" and (
                len(current_traj.phases) > 1
                and current_traj.phases[1].name in ("lift", "recovery_lift")
            ):
                current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                continue

            if completed == "carry" and (
                len(current_traj.phases) > 1 and current_traj.phases[1].name == "drop_descent"
            ):
                current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                continue

            if completed == "drop_descent" and (
                len(current_traj.phases) > 1 and current_traj.phases[1].name == "release"
            ):
                current_traj = dataclasses.replace(current_traj, phases=current_traj.phases[1:])
                continue

            if len(current_traj.phases) <= 1:
                status = "success"
                break

            measured_joints = believed_arm_joints()
            measured_gripper = get_joint(model, data, "gripper")
            if verbose:
                print(f"Replanning remaining trajectory after {completed}...")
            candidate_traj = None
            for replan_traj in replan_remaining_candidates(
                kinematics,
                measured_joints,
                measured_gripper,
                completed,
                dynamic_source,
                episode.believed_target,
                dynamic_grasp,
                episode.end_joints,
                episode.end_gripper,
            ):
                events = _preflight(
                    model,
                    replan_traj,
                    actuator_id,
                    episode.robot_geom_ids,
                    episode.env_geom_ids,
                )
                if not any(is_unexpected(n1, n2) for _, n1, n2 in events):
                    candidate_traj = replan_traj
                    break
            if candidate_traj is None:
                if verbose:
                    print(f"No clean replan after {completed}; aborting episode.")
                return "restart"
            current_traj = candidate_traj
    finally:
        if servo_renderer is not None:
            servo_renderer.close()
        if show_wrist_mixed:
            import cv2

            cv2.destroyAllWindows()

    if verbose:
        print(placement_error(model, data, episode.target).summary())
    return "success" if status == "success" else "restart"
