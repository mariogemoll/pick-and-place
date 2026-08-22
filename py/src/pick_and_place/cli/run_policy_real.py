# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a learned policy on the physical SO-101, closed-loop.

Two leaves, because what a checkpoint is and how it is queried is not something
the two controllers share.

``lerobot`` loads a LeRobot checkpoint (ACT, SmolVLA, pi0.5, ...); pass
``--checkpoint`` to run a fine-tune, else the base ``lerobot/smolvla_base``,
an un-finetuned plumbing spike (the arm moves but does not solve the task).
``flow-image`` runs an image-conditioned flow policy in this process, from a
checkpoint and the dataset export it was trained on (``--flow-export``); it is
queried at the policy's trained rate, and its live camera frames are reduced
through the export's recorded video resolution before reaching the model's
input resolution.

Everything about the rig and the run is declared once and shared by both leaves.

The hardware counterpart to ``run_policy_sim.py``. Where the sim run renders MuJoCo
cameras and integrates physics, this reads two real cameras and a real arm: each
control tick it snapshots the latest overhead and wrist frames, reads the
follower's joints, asks the selected controller for an action, and streams it back
to the arm as position targets.

By default it is *simpler* than the sim run on the proprioception side. The
follower already reports and accepts the exact real frame the dataset was
recorded in — arm joints in degrees, gripper as a 0-100 position — so for a
checkpoint fine-tuned on this rig's own recordings the follower's reading is the
observation state verbatim and the predicted action is sent verbatim, clamped to
the joint limits. No radians appear anywhere on the policy path; MuJoCo is
loaded only to derive those limits and the neutral start pose.

**A checkpoint trained in simulation needs two corrections that a real-trained
one must not get.** It learned a world where the state, the images and the
command all agree, and on hardware they do not: the servo readback differs from
the model frame by the session's joint zeros, which drift day to day, and the
joint settles a fitted tracking bias away from whatever it was commanded. Pass
``--joint-zeros`` and ``--tracking-bias-scale 1`` to correct both. They compose
on the command — a joint settles at model angle ``command + zero + bias`` — and
the joint zeros additionally shift the state the policy is shown.

The cameras do need conversion: each raw, lens-distorted frame is undistorted
with its calibrated intrinsics, center-cropped to the policy's aspect ratio, and
resized to its input resolution every tick, via the same geometry
the dataset conversion applies to recorded datasets — so the live
frames fed to the policy match the ones it was fine-tuned on, pixel-geometry for
pixel-geometry. The resolution defaults to whatever the checkpoint was trained on.

``--save-video`` records policy inputs, ``--record-video`` captures continuous
native camera footage, and ``--action-log`` stores measured states, predicted
actions, sent commands, and raw prediction chunks for each attempt.

Each attempt finds the target and cube, moves to a fresh near-neutral pose, and
runs until placement succeeds, the timeout expires, or the operator presses
Enter. The overhead camera verifies that the cube was set down at the target;
camera extrinsics are solved at startup and checked for drift between attempts.
Timed-out attempts return to neutral and retry. ``--loop`` continues after a
success instead of exiting.

That scoring reads the tagged cube's pose, so ``--no-measure-scene`` is needed for
a policy trained on the plain blue cube, which has none: attempts then run
unscored and the operator judges them.

Safety: the arm ramps smoothly from wherever it is parked onto each start pose
before the policy takes over, and on exit (success, Ctrl-C or step budget) it
parks NEUTRAL -> REST and releases torque. Every command is clamped to the
model's joint limits.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
from types import SimpleNamespace

# Some SmolVLM backbone ops are not implemented for Apple MPS; fall back to CPU
# for just those ops instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mujoco
import numpy as np

from pick_and_place.sim.scene import build_scene
from pick_and_place.core.camera_calibration import (
    LOCAL_CAMERA_INTRINSICS_DIR,
    load_camera_intrinsics,
    load_local_camera_intrinsics,
)
from pick_and_place.policies.dataset_export import resolve_recording_hw
from pick_and_place.policies.flow_image_policy import FlowImagePolicyController
from pick_and_place.scripted.episode_sampling import sample_hunt_pose, sample_near_neutral
from pick_and_place.cli.run_policy_real_parser import build_parser, validate
from pick_and_place.runtime.action_log import ActionLog
from pick_and_place.runtime.frame_reader import open_frame_reader
from pick_and_place.runtime.ramp import ramp_follower
from pick_and_place.core.robot_dynamics import (
    load_robot_dynamics_config,
    tracking_bias_deg,
    tracking_bias_vector,
)
from pick_and_place.scripted.motion import ramp_setpoints
from pick_and_place.spec.robot import CONTROL_HZ, GRIPPER_INDEX, JOINT_NAMES
from pick_and_place.core.joint_frames import (
    action_to_joints,
    clamp_and_warn,
    follower_clamp_limits,
    joints_to_action,
    load_joint_zero_offsets,
    sim_frame_to_real,
)
from pick_and_place.hardware.follower import make_so101_follower
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.perception.image_rectify import (
    build_undistort_map,
    center_crop_and_resize,
    transform_frame,
)
from pick_and_place.sim.derive_kinematics import derive_kinematics
from pick_and_place.runtime.overhead_detection import OperatorNotifier
from pick_and_place.policies.policy import (
    DEFAULT_CHECKPOINT,
    make_policy,
    resolve_checkpoint_cameras,
    select_device,
)
from pick_and_place.spec.controller import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.spec.robot import (
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)

# During the settle phase the arm's peak joint speed must stay below
# ``--settle-speed`` continuously for this long before the placement counts as
# finished, so a momentary pause mid-retreat does not end the attempt early.
SETTLE_STILL_HOLD = 1.0


def launch_flow_image_controller(
    args: argparse.Namespace,
    *,
    override_hw: tuple[int, int] | None,
) -> tuple[FlowImagePolicyController, tuple[int, int]]:
    """Resolve the recording resolution and load the controller."""
    try:
        recording_hw = resolve_recording_hw(
            args.flow_export / "normalization.npz", args.recording_hw
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    device = select_device(args.device)
    print(f"Loading the flow policy {args.checkpoint} on {device}...")
    controller = FlowImagePolicyController.from_export(
        args.checkpoint,
        args.flow_export,
        act_steps=args.flow_act_steps,
        integration_steps=args.flow_integration_steps,
        device=device,
        seed=args.flow_seed,
        noise_correlation=args.flow_noise_correlation,
    )
    if override_hw is not None and override_hw != controller.image_hw:
        raise SystemExit(
            f"--image-height/--image-width {override_hw} do not match the "
            f"model's trained image size {controller.image_hw}"
        )
    return controller, recording_hw


def _drain_stdin_lines() -> bool:
    """Return True if the operator has typed a line on stdin, consuming all
    pending lines. Non-blocking, so it can be polled from the control loop
    without stalling a tick."""
    typed = False
    while select.select([sys.stdin], [], [], 0)[0]:
        if not sys.stdin.readline():
            break  # EOF
        typed = True
    return typed


def run(args: argparse.Namespace) -> None:
    """Drive the chosen controller on the rig."""
    override = (args.image_height, args.image_width)
    override_hw = (args.image_height, args.image_width) if all(override) else None

    # The flow-image controller predicts an action horizon and hands it out over
    # the following ticks; the LeRobot path is queried one action at a time.
    chunked_controller = None
    recording_hw = None
    if args.controller == "flow-image":
        chunked_controller, recording_hw = launch_flow_image_controller(
            args, override_hw=override_hw
        )
        img_h, img_w = chunked_controller.image_hw
        overhead_key, wrist_key = OVERHEAD_FEATURE, WRIST_FEATURE
        control_hz = chunked_controller.policy_hz
    else:
        (img_h, img_w), (overhead_key, wrist_key) = resolve_checkpoint_cameras(
            args.checkpoint, override_hw=override_hw
        )
        control_hz = CONTROL_HZ

    import cv2

    device = None
    if args.controller == "lerobot":
        device = select_device(args.device)
        print(f"Loading {args.checkpoint} on {device} (first run downloads the weights)...")
    print(
        f"Feeding {img_w}x{img_h} (WxH) frames as {overhead_key!r} (overhead) "
        f"and {wrist_key!r} (wrist)."
    )
    if recording_hw is not None:
        print(
            f"Downsampling live frames through the training dataset's "
            f"{recording_hw[1]}x{recording_hw[0]} recording resolution."
        )
    print(f"Policy control rate: {control_hz:g} Hz.")
    if args.send_substeps > 1:
        print(
            f"Ramping each setpoint over {args.send_substeps} sends "
            f"({control_hz * args.send_substeps:g} Hz to the servos)."
        )

    # MuJoCo is used only for the joint limits (to clamp commands) and to map the
    # neutral sim pose into the real frame for the start ramp — never stepped.
    model = build_scene(include_environment=True).compile()
    kinematics = derive_kinematics(model)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()
    # A real servo settles a fitted bias away from what it was commanded, which
    # simulation does not reproduce, so a policy trained there aims short on
    # hardware. Subtracting it makes the arm land where the policy asked.
    tracking_bias = tracking_bias_vector(
        tracking_bias_deg(load_robot_dynamics_config(), scale=args.tracking_bias_scale),
        JOINT_NAMES,
    )
    if args.tracking_bias_scale:
        print(f"Compensating the fitted servo tracking bias at {args.tracking_bias_scale:g}x.")
    # A servo readback and the model frame differ by the session's joint zeros,
    # which drift day to day. A checkpoint fine-tuned on this rig's own
    # recordings learned the servo frame and wants none of this; one trained in
    # simulation learned a world where the state, the images and the command all
    # agree, and on hardware they do not until the zeros are applied.
    joint_zero_offsets = np.zeros(len(JOINT_NAMES))
    if args.joint_zeros is not None:
        offsets = load_joint_zero_offsets(args.joint_zeros)
        joint_zero_offsets = np.array([offsets.get(name, 0.0) for name in JOINT_NAMES])
        print(f"Applying session joint zeros from {args.joint_zeros}: {offsets}")
    neutral_real = sim_frame_to_real(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER)
    rest_real = sim_frame_to_real(REST_ARM_JOINTS, REST_GRIPPER)

    predict_action = None
    if args.controller == "lerobot":
        from lerobot.utils.control_utils import predict_action

    intrinsics_by_camera = load_local_camera_intrinsics()
    missing = [cam for cam in ("overhead_camera", "wrist_camera") if cam not in intrinsics_by_camera]
    if missing:
        raise SystemExit(f"no calibrated intrinsics for {missing}; cannot undistort")
    workspace_intrinsics = None
    if args.workspace_camera is not None:
        workspace_intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / "workspace_camera.json"
        if not workspace_intrinsics_path.exists():
            raise SystemExit(
                f"no calibrated intrinsics at {workspace_intrinsics_path}; cannot undistort"
            )
        workspace_intrinsics = load_camera_intrinsics(workspace_intrinsics_path)

    print("Opening cameras...")
    overhead = open_frame_reader(args.camera, 1920, 1080, "overhead")
    wrist = open_frame_reader(args.wrist_camera, 1280, 720, "wrist")
    first_overhead = overhead.wait_for_frame().bgr
    first_wrist = wrist.wait_for_frame().bgr
    workspace = first_workspace = None
    if args.workspace_camera is not None:
        workspace = open_frame_reader(args.workspace_camera, 1920, 1080, "workspace")
        first_workspace = workspace.wait_for_frame().bgr

    # Every frame is rectified to the same pinhole view the offline dataset
    # conversion produces, at the policy's input resolution, so the policy loads
    # against a fixed shape regardless of either camera's native resolution.
    overhead_undistort_map = build_undistort_map(
        intrinsics_by_camera["overhead_camera"], first_overhead.shape[1], first_overhead.shape[0], cv2
    )
    wrist_undistort_map = build_undistort_map(
        intrinsics_by_camera["wrist_camera"], first_wrist.shape[1], first_wrist.shape[0], cv2
    )

    def policy_frame(rgb, undistort_map):
        if recording_hw is None:
            return transform_frame(rgb, undistort_map, img_w, img_h, cv2)
        recorded = transform_frame(
            rgb,
            undistort_map,
            recording_hw[1],
            recording_hw[0],
            cv2,
        )
        return center_crop_and_resize(recorded, img_w, img_h, cv2)

    if chunked_controller is None:
        policy, preprocessor, postprocessor = make_policy(
            args.checkpoint,
            (img_h, img_w),
            (overhead_key, wrist_key),
            device,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            base_checkpoint=args.base_checkpoint,
        )
    else:
        policy = chunked_controller
        preprocessor = postprocessor = None
    policy.reset()
    if chunked_controller is not None:
        print(
            f"Policy chunks: predicts {policy.prediction_steps}, "
            f"executes {policy.act_steps} before re-query "
            f"with {policy.observation_steps} observation steps "
            f"({policy.integration_steps} flow integration steps)."
        )
    elif hasattr(policy.config, "chunk_size") and hasattr(policy.config, "n_action_steps"):
        print(
            f"Policy chunks: predicts {policy.config.chunk_size}, "
            f"executes {policy.config.n_action_steps} before re-query."
        )
    if (
        chunked_controller is None
        and getattr(policy.config, "temporal_ensemble_coeff", None) is not None
    ):
        print(f"Temporal ensembling coeff: {policy.config.temporal_ensemble_coeff}")

    # Action logging: capture every raw chunk the model predicts by wrapping
    # predict_action_chunk, which select_action calls both under temporal
    # ensembling (every tick, before the ensembler averages it away) and in
    # queued mode (once per re-query). The control loop drains the capture each
    # tick and unnormalizes it through the same postprocessor as the returned
    # action, so the log compares like with like in real units.
    action_log = None
    captured_chunk: list = [None]
    if args.action_log is not None:
        import datetime

        log_dir = args.action_log / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        action_log = ActionLog(log_dir)
        print(f"Logging per-attempt actions and raw chunks to {log_dir}")
        if chunked_controller is not None:
            print("Chunked-controller action logs include every raw predicted horizon.")
        elif hasattr(policy, "predict_action_chunk"):
            _predict_action_chunk = policy.predict_action_chunk

            def _capture_predict_action_chunk(batch):
                chunk = _predict_action_chunk(batch)
                captured_chunk[0] = chunk
                return chunk

            policy.predict_action_chunk = _capture_predict_action_chunk
        else:
            print(
                "Warning: policy has no predict_action_chunk; logging without raw chunks."
            )

    print("Connecting to follower...")
    # Keep torque on a plain disconnect so the arm holds rather than going limp;
    # torque is only released deliberately at REST in the finally block.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    wrist_writer = overhead_writer = None
    if args.save_video is not None:
        import imageio.v2 as imageio

        args.save_video.mkdir(parents=True, exist_ok=True)
        wrist_writer = imageio.get_writer(args.save_video / "wrist.mp4", fps=control_hz)
        overhead_writer = imageio.get_writer(args.save_video / "overhead.mp4", fps=control_hz)
        print(f"Saving observation frames to {args.save_video}/{{wrist,overhead}}.mp4")

    # Continuous run recording: every camera's full native-rate, undistorted
    # stream (no cropping/resizing) on a shared clock, with optional audio.
    # Frames are submitted from the reader threads, so the recording sees every
    # captured frame, not just the ones the control loop happened to snapshot.
    recorder = None
    record_dir = None
    if args.record_video is not None:
        import datetime

        from pick_and_place.analysis.episode_video import LiveVideoRecorder

        record_dir = args.record_video / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        record_maps = {"overhead": overhead_undistort_map, "wrist": wrist_undistort_map}
        if workspace is not None:
            record_maps["workspace"] = build_undistort_map(
                workspace_intrinsics, first_workspace.shape[1], first_workspace.shape[0], cv2
            )
        recorder = LiveVideoRecorder(
            record_dir,
            record_maps,
            audio=args.record_audio,
            audio_device=(
                int(args.audio_device)
                if args.audio_device is not None and args.audio_device.isdecimal()
                else args.audio_device
            ),
        )
        overhead.on_frame = lambda bgr, t: recorder.submit("overhead", bgr, t)
        wrist.on_frame = lambda bgr, t: recorder.submit("wrist", bgr, t)
        if workspace is not None:
            workspace.on_frame = lambda bgr, t: recorder.submit("workspace", bgr, t)
        cams = "/".join(record_maps)
        audio_note = " with audio" if args.record_audio else ""
        print(f"Recording the {cams} cameras{audio_note} to {record_dir}")

    # Automatic attempt setup and success checks read the tagged cube's pose from the
    # overhead camera, so they are available to any controller trained against it —
    # what decides this is the cube in the scene, not which policy is driving. A plain
    # blue cube has no measurable pose and has to be scored by the operator.
    measure_scene = args.measure_scene
    rng = np.random.default_rng()
    from pick_and_place.calibration.camera_compare import load_intrinsics
    from pick_and_place.calibration.cam_align_solve import (
        ExtrinsicsSolveError,
        apply_solve_result,
        check_solve_plausible,
        solve_overhead_extrinsics,
    )
    from pick_and_place.core.rotations import pose_delta_mm_deg
    from pick_and_place.core.camera_calibration import load_local_camera_extrinsics
    from pick_and_place.sim.camera_extrinsics import apply_camera_extrinsics_to_model
    from pick_and_place.perception.cube_detection import (
        cube_pose_to_world,
        estimate_cube_pose,
        make_cube_detector,
    )
    from pick_and_place.runtime.overhead_detection import (
        CUBE_LOOK_TIMEOUT,
        track_cube,
        track_drop_zone_square,
    )
    from pick_and_place.perception.paper_detection import PaperTracker

    # The success scan reads the cube pose in world coordinates, so the model's
    # overhead camera must sit where the real one does. Start from the saved
    # extrinsics; unless --no-recalibrate, they are re-solved live from the
    # workspace-frame tags at startup (see solve_startup_extrinsics below).
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if cam_id < 0:
        raise SystemExit(f"No camera named {args.camera_name!r} in the model.")
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
    # Set by the startup overhead solve; the periodic drift check compares against it.
    startup_extrinsics: tuple[np.ndarray, np.ndarray] | None = None
    last_drift_check = 0.0  # monotonic time of the last drift solve

    det_intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{args.camera_name}.json"
    if not det_intrinsics.exists():
        raise SystemExit(f"Missing {args.camera_name} intrinsics at {det_intrinsics}.")
    det_matrix, det_map = load_intrinsics(det_intrinsics, 1920, 1080, cv2)
    detector = make_cube_detector() if measure_scene else None
    drop_zone_tracker = PaperTracker()
    notifier = OperatorNotifier(enabled=args.operator_alerts, sound_path=args.alert_sound)

    def scan_cube_world():
        """Detect the cube on the latest overhead frame and return its (x, y, z)
        world position, or None if it is not currently visible."""
        frame = overhead.latest()
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.remap(rgb, *det_map, cv2.INTER_LINEAR)
        estimate = estimate_cube_pose(rgb, detector, det_matrix)
        if estimate is None:
            return None
        _, pos = cube_pose_to_world(estimate, cam_pos, cam_rot)
        return float(pos[0]), float(pos[1]), float(pos[2])

    if args.controller == "lerobot":
        print(f"Instruction: {args.instruction!r}")
    if args.checkpoint == DEFAULT_CHECKPOINT:
        print("Running closed-loop. Actions are NOT task-calibrated (un-finetuned base).")
    else:
        print(f"Running closed-loop with fine-tuned checkpoint {args.checkpoint!r}.")

    period = 1.0 / control_hz
    tick = 0
    parked = False

    def run_attempt(target_xy) -> str:
        """Drive the policy closed-loop for one attempt.

        Returns ``"steps"`` when the global ``--steps`` budget is hit, ``"timeout"``
        when ``--attempt-timeout`` elapses without a placement, ``"abandoned"``
        when the operator presses Enter to declare the attempt failed, or ``"success"``
        once the cube has been set down at the target (within ``--success-tolerance``
        in xy and ``--place-height-tolerance`` of its resting height). Placement is
        confirmed after ``--success-dwell`` seconds; the arm slow-down runs in
        parallel over the same window and is soft — success fires as soon as the
        cube has been placed for the dwell and the arm has slowed, or at
        ``--settle-timeout`` regardless. The timeout is disabled when
        ``--attempt-timeout`` is <= 0.
        """
        nonlocal tick
        # Discard any stale lines typed during the ramp/hunt so an old Enter
        # press cannot instantly abandon the attempt that is just starting.
        _drain_stdin_lines()
        print("Press Enter at any time to stop this attempt.")
        attempt_start = time.monotonic()
        next_tick = time.monotonic()
        report_time, report_tick, infer_seconds = next_tick, tick, 0.0

        # Run the overhead placement scan on its own thread: the AprilTag detection
        # on the full-resolution frame takes tens of milliseconds, which would
        # stall the 30 Hz control loop if done inline. The heavy work (detection,
        # remap) releases the GIL, so it overlaps the policy inference cleanly. The
        # scan keeps a single shared timestamp — when the cube was first seen
        # continuously placed — which the control loop reads to time out placement,
        # confirm it, and monitor slow-down all against the same clock.
        placement = SimpleNamespace(since=None)
        stop_scan = threading.Event()

        def scan_loop() -> None:
            # The cube counts as placed only when it is at the target in xy *and*
            # back near its resting height (set down, not still carried above the
            # target) — so a fly-through during the carry never counts. Once a
            # placement has started, a lost sighting does NOT clear it: the
            # retreating arm routinely occludes a cube it just set down. Only a
            # clear "visible but not placed" reading (moved away or lifted) clears
            # it, which also lets the control loop reset if the cube moves again.
            while not stop_scan.wait(args.scan_interval):
                pose = scan_cube_world()
                if pose is None:
                    print("success scan: cube not visible")
                    continue
                x, y, z = pose
                dist = float(np.hypot(x - target_xy[0], y - target_xy[1]))
                above = z - CUBE_HALF_SIZE
                at_target = dist <= args.success_tolerance
                set_down = abs(above) <= args.place_height_tolerance
                print(
                    f"success scan: cube ({x:.3f}, {y:.3f}) {dist * 100.0:.1f} cm from "
                    f"target, {above * 100.0:+.1f} cm above rest"
                )
                if at_target and set_down:
                    if placement.since is None:
                        placement.since = time.monotonic()
                else:
                    placement.since = None

        scanner = None
        if target_xy is not None:
            scanner = threading.Thread(target=scan_loop, daemon=True)
            scanner.start()
        if action_log is not None:
            action_log.start_attempt()
        outcome = "error"
        try:
            outcome = _control_loop(attempt_start, next_tick,
                                    report_time, report_tick, infer_seconds, placement)
            return outcome
        finally:
            stop_scan.set()
            if scanner is not None:
                scanner.join(timeout=2.0)
            if action_log is not None:
                action_log.end_attempt(outcome)

    def _control_loop(attempt_start, next_tick,
                      report_time, report_tick, infer_seconds, placement) -> str:
        nonlocal tick
        # Placement confirmation and arm slow-down run concurrently from the moment
        # the cube is first seen placed (``placement.since``): the policy keeps
        # driving so the arm retreats, but the placement is the success — the
        # slow-down is soft and only trims the tail, never a hard requirement.
        still_since = None
        prev_arm = None
        prev_t = None
        # Where the ramp to each new setpoint starts. None until the first send of
        # the attempt, which ramps from the arm's measured pose instead.
        last_sent = None
        announced = False
        raw_lag = None  # newest chunk's first action vs the ensembled one, deg
        while True:
            if _drain_stdin_lines():
                return "abandoned"
            overhead_rgb = cv2.cvtColor(overhead.wait_for_frame().bgr, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist.wait_for_frame().bgr, cv2.COLOR_BGR2RGB)
            overhead_rgb = policy_frame(overhead_rgb, overhead_undistort_map)
            wrist_rgb = policy_frame(wrist_rgb, wrist_undistort_map)

            state = action_to_joints(follower.get_observation(), neutral_real).astype(np.float32)
            # `state` stays in the servo frame for the velocity cap, the safety
            # checks and the log, because that is what the hardware reported.
            # Only the policy sees the model frame.
            policy_state = (state + joint_zero_offsets).astype(np.float32)
            observation = {
                STATE_FEATURE: policy_state,
                OVERHEAD_FEATURE: overhead_rgb,
                WRIST_FEATURE: wrist_rgb,
            }
            if wrist_writer is not None:
                wrist_writer.append_data(wrist_rgb)
                overhead_writer.append_data(overhead_rgb)

            infer_start = time.monotonic()
            if chunked_controller is None:
                assert predict_action is not None
                lerobot_observation = {
                    STATE_FEATURE: policy_state,
                    overhead_key: overhead_rgb,
                    wrist_key: wrist_rgb,
                }
                action = predict_action(
                    lerobot_observation,
                    policy,
                    device,
                    preprocessor,
                    postprocessor,
                    use_amp=False,
                    task=args.instruction,
                    robot_type="so101",
                )
                action_real = action.to("cpu").numpy().reshape(-1)[: len(JOINT_NAMES)]
            else:
                action_real = policy.act(observation)
            infer_seconds += time.monotonic() - infer_start
            # Both corrections land on the command, and compose: a joint settles
            # at model angle ``command + zero_offset + tracking_bias``, so
            # reaching the policy's target means subtracting both. Before
            # clamping, so the joint limits still bind what is actually sent,
            # and before the velocity cap, so the cap bounds real travel rather
            # than the uncorrected request.
            target = clamp_and_warn(
                action_real - joint_zero_offsets - tracking_bias, clamp_low, clamp_high, clip_warned
            )
            # Velocity cap: never command an arm joint more than one tick's worth
            # of travel beyond where the arm actually is. This bounds both speed
            # and the servo's position error regardless of what the policy asks
            # for. The gripper passes through (open/close should stay timely).
            commanded = target.copy()
            if args.max_joint_speed > 0:
                max_step = args.max_joint_speed / control_hz
                arm_delta = target[:GRIPPER_INDEX] - state[:GRIPPER_INDEX]
                commanded[:GRIPPER_INDEX] = state[:GRIPPER_INDEX] + np.clip(
                    arm_delta, -max_step, max_step
                )
            # One setpoint per policy period would be a single step the servos chase
            # and then hold, so the arm tracks a staircase. Ramp to it across the
            # period instead, pacing the sends to absolute deadlines so the extra
            # sends consume the period's slack rather than adding to it. Total travel
            # per period — and so the velocity cap above — is unchanged.
            sends = ramp_setpoints(last_sent if last_sent is not None else state,
                                   commanded, args.send_substeps)
            sub_period = period / len(sends)
            for i, setpoint in enumerate(sends, start=1):
                # The gripper takes its new value on the first send: like the
                # velocity cap above, opening and closing stay timely.
                setpoint[GRIPPER_INDEX] = commanded[GRIPPER_INDEX]
                follower.send_action(joints_to_action(setpoint))
                if i < len(sends):
                    slack = (next_tick + i * sub_period) - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
            last_sent = commanded.copy()

            # Drain the chunk captured during this tick's inference (if any) and
            # unnormalize the whole (chunk, dim) sequence in one pass — the
            # postprocessor's stats broadcast per action dimension. Row 0 is the
            # model's freshest prediction for this very tick, so its gap to the
            # returned (ensembled) action is the ensemble lag.
            chunk_real = None
            if chunked_controller is not None and policy.latest_prediction is not None:
                chunk_real = policy.latest_prediction.copy()
                raw_lag = float(
                    np.max(np.abs(chunk_real[0, :GRIPPER_INDEX] - action_real[:GRIPPER_INDEX]))
                )
            elif captured_chunk[0] is not None:
                chunk_real = (
                    postprocessor(captured_chunk[0].squeeze(0))
                    .to("cpu")
                    .numpy()[:, : len(JOINT_NAMES)]
                )
                captured_chunk[0] = None
                raw_lag = float(
                    np.max(np.abs(chunk_real[0, :GRIPPER_INDEX] - action_real[:GRIPPER_INDEX]))
                )
            if action_log is not None:
                action_log.log_tick(
                    tick, time.monotonic(), state, action_real, commanded, chunk_real
                )

            if tick % 10 == 0:
                np.set_printoptions(precision=2, suppress=True)
                now = time.monotonic()
                ticks = tick - report_tick
                rate = f"  {ticks / (now - report_time):5.1f} Hz" if ticks else ""
                infer = f"  infer {infer_seconds / ticks * 1000.0:5.1f} ms" if ticks else ""
                lag = f"  raw-ens {raw_lag:4.1f}deg" if raw_lag is not None else ""
                if args.attempt_timeout > 0:
                    clock = f"  {now - attempt_start:4.1f}/{args.attempt_timeout:.0f}s"
                else:
                    clock = f"  {now - attempt_start:4.1f}s"
                print(f"tick {tick:4d}  action={commanded}{rate}{infer}{lag}{clock}")
                report_time, report_tick, infer_seconds = now, tick, 0.0

            tick += 1
            if args.steps and tick >= args.steps:
                return "steps"

            now = time.monotonic()
            arm = state[:GRIPPER_INDEX]

            # Track arm slow-down every tick so it is already known the moment the
            # placement dwell completes, rather than measured only afterwards.
            arm_settled = False
            if prev_arm is not None and now > prev_t:
                speed = float(np.max(np.abs(arm - prev_arm))) / (now - prev_t)
                if speed <= args.settle_speed:
                    if still_since is None:
                        still_since = now
                else:
                    still_since = None
                arm_settled = still_since is not None and now - still_since >= SETTLE_STILL_HOLD
            prev_arm, prev_t = arm.copy(), now

            since = placement.since
            if since is None:
                # Cube not (yet, or no longer) placed. Give up on the attempt once
                # the placement timeout passes with nothing set down.
                announced = False
                if args.attempt_timeout > 0 and now - attempt_start >= args.attempt_timeout:
                    return "timeout"
            else:
                if not announced:
                    print("Cube placed. Confirming placement while the arm slows down...")
                    announced = True
                placed_for = now - since
                # The dwell confirms the placement; the slow-down runs over the same
                # window and only trims the tail. Finish once the cube has held for
                # the dwell and the arm has slowed, or at --settle-timeout regardless
                # (so the policy never lingers in post-placement, off-distribution
                # territory). A cube that moves again clears placement.since above.
                if placed_for >= args.success_dwell and (arm_settled or placed_for >= args.settle_timeout):
                    reason = "arm settled" if arm_settled else "settle timeout"
                    print(f"Cube placed for {placed_for:.1f}s ({reason}). Success.")
                    return "success"

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                # Don't issue a burst of catch-up commands after a long stall.
                next_tick = time.monotonic()

    def go_neutral() -> None:
        ramp_follower(
            follower,
            neutral_real,
            clamp_low,
            clamp_high,
            clip_warned,
            max_joint_speed=args.max_joint_speed,
        )

    def hunt(label, detect):
        """Look for something on the overhead camera, panning the arm through fresh
        random search poses (the 'dance') to clear the view between tries. The arm
        can sit between the fixed overhead camera and the cube or square, so a look
        from one pose may be blocked while another is clear. Returns the detection
        or None after ``--max-hunt-tries`` looks."""
        for i in range(args.max_hunt_tries):
            if i > 0:
                arm, grip = sample_hunt_pose(rng)
                print(f"{label} look {i + 1}/{args.max_hunt_tries}: panning to a new search pose...")
                ramp_follower(
                    follower,
                    sim_frame_to_real(arm, grip),
                    clamp_low,
                    clamp_high,
                    clip_warned,
                    max_joint_speed=args.max_joint_speed,
                )
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"{label} look 1/{args.max_hunt_tries}: searching from the current pose...")
            result = detect()
            if result is not None:
                return result
        return None

    def find_or_prompt(label, detect, missing_message):
        """Hunt for a detection; if the dance comes up empty, ask the operator to
        make it visible and try again. Returns the detection or None on Ctrl-D."""
        while True:
            result = hunt(label, detect)
            if result is not None:
                return result
            notifier.alert(missing_message)
            try:
                input(f"Make the {label} visible, then press Enter (Ctrl-D to stop)...")
            except EOFError:
                return None

    def detect_target():
        return track_drop_zone_square(
            overhead, args.camera_name, model, data, drop_zone_tracker, args.drop_zone_color
        )

    def detect_cube():
        return track_cube(overhead, args.camera_name, model, data, CUBE_LOOK_TIMEOUT)

    def solve_startup_extrinsics() -> None:
        """Solve the overhead extrinsics live from the workspace-frame tags, validate
        them, and apply them to the model so the success scan back-projects the cube
        against where the camera actually is. Refuses to start on a failed or
        implausible solve."""
        nonlocal startup_extrinsics, last_drift_check, cam_pos, cam_rot
        print("Solving overhead camera extrinsics from the workspace-frame tags...")
        result = solve_overhead_extrinsics(
            model,
            data,
            overhead,
            camera_name=args.camera_name,
            intrinsics_path=args.overhead_intrinsics,
            samples=args.recalibrate_samples,
            max_seconds=args.recalibrate_max_seconds,
            cv2_module=cv2,
        )
        if result is None:
            raise SystemExit(
                "Overhead calibration failed: never saw all four workspace-frame tags "
                "in one frame. Clear the camera view and check the tags."
            )
        try:
            check_solve_plausible(result)
        except ExtrinsicsSolveError as exc:
            raise SystemExit(f"Overhead calibration rejected: {exc}") from exc
        apply_solve_result(model, data, args.camera_name, result)
        startup_extrinsics = (
            np.array(result.pos, dtype=float),
            np.array(result.quat, dtype=float),
        )
        # The success scan back-projects the cube through this pose, so refresh the
        # cached camera frame it reads to match the freshly solved extrinsics.
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
        last_drift_check = time.monotonic()
        print(
            f"Overhead extrinsics solved: {result.reprojection_error_px:.2f}px, "
            f"{result.nominal_delta.translation_m * 1000.0:.1f}mm / "
            f"{result.nominal_delta.rotation_deg:.2f}deg from nominal."
        )

    def check_overhead_drift() -> None:
        """Re-solve the overhead extrinsics from the current (near-neutral) pose and
        stop the run if the camera has drifted from the startup calibration. Skips
        quietly if the tags are occluded, and is rate-limited to
        --recalibrate-check-interval so it only runs occasionally between attempts."""
        nonlocal last_drift_check
        if (
            not args.recalibrate
            or startup_extrinsics is None
            or args.recalibrate_check_interval <= 0
            or time.monotonic() - last_drift_check < args.recalibrate_check_interval
        ):
            return
        print("Drift check: re-solving overhead extrinsics...")
        saved_pos = model.cam_pos[cam_id].copy()
        saved_quat = model.cam_quat[cam_id].copy()
        check = solve_overhead_extrinsics(
            model,
            data,
            overhead,
            camera_name=args.camera_name,
            intrinsics_path=args.overhead_intrinsics,
            samples=args.recalibrate_samples,
            max_seconds=args.recalibrate_max_seconds,
            cv2_module=cv2,
        )
        # The re-solve only decides whether to stop; the startup calibration stays
        # live and is never re-applied mid-run.
        model.cam_pos[cam_id] = saved_pos
        model.cam_quat[cam_id] = saved_quat
        mujoco.mj_forward(model, data)
        last_drift_check = time.monotonic()
        if check is None:
            print("Drift check skipped: could not see all four tags (occluded). Continuing.")
            return
        drift_mm, drift_deg = pose_delta_mm_deg(
            startup_extrinsics[0],
            startup_extrinsics[1],
            np.array(check.pos, dtype=float),
            np.array(check.quat, dtype=float),
        )
        print(f"Overhead drift vs startup: {drift_mm:.1f}mm / {drift_deg:.2f}deg.")
        if drift_mm > args.recalibrate_drift_mm or drift_deg > args.recalibrate_drift_deg:
            raise SystemExit(
                f"Overhead camera drifted {drift_mm:.1f}mm / {drift_deg:.2f}deg since startup "
                f"(limits {args.recalibrate_drift_mm:.0f}mm / {args.recalibrate_drift_deg:.1f}deg). "
                "Stopping so the operator can check the mount and recalibrate."
            )

    try:
        print("Homing to the neutral pose...")
        go_neutral()
        if args.recalibrate and measure_scene:
            solve_startup_extrinsics()
        attempt = 0
        while True:
            attempt += 1
            budget = f"timeout {args.attempt_timeout:.0f}s" if args.attempt_timeout > 0 else "no timeout"
            print(f"\n=== Attempt {attempt} ({budget}) ===")

            target_xy = None
            if measure_scene:
                check_overhead_drift()
                target = find_or_prompt(
                    "drop-zone square", detect_target, "Drop-zone square not visible."
                )
                if target is None:
                    break
                target_xy = (float(target.x), float(target.y))
                print(f"Target drop zone at ({target_xy[0]:.3f}, {target_xy[1]:.3f}).")
                cube = find_or_prompt(
                    "cube", detect_cube, "Cube not visible in the pickup zone. Please reset it."
                )
                if cube is None:
                    break
                print(f"Cube at ({cube.x:.3f}, {cube.y:.3f}).")
            else:
                print("Running without cube/target measurement or automatic success detection.")

            # Each attempt starts from a fresh randomish near-neutral pose: the
            # policy is strongest early on, so a timed-out attempt is abandoned and
            # simply retried from a new start rather than left to flail.
            arm, grip = sample_near_neutral(rng)
            print("Ramping to a randomish start pose...")
            ramp_follower(
                follower,
                sim_frame_to_real(arm, grip),
                clamp_low,
                clamp_high,
                clip_warned,
                max_joint_speed=args.max_joint_speed,
            )
            policy.reset()

            outcome = run_attempt(target_xy)

            if outcome == "steps":
                break
            if not measure_scene:
                print(f"Unmeasured rollout ended: {outcome}.")
                if not args.loop:
                    break
                go_neutral()
                try:
                    input("Reset the scene, then press Enter for the next attempt "
                          "(Ctrl-C to stop)...")
                except EOFError:
                    break
                continue
            if outcome == "timeout":
                print(f"TIMEOUT — no success within {args.attempt_timeout:.0f}s. "
                      "Returning to neutral and retrying.")
                go_neutral()
                continue
            if outcome == "abandoned":
                print("ABANDONED — operator declared the attempt failed. "
                      "Returning to neutral and retrying.")
                go_neutral()
                continue

            # Success. Exit by default; with --loop, hand the scene back to the
            # operator to reset and keep going.
            print("SUCCESS — the cube reached the target.")
            if not args.loop:
                break
            notifier.alert("Success. Please reset the cube and target for the next attempt.")
            print("Parking to the neutral pose before the next attempt...")
            go_neutral()
            try:
                input("Reset the scene, then press Enter for the next attempt (Ctrl-C to stop)...")
            except EOFError:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        overhead.close()
        wrist.close()
        if workspace is not None:
            workspace.close()
        if recorder is not None:
            print(f"Finalizing the run recording at {record_dir}...")
            recorder.close()
        if wrist_writer is not None:
            wrist_writer.close()
            overhead_writer.close()
        try:
            print("Parking to NEUTRAL then REST...")
            follower.bus.enable_torque()
            ramp_follower(
                follower,
                neutral_real,
                clamp_low,
                clamp_high,
                clip_warned,
                max_joint_speed=args.max_joint_speed,
            )
            ramp_follower(
                follower,
                rest_real,
                clamp_low,
                clamp_high,
                clip_warned,
                max_joint_speed=args.max_joint_speed,
            )
            parked = True
        except Exception as exc:  # noqa: BLE001 - best-effort park before release
            print(f"Warning: could not park the arm: {exc}")
        if parked:
            print("At REST — releasing torque.")
            try:
                follower.bus.disable_torque()
            except Exception as exc:  # noqa: BLE001 - best-effort torque release
                print(f"Warning: could not release torque: {exc}")
        print("Disconnecting hardware...")
        follower.disconnect()
        if chunked_controller is not None:
            chunked_controller.close()
    print(f"Ran {tick} control ticks.")


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    validate(parser, args)
    run(args)


if __name__ == "__main__":
    main()
