# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Drive the physical SO-101 follower through a prepared pick-and-carry episode.

This is the home of the hardware execution path. The rig is the plant: the
trajectory's joint set points stream out to the real arm at ``CONTROL_HZ``.

Feedback is applied at two points, not continuously across the whole episode:

- **Descent (wrist-camera PBVS).** During the descent onto the cube, a wrist
  camera worker detects the cube as fast as frames and AprilTag solving allow.
  The control loop consumes the latest published estimate each tick, low-pass
  filters it into the live source pose, re-derives the locked face/elbow grasp,
  and ``DescentPhase.evaluate`` re-solves IK toward the updated grasp. See
  :mod:`pick_and_place.runtime.wrist_servo` and
  :mod:`pick_and_place.scripted.descent`.
- **Phase boundaries (checkpoint replanning).** After a completed phase the
  measured joints are sensed and the remaining trajectory is replanned and
  preflighted before continuing (sense → plan → execute → re-seed). Which
  boundaries get a checkpoint, and why the rest do not, is
  :mod:`pick_and_place.rollout.checkpoint`.

The other phases (hover, carry, release, lift) are feedforward playback. Motor
readback is logged every tick and, at checkpoints, fed back into the replan.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from pick_and_place.core.joint_frames import (
    GRIPPER_READBACK_CLOSED,
    action_to_joints,
    clamp_and_warn,
    follower_clamp_limits,
    sim_frame_to_real,
)
from pick_and_place.data.recorder import EpisodeRecorder
from pick_and_place.data.recording import RecordingSession
from pick_and_place.rollout.checkpoint import replan_from_checkpoint
from pick_and_place.scripted.checkpoint import fuses_into_next
from pick_and_place.scripted.descent import regrasp_after_descent
from pick_and_place.runtime.episodes import Episode
from pick_and_place.plant.real import RealPlant
from pick_and_place.data.trajectory_artifact import TrajectoryWriter
from pick_and_place.rollout.phase import Run, play_phase
from pick_and_place.runtime.wrist_preview import WristView, show_preview
from pick_and_place.runtime.ramp import ramp_follower
from pick_and_place.rollout.real_dataset import TickRecorder
from pick_and_place.runtime.wrist_servo import WristServo, open_sim_view, open_wrist_servo
from pick_and_place.spec.robot import (
    CONTROL_HZ,
    GRIPPER_INDEX,
    HARDWARE_SIMULATION_HZ,
    JOINT_NAMES,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)

# Default playback pace for the physical arm: run at the planner's nominal speed.
# Scaling the trajectory clock slows every phase uniformly without touching the
# planner. Override with ``speed``.
REAL_ARM_DEFAULT_SPEED = 1.0
# Logging-only pickup heuristic. A held cube should keep the physical gripper
# encoder noticeably more open than an empty close.
PICKUP_GRIPPER_MARGIN = 5.0


def ramp_to_resting(
    follower,
    target_real: np.ndarray,
    target_sim_joints: dict[str, float],
    target_sim_gripper: float,
    actuator_id: dict[str, int],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    low: np.ndarray,
    high: np.ndarray,
    warned: set[str],
    on_tick: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Ramp the real arm onto a pose while walking the sim onto its match.

    Both are driven off the same eased fraction, so the viewer shows what the arm
    is doing rather than jumping to the end pose; the sim is stepped and synced
    inside the ramp's tick budget.
    """
    current_sim_joints = {name: data.ctrl[actuator_id[name]] for name in target_sim_joints}
    current_sim_gripper = data.ctrl[actuator_id["gripper"]]

    def follow_in_sim(alpha: float, commanded: np.ndarray) -> None:
        for name in target_sim_joints:
            data.ctrl[actuator_id[name]] = current_sim_joints[name] + alpha * (
                target_sim_joints[name] - current_sim_joints[name]
            )
        data.ctrl[actuator_id["gripper"]] = current_sim_gripper + alpha * (
            target_sim_gripper - current_sim_gripper
        )
        mujoco.mj_step(model, data)
        if on_tick is not None:
            on_tick(commanded)
        viewer.sync()

    ramp_follower(follower, target_real, low, high, warned, viewer=viewer, on_tick=follow_in_sim)


def _report_tracking(recorder: EpisodeRecorder) -> None:
    """Print a per-joint desired-vs-actual error summary over the recorded run."""
    if len(recorder) == 0:
        print("No follower samples recorded.")
        return
    stacked = recorder.stacked()
    t, commanded, measured = stacked["t"], stacked["commanded"], stacked["measured"]
    error = measured - commanded
    print("\nPer-joint tracking (actual − commanded):")
    print(f"  {'joint':<14}{'unit':<5}{'max|err|':>10}{'mean|err|':>11}{'mean err':>10}")
    for i, name in enumerate(JOINT_NAMES):
        unit = "pos" if i == GRIPPER_INDEX else "deg"
        col = error[:, i]
        print(
            f"  {name:<14}{unit:<5}{np.max(np.abs(col)):>10.2f}"
            f"{np.mean(np.abs(col)):>11.2f}{np.mean(col):>10.2f}"
        )
    print(f"  ({len(recorder)} samples over {t[-1]:.2f}s)")
    if "wall_t" in stacked and len(stacked["wall_t"]) > 1:
        wall_dt = np.diff(stacked["wall_t"])
        missed = int(np.count_nonzero(wall_dt > 1.5 / CONTROL_HZ))
        print(
            f"  wall cadence: median {np.median(wall_dt) * 1000:.1f} ms "
            f"({1.0 / np.median(wall_dt):.1f} Hz), "
            f"p95 {np.percentile(wall_dt, 95) * 1000:.1f} ms, "
            f"{missed} missed tick(s)"
        )
    print("  (mean err is each joint's sim→real tracking bias)")


def _report_grasp(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Print where the jaws closed, in the cube's own frame.

    The one number that says whether a grasp was central or glancing, and the
    first thing to look at when an episode lifts nothing.
    """
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    if gripper_id < 0 or cube_id < 0:
        return

    from pick_and_place.core.geometry import JAW_CONTACT_POSITION

    tcp_world = data.xpos[gripper_id] + data.xmat[gripper_id].reshape(3, 3) @ JAW_CONTACT_POSITION
    tcp_cube = data.xmat[cube_id].reshape(3, 3).T @ (tcp_world - data.xpos[cube_id])
    print("\n--- GRASP DIAGNOSTICS ---")
    print("TCP position in cube local frame:")
    for axis, value in zip("xyz", tcp_cube):
        print(f"  {axis}={value * 1000:+.1f} mm")
    print("-------------------------\n")


def _grasp_and_lift(kinematics, grasp, free_grasp: bool) -> tuple[Any, Any]:
    """The close-and-lift pair, built from a grasp the descent settled on."""
    from pick_and_place.scripted.trajectory import GraspPhase, LiftPhase, RecoveryLiftPhase
    from pick_and_place.spec.robot import GRIPPER_OPEN

    lift_cls = RecoveryLiftPhase if free_grasp else LiftPhase
    return (
        GraspPhase(grasp.grasp_joints, start_gripper=GRIPPER_OPEN),
        lift_cls(kinematics, grasp.grasp_joints, grasp.lift_joints),
    )


def _pickup_report(
    phase_name: str, gripper_position: float, empty_position: float, margin: float
) -> dict[str, Any]:
    """Judge whether the gripper is holding anything, from its encoder alone.

    A held cube keeps the jaws further open than an empty close, so the gap
    between them is evidence of a pickup. It is only ever logged — nothing
    branches on it — because the encoder also reads open when the jaws jam.
    """
    delta = gripper_position - empty_position
    confidence = delta / margin if margin > 0.0 else float("inf")
    print(
        f"Pickup check after {phase_name}: gripper={gripper_position:.1f}, "
        f"empty={empty_position:.1f}, delta={delta:+.1f} (margin {margin:.1f})"
    )
    return {
        "pickup_check_phase": phase_name,
        "pickup_gripper_position": gripper_position,
        "pickup_empty_gripper_position": float(empty_position),
        "pickup_gripper_margin": float(margin),
        "pickup_gripper_delta": delta,
        "pickup_confidence": confidence,
    }


def execute_episode(
    episode: Episode,
    *,
    follower,
    viewer,
    recording: RecordingSession | None = None,
    overhead_camera_cap=None,
    workspace_camera_cap=None,
    speed: float | None = None,
    wrist_camera: str | None = None,
    wrist_intrinsics: str | None = None,
    show_wrist_cam: bool = False,
    show_wrist_mixed: bool = False,
    failed_trajectory_dir: Path | str | None = None,
    free_grasp: bool = False,
    pickup_empty_gripper_position: float = GRIPPER_READBACK_CLOSED,
    pickup_gripper_margin: float = PICKUP_GRIPPER_MARGIN,
    success_metadata: Callable[[], dict[str, Any]] | None = None,
    record_rest_to_rest: bool = False,
    joint_offsets_deg: dict[str, float] | None = None,
) -> str:
    """Run one pass of a prepared episode on an already-connected follower.

    The caller owns the ``follower`` (connected) and the ``viewer`` (a launched
    passive viewer, or a mock exposing ``is_running``/``sync``). This ramps the
    real arm onto the trajectory start pose, then steps the sim (the plant) while
    streaming set points to the arm at ``CONTROL_HZ`` and logging motor readback.
    MuJoCo advances in a batch of high-rate physics substeps per control tick.

    If ``recording`` is given, the episode is written straight into its
    ``LeRobotDataset`` and both the wrist camera (opened here when
    ``wrist_camera`` is set) and ``overhead_camera_cap`` (owned by the caller)
    are required; ``workspace_camera_cap`` is an optional third. See
    :class:`~pick_and_place.rollout.real_dataset.TickRecorder` for what a
    recorded episode guarantees.

    Returns ``"success"`` when the trajectory ran to completion, or ``"restart"``
    when a checkpoint replan failed and the caller should abort and re-home. A
    ``KeyboardInterrupt`` propagates (after camera cleanup) so the caller can park.
    When ``record_rest_to_rest`` is set, the recording includes the motion from
    REST to the episode's start pose and the final return to REST. The caller
    must have already parked the physical arm at REST before calling. Does not
    connect/disconnect the follower or otherwise move it to REST.

    ``joint_offsets_deg`` (from the session calibration) is applied feed-forward
    to every arm command and to the readback the checkpoint replanner starts
    from, so open-loop reaching lands near the planned model pose instead of the
    servos' miscalibrated zero. Left ``None`` the arm runs on its raw servo
    calibration. The image-space descent servo still corrects the residual.
    """
    model, data = episode.model, episode.data
    kinematics, actuator_id = episode.kinematics, episode.actuator_id
    robot_geom_ids, env_geom_ids = episode.robot_geom_ids, episode.env_geom_ids

    speed = speed if speed is not None else REAL_ARM_DEFAULT_SPEED
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if record_rest_to_rest and recording is None:
        raise ValueError("record_rest_to_rest requires recording")
    simulation_steps_per_tick = round(HARDWARE_SIMULATION_HZ / CONTROL_HZ)
    control_period = 1.0 / CONTROL_HZ
    if not math.isclose(model.opt.timestep * simulation_steps_per_tick, control_period):
        raise ValueError(
            f"MuJoCo timestep {model.opt.timestep:g}s cannot produce {CONTROL_HZ:g} Hz exactly"
        )

    failed_trajectory_path = (
        Path(failed_trajectory_dir) if failed_trajectory_dir is not None else None
    )
    if failed_trajectory_path is not None:
        failed_trajectory_path.mkdir(parents=True, exist_ok=True)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()
    # Per-tick log of (trajectory time, commanded real joints, motor readback).
    recorder = EpisodeRecorder()
    print(
        f"Playback speed: {speed:g}× nominal  "
        f"(run ≈ {episode.trajectory.duration / speed:.1f}s)"
    )

    show_wrist = show_wrist_cam or show_wrist_mixed
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    servo: WristServo | None = None
    wrist_renderer: mujoco.Renderer | None = None
    if wrist_camera is not None and wrist_cam_id >= 0:
        servo = open_wrist_servo(
            wrist_camera,
            wrist_intrinsics,
            annotate=show_wrist,
            on_frame=(
                None
                if recording is None
                else lambda bgr, t: recording.record_live_frame("wrist", bgr, t)
            ),
            on_overlay=None if recording is None else recording.record_visual_servo_overlay,
        )
        if servo is not None:
            wrist_renderer = open_sim_view(model, wrist_cam_id, servo, show_wrist_mixed)

    tick_recorder: TickRecorder | None = None
    episode_status = "incomplete"
    episode_metadata: dict[str, Any] | None = None
    pickup_metadata: dict[str, Any] | None = None
    try:
        wall_start = time.monotonic()
        if recording is not None:
            tick_recorder = TickRecorder.open(
                recording,
                wrist_reader=None if servo is None else servo.reader,
                overhead_capture=overhead_camera_cap,
                workspace_capture=workspace_camera_cap,
                wall_start=wall_start,
            )

        def record_tick(commanded: np.ndarray, actual: np.ndarray, **kwargs) -> None:
            if tick_recorder is not None:
                tick_recorder.record(commanded, actual, data.qpos.copy(), **kwargs)

        def log_tick(commanded: np.ndarray, actual: np.ndarray, **kwargs) -> None:
            """One control tick's outcome: into the tracking log, and into the dataset."""
            recorder.log(
                commanded=commanded,
                measured=actual,
                t=data.time,
                wall_t=time.monotonic() - wall_start,
            )
            record_tick(commanded, actual, **kwargs)

        def readback(commanded: np.ndarray) -> np.ndarray:
            return action_to_joints(follower.get_observation(), commanded)

        def to_real(joints: dict[str, float], gripper: float) -> np.ndarray:
            return clamp_and_warn(
                sim_frame_to_real(joints, gripper, joint_offsets_deg),
                clamp_low,
                clamp_high,
                clip_warned,
            )

        # In rest-to-rest mode the physical arm is already at REST. Start the
        # cameras there, then include the transition to the planned start pose.
        start_real = to_real(episode.start_joints, episode.start_gripper)
        if record_rest_to_rest:
            rest_real = to_real(REST_ARM_JOINTS, REST_GRIPPER)
            record_tick(rest_real, readback(rest_real))
        print("Ramping real arm to the trajectory start pose...")
        ramp_to_resting(
            follower,
            start_real,
            episode.start_joints,
            episode.start_gripper,
            actuator_id,
            model,
            data,
            viewer,
            clamp_low,
            clamp_high,
            clip_warned,
            on_tick=(
                (lambda commanded: record_tick(commanded, readback(commanded)))
                if record_rest_to_rest
                else None
            ),
        )

        plant = RealPlant(
            model,
            data,
            follower=follower,
            actuator_id=actuator_id,
            robot_geom_ids=robot_geom_ids,
            env_geom_ids=env_geom_ids,
            kinematics=kinematics,
            substeps_per_tick=simulation_steps_per_tick,
            clamp_low=clamp_low,
            clamp_high=clamp_high,
            wrist_camera_id=wrist_cam_id,
            servo=servo,
            joint_offsets_deg=joint_offsets_deg,
            speed=speed,
        )
        wrist = WristView(renderer=wrist_renderer, show=show_wrist)
        # A hardware run produces the same artifact a sim run does, minus the
        # ground truth no rig has: the true arm pose is the servo readback,
        # because that is all the arm can tell you about itself.
        artifact = TrajectoryWriter()
        run = Run(
            on_tick=lambda tick: log_tick(
                tick.action,
                tick.observation.state,
                servo_active=tick.is_descent,
                servo_source=(
                    np.array(
                        [
                            tick.tracked_source.x,
                            tick.tracked_source.y,
                            tick.tracked_source.z,
                            tick.tracked_source.yaw,
                        ]
                    )
                    if tick.is_descent
                    else None
                ),
            ),
            on_look=lambda plant, sighting, tracked, is_descent: show_preview(
                plant, wrist, is_descent
            ),
            sync=viewer.sync,
            should_stop=lambda: not viewer.is_running(),
        )

        current_traj = episode.trajectory
        tracked_source = episode.source
        tracked_grasp = current_traj.grasp
        commanded = start_real
        contacts: set[tuple[str, str]] = set()

        while current_traj.phases and viewer.is_running():
            phase = current_traj.phases[0]
            print(f"Executing phase: {phase.name}")
            played = play_phase(
                plant,
                phase,
                run=run,
                artifact=artifact,
                tracked_source=tracked_source,
                contacts=contacts,
                action=commanded,
                watch=show_wrist,
            )
            phase = played.phase
            tracked_source = played.tracked_source
            commanded = played.action
            contacts = played.contacts
            if played.outcome == "restart":
                episode_status = "restart"
                return "restart"
            if played.outcome == "stopped":
                break

            completed = phase.name
            remaining_phases = current_traj.phases[1:]

            if completed == "grasp":
                _report_grasp(model, data)
            if remaining_phases and fuses_into_next(completed, remaining_phases[0].name):
                current_traj = dataclasses.replace(current_traj, phases=remaining_phases)
                continue

            if completed == "descent":
                # The descent ended at the servo-corrected grasp pose. Rebuild just
                # the grasp and lift from there and run them from the locked command
                # — grasp + lift is contact-critical. The carry onward is replanned
                # from measured state after the lift, so it is left alone here.
                tracked_grasp = regrasp_after_descent(
                    phase, tracked_source, kinematics,
                    free_grasp=free_grasp, current=tracked_grasp,
                )
                current_traj = dataclasses.replace(
                    current_traj,
                    phases=(*_grasp_and_lift(kinematics, tracked_grasp, free_grasp),
                            *current_traj.phases[3:]),
                    grasp=tracked_grasp,
                )
                continue

            if completed in ("lift", "recovery_lift") and pickup_metadata is None:
                pickup_metadata = _pickup_report(
                    completed,
                    float(readback(commanded)[GRIPPER_INDEX]),
                    pickup_empty_gripper_position,
                    pickup_gripper_margin,
                )

            if not remaining_phases:
                episode_status = "success"
                break

            measured_joints, measured_gripper = plant.measured()
            print(f"Replanning remaining trajectory after {completed}...")
            candidate = replan_from_checkpoint(
                model,
                kinematics=kinematics,
                actuator_id=actuator_id,
                robot_geom_ids=robot_geom_ids,
                env_geom_ids=env_geom_ids,
                measured_joints=measured_joints,
                measured_gripper=measured_gripper,
                completed_phase_name=completed,
                source=tracked_source,
                target=episode.target,
                grasp=tracked_grasp,
                end_joints=episode.end_joints,
                end_gripper=episode.end_gripper,
                free_grasp=free_grasp,
                failed_trajectory_dir=failed_trajectory_path,
            )
            if candidate is None:
                episode_status = "restart"
                return "restart"
            current_traj = candidate

        if episode_status == "success" and record_rest_to_rest and viewer.is_running():
            print("Returning real arm to REST...")
            ramp_to_resting(
                follower,
                to_real(REST_ARM_JOINTS, REST_GRIPPER),
                REST_ARM_JOINTS,
                REST_GRIPPER,
                actuator_id,
                model,
                data,
                viewer,
                clamp_low,
                clamp_high,
                clip_warned,
                on_tick=lambda command: record_tick(command, readback(command)),
            )

    except KeyboardInterrupt:
        # Let the caller park the arm; clean up cameras on the way out.
        print("\nInterrupted during episode.")
        raise
    finally:
        # Stop optional audio before draining the asynchronous video writer. The
        # muxer pads/trims it to the frame-derived MP4 duration. Then stop every
        # producer — the servo thread and its camera first, then the recorder's —
        # so nothing is still arriving when the queue is drained and the episode
        # is committed.
        if recording is not None:
            recording.stop_audio_capture()
        if servo is not None:
            servo.close()
        if tick_recorder is not None:
            tick_recorder.close()
        if episode_status == "success":
            episode_metadata = dict(pickup_metadata or {})
            if success_metadata is not None:
                episode_metadata.update(success_metadata())
            episode_metadata = episode_metadata or None
        _report_tracking(recorder)
        if tick_recorder is not None:
            tick_recorder.finish(episode_status, episode_metadata, len(recorder))
        if show_wrist:
            import cv2

            cv2.destroyAllWindows()
        if wrist_renderer is not None:
            wrist_renderer.close()

    return "success" if episode_status == "success" else "restart"
