#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run the analytic pick-and-place on the physical SO-101 as a continuous loop.

Refuses to start unless the full rig is present: both the overhead and wrist
cameras open and both have calibrated intrinsics. It then solves the overhead
camera extrinsics from the workspace-frame AprilTags and refuses to start if the
tags are missing or implausible (swapped/rotated). During a long-enough cooldown
it re-solves and stops the run if the camera has drifted from that startup pose.

Homes the arm to a near-neutral pose, then repeats: look for the cube from the
current pose (re-homing to fresh near-neutral poses if it can't be seen), plan a
collision-free ``pick_and_carry`` episode from wherever the arm currently is, and
run it on the real arm via ``pick_and_place.executor``. A failed plan or a failed
checkpoint replan aborts the episode and re-homes. Every ``--rest-every`` episodes
the arm takes a torque-off cooldown at REST; the operator is expected to move the
target plate during that window, and the run stays halted with repeated audible
alerts until the plate has moved far enough. Press Ctrl-C to stop: the arm parks
to NEUTRAL, then REST, and releases torque.

The runner owns the follower, the viewer and the cameras and keeps them alive
across the whole loop; ``execute_episode`` runs a single pass against them. For
sim-only playback (no arm) use ``sim.py``.

Every run records straight into a LeRobotDataset (``datasets/<timestamp>/`` by
default): each successful episode is committed with one frame per control tick
(measured joints as state, commanded joints as action, plus the wrist and
overhead camera frames); aborted/restarted episodes are discarded. See
``execute_episode``'s docstring.
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place.episode_loop import episode_loop
from pick_and_place.episodes import (
    EpisodeSamplingError,
    PlacementError,
    _build_model,
    prepare_episode,
    sample_cube,
    sample_hunt_pose,
    sample_near_neutral,
)
from pick_and_place.executor import (
    CONTROL_HZ,
    HARDWARE_SIMULATION_HZ,
    RecordingSession,
    clamp_and_warn,
    execute_episode,
    follower_clamp_limits,
    ramp_to_resting,
)
from pick_and_place.follower import (
    action_to_joints,
    load_follower_joint_offsets,
    make_so101_follower,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.paper_detection import (
    PaperTarget,
    PaperTracker,
    detect_paper_target,
    project_to_pixel,
    set_paper_target_marker,
)
from pick_and_place.safety import EpisodeAborted, recover_on
from pick_and_place.trajectory import (
    GRIPPER_OPEN,
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)

# How long a single look attempt stares at the camera feed before giving up and
# moving to a fresh near-neutral pose to try again.
CUBE_LOOK_TIMEOUT = 2.0
# Plan-search budget per episode: how many source/target/end resamples to try
# before declaring the cube unreachable from the current pose and aborting.
EPISODE_MAX_ATTEMPTS = 40
# Cube-recovery relocation retries. The move is unrecorded but required so
# unattended collection can continue from a cube location that is usable for the
# next recorded pickup.
CUBE_RECOVERY_MAX_ATTEMPTS = 3
DEFAULT_ALERT_SOUND = "/System/Library/Sounds/Blow.aiff"


@dataclass
class OverheadDetectionDebug:
    """One camera-space detection snapshot for operator verification."""

    bgr: np.ndarray
    camera_matrix: np.ndarray
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    cube: CubePose | None = None
    target: PaperTarget | None = None


def _draw_overhead_debug_overlay(
    debug: OverheadDetectionDebug,
    *,
    show_distance: bool = False,
) -> np.ndarray:
    """Draw the accepted cube and target poses onto an overhead BGR frame."""
    import cv2

    image = debug.bgr.copy()
    height, width = image.shape[:2]
    scale = max(width, height) / 1080.0
    line = max(2, int(round(2 * scale)))
    font_scale = 0.65 * scale
    font_thickness = max(1, int(round(2 * scale)))

    def label(text: str, px: np.ndarray, color: tuple[int, int, int]) -> None:
        point = tuple(np.round(px).astype(int))
        cv2.putText(
            image,
            text,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    target_center_px = None
    cube_center_px = None
    if debug.target is not None:
        corners_px = project_to_pixel(
            debug.target.corners_world,
            debug.camera_matrix,
            debug.camera_position,
            debug.camera_rotation,
        )
        center_px = project_to_pixel(
            debug.target.center_world,
            debug.camera_matrix,
            debug.camera_position,
            debug.camera_rotation,
        )[0]
        target_center_px = center_px
        corners = np.round(corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [corners], True, (255, 255, 0), line, cv2.LINE_AA)
        cv2.line(
            image,
            tuple(corners[0, 0]),
            tuple(corners[1, 0]),
            (0, 255, 255),
            line + 1,
            cv2.LINE_AA,
        )
        cv2.circle(image, tuple(np.round(center_px).astype(int)), line + 3, (0, 0, 255), -1)
        label("plate", center_px, (255, 255, 0))
        label("target", center_px + np.array((0.0, 24.0 * scale)), (0, 0, 255))

    if debug.cube is not None:
        half = CUBE_HALF_SIZE
        yaw = float(debug.cube.yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        corners_world = np.zeros((4, 3))
        corners_world[:, :2] = np.array([debug.cube.x, debug.cube.y]) + local @ rot.T
        corners_world[:, 2] = debug.cube.z
        center_world = np.array([[debug.cube.x, debug.cube.y, debug.cube.z]])

        corners_px = project_to_pixel(
            corners_world,
            debug.camera_matrix,
            debug.camera_position,
            debug.camera_rotation,
        )
        center_px = project_to_pixel(
            center_world,
            debug.camera_matrix,
            debug.camera_position,
            debug.camera_rotation,
        )[0]
        cube_center_px = center_px
        poly = np.round(corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [poly], True, (0, 165, 255), line, cv2.LINE_AA)
        cv2.circle(image, tuple(np.round(center_px).astype(int)), line + 3, (0, 0, 255), -1)
        label("cube", center_px, (0, 165, 255))

    if (
        show_distance
        and debug.target is not None
        and debug.cube is not None
        and target_center_px is not None
        and cube_center_px is not None
    ):
        start = tuple(np.round(cube_center_px).astype(int))
        end = tuple(np.round(target_center_px).astype(int))
        cv2.line(image, start, end, (255, 0, 255), line + 1, cv2.LINE_AA)
        midpoint = (cube_center_px + target_center_px) / 2.0
        distance_m = float(
            np.linalg.norm(
                np.array([debug.cube.x, debug.cube.y])
                - np.asarray(debug.target.center_world[:2], dtype=float)
            )
        )
        label(f"{distance_m * 1000.0:.1f} mm", midpoint, (255, 0, 255))

    return image


def _write_overhead_debug_image(
    path: Path,
    debug: OverheadDetectionDebug,
    *,
    show_distance: bool = False,
) -> None:
    """Write an overhead verification overlay without touching OpenCV HighGUI."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = _draw_overhead_debug_overlay(debug, show_distance=show_distance)
    if not cv2.imwrite(str(path), image):
        print(f"Warning: could not write overhead debug image: {path}")


def _empty_overhead_debug() -> OverheadDetectionDebug:
    """Create a mutable preflight debug snapshot, filled by detections."""
    return OverheadDetectionDebug(
        bgr=np.empty((0, 0, 3), dtype=np.uint8),
        camera_matrix=np.eye(3),
        camera_position=np.zeros(3),
        camera_rotation=np.eye(3),
    )


class OperatorNotifier:
    """Best-effort audible notices for long unattended hardware runs."""

    def __init__(self, *, enabled: bool, sound_path: str | None) -> None:
        self.enabled = enabled
        self.sound_path = sound_path if sound_path and Path(sound_path).exists() else None
        self._afplay = shutil.which("afplay") if sys.platform == "darwin" else None
        self._say = shutil.which("say") if sys.platform == "darwin" else None

    def alert(self, message: str, *, repeat_sound: int = 1) -> None:
        """Print ``message`` and announce it audibly when supported."""
        print(f"Operator alert: {message}")
        if not self.enabled:
            return
        if self._afplay and self.sound_path:
            for _ in range(max(1, repeat_sound)):
                subprocess.run([self._afplay, self.sound_path], check=False)
        else:
            print("\a", end="", flush=True)
        if self._say:
            subprocess.run([self._say, message], check=False)


class MockViewer:
    """Stand-in viewer for headless runs and for parking after the real viewer
    has closed. Exposes the ``is_running``/``sync`` surface the ramps rely on."""

    def __init__(self) -> None:
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def sync(self) -> None:
        pass

    def close(self) -> None:
        self._running = False

    def __enter__(self) -> "MockViewer":
        return self

    def __exit__(self, *args) -> None:
        pass


def track_cube(
    cap,
    camera_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    timeout: float,
    *,
    free_grasp: bool = False,
    return_out_of_zone: bool = False,
    debug: OverheadDetectionDebug | None = None,
) -> CubePose | None:
    """Look for the cube on ``cap`` for up to ``timeout`` seconds.

    Returns the cube pose if it is detected inside the allowed workspace, or
    when ``return_out_of_zone`` is set. Otherwise returns ``None`` if nothing
    usable is seen before the timeout (not visible, or outside the workspace)."""
    import cv2
    from scipy.spatial.transform import Rotation

    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.cube_detection import (
        cube_pose_to_world,
        estimate_cube_pose,
        make_cube_detector,
    )
    from pick_and_place.workspace_overlays import is_cube_pickup_allowed

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()

    intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
    if intrinsics.exists():
        camera_matrix, undistort_map = load_intrinsics(intrinsics, 1920, 1080, cv2)
    else:
        focal = (1080 / 2.0) / np.tan(np.radians(model.cam_fovy[camera_id]) / 2.0)
        camera_matrix = np.array(
            [[focal, 0, 1920 / 2.0], [0, focal, 1080 / 2.0], [0, 0, 1]], dtype=float
        )
        undistort_map = None

    detector = make_cube_detector()

    # Flush a few stale frames so we read what the arm sees now, not buffered.
    for _ in range(5):
        cap.read()

    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)

        estimate = estimate_cube_pose(rgb, detector, camera_matrix)
        if estimate is None:
            continue

        rotation, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
        allowed = is_cube_pickup_allowed
        if not allowed(float(position[0]), float(position[1])):
            print(
                f"Cube seen at ({position[0]:.3f}, {position[1]:.3f}) but outside the "
                "allowed pick-up zone."
            )
            if not return_out_of_zone:
                continue

        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
        print(f"Tracked cube: pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        cube = CubePose(
            x=float(position[0]),
            y=float(position[1]),
            z=CUBE_HALF_SIZE,
            roll=float(roll) if free_grasp else 0.0,
            pitch=float(pitch) if free_grasp else 0.0,
            yaw=float(yaw),
        )
        if debug is not None:
            debug.bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            debug.camera_matrix = camera_matrix.copy()
            debug.camera_position = cam_pos.copy()
            debug.camera_rotation = cam_rot.copy()
            debug.cube = cube
        return cube

    return None


def track_drop_zone_square(
    cap,
    camera_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tracker: PaperTracker,
    target_color: str,
    timeout: float = CUBE_LOOK_TIMEOUT,
    debug: OverheadDetectionDebug | None = None,
) -> CubePose | None:
    """Look for a black/white drop-zone square and return its center as a target."""
    import cv2

    from pick_and_place.camera_compare import load_intrinsics
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.workspace_overlays import (
        is_cube_drop_allowed,
        workspace_interior_corners_world,
    )

    workspace_corners = workspace_interior_corners_world()

    # This function is called once per episode. Do not let the tracker's cached
    # estimate satisfy a new episode when the square is no longer visible.
    tracker.reset()

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()

    intrinsics = LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
    if intrinsics.exists():
        camera_matrix, undistort_map = load_intrinsics(intrinsics, 1920, 1080, cv2)
    else:
        focal = (1080 / 2.0) / np.tan(np.radians(model.cam_fovy[camera_id]) / 2.0)
        camera_matrix = np.array(
            [[focal, 0, 1920 / 2.0], [0, focal, 1080 / 2.0], [0, 0, 1]], dtype=float
        )
        undistort_map = None

    for _ in range(5):
        cap.read()

    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)

        raw_target = detect_paper_target(
            rgb,
            camera_matrix,
            cam_pos,
            cam_rot,
            target_color=target_color,
            workspace_corners_world=workspace_corners,
        )
        target = tracker.update(raw_target)
        if target is None:
            continue

        usable = is_cube_drop_allowed(*target.xy)
        set_paper_target_marker(model, data, target, usable=usable)
        if not usable:
            print(
                f"Drop zone seen at ({target.xy[0]:.3f}, {target.xy[1]:.3f}) "
                "but outside the allowed drop zone."
            )
            continue

        print(f"Tracked drop zone: pos=({target.xy[0]:.3f}, {target.xy[1]:.3f})")
        if debug is not None:
            debug.bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            debug.camera_matrix = camera_matrix.copy()
            debug.camera_position = cam_pos.copy()
            debug.camera_rotation = cam_rot.copy()
            debug.target = target
        return CubePose(x=target.xy[0], y=target.xy[1], z=CUBE_HALF_SIZE)

    return None


def final_placement_metadata(
    cube: CubePose | None,
    target: CubePose,
    *,
    check_error: str = "",
) -> dict[str, object]:
    """Episode metadata for the physical cube's final overhead-camera pose."""
    target_xyz = (float(target.x), float(target.y), float(CUBE_HALF_SIZE))
    if cube is None:
        print("placement error: cube not detected after release")
        nan = float("nan")
        return {
            "placement_detected": False,
            "placement_check_error": check_error,
            "placement_cube_x": nan,
            "placement_cube_y": nan,
            "placement_cube_z": nan,
            "placement_target_x": target_xyz[0],
            "placement_target_y": target_xyz[1],
            "placement_target_z": target_xyz[2],
            "placement_dx": nan,
            "placement_dy": nan,
            "placement_dz": nan,
            "placement_xy": nan,
        }

    cube_xyz = (float(cube.x), float(cube.y), float(cube.z))
    error = PlacementError(
        cube_xyz=cube_xyz,
        target_xyz=target_xyz,
        dx=cube_xyz[0] - target_xyz[0],
        dy=cube_xyz[1] - target_xyz[1],
        dz=cube_xyz[2] - target_xyz[2],
        xy=float(np.linalg.norm(np.asarray(cube_xyz[:2]) - np.asarray(target_xyz[:2]))),
    )
    print(error.summary())
    return {
        "placement_detected": True,
        "placement_check_error": check_error,
        "placement_cube_x": error.cube_xyz[0],
        "placement_cube_y": error.cube_xyz[1],
        "placement_cube_z": error.cube_xyz[2],
        "placement_target_x": error.target_xyz[0],
        "placement_target_y": error.target_xyz[1],
        "placement_target_z": error.target_xyz[2],
        "placement_dx": error.dx,
        "placement_dy": error.dy,
        "placement_dz": error.dz,
        "placement_xy": error.xy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="number of episodes to run; 0 means loop until Ctrl-C (default: 0)",
    )
    parser.add_argument(
        "--max-hunt-tries",
        type=int,
        default=5,
        help="near-neutral poses to try while looking for the cube before giving up (default: 5)",
    )
    parser.add_argument(
        "--rest-every",
        type=int,
        default=10,
        help="episodes between cooldown rests; 0 to disable (default: 10)",
    )
    parser.add_argument(
        "--rest-duration",
        type=float,
        default=30.0,
        help="cooldown rest duration in seconds, torque off at REST (default: 30.0)",
    )
    parser.add_argument(
        "--operator-alerts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="play a signal sound and speak operator alerts on macOS (default: on)",
    )
    parser.add_argument(
        "--alert-sound",
        default=DEFAULT_ALERT_SOUND,
        help=f"sound file to play before spoken alerts (default: {DEFAULT_ALERT_SOUND})",
    )
    parser.add_argument(
        "--target-change-min-distance",
        type=float,
        default=0.03,
        help="minimum cooldown target-plate center movement before the run resumes, in metres "
        "(default: 0.03)",
    )
    parser.add_argument(
        "--target-change-alert-min-seconds",
        type=float,
        default=10.0,
        help="initial backoff between repeated target-plate alerts (default: 10)",
    )
    parser.add_argument(
        "--target-change-alert-max-seconds",
        type=float,
        default=120.0,
        help="maximum backoff between repeated target-plate alerts (default: 120)",
    )
    parser.add_argument(
        "--source",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="pin the source cube (x, y) on the floor; omit to track it with the camera",
    )
    parser.add_argument(
        "--drop-zone-color",
        dest="drop_zone_color",
        choices=("black", "white"),
        default="black",
        help="color of the drop-zone square to detect (default: black)",
    )
    parser.add_argument("--follower-port", required=True, help="serial port of the SO-101 follower")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id (default: folly)")
    parser.add_argument(
        "--offsets-path",
        default=None,
        help="JSON of per-joint sim→real degree offsets (default: zero offsets)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier of nominal pace (1.0 = nominal; default: 1.0)",
    )
    parser.add_argument(
        "--pickup-empty-gripper-position",
        type=float,
        default=2.3,
        help="physical gripper readback expected after an empty close (default: 2.3)",
    )
    parser.add_argument(
        "--pickup-gripper-margin",
        type=float,
        default=5.0,
        help="minimum readback above empty-close to log pickup_detected=true (default: 5.0)",
    )
    parser.add_argument("--camera", default="0", help="OpenCV index/path of the overhead camera (default: 0)")
    parser.add_argument("--camera-name", default="overhead_camera", help="camera name in the model")
    parser.add_argument(
        "--recalibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solve the overhead camera extrinsics live from the workspace-frame AprilTags at "
        "startup and refuse to start if they are missing or implausible (default: on; "
        "--no-recalibrate uses the saved sidecar extrinsics instead)",
    )
    parser.add_argument(
        "--recalibrate-samples",
        type=int,
        default=10,
        help="overhead frames to average per extrinsics solve (default: 10)",
    )
    parser.add_argument(
        "--recalibrate-max-seconds",
        type=float,
        default=15.0,
        help="time budget to gather the solve frames before giving up (default: 15)",
    )
    parser.add_argument(
        "--overhead-intrinsics",
        type=Path,
        default=None,
        help="overhead camera intrinsics JSON for the solve (default: local sidecar)",
    )
    parser.add_argument(
        "--recalibrate-check-min-cooldown",
        type=float,
        default=15.0,
        help="during a cooldown at least this long (s), re-solve the overhead extrinsics at "
        "near-neutral and stop the run if the camera has drifted past the threshold; 0 disables "
        "the cooldown drift check (default: 15)",
    )
    parser.add_argument(
        "--recalibrate-drift-mm",
        type=float,
        default=10.0,
        help="translation drift from the startup solve that stops the run (default: 10 mm)",
    )
    parser.add_argument(
        "--recalibrate-drift-deg",
        type=float,
        default=2.0,
        help="rotation drift from the startup solve that stops the run (default: 2 deg)",
    )
    parser.add_argument("--wrist-camera", default="1", help="OpenCV index/path of the wrist camera (default: 1)")
    parser.add_argument("--wrist-intrinsics", default=None, help="path to wrist camera intrinsics JSON")
    parser.add_argument("--show-wrist-cam", action="store_true", help="show the live wrist camera feed")
    parser.add_argument("--show-wrist-mixed", action="store_true", help="overlay the sim render on the wrist feed")
    parser.add_argument(
        "--save-overhead-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save preflight/final overhead verification images into the run directory "
        "(default: on)",
    )
    parser.add_argument(
        "--overhead-debug-dir",
        type=Path,
        default=None,
        help="directory for overhead verification images "
        "(default: <dataset-root>/overhead_debug)",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show the 3D MuJoCo viewer (default: off, headless)",
    )
    parser.add_argument(
        "--preflight-debug",
        action="store_true",
        help="print detailed collision diagnostics for rejected trajectory candidates",
    )
    parser.add_argument(
        "--preflight-debug-limit",
        type=int,
        default=12,
        help="maximum detailed contact rows to print per rejected candidate",
    )
    parser.add_argument(
        "--save-failed-trajectories",
        type=Path,
        default=None,
        help="directory for replayable .npz rollouts of rejected preflight candidates",
    )
    parser.add_argument(
        "--failed-trajectory-limit",
        type=int,
        default=8,
        help="maximum rejected candidates to save",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="output dir for the LeRobotDataset (default: datasets/<timestamp>)",
    )
    parser.add_argument(
        "--repo-id",
        default="local/pick-and-place-so101",
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
        help="LeRobot video codec (default: auto = best available HW encoder, "
        "e.g. h264_videotoolbox on macOS)",
    )
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode video in real time during capture (near-instant save_episode, "
        "no PNG scratch files); --no-streaming-encoding falls back to PNG-then-encode",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="background image-writer threads LeRobot uses for PNG-then-encode mode",
    )
    args = parser.parse_args()
    if args.target_change_min_distance < 0:
        parser.error("--target-change-min-distance must be non-negative")
    if args.target_change_alert_min_seconds <= 0:
        parser.error("--target-change-alert-min-seconds must be positive")
    if args.target_change_alert_max_seconds < args.target_change_alert_min_seconds:
        parser.error("--target-change-alert-max-seconds must be at least the minimum")
    notifier = OperatorNotifier(
        enabled=args.operator_alerts,
        sound_path=args.alert_sound,
    )

    import cv2

    from pick_and_place.cam_align_solve import (
        ExtrinsicsSolveError,
        apply_solve_result,
        check_solve_plausible,
        parse_index_or_path,
        pose_delta_mm_deg,
        solve_overhead_extrinsics,
    )
    from pick_and_place.camera_extrinsics import (
        apply_camera_extrinsics_to_model,
        load_local_camera_extrinsics,
    )
    from pick_and_place.workspace_overlays import PAN_AXIS

    # One persistent scene for the whole loop: the cube is a freejoint that
    # prepare_episode repositions per episode, so a single viewer can stay bound.
    print("Building scene...")
    dummy_source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(
        dummy_source,
        include_environment=True,
        paper_target_marker=True,
    )
    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)

    kinematics = derive_kinematics(model)
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    offsets = load_follower_joint_offsets(args.offsets_path)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()
    rng = np.random.default_rng()
    # Set by the startup overhead solve; the cooldown drift check compares against it.
    startup_extrinsics: tuple[np.ndarray, np.ndarray] | None = None
    # Cooldowns require the operator to move away from the most recent
    # successfully completed episode's target before the next episode is planned.
    cooldown_reference_target: CubePose | None = None

    # Refuse to start unless the full rig is present: both cameras open and both
    # have calibrated intrinsics. The overhead extrinsics are solved from the
    # workspace-frame tags once the viewer is up; the wrist camera is opened per
    # episode by the executor, so it is only probed here.
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR

    def require_intrinsics(camera_name: str, override) -> None:
        path = Path(override) if override is not None else LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
        if not path.exists():
            raise SystemExit(f"Missing {camera_name} intrinsics at {path}. Calibrate the camera first.")

    require_intrinsics(args.camera_name, args.overhead_intrinsics)
    require_intrinsics("wrist_camera", args.wrist_intrinsics)

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    print("Opening overhead camera...")
    overhead_cap = cv2.VideoCapture(parse_index_or_path(args.camera), backend)
    overhead_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    overhead_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not overhead_cap.isOpened():
        overhead_cap.release()
        raise SystemExit(f"Could not open the overhead camera {args.camera!r}.")

    print("Checking wrist camera...")
    wrist_probe = cv2.VideoCapture(parse_index_or_path(args.wrist_camera), backend)
    wrist_open = wrist_probe.isOpened()
    wrist_probe.release()
    if not wrist_open:
        overhead_cap.release()
        raise SystemExit(f"Could not open the wrist camera {args.wrist_camera!r}.")

    print("Connecting to follower...")
    # Keep torque on a plain disconnect (crash / mid-loop exit) so the arm holds
    # rather than going limp; torque is only released deliberately at REST.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = (
        args.dataset_root
        if args.dataset_root is not None
        else Path(__file__).resolve().parents[2] / "datasets" / timestamp
    )
    overhead_debug_dir = (
        args.overhead_debug_dir
        if args.overhead_debug_dir is not None
        else dataset_root / "overhead_debug"
    )
    recording = RecordingSession(
        repo_id=args.repo_id,
        root=dataset_root,
        task=args.task,
        fps=CONTROL_HZ,
        vcodec=args.vcodec,
        streaming_encoding=args.streaming_encoding,
        image_writer_threads=args.image_writer_threads,
    )
    print(f"Recording into LeRobotDataset at: {dataset_root}")

    drop_zone_tracker = PaperTracker()

    def read_current_sim_pose() -> tuple[dict[str, float], float]:
        """Read the real arm and convert to the sim joint frame."""
        actual = action_to_joints(follower.get_observation(), clamp_low)
        return real_frame_to_sim(actual, offsets)

    def move_to(arm_joints: dict[str, float], gripper: float, viewer) -> None:
        """Smoothly ramp the real arm and the sim onto ``arm_joints``/``gripper``."""
        target_real = clamp_and_warn(
            sim_frame_to_real(arm_joints, gripper, offsets), clamp_low, clamp_high, clip_warned
        )
        ramp_to_resting(
            follower, target_real, arm_joints, gripper, actuator_id, model, data, viewer
        )

    def open_gripper_in_place(viewer) -> None:
        """Open the gripper while holding the current arm pose (drop any cube)."""
        arm, _ = read_current_sim_pose()
        move_to(arm, GRIPPER_OPEN, viewer)

    def abort_to_near_neutral(viewer) -> None:
        """Recover from a failed episode: drop the cube and re-home near neutral."""
        print("Aborting episode: opening gripper and re-homing to near-neutral...")
        open_gripper_in_place(viewer)
        arm, grip = sample_near_neutral(rng)
        move_to(arm, grip, viewer)

    def check_overhead_drift() -> None:
        """Re-solve the overhead extrinsics from the current (near-neutral) pose and
        stop the run if the camera has drifted from the startup calibration. Skips
        quietly if the tags are occluded."""
        if not args.recalibrate or startup_extrinsics is None:
            return
        print("Cooldown drift check: re-solving overhead extrinsics...")
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
        saved_pos = model.cam_pos[cam_id].copy()
        saved_quat = model.cam_quat[cam_id].copy()
        check = solve_overhead_extrinsics(
            model,
            data,
            overhead_cap,
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

    def target_distance(a: CubePose, b: CubePose) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def wait_for_target_plate_change(viewer) -> None:
        """Pause after cooldown until the operator has moved the target plate."""
        reference = cooldown_reference_target
        if reference is None or args.target_change_min_distance == 0:
            return

        def look_from_current_pose() -> CubePose | None:
            if not viewer.is_running():
                return None
            print("Checking target plate movement from the current near-neutral pose...")
            return track_drop_zone_square(
                overhead_cap,
                args.camera_name,
                model,
                data,
                drop_zone_tracker,
                args.drop_zone_color,
            )

        threshold = args.target_change_min_distance
        backoff = args.target_change_alert_min_seconds
        notifier.alert(
            "Please move the target plate to a substantially different position.",
            repeat_sound=2,
        )
        while viewer.is_running():
            target = look_from_current_pose()
            if not viewer.is_running():
                return
            if target is None:
                notifier.alert(
                    "Target plate is not visible. Move it into view before the run can continue."
                )
            else:
                moved = target_distance(reference, target)
                if moved >= threshold:
                    print(
                        f"Target plate moved {moved * 100.0:.1f}cm "
                        f"(required {threshold * 100.0:.1f}cm). Resuming."
                    )
                    return
                notifier.alert(
                    "Target plate has not moved enough. Move it to a new position before "
                    "the run can continue."
                )
                print(
                    f"Plate movement: {moved * 100.0:.1f}cm "
                    f"(required {threshold * 100.0:.1f}cm); "
                    f"from ({reference.x:.3f}, {reference.y:.3f}) "
                    f"to ({target.x:.3f}, {target.y:.3f})."
                )
            time.sleep(backoff)
            backoff = min(backoff * 2.0, args.target_change_alert_max_seconds)

    def cooldown(viewer) -> None:
        """Park at REST with torque off for the cooldown, then re-home near neutral.
        A long-enough cooldown doubles as an overhead-camera drift check. The
        cooldown also gives the operator time to move the target plate; the run
        remains paused until that movement is confirmed by the overhead camera."""
        print(f"Cooldown: resting with torque off for {args.rest_duration:.0f}s...")
        if cooldown_reference_target is not None and args.target_change_min_distance > 0:
            notifier.alert(
                "Cooldown started. Move the target plate before the next episode.",
                repeat_sound=2,
            )
        move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)
        follower.bus.disable_torque()
        time.sleep(args.rest_duration)
        follower.bus.enable_torque()
        arm, grip = sample_near_neutral(rng)
        move_to(arm, grip, viewer)
        if (
            args.recalibrate
            and args.recalibrate_check_min_cooldown > 0
            and args.rest_duration >= args.recalibrate_check_min_cooldown
        ):
            check_overhead_drift()
        wait_for_target_plate_change(viewer)

    def park_from_interrupt() -> None:
        """User ended the loop: park to NEUTRAL, then REST. The real viewer has
        already been torn down by the `with`, so park against a mock one. Make
        sure torque is on first — a Ctrl-C during the cooldown sleep leaves it
        off, and parking commands would be ignored by a limp arm."""
        nonlocal ended_at_rest
        print("\nCtrl-C: parking to NEUTRAL then REST...")
        try:
            follower.bus.enable_torque()
        except Exception as exc:  # noqa: BLE001 - best-effort re-enable before parking
            print(f"Warning: could not enable torque before parking: {exc}")
        park = MockViewer()
        move_to(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER, park)
        move_to(REST_ARM_JOINTS, REST_GRIPPER, park)
        ended_at_rest = True

    from pick_and_place.workspace_overlays import is_cube_pickup_allowed

    def hunt_for_cube(
        viewer,
        *,
        free_grasp: bool = False,
        return_out_of_zone: bool = False,
        debug: OverheadDetectionDebug | None = None,
    ) -> CubePose | None:
        """Look for the cube from the current pose, re-homing near neutral up to
        ``--max-hunt-tries`` times. Returns the cube pose or ``None`` if not found."""
        if args.source is not None and not free_grasp:
            source = CubePose(x=args.source[0], y=args.source[1], z=CUBE_HALF_SIZE)
            if debug is not None:
                debug.cube = source
            return source
        for attempt in range(args.max_hunt_tries):
            if not viewer.is_running():
                return None
            if attempt > 0:
                arm, grip = sample_hunt_pose(rng)
                print(f"Look {attempt + 1}/{args.max_hunt_tries}: panning to a new search pose...")
                move_to(arm, grip, viewer)
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"Look {attempt + 1}/{args.max_hunt_tries}: searching from current pose...")
            source = track_cube(
                overhead_cap,
                args.camera_name,
                model,
                data,
                CUBE_LOOK_TIMEOUT,
                free_grasp=free_grasp,
                return_out_of_zone=return_out_of_zone,
                debug=debug,
            )
            if source is not None:
                return source
        return None

    def hunt_for_drop_zone(
        viewer,
        *,
        debug: OverheadDetectionDebug | None = None,
    ) -> CubePose | None:
        """Look for the drop-zone square on the overhead camera.

        The arm can sit between the overhead camera and the square, so we re-home
        to fresh near-neutral poses up to ``--max-hunt-tries`` times to clear the
        view, exactly like ``hunt_for_cube``. Returns the target or ``None``."""
        tries = args.max_hunt_tries
        for attempt in range(tries):
            if not viewer.is_running():
                return None
            if attempt > 0:
                arm, grip = sample_hunt_pose(rng)
                print(
                    f"Drop-zone look {attempt + 1}/{tries}: panning to a new search pose..."
                )
                move_to(arm, grip, viewer)
                time.sleep(0.5)  # let the camera settle
            else:
                print(f"Drop-zone look {attempt + 1}/{tries}: searching from current pose...")
            target = track_drop_zone_square(
                overhead_cap,
                args.camera_name,
                model,
                data,
                drop_zone_tracker,
                args.drop_zone_color,
                debug=debug,
            )
            if target is not None:
                return target
        return None

    def recover_cube(viewer) -> bool:
        """Move the cube to a fresh random source pose before the next episode."""
        for recovery_attempt in range(1, CUBE_RECOVERY_MAX_ATTEMPTS + 1):
            print(
                f"\n--- Cube recovery {recovery_attempt}/{CUBE_RECOVERY_MAX_ATTEMPTS} "
                "(not recorded) ---"
            )
            recovery_source = hunt_for_cube(viewer, free_grasp=True)
            if not viewer.is_running():
                return False
            if recovery_source is None:
                print("Cube recovery could not locate the cube.")
                continue

            current_joints, current_gripper = read_current_sim_pose()
            with recover_on(
                EpisodeSamplingError,
                EpisodeAborted,
                recover=lambda: abort_to_near_neutral(viewer),
            ):
                try:
                    recovery = prepare_episode(
                        rng,
                        recovery_source,
                        start_joints=current_joints,
                        start_gripper=current_gripper,
                        model=model,
                        data=data,
                        max_attempts=EPISODE_MAX_ATTEMPTS,
                        verbose=True,
                        include_environment=True,
                        preflight_debug=args.preflight_debug,
                        preflight_debug_limit=args.preflight_debug_limit,
                        failed_trajectory_dir=args.save_failed_trajectories,
                        failed_trajectory_limit=args.failed_trajectory_limit,
                        free_grasp=True,
                        target_sampler=sample_cube,
                    )
                except EpisodeSamplingError:
                    print("Cube recovery: no feasible relocation plan.")
                    raise

                print(
                    f"Recovering cube to pickup zone "
                    f"({recovery.target.x:.3f}, {recovery.target.y:.3f})."
                )
                status = execute_episode(
                    recovery,
                    follower=follower,
                    viewer=viewer,
                    offsets_path=args.offsets_path,
                    speed=args.speed,
                    wrist_camera=args.wrist_camera,
                    wrist_intrinsics=args.wrist_intrinsics,
                    show_wrist_cam=args.show_wrist_cam,
                    show_wrist_mixed=args.show_wrist_mixed,
                    failed_trajectory_dir=args.save_failed_trajectories,
                    free_grasp=True,
                    pickup_empty_gripper_position=args.pickup_empty_gripper_position,
                    pickup_gripper_margin=args.pickup_gripper_margin,
                )
                if status == "restart":
                    raise EpisodeAborted
                return True

        return False

    disable_viewer = (not args.viewer) or (
        (args.show_wrist_cam or args.show_wrist_mixed) and sys.platform == "darwin"
    )
    viewer_ctx = MockViewer() if disable_viewer else mujoco.viewer.launch_passive(model, data)

    # Run-level session: home near-neutral on entry; whatever ends the run
    # (episode budget met, cube lost, viewer closed, or Ctrl-C) flows to REST
    # before the hardware is released in the `finally` below.
    ended_at_rest = False
    try:
        with recover_on(KeyboardInterrupt, recover=park_from_interrupt):
            with viewer_ctx as viewer:
                if args.recalibrate:
                    print("Solving overhead camera extrinsics from the workspace-frame tags...")
                    result = solve_overhead_extrinsics(
                        model,
                        data,
                        overhead_cap,
                        camera_name=args.camera_name,
                        intrinsics_path=args.overhead_intrinsics,
                        samples=args.recalibrate_samples,
                        max_seconds=args.recalibrate_max_seconds,
                        cv2_module=cv2,
                    )
                    if result is None:
                        raise SystemExit(
                            "Overhead calibration failed: never saw all four workspace-frame "
                            "tags in one frame. Clear the camera view and check the tags."
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
                    print(
                        f"Overhead extrinsics solved: {result.reprojection_error_px:.2f}px, "
                        f"{result.nominal_delta.translation_m * 1000.0:.1f}mm / "
                        f"{result.nominal_delta.rotation_deg:.2f}deg from nominal."
                    )

                print("Homing to near-neutral...")
                arm, grip = sample_near_neutral(rng)
                move_to(arm, grip, viewer)

                for ep in episode_loop(
                    target=args.episodes,
                    rest_every=args.rest_every,
                    cooldown=lambda: cooldown(viewer),
                    should_continue=viewer.is_running,
                ):
                    overhead_debug = _empty_overhead_debug()
                    episode_target = hunt_for_drop_zone(viewer, debug=overhead_debug)
                    if not viewer.is_running():
                        break
                    if episode_target is None:
                        notifier.alert(
                            f"Drop zone square not found after {args.max_hunt_tries} looks. "
                            "The run is stopping."
                        )
                        break

                    source = hunt_for_cube(
                        viewer,
                        return_out_of_zone=True,
                        debug=overhead_debug,
                    )
                    if not viewer.is_running():
                        break
                    if source is None:
                        notifier.alert(
                            f"Cube not found after {args.max_hunt_tries} looks. "
                            "The run is stopping."
                        )
                        break
                    if not is_cube_pickup_allowed(source.x, source.y):
                        notifier.alert(
                            "Cube is outside the pickup zone. Running an unrecorded recovery."
                        )
                        if not recover_cube(viewer):
                            notifier.alert("Cube recovery failed after retries. The run is stopping.")
                            break
                        source = hunt_for_cube(viewer, debug=overhead_debug)
                        if not viewer.is_running():
                            break
                        if source is None:
                            notifier.alert(
                                f"Cube not found after recovery and {args.max_hunt_tries} looks. "
                                "The run is stopping."
                            )
                            break

                    current_joints, current_gripper = read_current_sim_pose()
                    # Per-episode guard: an infeasible plan or a failed checkpoint
                    # replan opens the gripper and re-homes, then this attempt is
                    # abandoned without calling `ep.complete()`.
                    with recover_on(
                        EpisodeSamplingError, EpisodeAborted,
                        recover=lambda: abort_to_near_neutral(viewer),
                    ):
                        try:
                            episode = prepare_episode(
                                rng,
                                source,
                                episode_target,
                                start_joints=current_joints,
                                start_gripper=current_gripper,
                                model=model,
                                data=data,
                                max_attempts=EPISODE_MAX_ATTEMPTS,
                                verbose=True,
                                include_environment=True,
                                preflight_debug=args.preflight_debug,
                                preflight_debug_limit=args.preflight_debug_limit,
                                failed_trajectory_dir=args.save_failed_trajectories,
                                failed_trajectory_limit=args.failed_trajectory_limit,
                            )
                        except EpisodeSamplingError:
                            notifier.alert(
                                "No feasible plan from the current pose. Re-homing the arm."
                            )
                            print("No feasible plan from the current pose.")
                            raise

                        print(f"\n--- Episode {ep.index}"
                              f"{f'/{args.episodes}' if args.episodes else ''} ---")
                        initial_overhead_debug = overhead_debug
                        preflight_debug_written = False

                        def check_final_placement() -> dict[str, object]:
                            nonlocal preflight_debug_written

                            if (
                                args.save_overhead_debug
                                and initial_overhead_debug.bgr.size
                                and not preflight_debug_written
                            ):
                                path = (
                                    overhead_debug_dir
                                    / f"episode_{ep.index:05d}_preflight.jpg"
                                )
                                _write_overhead_debug_image(path, initial_overhead_debug)
                                print(f"Saved overhead preflight debug image: {path}")
                                preflight_debug_written = True

                            print("Checking final cube placement from the overhead camera...")
                            final_debug = OverheadDetectionDebug(
                                bgr=initial_overhead_debug.bgr,
                                camera_matrix=initial_overhead_debug.camera_matrix,
                                camera_position=initial_overhead_debug.camera_position,
                                camera_rotation=initial_overhead_debug.camera_rotation,
                                target=initial_overhead_debug.target,
                            )
                            try:
                                final_cube = track_cube(
                                    overhead_cap,
                                    args.camera_name,
                                    model,
                                    data,
                                    CUBE_LOOK_TIMEOUT,
                                    return_out_of_zone=True,
                                    debug=final_debug,
                                )
                            except Exception as exc:
                                print(f"placement error: final cube check failed: {exc}")
                                if args.save_overhead_debug and final_debug.bgr.size:
                                    path = (
                                        overhead_debug_dir
                                        / f"episode_{ep.index:05d}_final_failed.jpg"
                                    )
                                    _write_overhead_debug_image(
                                        path,
                                        final_debug,
                                        show_distance=True,
                                    )
                                    print(f"Saved overhead final debug image: {path}")
                                return final_placement_metadata(
                                    None,
                                    episode.target,
                                    check_error=str(exc),
                                )
                            if args.save_overhead_debug and final_debug.bgr.size:
                                path = overhead_debug_dir / f"episode_{ep.index:05d}_final.jpg"
                                _write_overhead_debug_image(
                                    path,
                                    final_debug,
                                    show_distance=final_cube is not None,
                                )
                                print(f"Saved overhead final debug image: {path}")
                            return final_placement_metadata(final_cube, episode.target)

                        status = execute_episode(
                            episode,
                            follower=follower,
                            viewer=viewer,
                            offsets_path=args.offsets_path,
                            recording=recording,
                            overhead_camera_cap=overhead_cap,
                            speed=args.speed,
                            wrist_camera=args.wrist_camera,
                            wrist_intrinsics=args.wrist_intrinsics,
                            show_wrist_cam=args.show_wrist_cam,
                            show_wrist_mixed=args.show_wrist_mixed,
                            failed_trajectory_dir=args.save_failed_trajectories,
                            pickup_empty_gripper_position=args.pickup_empty_gripper_position,
                            pickup_gripper_margin=args.pickup_gripper_margin,
                            success_metadata=check_final_placement,
                        )

                        if status == "restart":
                            notifier.alert("Episode restarted or aborted. Re-homing the arm.")
                            raise EpisodeAborted

                        ep.complete()
                        cooldown_reference_target = episode_target
                        is_last = args.episodes != 0 and ep.index >= args.episodes
                        if not is_last:
                            if not recover_cube(viewer):
                                notifier.alert("Cube recovery failed after retries. The run is stopping.")
                                break

                # Normal end (episode budget met, cube lost, or viewer closed): the
                # arm is at the last near-neutral pose — flow it straight to REST.
                if viewer.is_running():
                    notifier.alert("Collection loop is done. Moving the arm to rest.")
                    print("Loop done. Moving to REST...")
                    move_to(REST_ARM_JOINTS, REST_GRIPPER, viewer)
                    ended_at_rest = True
    finally:
        if recording.dataset is not None:
            print("Finalizing dataset...")
            recording.finalize()
            print(f"Dataset written to {dataset_root}")
        if ended_at_rest:
            print("At REST — releasing torque.")
            try:
                follower.bus.disable_torque()
            except Exception as exc:  # noqa: BLE001 - best-effort torque release
                print(f"Warning: could not release torque: {exc}")
        print("Disconnecting hardware...")
        follower.disconnect()
        if overhead_cap is not None:
            overhead_cap.release()


if __name__ == "__main__":
    main()
