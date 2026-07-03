# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Overhead-camera cube/drop-zone detection and operator helpers.

Shared by the hardware collection scripts: the analytic runner (``real.py``)
and the teleoperated recorder (``record_teleop.py``). Holds the overhead cube
and drop-zone trackers, the verification-overlay writer, the placement-error
metadata builder, the best-effort audible operator notifier, and the mock
viewer used for headless runs and post-loop parking.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pick_and_place.dataset_metadata import placement_error_metadata
from pick_and_place.episodes import PlacementError
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.paper_detection import (
    PaperTarget,
    PaperTracker,
    detect_paper_target,
    project_to_pixel,
    set_paper_target_marker,
)

# How long a single look attempt stares at the camera feed before giving up.
CUBE_LOOK_TIMEOUT = 2.0
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


def write_overhead_debug_image(
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


def empty_overhead_debug() -> OverheadDetectionDebug:
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

    def chirp(self, sound_path: str | None = None) -> None:
        """Play a short sound with no speech -- a low-latency cue.

        Speech (``say``) always lags by a second or so, so time-critical cues
        (e.g. "recording started now") use this instead. Falls back to the
        terminal bell when no player/sound is available.
        """
        if not self.enabled:
            return
        path = sound_path if sound_path is not None else self.sound_path
        if self._afplay and path and Path(path).exists():
            subprocess.run([self._afplay, path], check=False)
        else:
            print("\a", end="", flush=True)


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
    if not intrinsics.exists():
        raise RuntimeError(f"Missing {camera_name} intrinsics at {intrinsics}")
    camera_matrix, undistort_map = load_intrinsics(intrinsics, 1920, 1080, cv2)

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
        if not is_cube_pickup_allowed(float(position[0]), float(position[1])):
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
    if not intrinsics.exists():
        raise RuntimeError(f"Missing {camera_name} intrinsics at {intrinsics}")
    camera_matrix, undistort_map = load_intrinsics(intrinsics, 1920, 1080, cv2)

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


def final_placement_metadata(cube: CubePose | None, target: CubePose) -> dict[str, Any]:
    """Episode metadata for the physical cube's final overhead-camera pose."""
    target_xyz = (float(target.x), float(target.y), float(CUBE_HALF_SIZE))
    if cube is None:
        print("placement error: cube not detected after release")
        return placement_error_metadata(None, detected=False)

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
    return placement_error_metadata(error, detected=True)
