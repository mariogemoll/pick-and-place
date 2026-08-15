# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Detect the cube in the wrist camera while the arm descends onto it.

The descent is the one phase that cannot be played back open-loop: by then the
jaws are centimetres from a cube whose measured pose carries the whole rig's
calibration error, and a few millimetres decide whether the grasp closes on it
or knocks it over. So a worker thread runs the AprilTag detector on wrist frames
as fast as they arrive and publishes its newest world-frame estimate; the
control loop picks that up once per tick and re-solves the grasp toward it.

Detection and control run at unrelated rates on purpose. Tag solving takes tens
of milliseconds and the loop must not wait for it, so the two meet through a
single published :class:`WristServoEstimate` — the loop always reads the freshest
one and never blocks, and a frame that took too long is simply superseded.
Each estimate carries the camera frame index it came from, which is how the loop
tells a new reading from one it has already folded in.

The worker only runs while the loop says the descent is live. Off the descent it
idles, because the detector's cost is real and nothing consumes its answer.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.scripted.visual_servo import WristServoEstimate, WristServoPreview
from pick_and_place.runtime.frame_reader import FrameReader, open_capture
from pick_and_place.spec.workspace import CUBE_HALF_SIZE

#: The wrist module is opened at its native size and the undistort map is built
#: for it, so intrinsics and frames agree regardless of what the loop displays.
WRIST_CAPTURE_WIDTH = 1280
WRIST_CAPTURE_HEIGHT = 720

#: Corners of a unit cube, scaled by the cube's half-size and projected to draw
#: the detected pose back onto the preview.
_CUBE_CORNERS = np.float32(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]
)
_CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def draw_tags(bgr: np.ndarray, detections, cv2) -> None:
    """Outline each detected tag and label it with its id, in place."""
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


def project_cube(
    estimate,
    camera_matrix: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    cv2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the solved cube's corners into the image.

    The estimate is in world coordinates; OpenCV wants it in the camera's own
    frame, and OpenCV's camera looks down +z with +y down while MuJoCo's looks
    down -z with +y up, hence the axis flip. Returns the rotation and translation
    vectors alongside the projected corners, since drawing the pose axes needs
    them too.
    """
    cv_to_mj = np.diag([1.0, -1.0, -1.0])
    position_in_camera = camera_rotation.T @ (estimate.position - camera_position)
    rotation_in_camera = camera_rotation.T @ estimate.rotation
    tvec = cv_to_mj @ position_in_camera
    rvec, _ = cv2.Rodrigues(cv_to_mj @ rotation_in_camera)
    points, _ = cv2.projectPoints(
        _CUBE_CORNERS * CUBE_HALF_SIZE, rvec, tvec, camera_matrix, np.zeros(5)
    )
    return rvec, tvec, points.reshape(-1, 2).astype(int)


def draw_cube(bgr, rvec, tvec, points, camera_matrix: np.ndarray, cv2) -> None:
    """Draw the solved cube's pose axes and wireframe, in place."""
    cv2.drawFrameAxes(bgr, camera_matrix, np.zeros(5), rvec, tvec, 0.03, 2)
    for i, j in _CUBE_EDGES:
        cv2.line(bgr, tuple(points[i]), tuple(points[j]), (0, 165, 255), 2, cv2.LINE_AA)


class WristServo:
    """A detector thread over ``reader``, publishing its newest cube estimate.

    Owns the thread from construction until :meth:`close`. The control loop drives
    it through :meth:`begin_phase` (is the descent live, and what has already been
    seen) and :meth:`sample` (here is where the camera is now, give me your latest),
    both of which take the lock once so the loop never sees a half-updated pair.

    ``annotate`` turns on the preview overlays, which cost a per-frame draw and are
    only worth paying for when a window is actually showing them. ``on_overlay``,
    if given, receives ``(captured_at, primitives)`` for every processed frame, so
    a recording can store what the servo saw alongside the video.
    """

    def __init__(
        self,
        reader: FrameReader,
        tracker: Any,
        camera_matrix: np.ndarray,
        undistort_map: tuple[np.ndarray, np.ndarray] | None,
        *,
        annotate: bool = False,
        on_overlay: Callable[[float, dict[str, Any]], None] | None = None,
    ) -> None:
        self.reader = reader
        self.tracker = tracker
        self.camera_matrix = camera_matrix
        self.undistort_map = undistort_map
        self.annotate = annotate
        self.on_overlay = on_overlay

        self._lock = threading.Lock()
        self._active = False
        self._running = True
        self._camera_position: np.ndarray | None = None
        self._camera_rotation: np.ndarray | None = None
        self._estimate: WristServoEstimate | None = None
        self._preview: WristServoPreview | None = None
        self._thread = threading.Thread(target=self._loop, name="wrist-servo", daemon=True)
        self._thread.start()

    def begin_phase(self, active: bool) -> tuple[int, int]:
        """Arm or idle the worker for a phase, and report what it has published.

        The returned estimate and preview frame ids are what the loop compares
        against to spot a *new* reading. Taken here, under the same lock that
        clears the stale camera pose, so nothing published for the previous phase
        can be mistaken for a fresh answer in this one.
        """
        with self._lock:
            self._active = active
            self._camera_position = None
            self._camera_rotation = None
            estimate_id = self._estimate.frame_id if self._estimate is not None else -1
            preview_id = self._preview.frame_id if self._preview is not None else -1
        return estimate_id, preview_id

    def sample(
        self, camera_position: np.ndarray, camera_rotation: np.ndarray
    ) -> tuple[WristServoEstimate | None, WristServoPreview | None]:
        """Publish where the wrist camera is now and take the worker's latest output.

        The pose has to be the one from this tick: the worker maps a detection out
        of the image and into the world through it, so a stale pose would place a
        correctly detected cube in the wrong place.
        """
        with self._lock:
            self._camera_position = camera_position
            self._camera_rotation = camera_rotation
            return self._estimate, self._preview

    def close(self) -> None:
        """Stop the worker and release the camera."""
        with self._lock:
            self._active = False
        self._running = False
        self._thread.join(timeout=1.0)
        self.reader.close()

    def _loop(self) -> None:
        import cv2
        from scipy.spatial.transform import Rotation

        from pick_and_place.perception.cube_detection import detect_cube_faces

        last_frame_id = -1
        while self._running:
            with self._lock:
                active = self._active
                position = None if self._camera_position is None else self._camera_position.copy()
                rotation = None if self._camera_rotation is None else self._camera_rotation.copy()
            if not active or position is None or rotation is None:
                time.sleep(0.002)
                continue

            # Only ever run the detector on a frame it has not already seen; this
            # worker and the camera run at unrelated rates.
            frame = self.reader.latest()
            if frame is None or frame.index == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame.index

            # The overlays below draw into ``bgr`` in place, and the published
            # frame is shared with the recorder, so this must not alias it.
            # ``remap`` already yields a fresh array; the unrectified path copies.
            bgr = (
                cv2.remap(frame.bgr, *self.undistort_map, cv2.INTER_LINEAR)
                if self.undistort_map is not None
                else frame.bgr.copy()
            )
            detections = detect_cube_faces(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.tracker.detector)
            if self.annotate:
                draw_tags(bgr, detections, cv2)

            estimate = self.tracker.update(
                detections, self.camera_matrix, position, rotation, dist=None
            )
            overlay: dict[str, Any] = {
                "tags": [np.asarray(det.corners).tolist() for det in detections]
            }
            if estimate is not None:
                _, _, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
                source = CubePose(
                    x=float(estimate.position[0]),
                    y=float(estimate.position[1]),
                    z=CUBE_HALF_SIZE,
                    roll=0.0,
                    pitch=0.0,
                    yaw=float(yaw),
                )
                with self._lock:
                    self._estimate = WristServoEstimate(last_frame_id, source)
                # Projecting the cube is cheap and is what a viewer needs to see
                # what the servo saw, so it happens whether or not anything is
                # being drawn on screen right now.
                rvec, tvec, points = project_cube(
                    estimate, self.camera_matrix, position, rotation, cv2
                )
                overlay["cube_edges"] = [
                    [points[i].tolist(), points[j].tolist()] for i, j in _CUBE_EDGES
                ]
                if self.annotate:
                    draw_cube(bgr, rvec, tvec, points, self.camera_matrix, cv2)

            if self.on_overlay is not None:
                self.on_overlay(frame.captured_at, overlay)

            if self.annotate:
                with self._lock:
                    self._preview = WristServoPreview(last_frame_id, bgr)


def open_wrist_servo(
    source: str,
    intrinsics: str | Path | None,
    *,
    annotate: bool = False,
    on_frame: Callable[[np.ndarray, float], None] | None = None,
    on_overlay: Callable[[float, dict[str, Any]], None] | None = None,
) -> WristServo | None:
    """Open the wrist camera and start servoing off it, or ``None`` if it will not open.

    A camera that is unplugged or renumbered is a routine rig fault and not worth
    aborting an episode for — the descent falls back to open-loop playback, which
    is what a run without a wrist camera does anyway. Missing *intrinsics*, by
    contrast, is a setup mistake: the frames would be there but every pose solved
    from them would be silently wrong, so that raises.
    """
    import cv2

    from pick_and_place.calibration.camera_compare import load_intrinsics
    from pick_and_place.core.camera_calibration import LOCAL_CAMERA_INTRINSICS_DIR
    from pick_and_place.perception.cube_detection import CubeTracker

    try:
        capture = open_capture(source, WRIST_CAPTURE_WIDTH, WRIST_CAPTURE_HEIGHT, "wrist")
    except RuntimeError:
        print(f"Warning: could not open wrist camera {source!r}")
        return None

    request_camera_fps(capture, "wrist")
    path = (
        Path(intrinsics)
        if intrinsics is not None
        else LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
    )
    if not path.exists():
        capture.release()
        raise RuntimeError(f"Missing wrist camera intrinsics at {path}")
    camera_matrix, undistort_map = load_intrinsics(
        path, WRIST_CAPTURE_WIDTH, WRIST_CAPTURE_HEIGHT, cv2
    )

    return WristServo(
        FrameReader(capture, "wrist", on_frame=on_frame),
        CubeTracker(smooth=0.95),
        camera_matrix,
        undistort_map,
        annotate=annotate,
        on_overlay=on_overlay,
    )


def open_sim_view(
    model, camera_id: int, servo: WristServo, show_mixed: bool
):
    """Match the sim's wrist camera to the real one, and build the blend renderer.

    The rectified frames the servo works on have a different field of view from
    the raw ones, so the sim camera is re-derived from the rectified focal length.
    Without that a sim/real overlay would not line up even at a perfect pose.
    """
    import mujoco

    rect_fy = float(servo.camera_matrix[1, 1])
    model.cam_fovy[camera_id] = float(
        np.degrees(2.0 * np.arctan((WRIST_CAPTURE_HEIGHT / 2.0) / rect_fy))
    )
    if not show_mixed:
        return None
    scale = min(
        1.0,
        int(model.vis.global_.offwidth) / WRIST_CAPTURE_WIDTH,
        int(model.vis.global_.offheight) / WRIST_CAPTURE_HEIGHT,
    )
    return mujoco.Renderer(
        model,
        width=max(1, round(WRIST_CAPTURE_WIDTH * scale)),
        height=max(1, round(WRIST_CAPTURE_HEIGHT * scale)),
    )


def show_frame(
    bgr: np.ndarray, *, renderer, model, data, undistort_map, rectify: bool
) -> None:
    """Display one wrist frame, optionally blended with the sim's view of the same pose.

    The descent's frames are already rectified by the worker, so ``rectify`` is
    what keeps them from being remapped twice.
    """
    import cv2

    if rectify and undistort_map is not None:
        bgr = cv2.remap(bgr, *undistort_map, cv2.INTER_LINEAR)
    if renderer is not None:
        renderer.update_scene(data, camera="wrist_camera")
        sim_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        if sim_bgr.shape[:2] != bgr.shape[:2]:
            sim_bgr = cv2.resize(
                sim_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        bgr = cv2.addWeighted(bgr, 0.6, sim_bgr, 0.4, 0.0)
    cv2.imshow("Wrist Cam", bgr)
    cv2.waitKey(1)


def request_camera_fps(capture: Any, label: str) -> None:
    """Ask the camera to capture at ``CONTROL_HZ`` and report what it grants.

    Pinning the capture rate near the tick rate keeps the latest-frame buffer
    fresh and avoids wasting captures. Sync no longer depends on the camera
    honoring it — the control loop logs exactly one buffered frame per tick
    either way — so a mismatch is a warning, not a failure.
    """
    import math

    import cv2

    from pick_and_place.spec.robot import CONTROL_HZ

    capture.set(cv2.CAP_PROP_FPS, float(CONTROL_HZ))
    actual = capture.get(cv2.CAP_PROP_FPS)
    if not actual or actual <= 0 or math.isnan(actual):
        print(f"warning: {label} camera did not report an FPS after requesting {CONTROL_HZ:g}")
    elif not math.isclose(actual, CONTROL_HZ, rel_tol=1e-2):
        print(
            f"warning: {label} camera reports {actual:g} fps, not the requested "
            f"{CONTROL_HZ:g}; frames are still logged one per control tick"
        )
