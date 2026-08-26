# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Locate the cube for the simulated wrist-camera descent servo.

This is the simulated half of :mod:`pick_and_place.runtime.wrist_servo`. Both
answer the same question — where is the cube, seen from the jaws — and the point
of doing it in sim is that the answer must inherit the same errors. So the frame
is rendered from the camera where it *truly* is, and the detection is mapped into
the world through where the controller *believes* it is:

- a kinematics-only shadow of the scene is posed at the believed joints and the
  believed cube, and its wrist camera pose is what the solve is projected
  through, so the estimate carries the whole hand-eye error;
- the render itself comes from the true scene, including any perturbed physical
  camera mount a domain-randomization draw moved.

Unlike the hardware servo there is no thread: whichever estimator is selected
runs inline, once per control tick. That keeps a recorded episode a pure
function of its seed.

The default geometric mode reproduces the ideal output of pose-based visual
servoing without rendering a second wrist image or running an AprilTag detector:
it expresses the true cube pose in the true camera frame, then maps that pose
back through the camera frame the controller believes. Camera-mount and joint
calibration errors therefore remain in the estimate exactly where visual
servoing would put them. Detector mode remains available for matched validation.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.perception.cube_detection import CubeTracker, detect_cube_faces
from pick_and_place.perception.detector_process import DetectorProcess
from pick_and_place.runtime.believed_frame import BelievedFrame
from pick_and_place.sim.model import get_joint, set_cube_pose, set_joint
from pick_and_place.spec.workspace import CUBE_REST_Z

#: The working resolution the real servo detects at. The render is capped to the
#: model's offscreen buffer, which is what actually limits it.
SERVO_RENDER_WIDTH = 1280
SERVO_RENDER_HEIGHT = 720

#: How much of a new reading the tracker keeps. High, because a single frame's
#: solve is noisy and the descent has many of them.
TRACKER_SMOOTHING = 0.95

WRIST_SERVO_MODES = ("geometric", "detector")


@dataclasses.dataclass(frozen=True)
class CubeSighting:
    """One tick's look at the cube.

    ``pose`` is the world-frame answer the descent steers by, pinned to the
    cube's resting height. ``estimate`` is the tracker's own result, kept because
    projecting the solve back into the image needs its full rotation.
    """

    detections: list
    pose: CubePose | None
    estimate: Any | None


class SimWristServo:
    """Produce wrist-servo cube sightings from geometry or rendered tags.

    Detector mode owns a renderer and detector process. Both modes own the
    believed-world shadow from construction until :meth:`close`.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        belief: BelievedFrame,
        camera_name: str,
        *,
        mode: str = "geometric",
        believed_camera_pose: tuple[np.ndarray, np.ndarray] | None = None,
        detector_crash_dump_dir: str | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.belief = belief
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera_name = camera_name
        if mode not in WRIST_SERVO_MODES:
            raise ValueError(f"wrist servo mode must be one of {WRIST_SERVO_MODES}, got {mode!r}")
        self.mode = mode
        self._believed_camera_pose = believed_camera_pose

        width = min(SERVO_RENDER_WIDTH, int(model.vis.global_.offwidth))
        height = min(SERVO_RENDER_HEIGHT, int(model.vis.global_.offheight))
        self._renderer = (
            mujoco.Renderer(model, width=width, height=height)
            if mode == "detector"
            else None
        )
        fovy = math.radians(float(model.cam_fovy[self.camera_id]))
        fy = (height / 2.0) / math.tan(fovy / 2.0)
        self.camera_matrix = np.array(
            [[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]]
        )

        # nthreads=1 because sharding already saturates the machine with workers.
        self._detector = (
            DetectorProcess(nthreads=1, crash_dump_dir=detector_crash_dump_dir)
            if mode == "detector"
            else None
        )
        self.tracker = (
            CubeTracker(smooth=TRACKER_SMOOTHING, detector=self._detector)
            if self._detector is not None
            else None
        )
        self._shadow = mujoco.MjData(model)

    def believed_camera(self, believed_cube: CubePose) -> tuple[np.ndarray, np.ndarray]:
        """Pose the believed-world shadow and return its wrist camera pose."""
        for name, value in self.belief.arm_joints().items():
            set_joint(self.model, self._shadow, name, value)
        set_joint(self.model, self._shadow, "gripper", get_joint(self.model, self.data, "gripper"))
        set_cube_pose(self.model, self._shadow, believed_cube)
        mujoco.mj_kinematics(self.model, self._shadow)
        # Body kinematics alone leaves camera poses unset; this fills cam_xpos/xmat.
        if self._believed_camera_pose is None:
            mujoco.mj_camlight(self.model, self._shadow)
            return self._shadow_camera()

        # Rendering uses the perturbed physical camera stored in the model. The
        # controller maps detections through the nominal mount calibration, so
        # temporarily restore that local pose only while updating the believed
        # shadow's camera transform.
        true_pos = self.model.cam_pos[self.camera_id].copy()
        true_quat = self.model.cam_quat[self.camera_id].copy()
        try:
            self.model.cam_pos[self.camera_id] = self._believed_camera_pose[0]
            self.model.cam_quat[self.camera_id] = self._believed_camera_pose[1]
            mujoco.mj_camlight(self.model, self._shadow)
            return self._shadow_camera()
        finally:
            self.model.cam_pos[self.camera_id] = true_pos
            self.model.cam_quat[self.camera_id] = true_quat

    def _shadow_camera(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._shadow.cam_xpos[self.camera_id].copy(),
            self._shadow.cam_xmat[self.camera_id].reshape(3, 3).copy(),
        )

    def look(self, believed_cube: CubePose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render the true world from the wrist, reporting the believed camera pose.

        The pair goes together: the image is what the camera sees, the pose is
        what the controller thinks it saw it from, and solving one against the
        other is the whole simulation of a miscalibrated rig.
        """
        camera_position, camera_rotation = self.believed_camera(believed_cube)
        if self._renderer is None:
            raise RuntimeError("wrist RGB is only available in detector mode")
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render(), camera_position, camera_rotation

    def render_believed(self, believed_cube: CubePose) -> np.ndarray:
        """Render the *believed* world — the arm and cube where the controller thinks they are."""
        self.believed_camera(believed_cube)
        if self._renderer is None:
            raise RuntimeError("believed wrist RGB is only available in detector mode")
        self._renderer.update_scene(self._shadow, camera=self.camera_name)
        return self._renderer.render()

    def solve(
        self, rgb: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray
    ) -> CubeSighting:
        """Detect the cube's tags in ``rgb`` and turn them into a world pose."""
        from scipy.spatial.transform import Rotation

        if self.tracker is None:
            raise RuntimeError("AprilTag solving is only available in detector mode")
        detections = detect_cube_faces(rgb, self.tracker.detector)
        estimate = self.tracker.update(
            detections, self.camera_matrix, camera_position, camera_rotation, dist=None
        )
        if estimate is None:
            return CubeSighting(detections, None, None)
        _, _, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
        pose = CubePose(
            x=float(estimate.position[0]),
            y=float(estimate.position[1]),
            z=CUBE_REST_Z,
            yaw=float(yaw),
        )
        return CubeSighting(detections, pose, estimate)

    def geometric_sighting(self, true_cube: CubePose, believed_cube: CubePose) -> CubeSighting:
        """Map the true cube through true and believed wrist-camera frames.

        This is the noiseless pose a perfect camera-relative detector would
        return. It deliberately does not copy world truth into the belief: a
        physical camera-mount perturbation and a wrong joint frame separate the
        two camera transforms and survive in the resulting estimate.
        """
        from scipy.spatial.transform import Rotation

        mujoco.mj_camlight(self.model, self.data)
        true_camera_position = self.data.cam_xpos[self.camera_id].copy()
        true_camera_rotation = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        believed_camera_position, believed_camera_rotation = self.believed_camera(
            believed_cube
        )
        true_position = np.array((true_cube.x, true_cube.y, true_cube.z))
        true_rotation = Rotation.from_euler(
            "xyz", (true_cube.roll, true_cube.pitch, true_cube.yaw)
        ).as_matrix()
        camera_position = true_camera_rotation.T @ (true_position - true_camera_position)
        camera_rotation = true_camera_rotation.T @ true_rotation
        estimate_position = believed_camera_position + believed_camera_rotation @ camera_position
        estimate_rotation = believed_camera_rotation @ camera_rotation
        _, _, yaw = Rotation.from_matrix(estimate_rotation).as_euler("xyz")
        pose = CubePose(
            x=float(estimate_position[0]),
            y=float(estimate_position[1]),
            z=CUBE_REST_Z,
            yaw=float(yaw),
        )
        return CubeSighting([], pose, None)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        if self._detector is not None:
            self._detector.close()
