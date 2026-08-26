# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turn a wrist image into a cube pose, for a controller that only consumes them.

Detection belongs on this side of the line. A controller that produced its own
sightings would need a camera model, a tag detector and a kinematic mirror to
project through — three capabilities it has no other use for — and the
observation-driven controller would stop being drivable from images alone.

The mirror here is deliberately *nominal*: reported joints pose a camera at the
mount the calibration says is there, so a physical or simulated mount
perturbation stays hidden inside the image rather than leaking out through
extrinsics the environment supplies. That is what makes the hand-eye error
something the controller has to servo through rather than something it is told.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.core.geometry import CubePose
from pick_and_place.perception.cube_detection import CubeTracker
from pick_and_place.sim.model import set_joint
from pick_and_place.spec.workspace import CUBE_REST_Z

#: What a controller asks for: an image, where it thinks its joints are, and its
#: prior on the cube; back comes a world pose, or nothing.
WristLocalization = Callable[
    [np.ndarray, dict[str, float], float, CubePose], CubePose | None
]


class WristCameraLocalizer:
    """Map wrist RGB into world poses through fixed nominal calibration.

    The model is a controller-owned kinematic mirror. Reported joints pose its
    nominal wrist camera each tick, so physical or simulated camera-mount
    perturbations remain hidden in the observation image instead of leaking
    through extrinsics supplied by the environment.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_matrix: np.ndarray,
        *,
        camera_name: str = "wrist_camera",
        tracker_factory: Callable[[], Any] | None = None,
        free_grasp: bool = False,
    ) -> None:
        matrix = np.asarray(camera_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must have finite shape (3, 3)")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"model has no camera named {camera_name!r}")
        self.model = model
        self.camera_matrix = matrix.copy()
        self.camera_id = camera_id
        self.free_grasp = free_grasp
        self._tracker_factory = tracker_factory or (lambda: CubeTracker(smooth=0.95))
        self._shadow = mujoco.MjData(model)
        self.reset()

    def reset(self) -> None:
        """Clear tracking history between controller episodes."""
        self._tracker = self._tracker_factory()

    def __call__(
        self,
        image: np.ndarray,
        reported_joints: dict[str, float],
        reported_gripper: float,
        prior: CubePose,
    ) -> CubePose | None:
        del prior
        for name, value in reported_joints.items():
            set_joint(self.model, self._shadow, name, value)
        set_joint(self.model, self._shadow, "gripper", reported_gripper)
        mujoco.mj_forward(self.model, self._shadow)
        estimate = self._tracker.update_frame(
            image,
            self.camera_matrix,
            self._shadow.cam_xpos[self.camera_id],
            self._shadow.cam_xmat[self.camera_id].reshape(3, 3),
            dist=None,
        )
        if estimate is None:
            return None
        roll, pitch, yaw = Rotation.from_matrix(estimate.rotation).as_euler("xyz")
        return CubePose(
            x=float(estimate.position[0]),
            y=float(estimate.position[1]),
            z=CUBE_REST_Z,
            roll=float(roll) if self.free_grasp else 0.0,
            pitch=float(pitch) if self.free_grasp else 0.0,
            yaw=float(yaw),
        )


class AsyncWristLocalization:
    """Keep wrist detection off the command-critical control thread."""

    def __init__(self, localizer: WristLocalization) -> None:
        self._localizer = localizer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wrist-localizer")
        self._future: Future[CubePose | None] | None = None
        self._latest: CubePose | None = None

    def __call__(
        self,
        frame_rgb: np.ndarray,
        joints: dict[str, float],
        gripper: float,
        prior: CubePose,
    ) -> CubePose | None:
        if self._future is not None and self._future.done():
            self._latest = self._future.result()
            self._future = None
        if self._future is None:
            self._future = self._executor.submit(
                self._localizer,
                np.asarray(frame_rgb).copy(),
                dict(joints),
                float(gripper),
                prior,
            )
        return self._latest

    def reset(self) -> None:
        if self._future is not None:
            if not self._future.cancel():
                self._future.result()
            self._future = None
        self._latest = None
        reset = getattr(self._localizer, "reset", None)
        if reset is not None:
            reset()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
