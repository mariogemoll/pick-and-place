# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Find the cube and the drop plate the way the rig does: render, detect, solve.

The wrist loop was always genuinely simulated. Overhead localization was not:
the code took the true cube pose, added a few millimetres of Gaussian noise and
handed that to the planner. No overhead image was rendered and no detector ran,
which meant the one visual pipeline the real rig depends on most was assumed
correct rather than tested.

This runs it for real. The image comes from the true world; the solve is
projected through where the camera is *believed* to be, which is what turns a
calibration error into a localization error instead of leaving it invisible.

**Simulating the loop removes error unless the causes go back in.** In a clean
scene the extrinsics are exact and the workspace frame is exactly where the
model puts it, so an honest render-and-detect comes out *more* accurate than the
rig — the wrong direction. So the believed camera pose is displaced from the
true one by a drawn
:class:`~pick_and_place.core.miscalibration.OverheadCameraError`, and the
resulting localization error is an outcome that can be measured against the
rig's rather than a number that was assumed.

**The arm can block the view.** That is why the rig has a hunt behavior — pan
through wide search poses until the cube and the plate are both visible — and
now it is why sim needs one too. :meth:`SimOverheadPerception.look` simply
reports what it could and could not see; moving the arm is the caller's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.miscalibration import OverheadCameraError
from pick_and_place.core.workspace_bounds import workspace_interior_corners_world
from pick_and_place.perception.detector_process import DetectorProcess
from pick_and_place.perception.overhead_localization import OverheadLocalizer
from pick_and_place.spec.drop_zone import PaperTarget

OVERHEAD_CAMERA = "overhead_camera"

#: The resolution the rig's extrinsics solve and tag detection actually run at.
#: Detection needs more pixels than a recorded observation carries, so overhead
#: perception renders separately, the way the wrist servo already does. Capped
#: by the model's offscreen buffer, which is what really limits it.
DETECTION_WIDTH = 1920
DETECTION_HEIGHT = 1080


@dataclass(frozen=True)
class OverheadReading:
    """One look at the workspace from above.

    Either field may be ``None``: the arm can stand between the camera and the
    cube, or between it and the plate, and on the rig that is ordinary rather
    than exceptional.
    """

    cube: CubePose | None
    target: PaperTarget | None
    rgb: np.ndarray | None = None

    @property
    def complete(self) -> bool:
        """Whether this look found everything an episode needs to be planned."""
        return self.cube is not None and self.target is not None


def believed_camera_pose(
    position: np.ndarray, rotation: np.ndarray, error: OverheadCameraError
) -> tuple[np.ndarray, np.ndarray]:
    """Where the calibration says the camera is, given where it actually is.

    A rigid offset, because that is what both causes reduce to: a camera that
    moved between sessions and a frame lying off its authored place both leave
    the solved pose displaced from the true one.
    """
    delta = Rotation.from_euler("xyz", error.rotation_deg, degrees=True)
    return (
        np.asarray(position, dtype=float) + np.asarray(error.position_m, dtype=float),
        delta.as_matrix() @ np.asarray(rotation, dtype=float),
    )


def camera_matrix_for(model: mujoco.MjModel, camera: int, width: int, height: int) -> np.ndarray:
    """The pinhole intrinsics of a MuJoCo camera rendered at ``width`` x ``height``.

    MuJoCo cameras are ideal pinholes with the principal point at the image
    centre, which is also what the rig's rectified frames are — so a solve
    against this matrix is the same solve the rig runs.
    """
    fovy = math.radians(float(model.cam_fovy[camera]))
    fy = (height / 2.0) / math.tan(fovy / 2.0)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


class SimOverheadPerception:
    """Renders the overhead camera and localizes the cube and plate out of it.

    Owns a renderer and a localizer from construction until :meth:`close`.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        error: OverheadCameraError = OverheadCameraError((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        target_color: str = "black",
        detector: Any = None,
        width: int = DETECTION_WIDTH,
        height: int = DETECTION_HEIGHT,
    ) -> None:
        self.model = model
        self.data = data
        self.target_color = target_color
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, OVERHEAD_CAMERA)
        if self.camera_id < 0:
            raise ValueError(f"model has no camera named {OVERHEAD_CAMERA!r}")

        # Detection runs out of process. libapriltag segfaults on rare inputs and
        # its C destructor segfaults when a detector is collected inside a pool
        # worker; either would take down the whole shard and corrupt its
        # metadata. Behind the process boundary that costs one detection.
        self._owned_detector = None if detector is not None else DetectorProcess(nthreads=1)
        detector = detector if detector is not None else self._owned_detector

        self.width = min(width, int(model.vis.global_.offwidth))
        self.height = min(height, int(model.vis.global_.offheight))
        self._renderer = mujoco.Renderer(model, width=self.width, height=self.height)

        mujoco.mj_forward(model, data)
        # The overhead camera is bolted to the world, so its true pose is read
        # once here rather than per tick the way the wrist camera's is.
        self.true_position = data.cam_xpos[self.camera_id].copy()
        self.true_rotation = data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        self.localizer = OverheadLocalizer(
            camera_matrix_for(model, self.camera_id, self.width, self.height),
            self.true_position,
            self.true_rotation,
            detector_factory=lambda: detector,
        )
        self.workspace_corners = workspace_interior_corners_world()
        self.set_error(error)

    def set_error(self, error: OverheadCameraError) -> None:
        """Recalibrate: draw a fresh gap between where the camera is and is thought to be.

        Once per session on the rig, so once per episode here — the extrinsics
        are solved at the start and hold for everything that follows.
        """
        self.error = error
        position, rotation = believed_camera_pose(
            self.true_position, self.true_rotation, error
        )
        self.localizer.camera_position = position
        self.localizer.camera_rotation = rotation

    def reset(self) -> None:
        """Forget detections from the previous episode."""
        self.localizer.reset()

    def look(self, *, keep_image: bool = False) -> OverheadReading:
        """Render the workspace from above and solve what is visible in it."""
        self._renderer.update_scene(self.data, camera=OVERHEAD_CAMERA)
        rgb = self._renderer.render()
        return OverheadReading(
            cube=self.localizer.localize_cube(rgb),
            target=self.localizer.localize_drop_target(
                rgb,
                target_color=self.target_color,
                workspace_corners_world=self.workspace_corners,
            ),
            rgb=rgb if keep_image else None,
        )

    def close(self) -> None:
        self._renderer.close()
        if self._owned_detector is not None:
            self._owned_detector.close()
