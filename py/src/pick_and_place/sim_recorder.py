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

The image features are 512x512 squares — the input size of the SmolVLA vision
tower. Each camera's vertical field of view is set from its calibrated
intrinsics, so a sim frame matches a real frame that has been undistorted,
center-cropped to a square, and resized to 512x512. (Undistortion is a no-op in
sim: the sim camera is an ideal pinhole, so there is nothing to fake-then-invert
— matching the field of view is all that is needed.)
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from pick_and_place.episodes import (
    Episode,
    get_joint,
    is_unexpected,
    placement_error,
    scan_contacts,
)
from pick_and_place.executor import (
    CONTROL_HZ,
    HARDWARE_SIMULATION_HZ,
)
from pick_and_place.recording import RecordingSession
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    sim_frame_to_real,
)
from pick_and_place.image_rectify import SQUARE_SIZE

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
    """Offscreen renderers for the wrist and overhead cameras at 512x512.

    Each camera's ``cam_fovy`` is overridden from its calibrated intrinsics (when
    available) so the square sim render matches a square crop of the real
    undistorted feed. Falls back to the model's built-in fovy for any camera
    without local intrinsics.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        intrinsics_by_name: dict[str, dict[str, Any]] | None = None,
        size: int = SQUARE_SIZE,
    ) -> None:
        self.size = size
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
        self._renderer = mujoco.Renderer(model, width=size, height=size)

    def render(self, data: mujoco.MjData, camera: str) -> np.ndarray:
        """Render one camera to an ``(size, size, 3)`` uint8 RGB array."""
        self._renderer.update_scene(data, camera=camera)
        return self._renderer.render()

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Render both cameras, returning ``(wrist_rgb, overhead_rgb)``."""
        return self.render(data, WRIST_CAMERA), self.render(data, OVERHEAD_CAMERA)

    def close(self) -> None:
        self._renderer.close()


def record_episode(
    episode: Episode,
    *,
    recording: RecordingSession,
    rig: SimCameraRig,
    viewer: Any = None,
    speed: float = 1.0,
) -> None:
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

    The dataset is created lazily on the first episode once the (fixed) 512x512
    frame shape is known. The caller commits the episode with ``save_episode``
    and ends the run with :meth:`RecordingSession.finalize`.
    """
    if speed <= 0.0:
        raise ValueError("speed must be positive")

    model = episode.model
    data = episode.data
    actuator_id = episode.actuator_id
    trajectory = episode.trajectory

    simulation_steps_per_tick = round(HARDWARE_SIMULATION_HZ / CONTROL_HZ)
    control_period = 1.0 / CONTROL_HZ
    if not math.isclose(model.opt.timestep * simulation_steps_per_tick, control_period):
        raise ValueError(
            f"MuJoCo timestep {model.opt.timestep:g}s cannot produce {CONTROL_HZ:g} Hz exactly"
        )

    if recording.dataset is None:
        recording.create_dataset((rig.size, rig.size, 3), (rig.size, rig.size, 3))

    prev_contacts: set[tuple[str, str]] = set()
    playback_start = data.time
    while True:
        traj_t = (data.time - playback_start) * speed
        frame = trajectory.evaluate(traj_t)

        measured_arm = {name: get_joint(model, data, name) for name in ARM_JOINT_NAMES}
        measured_gripper = get_joint(model, data, "gripper")
        state = sim_frame_to_real(measured_arm, measured_gripper)
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
                f"Streaming video encoder dropped {dropped} frame(s) at t={traj_t:.3f}s: "
                "the encoder cannot keep pace with capture, which would desync the video "
                "from the recorded frames. Use a hardware vcodec (auto) or raise the "
                "encoder queue size."
            )

        if traj_t >= trajectory.duration:
            break

        for name, value in frame.joints.items():
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, data, nstep=simulation_steps_per_tick)

        curr_contacts = {
            (min(n1, n2), max(n1, n2))
            for n1, n2 in scan_contacts(
                model, data, episode.robot_geom_ids, episode.env_geom_ids
            )
            if is_unexpected(n1, n2)
        }
        for pair in curr_contacts - prev_contacts:
            print(f"collision t={traj_t:.3f}s  {pair[0]} ↔ {pair[1]}")
        prev_contacts = curr_contacts

        if viewer is not None:
            viewer.sync()

    print(placement_error(model, data, episode.target).summary())
