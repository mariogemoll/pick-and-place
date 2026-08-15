# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turn a simulated control tick into one dataset row.

The schema is the real recordings' schema, which is the whole point: a policy
trained on sim frames has to see what it would see on the rig. Each row holds the
measured joints as ``observation.state``, the set point the same tick is about to
command as ``action``, both cameras, and — privileged, sim-only — the true cube
pose as ``observation.environment_state``.

The joints and poses arrive already read: the loop captures each tick's ground
truth once, into the trajectory artifact, and passes the same values here. That
ordering matters as much as the values do. State and images belong to the moment
*before* the tick's command is applied, exactly as a real recording does where
the motors are read before they have tracked the new set point, so a row pairs
the observation at time t with the action issued from it.

Phase structure lives with the artifact rather than here, since an episode has
exact spans whether or not anything is capturing images from it.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

#: Per-frame privileged ground truth stored as observation.environment_state:
#: the true simulator cube pose, valid in every phase (including in-gripper).
CUBE_POSE_STATE_NAMES = (
    "cube_x",
    "cube_y",
    "cube_z",
    "cube_qw",
    "cube_qx",
    "cube_qy",
    "cube_qz",
)


class SimTickRecorder:
    """Writes one row per control tick into ``recording``.

    The dataset is created here, lazily, because its image shape is the rig's
    output size and nothing knows that until a rig exists. The remaining
    bookkeeping is the check that the video encoder is keeping up.
    """

    def __init__(self, recording: Any, rig: Any, data: mujoco.MjData) -> None:
        self.recording = recording
        self.rig = rig
        self.data = data

        if recording.dataset is None:
            image_shape = (rig.height, rig.width, 3)
            recording.create_dataset(
                image_shape, image_shape, environment_state_names=CUBE_POSE_STATE_NAMES
            )

    def record(
        self, *, believed_state: np.ndarray, action: np.ndarray, true_cube_pose: np.ndarray
    ) -> None:
        """Render both cameras and store them against this tick's state and command."""
        wrist_rgb, overhead_rgb = self.rig.capture(self.data)
        self.recording.dataset.add_frame(
            {
                "observation.state": believed_state.astype(np.float32),
                "action": action.astype(np.float32),
                "observation.environment_state": true_cube_pose.astype(np.float32),
                "observation.images.wrist": wrist_rgb,
                "observation.images.overhead": overhead_rgb,
                "task": self.recording.task,
            }
        )

        # A dropped encoder frame would leave the video shorter than the recorded
        # rows; rather than write a corrupt episode, fail the moment it happens.
        dropped = self.recording.dropped_frame_count()
        if dropped:
            raise RuntimeError(
                f"Streaming video encoder dropped {dropped} frame(s): the encoder "
                "cannot keep pace with capture, which would desync the video from "
                "the recorded frames. Use a hardware vcodec (auto) or raise the "
                "encoder queue size."
            )
