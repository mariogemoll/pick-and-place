# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Turn a simulated control tick into one dataset row.

The schema is the real recordings' schema, which is the whole point: a policy
trained on sim frames has to see what it would see on the rig. Each row holds
the measured joints as ``observation.state``, the set point the same tick is
about to command as ``action``, both cameras, and — privileged, sim-only — the
true cube pose as ``observation.environment_state``.

State and images are captured *before* the tick's command is applied, so a row
pairs the observation at time t with the action issued from it, exactly as a real
recording does where the state is read before the servos have tracked the new set
point. Keeping that ordering is why recording is its own step in the loop rather
than a side effect of stepping physics.

The rows also carry their own phase structure: each contiguous run of ticks
driven by the same named trajectory phase becomes a :class:`~pick_and_place.core.task_phases.PhaseSpan`,
taken from the controller rather than reconstructed from the gripper trace
afterwards.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.core.task_phases import PhaseSpan
from pick_and_place.sim.model import get_joint
from pick_and_place.spec.robot import ARM_JOINT_NAMES

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
    """Writes one row per control tick into ``recording``, and tracks the spans.

    The dataset is created here, lazily, because its image shape is the rig's
    output size and nothing knows that until a rig exists. Everything else is
    bookkeeping the caller should not have to repeat: the frame count, the phase
    spans, and the check that the video encoder is keeping up.
    """

    def __init__(self, recording: Any, rig: Any, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.recording = recording
        self.rig = rig
        self.model = model
        self.data = data
        self.frames = 0
        self._spans: list[PhaseSpan] = []

        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        self._cube_qpos_adr = int(model.jnt_qposadr[model.body_jntadr[cube_body_id]])

        if recording.dataset is None:
            image_shape = (rig.height, rig.width, 3)
            recording.create_dataset(
                image_shape, image_shape, environment_state_names=CUBE_POSE_STATE_NAMES
            )

    @property
    def spans(self) -> tuple[PhaseSpan, ...]:
        """The phase spans of the frames recorded so far."""
        return tuple(self._spans)

    def record(self, frame, phase_name: str, offsets_deg: dict[str, float] | None) -> None:
        """Capture the state and cameras, then store them against ``frame``'s command."""
        measured_arm = {name: get_joint(self.model, self.data, name) for name in ARM_JOINT_NAMES}
        measured_gripper = get_joint(self.model, self.data, "gripper")
        state = sim_frame_to_real(measured_arm, measured_gripper, offsets_deg)
        action = sim_frame_to_real(frame.joints, frame.gripper)
        cube_pose = np.asarray(
            self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7], dtype=np.float32
        ).copy()

        wrist_rgb, overhead_rgb = self.rig.capture(self.data)
        self.recording.dataset.add_frame(
            {
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "observation.environment_state": cube_pose,
                "observation.images.wrist": wrist_rgb,
                "observation.images.overhead": overhead_rgb,
                "task": self.recording.task,
            }
        )
        if not self._spans or self._spans[-1].name != phase_name:
            self._spans.append(PhaseSpan(name=phase_name, start_frame=self.frames))
        self.frames += 1

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
