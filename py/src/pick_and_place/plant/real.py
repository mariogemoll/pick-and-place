# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The rig as a plant: a physical arm, and a MuJoCo shadow of what it believes.

The shadow is stepped at the *commanded* joints, so it holds the arm as the
controller believes it to be — which is what the wrist camera pose is taken from,
and what collision reporting runs against. The physical arm is the true world,
and the only thing that ever reads it is the servo readback.

The detector runs on **its own thread** over the live camera, so a tick receives
whatever it has most recently produced rather than waiting for a fresh solve. A
reading is therefore often repeated across ticks, and this is the one place that
knows it: :attr:`Sighting.fresh` is false the second time the same solve comes
back, so a single detection cannot pull the grasp twice.

The clock is the shadow's simulated time, paced against wall time. Trajectory
time advancing with physics rather than with the wall is what makes a phase play
at the same rate here as it does in a pure sim run.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import (
    action_to_joints,
    clamp_and_warn,
    joints_to_action,
    sim_frame_to_real,
)
from pick_and_place.plant.interface import NOTHING_SEEN, Sighting
from pick_and_place.runtime.checkpoint import measured_sim_state
from pick_and_place.runtime.wrist_servo import WristServo
from pick_and_place.sim.collisions import unexpected_contact_pairs
from pick_and_place.sim.model import set_cube_pose
from pick_and_place.spec.robot import CONTROL_HZ


class RealPlant:
    """Commands the follower, steps the believed shadow, and reads the servos back.

    ``joint_offsets_deg`` is the session calibration: the command is
    ``degrees(planned) - offset``, so the joint lands where the plan wanted it
    rather than where its zero happens to be.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        follower: Any,
        actuator_id: dict[str, int],
        robot_geom_ids: Any,
        env_geom_ids: Any,
        kinematics: Any,
        substeps_per_tick: int,
        clamp_low: np.ndarray,
        clamp_high: np.ndarray,
        wrist_camera_id: int,
        servo: WristServo | None = None,
        joint_offsets_deg: dict[str, float] | None = None,
        speed: float = 1.0,
    ) -> None:
        self.model = model
        self.data = data
        self.follower = follower
        self.actuator_id = actuator_id
        self.robot_geom_ids = robot_geom_ids
        self.env_geom_ids = env_geom_ids
        self.kinematics = kinematics
        self.substeps_per_tick = substeps_per_tick
        self.clamp_low = clamp_low
        self.clamp_high = clamp_high
        self.wrist_camera_id = wrist_camera_id
        self.servo = servo
        self.joint_offsets_deg = joint_offsets_deg
        self.speed = speed
        self._clip_warned: set[str] = set()
        self._commanded = np.zeros(len(clamp_low))
        self._last_estimate_id = -1
        #: The newest annotated frame the detector thread produced, kept for the
        #: preview window. Nothing in control reads it.
        self.last_preview: Any = None
        self._next_tick = time.monotonic()

    @property
    def time(self) -> float:
        """The shadow's simulated time, which is what the pacing below tracks."""
        return float(self.data.time)

    @property
    def commanded(self) -> np.ndarray:
        """The last real-frame command issued, which a readback is filled in from."""
        return self._commanded

    def step(self, joints: Mapping[str, float], gripper: float) -> np.ndarray:
        """Command the shadow and the arm, step physics, and pace to the control rate."""
        for name, value in joints.items():
            self.data.ctrl[self.actuator_id[name]] = value
        self.data.ctrl[self.actuator_id["gripper"]] = gripper
        mujoco.mj_step(self.model, self.data, nstep=self.substeps_per_tick)

        self._commanded = clamp_and_warn(
            sim_frame_to_real(joints, gripper, self.joint_offsets_deg),
            self.clamp_low,
            self.clamp_high,
            self._clip_warned,
        )
        self.follower.send_action(joints_to_action(self._commanded))
        self._pace()
        return self._commanded

    def _pace(self) -> None:
        """Sleep out the rest of the tick, without bursting after a stall.

        A long vision, I/O or scheduling stall would otherwise be followed by a
        burst of catch-up commands, which the arm would try to execute at once.
        """
        control_period = 1.0 / CONTROL_HZ
        self._next_tick += control_period
        remaining = self._next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -control_period:
            self._next_tick = time.monotonic()

    def resync_clock(self) -> None:
        """Restart the pacing clock after a deliberate pause, so it does not burst."""
        self._next_tick = time.monotonic()

    def begin_phase(self, descent_active: bool) -> int:
        """Arm or idle the detector for a phase; return the preview id already shown.

        Taken under the servo's own lock together with the stale camera pose, so
        nothing published for the previous phase can be mistaken for a fresh
        answer in this one.
        """
        if self.servo is None:
            return -1
        self._last_estimate_id, preview_id = self.servo.begin_phase(descent_active)
        return preview_id

    def readback(self) -> np.ndarray:
        """The servos' real-frame report, filled in from the last command.

        A partial observation keeps its missing joints at what was commanded
        rather than silently reading zero.
        """
        return action_to_joints(self.follower.get_observation(), self._commanded)

    def measured(self) -> tuple[dict[str, float], float]:
        """One motor read, mapped into the sim frame through the session calibration."""
        return measured_sim_state(self.model, self.readback(), self.joint_offsets_deg)

    def sighting(self, believed_cube: CubePose) -> Sighting:
        """Whatever the detector thread has most recently produced.

        ``believed_cube`` is unused here: the shadow is already posed at the
        commanded joints by :meth:`step`, so the camera pose the solve is
        projected through is simply read off it.
        """
        if self.servo is None:
            return NOTHING_SEEN
        estimate, self.last_preview = self.servo.sample(
            self.data.cam_xpos[self.wrist_camera_id].copy(),
            self.data.cam_xmat[self.wrist_camera_id].reshape(3, 3).copy(),
        )
        if estimate is None:
            return NOTHING_SEEN
        fresh = estimate.frame_id != self._last_estimate_id
        self._last_estimate_id = estimate.frame_id
        return Sighting(pose=estimate.source, fresh=fresh, estimate=estimate)

    def set_believed_cube(self, pose: CubePose) -> None:
        """Move the cube in the shadow, so the rendered view matches the belief."""
        set_cube_pose(self.model, self.data, pose)

    def new_contacts(self) -> set[tuple[str, str]]:
        return unexpected_contact_pairs(
            self.model, self.data, self.robot_geom_ids, self.env_geom_ids
        )

    def close(self) -> None:
        if self.servo is not None:
            self.servo.close()
