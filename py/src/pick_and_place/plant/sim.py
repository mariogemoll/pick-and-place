# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The simulator as a plant: a true world, and a believed shadow over it.

Physics runs the true joints; commands and readback live in the believed frame,
so a controller sees exactly what it would see on a rig whose servo zeros are
off. Without a miscalibration draw the two frames coincide and the whole thing
degenerates to feedforward playback, which is what a plain recording wants.

The detector runs **inline**, once per tick, rather than on a thread. Frames do
not arrive on their own here — they cost what they cost — and running them in
the loop is what keeps a recorded episode a pure function of its seed.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.plant.interface import NOTHING_SEEN, Sighting
from pick_and_place.runtime.believed_frame import BelievedFrame
from pick_and_place.runtime.sim_wrist_servo import SimWristServo
from pick_and_place.sim.collisions import unexpected_contact_pairs
from pick_and_place.sim.model import get_joint
from pick_and_place.spec.robot import CONTROL_HZ


class SimPlant:
    """Commands ``data.ctrl``, steps physics, and renders the wrist camera itself.

    ``speed`` scales trajectory time against simulated time; ``realtime`` sleeps
    out the rest of each tick so a viewer runs at the control rate, which a
    recording does not want.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        belief: BelievedFrame,
        actuator_id: dict[str, int],
        robot_geom_ids: Any,
        env_geom_ids: Any,
        kinematics: Any,
        substeps_per_tick: int,
        servo: SimWristServo | None = None,
        speed: float = 1.0,
        realtime: bool = False,
    ) -> None:
        self.model = model
        self.data = data
        self.belief = belief
        self.actuator_id = actuator_id
        self.robot_geom_ids = robot_geom_ids
        self.env_geom_ids = env_geom_ids
        self.kinematics = kinematics
        self.substeps_per_tick = substeps_per_tick
        self.servo = servo
        self.speed = speed
        self.realtime = realtime
        self._tick_started = time.monotonic()
        #: The last wrist render and the believed camera pose it was solved
        #: against. Kept because the mixed-view debugger needs both, and only
        #: this class ever has them.
        self.last_look: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def time(self) -> float:
        """Simulated time. A phase advances at the rate physics does."""
        return float(self.data.time)

    def step(self, joints: Mapping[str, float], gripper: float) -> np.ndarray:
        """Write the set point into the actuators and advance one control tick.

        The drawn joint-zero offsets are added on the way in, which is what puts
        physics in the true frame while the plan and the recorded rows stay in
        the believed one — a servo commanded ``theta`` rests at ``theta + offset``.
        """
        offsets = self.belief.offsets_rad()
        for name, value in joints.items():
            self.data.ctrl[self.actuator_id[name]] = value + offsets.get(name, 0.0)
        self.data.ctrl[self.actuator_id["gripper"]] = gripper
        mujoco.mj_step(self.model, self.data, nstep=self.substeps_per_tick)
        if self.realtime:
            remaining = 1.0 / CONTROL_HZ - (time.monotonic() - self._tick_started)
            if remaining > 0:
                time.sleep(remaining)
        self._tick_started = time.monotonic()
        return sim_frame_to_real(joints, gripper)

    def measured(self) -> tuple[dict[str, float], float]:
        """The servo-style readback: true joints minus the offsets in effect."""
        return self.belief.arm_joints(), get_joint(self.model, self.data, "gripper")

    def sighting(self, believed_cube: CubePose) -> Sighting:
        """Render the wrist camera and solve the cube out of it, inline.

        The image comes from the true world — including a perturbed physical
        camera mount, if one was drawn — while the solve is projected through the
        believed shadow's camera pose, so the estimate carries the hand-eye error
        exactly as on hardware.
        """
        if self.servo is None:
            return NOTHING_SEEN
        rgb, camera_position, camera_rotation = self.servo.look(believed_cube)
        self.last_look = (rgb, camera_position, camera_rotation)
        seen = self.servo.solve(rgb, camera_position, camera_rotation)
        return Sighting(
            pose=seen.pose,
            # Rendered and solved this tick, so a pose is always a new one.
            fresh=seen.pose is not None,
            detections=seen.detections,
            estimate=seen.estimate,
        )

    def set_believed_cube(self, pose: CubePose) -> None:
        """A no-op: the simulator's shadow is posed per sighting, not carried."""

    def new_contacts(self) -> set[tuple[str, str]]:
        return unexpected_contact_pairs(
            self.model, self.data, self.robot_geom_ids, self.env_geom_ids
        )

    def close(self) -> None:
        if self.servo is not None:
            self.servo.close()
