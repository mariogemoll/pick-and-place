# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The simulator as a plant: a true world, and a believed shadow over it.

Physics runs the true joints; commands and readback live in the believed frame,
so a controller sees exactly what it would see on a rig whose servo zeros are
off. Without a miscalibration draw the two frames coincide and the whole thing
degenerates to feedforward playback, which is what a plain recording wants.

Wrist localization runs **inline**, once per tick, rather than on a thread.
Running it in the loop is what keeps a recorded episode a pure function of its
seed; the selected localizer may use geometric camera-relative feedback or the
rendered AprilTag path.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.plant.interface import NOTHING_SEEN, Observation, Sighting
from pick_and_place.runtime.believed_frame import BelievedFrame
from pick_and_place.runtime.sim_wrist_servo import SimWristServo
from pick_and_place.sim.collisions import unexpected_contact_pairs
from pick_and_place.sim.model import get_cube_pose, get_cube_qpos, get_joint
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
        tracking_bias_rad: dict[str, float] | None = None,
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
        # Where a commanded joint actually settles. A *tracking* error, so it
        # goes into the command and stays out of the readback: the arm really
        # does fall short of what it was asked for, and reports that truthfully.
        self.tracking_bias_rad = tracking_bias_rad or {}
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

    @property
    def has_wrist_camera(self) -> bool:
        return self.servo is not None

    def begin_phase(self, descent_active: bool) -> None:
        """Nothing to arm: the detector runs inline, on demand, and holds no state."""

    def resync_clock(self) -> None:
        """Nothing to resync: the clock is simulated time, which does not run away."""
        self._tick_started = time.monotonic()

    def observe(self) -> Observation:
        """The tick's world in both frames, plus the cube's privileged true pose.

        Read once and handed to everything that wants it. Two readers would
        drift: the pan zero wanders with the clock, so a second read is a second
        draw of the offsets that separate the frames.
        """
        true_state, believed_state = self.belief.state_pair()
        return Observation(
            state=believed_state,
            true_state=true_state,
            true_cube_pose=get_cube_qpos(self.model, self.data),
        )

    def to_real(self, joints: Mapping[str, float], gripper: float) -> np.ndarray:
        """The command as a real-frame six-vector. Nothing clamps it: it was preflighted."""
        return sim_frame_to_real(joints, gripper)

    def step(self, joints: Mapping[str, float], gripper: float) -> np.ndarray:
        """Write the set point into the actuators and advance one control tick.

        Two things are folded in on the way, and they are not the same thing. The
        drawn joint-zero offsets put physics in the true frame while the plan and
        the recorded rows stay in the believed one — a servo commanded ``theta``
        rests at ``theta + offset`` and reports ``theta``. The tracking bias is a
        droop the arm never corrects and never hides, so it moves where the joint
        settles and the readback follows it.
        """
        offsets = self.belief.offsets_rad()
        for name, value in joints.items():
            self.data.ctrl[self.actuator_id[name]] = (
                value + offsets.get(name, 0.0) + self.tracking_bias_rad.get(name, 0.0)
            )
        self.data.ctrl[self.actuator_id["gripper"]] = gripper
        mujoco.mj_step(self.model, self.data, nstep=self.substeps_per_tick)
        if self.realtime:
            remaining = 1.0 / CONTROL_HZ - (time.monotonic() - self._tick_started)
            if remaining > 0:
                time.sleep(remaining)
        self._tick_started = time.monotonic()
        return self.to_real(joints, gripper)

    def measured(self) -> tuple[dict[str, float], float]:
        """The servo-style readback: true joints minus the offsets in effect."""
        return self.belief.arm_joints(), get_joint(self.model, self.data, "gripper")

    def sighting(self, believed_cube: CubePose) -> Sighting:
        """Locate the cube for wrist-servo feedback, inline.

        Both modes express the cube relative to the true camera and map the
        result through the believed shadow's camera pose, so a perturbed mount
        remains a hand-eye error. Detector mode obtains that relation from a
        rendered tag; geometric mode obtains it directly from simulation.
        """
        if self.servo is None:
            return NOTHING_SEEN
        if self.servo.mode == "geometric":
            seen = self.servo.geometric_sighting(
                get_cube_pose(self.model, self.data), believed_cube
            )
            return Sighting(pose=seen.pose, fresh=True)
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
