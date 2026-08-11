# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Helpers for applying fitted real-robot joint response in simulation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from pick_and_place.spec.robot import ARM_JOINT_NAMES

DEFAULT_ROBOT_DYNAMICS_PATH = (
    Path(__file__).resolve().parents[4] / "config" / "robot_dynamics" / "so101_follower.json"
)


def load_robot_dynamics_config(path: str | Path = DEFAULT_ROBOT_DYNAMICS_PATH) -> dict:
    """Load a fitted robot-dynamics JSON artifact."""
    return json.loads(Path(path).read_text())


def tracking_bias_rad(config: dict, *, scale: float = 1.0) -> dict[str, float]:
    """Where each arm joint settles relative to the position it was commanded.

    The fit behind `config/robot_dynamics/so101_follower.json` solves
    ``state[t+1] = state[t] + alpha * (action[t-delay] - state[t]) + beta``, so
    a joint holding still satisfies ``state = action + beta / alpha`` -- the
    ``steady_state_bias`` the artifact records. The real arm settles 2.16
    degrees from its shoulder_lift command and 1.37 from its elbow_flex one.
    Simulation reproduces none of that: its position actuators carry
    ``kp = 998`` against gravity torques of order a newton-metre, so a settled
    joint sits within 0.04 degrees of its command.

    **This is not the joint-zero offset in `miscalibration`, and the two must
    not be added to the same quantity.** A joint zero is a *sensing* error: the
    arm sits somewhere other than it believes, and the readback has to be
    corrected to match. This is a *tracking* error: the arm really does not
    reach what it was asked for, and it reports that truthfully. A policy
    emitting absolute joint targets has to push past the target to land on it,
    which is behavior it can only learn from data where commands fall short.

    That the bias is gravity droop rather than an artifact of a follower chasing
    a moving leader is what the fit's own pattern argues: shoulder_lift and
    elbow_flex carry the load and fit at 2.16 and 1.37 degrees, while
    shoulder_pan, whose axis is vertical and which therefore holds no gravity
    torque, fits at 0.03.

    ``scale`` multiplies every offset, which is how an episode draws an arm that
    droops more or less than the one that was measured. The gripper is excluded:
    the artifact labels every joint "degrees", but the gripper was fit in the
    hardware's 0-100 position units, so converting its value as an angle would
    be a unit error for a term worth 0.1 of those units anyway.
    """
    return {
        name: math.radians(value) for name, value in tracking_bias_deg(config, scale=scale).items()
    }


def tracking_bias_deg(config: dict, *, scale: float = 1.0) -> dict[str, float]:
    """The same bias in the hardware frame, which is where it was fitted.

    The arm joints map between frames by nothing but a degree-to-radian
    conversion, so this is the more primitive of the two. It is what the
    hardware runner needs: subtracting it from a command makes the servo settle
    on what the policy asked for instead of a bias away from it, which is the
    correction that makes a real arm behave like the simulated one a policy was
    trained against.
    """
    joints = config["joints"]
    return {
        name: float(joints[name]["steady_state_bias"]) * scale
        for name in ARM_JOINT_NAMES
        if name in joints
    }


def tracking_bias_vector(bias_rad: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    """Order a bias mapping into a control vector, zero where a joint is absent."""
    return np.array([bias_rad.get(name, 0.0) for name in names])


def set_actuator_activation(model, data, actuator_id: int, value: float) -> None:
    """Seed a filtered actuator's activation to match its current control.

    MuJoCo ``position`` actuators with ``timeconst`` filter ``ctrl`` through an
    activation state. Initialising that state avoids a fake startup transient
    from zero whenever a scene is reset directly to a pose.
    """
    actadr = int(model.actuator_actadr[actuator_id])
    if actadr >= 0:
        data.act[actadr] = float(value)
