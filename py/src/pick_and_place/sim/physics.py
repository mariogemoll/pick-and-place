# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Give a compiled scene the arm one episode drew, and take it back afterwards.

Applied to a *compiled* model rather than to the spec, because a recording run
keeps one persistent scene across every episode — recompiling per draw would
cost more than the episode does and would break a viewer bound to the model.

**This has to run before the trajectory is preflighted.** Preflight vets a
candidate under live physics, so vetting it against nominal physics when the
episode will run under a draw is checking a different world than the one that
follows. The order is: draw, apply, plan, preflight, run.

The tracking bias is deliberately not applied here. It is not a property of the
model — a MuJoCo position actuator at kp 998 settles within a fortieth of a
degree of its command whatever its mass is — it is a property of the *command*,
so the plant folds it in on the way to ``data.ctrl`` and leaves the readback
alone. A drooping servo reports where it really is.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pick_and_place.core.physics import PhysicsDraw
from pick_and_place.spec.robot import JOINT_NAMES


class PhysicsRandomizer:
    """Applies a physics draw to a compiled model, and restores it between draws.

    Restoring matters as much as applying: a scene is reused across episodes, so
    a scale that is not undone compounds onto the next draw and the envelope
    silently widens.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._body_mass = model.body_mass.copy()
        self._body_inertia = model.body_inertia.copy()
        self._geom_friction = model.geom_friction.copy()
        self._dof_damping = model.dof_damping.copy()
        self._dof_frictionloss = model.dof_frictionloss.copy()
        self._actuator_gainprm = model.actuator_gainprm.copy()
        self._actuator_biasprm = model.actuator_biasprm.copy()
        self._actuator_dynprm = model.actuator_dynprm.copy()
        self._actuator_id = {
            name: index
            for name in JOINT_NAMES
            if (index := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)) >= 0
        }
        self._dof_adr = {
            name: int(model.jnt_dofadr[joint])
            for name in JOINT_NAMES
            if (joint := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) >= 0
        }

    def reset(self) -> None:
        """Restore the arm the model compiled with."""
        model = self.model
        model.body_mass[:] = self._body_mass
        model.body_inertia[:] = self._body_inertia
        model.geom_friction[:] = self._geom_friction
        model.dof_damping[:] = self._dof_damping
        model.dof_frictionloss[:] = self._dof_frictionloss
        model.actuator_gainprm[:] = self._actuator_gainprm
        model.actuator_biasprm[:] = self._actuator_biasprm
        model.actuator_dynprm[:] = self._actuator_dynprm

    def apply(self, draw: PhysicsDraw) -> None:
        """Scale the model onto this episode's arm."""
        self.reset()
        if draw.is_nominal:
            return
        model = self.model

        # Mass and inertia together: scaling one without the other builds an arm
        # whose links weigh more but spin as though they did not.
        model.body_mass[:] = self._body_mass * draw.mass_scale
        model.body_inertia[:] = self._body_inertia * draw.mass_scale
        # Sliding, torsional and rolling friction all at once — a grippier
        # surface is grippier in every mode, and a cube that will not slide but
        # spins freely is not a surface anything has.
        model.geom_friction[:] = self._geom_friction * draw.friction_scale
        model.dof_damping[:] = self._dof_damping * draw.damping_scale

        for name, index in self._actuator_id.items():
            gain = draw.joint_gain_scale.get(name, 1.0)
            # A position actuator holds its gain in gainprm[0] and the matching
            # negative in biasprm[1]; scaling one alone turns it into a servo
            # that pulls harder than it resists, which is not a servo.
            model.actuator_gainprm[index, 0] = self._actuator_gainprm[index, 0] * gain
            model.actuator_biasprm[index, 1] = self._actuator_biasprm[index, 1] * gain
            model.actuator_dynprm[index, 0] = self._actuator_dynprm[
                index, 0
            ] * draw.joint_time_constant_scale.get(name, 1.0)

        for name, address in self._dof_adr.items():
            # Added to what the joint already has, not substituted for it: the
            # stock arm carries 0.052 and overwriting that with a small draw
            # would make a randomized arm *freer* than the nominal one.
            #
            # Only the arm's own joints. Dry friction on the cube's freejoint
            # would stop it sliding on the table, which is a property of the
            # table and is already covered by geom friction.
            model.dof_frictionloss[address] = self._dof_frictionloss[
                address
            ] + draw.extra_joint_friction.get(name, 0.0)


def tracking_bias_offsets(
    bias_rad: dict[str, float], draw: PhysicsDraw
) -> dict[str, float]:
    """This episode's droop: the fitted bias, scaled by what the draw asked for.

    A *tracking* error, not a sensing one. The arm genuinely fails to reach what
    it was asked for and reports that truthfully, so the command is pushed past
    the target and the readback is left alone — the opposite of a joint-zero
    offset, which is corrected out of the readback because the arm is where it
    was asked to be and only thinks otherwise.
    """
    if draw.tracking_bias_scale == 0.0:
        return {}
    return {name: value * draw.tracking_bias_scale for name, value in bias_rad.items()}


def actuator_gains(model: mujoco.MjModel) -> np.ndarray:
    """Each actuator's position gain, for a test or a report to read back."""
    return model.actuator_gainprm[:, 0].copy()
