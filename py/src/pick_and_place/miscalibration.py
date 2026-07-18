# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measured real-robot miscalibration, drawn per episode for injection into sim.

The real system's state estimate is systematically wrong in ways measured from
recorded episodes (see py/SIM2REAL.md): the servo joint zeros are offset from
the model frame and drift day to day, and the overhead cube/target localization
is off by millimetres. Sim episode generation and the RL env inject draws from
those measured distributions so that sim separates *true* state (what physics
and rendering use) from *believed* state (what the planner or policy acts on),
making open-loop reaching in sim miss the way real reaching misses.

Sign conventions match the session calibration (``follower.py``): a joint whose
servo command/readback reads ``theta`` sits physically at model angle
``theta + offset``. Injection therefore applies ``true = commanded + offset``
on the way into physics and ``believed = measured - offset`` on the way out.
The believed cube/target pose is ``true + error``.

The default joint-offset draws are zero-mean: real sessions run through the
session-start calibration, so the offset that remains at run time is the
calibration residual plus drift, whose spread is the measured day-to-day sigma.
A nonzero mean would only relabel the command frame — it is common to every
episode, so no policy can observe it. Set ``joint_offset_mean_deg`` to the
measured per-day means (pan ~+4.3 deg, elbow ~-3.6 deg) only to model deploying
against raw, uncalibrated servos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from pick_and_place.geometry import CubePose

# Fitted arm joints. wrist_roll is not observable by the hand-eye fit (it spins
# the camera about its own axis), so it carries no measured spread.
FITTED_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

# Day-to-day spread of the per-day fitted joint zeros over 20260701-04, degrees.
# The elbow value follows the SIM2REAL finding (~0.5 deg over the three
# well-sampled days; the 5-episode 20260704 fit is an outlier).
DEFAULT_JOINT_OFFSET_SIGMA_DEG = {
    "shoulder_pan": 1.5,
    "shoulder_lift": 1.0,
    "elbow_flex": 0.55,
    "wrist_flex": 1.8,
}

# Within-day spread of the pan zero (per-frame std of the offline fits),
# injected as a slowly wandering component on top of the per-episode constant.
DEFAULT_PAN_JITTER_SIGMA_DEG = 2.2
DEFAULT_PAN_JITTER_TAU_S = 10.0

# Believed-cube-pose error vs true (overhead localization + physical frame
# placement class of error): ~6-9 mm planar magnitude, ~3-5 mm vertical.
DEFAULT_CUBE_BELIEF_SIGMA_XY_M = 0.006
DEFAULT_CUBE_BELIEF_SIGMA_Z_M = 0.004
DEFAULT_CUBE_BELIEF_SIGMA_YAW_RAD = math.radians(2.0)
# The drop target is localized through the same overhead chain.
DEFAULT_TARGET_BELIEF_SIGMA_XY_M = 0.006


class SlowJitter:
    """Stationary Ornstein-Uhlenbeck process sampled at monotone times.

    Models the slow within-session wander of a joint zero: standard deviation
    ``sigma`` and correlation time ``tau`` seconds. ``value(t)`` may be called
    with any non-decreasing sequence of times.
    """

    def __init__(self, sigma: float, tau: float, rng: np.random.Generator) -> None:
        self._sigma = float(sigma)
        self._tau = float(tau)
        self._rng = rng
        self._t: float | None = None
        self._x = float(rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0

    def value(self, t: float) -> float:
        if self._sigma <= 0.0:
            return 0.0
        if self._t is None:
            self._t = float(t)
            return self._x
        dt = max(0.0, float(t) - self._t)
        self._t = float(t)
        if dt > 0.0:
            decay = math.exp(-dt / self._tau)
            noise = self._sigma * math.sqrt(1.0 - decay * decay)
            self._x = self._x * decay + float(self._rng.normal(0.0, 1.0)) * noise
        return self._x


@dataclass
class MiscalibrationDraw:
    """One episode's realization of the miscalibration model.

    ``base_offsets_deg`` is the per-episode constant joint-zero offset (the
    "add to the sim joints" sense); the pan additionally wanders via
    ``pan_jitter``. The belief errors are the constant per-episode offsets of
    the believed cube/target poses from the true ones.
    """

    base_offsets_deg: dict[str, float]
    pan_jitter: SlowJitter | None
    cube_belief_error: tuple[float, float, float, float]  # dx, dy, dz, dyaw
    target_belief_error: tuple[float, float]  # dx, dy

    def offsets_deg(self, t: float = 0.0) -> dict[str, float]:
        """Joint-zero offsets (degrees) in effect at episode time ``t``."""
        offsets = dict(self.base_offsets_deg)
        if self.pan_jitter is not None:
            offsets["shoulder_pan"] = (
                offsets.get("shoulder_pan", 0.0) + self.pan_jitter.value(t)
            )
        return offsets

    def offsets_rad(self, t: float = 0.0) -> dict[str, float]:
        return {name: math.radians(v) for name, v in self.offsets_deg(t).items()}

    def believe_cube(self, true_pose: CubePose) -> CubePose:
        """The cube pose the planner believes, given the true one."""
        dx, dy, dz, dyaw = self.cube_belief_error
        return replace(
            true_pose,
            x=true_pose.x + dx,
            y=true_pose.y + dy,
            z=true_pose.z + dz,
            yaw=true_pose.yaw + dyaw,
        )

    def believe_target(self, true_pose: CubePose) -> CubePose:
        """The drop target the planner believes, given the true one."""
        dx, dy = self.target_belief_error
        return replace(true_pose, x=true_pose.x + dx, y=true_pose.y + dy)


@dataclass(frozen=True)
class MiscalibrationModel:
    """Distributions of the measured miscalibration; ``sample`` draws an episode."""

    joint_offset_mean_deg: dict[str, float] = field(default_factory=dict)
    joint_offset_sigma_deg: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_OFFSET_SIGMA_DEG)
    )
    pan_jitter_sigma_deg: float = DEFAULT_PAN_JITTER_SIGMA_DEG
    pan_jitter_tau_s: float = DEFAULT_PAN_JITTER_TAU_S
    cube_belief_sigma_xy_m: float = DEFAULT_CUBE_BELIEF_SIGMA_XY_M
    cube_belief_sigma_z_m: float = DEFAULT_CUBE_BELIEF_SIGMA_Z_M
    cube_belief_sigma_yaw_rad: float = DEFAULT_CUBE_BELIEF_SIGMA_YAW_RAD
    target_belief_sigma_xy_m: float = DEFAULT_TARGET_BELIEF_SIGMA_XY_M

    def sample(self, rng: np.random.Generator) -> MiscalibrationDraw:
        # Sorted so a seeded rng assigns the same draw to the same joint on
        # every run (set order varies with the process hash seed).
        joint_names = sorted(
            set(self.joint_offset_sigma_deg) | set(self.joint_offset_mean_deg)
        )
        base = {
            name: self.joint_offset_mean_deg.get(name, 0.0)
            + float(rng.normal(0.0, self.joint_offset_sigma_deg.get(name, 0.0)))
            for name in joint_names
        }
        jitter = (
            SlowJitter(
                self.pan_jitter_sigma_deg,
                self.pan_jitter_tau_s,
                np.random.default_rng(rng.integers(2**63)),
            )
            if self.pan_jitter_sigma_deg > 0.0
            else None
        )
        return MiscalibrationDraw(
            base_offsets_deg=base,
            pan_jitter=jitter,
            cube_belief_error=(
                float(rng.normal(0.0, self.cube_belief_sigma_xy_m)),
                float(rng.normal(0.0, self.cube_belief_sigma_xy_m)),
                float(rng.normal(0.0, self.cube_belief_sigma_z_m)),
                float(rng.normal(0.0, self.cube_belief_sigma_yaw_rad)),
            ),
            target_belief_error=(
                float(rng.normal(0.0, self.target_belief_sigma_xy_m)),
                float(rng.normal(0.0, self.target_belief_sigma_xy_m)),
            ),
        )
