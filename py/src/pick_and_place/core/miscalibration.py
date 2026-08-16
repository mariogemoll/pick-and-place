# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measured real-robot miscalibration, drawn per episode for injection into sim.

The real system's state estimate is systematically wrong in ways measured from
recorded episodes: the servo joint zeros are offset from
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

from pick_and_place.core.geometry import CubePose

# Fitted arm joints. wrist_roll is not observable by the hand-eye fit (it spins
# the camera about its own axis), so it carries no measured spread.
FITTED_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")

# Day-to-day spread of the per-day fitted joint zeros over 20260701-04, degrees.
# The elbow value follows the measured spread (~0.5 deg over the three
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
#
# This is an *outcome*, not a cause. Injecting it directly is what a run does
# when nothing simulates the overhead camera; when something does, the causes
# below are perturbed instead and this becomes the target the emergent error is
# checked against.
DEFAULT_CUBE_BELIEF_SIGMA_XY_M = 0.006
DEFAULT_CUBE_BELIEF_SIGMA_Z_M = 0.004
DEFAULT_CUBE_BELIEF_SIGMA_YAW_RAD = math.radians(2.0)
# The drop target is localized through the same overhead chain.
DEFAULT_TARGET_BELIEF_SIGMA_XY_M = 0.006

# The causes of that outcome, both of which separate where the overhead camera
# is believed to be from where it actually sits.
#
# These are *residuals*, and that is the whole reason they are so much smaller
# than the raw figures they come from. The camera moves 25.6 mm and 1.9 degrees
# across the measured days and the physical workspace frame sits about 14 mm
# from where the model authors it — but the extrinsics are re-solved every
# session against that same frame, so most of both is measured rather than
# suffered. What is left over is what reaches the planner.
#
# How much is left over cannot be read off the rig directly, so it is fitted the
# other way round: these are the values at which simulated render-and-detect
# reproduces the 6-9 mm the rig actually misses by. See
# ``pick_and_place.plant.overhead_check``, which is the measurement, and
# ``scripts/check_overhead_localization.py``, which runs it.
DEFAULT_OVERHEAD_EXTRINSICS_SIGMA_M = 0.0045
DEFAULT_OVERHEAD_EXTRINSICS_SIGMA_DEG = 0.32
DEFAULT_WORKSPACE_FRAME_SIGMA_M = 0.0024


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


@dataclass(frozen=True)
class OverheadCameraError:
    """How far the overhead camera's calibration is from where it physically sits.

    Two causes, one effect. Whether the camera itself moved between sessions or
    the frame it was calibrated against is not where the model says, what
    reaches the planner is the same thing: poses solved through a camera pose
    that is slightly wrong. So they compose into one rigid offset from the true
    pose to the believed one.

    Injecting the causes rather than the outcome is the whole point. An honest
    render-and-detect in a clean scene would localize *better* than the real
    rig, because in sim the extrinsics are exact and the frame is exactly where
    the model puts it. Perturb the causes and the resulting localization error
    is something that can be measured and compared against the rig's, instead of
    a number that was assumed.
    """

    position_m: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    #: How much of ``position_m`` came from the frame sitting off its authored
    #: place, kept separate so a check can attribute the emergent error.
    frame_placement_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


#: A camera exactly where its calibration says it is.
NO_OVERHEAD_ERROR = OverheadCameraError((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class OverheadCameraModel:
    """The distribution the overhead calibration's residual error is drawn from.

    Separate from :class:`MiscalibrationModel`, and drawn from its own stream,
    for the reason every other draw here is: a camera sigma must not move any
    episode's cube. It also belongs to the *session* rather than to the arm —
    the extrinsics are solved once and hold for everything that follows.
    """

    extrinsics_sigma_m: float = DEFAULT_OVERHEAD_EXTRINSICS_SIGMA_M
    extrinsics_sigma_deg: float = DEFAULT_OVERHEAD_EXTRINSICS_SIGMA_DEG
    workspace_frame_sigma_m: float = DEFAULT_WORKSPACE_FRAME_SIGMA_M

    def sample(self, rng: np.random.Generator) -> OverheadCameraError:
        """Where the overhead camera is believed to be, relative to where it is.

        The frame placement error is drawn in the table plane only: the frame is
        a flat printed fixture that sits *on* the table, so it can be laid down
        in the wrong place but not at the wrong height.
        """
        extrinsics = rng.normal(0.0, self.extrinsics_sigma_m, size=3)
        frame = np.array(
            [
                rng.normal(0.0, self.workspace_frame_sigma_m),
                rng.normal(0.0, self.workspace_frame_sigma_m),
                0.0,
            ]
        )
        return OverheadCameraError(
            position_m=tuple(float(v) for v in extrinsics + frame),
            rotation_deg=tuple(
                float(rng.normal(0.0, self.extrinsics_sigma_deg)) for _ in range(3)
            ),
            frame_placement_m=tuple(float(v) for v in frame),
        )


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
