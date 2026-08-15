# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Measure the localization error simulated overhead perception actually produces.

This is the check that stops the new path from quietly being better than
reality. Rendering and detecting for real removes the injected error; putting
the *causes* back in is supposed to reproduce it, and nothing about that is
guaranteed — the drawn extrinsics and frame-placement spreads are a hypothesis
about what makes the rig miss by the amount it misses by.

So draw episodes, localize them, and compare the outcome against what
SIM2REAL measured on the rig: about 6 mm planar on the cube, a few millimetres
vertical, a couple of degrees of yaw, and the same 6 mm on the drop plate. If
the simulated chain comes out much tighter than that, sim is easier than the
rig and the demonstrations will not teach the recovery the rig needs; much
wider and it is teaching a correction the rig never has to make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.miscalibration import MiscalibrationModel
from pick_and_place.core.workspace_bounds import is_cube_drop_allowed
from pick_and_place.plant.overhead import SimOverheadPerception
from pick_and_place.scripted.episode_sampling import sample_cube, sample_hunt_pose, sample_target
from pick_and_place.sim.model import set_cube_pose, set_joint
from pick_and_place.sim.paper_target_marker import place_paper_target_marker
from pick_and_place.spec.workspace import DROP_ZONE_HALF_SIZE

#: How many search poses to try before giving up on an episode, matching the
#: rig's own patience: the arm can stand between the camera and either object,
#: and panning away is the whole remedy.
MAX_HUNT_POSES = 8


@dataclass(frozen=True)
class LocalizationError:
    """One episode's miss, in the quantities SIM2REAL reports."""

    cube_xy_m: float
    cube_z_m: float
    cube_yaw_rad: float
    target_xy_m: float
    hunts: int


@dataclass(frozen=True)
class ErrorSummary:
    """The distribution of misses over a run, next to how often anything was seen."""

    episodes: int
    localized: int
    cube_xy_median_m: float
    cube_xy_p90_m: float
    cube_z_median_m: float
    cube_yaw_median_rad: float
    target_xy_median_m: float
    mean_hunts: float

    def summary(self) -> str:
        return (
            f"{self.localized}/{self.episodes} localized "
            f"({self.mean_hunts:.2f} extra search poses each)\n"
            f"  cube planar : median {self.cube_xy_median_m * 1000:.1f} mm, "
            f"p90 {self.cube_xy_p90_m * 1000:.1f} mm\n"
            f"  cube height : median {self.cube_z_median_m * 1000:.1f} mm\n"
            f"  cube yaw    : median {math.degrees(self.cube_yaw_median_rad):.2f} deg\n"
            f"  target      : median {self.target_xy_median_m * 1000:.1f} mm"
        )


def measure_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: SimOverheadPerception,
    rng: np.random.Generator,
    *,
    max_hunts: int = MAX_HUNT_POSES,
) -> LocalizationError | None:
    """Place a random cube and plate, hunt until both are visible, and measure the miss.

    ``None`` when every search pose left something hidden — which is a real
    outcome, not a failure of the measurement, and is reported as such.
    """
    cube = sample_cube(rng)
    target = sample_target(rng)
    set_cube_pose(model, data, cube)
    place_paper_target_marker(
        model,
        (target.x, target.y),
        0.0,
        (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
        usable=is_cube_drop_allowed(target.x, target.y),
        alpha=1.0,
    )
    perception.reset()
    for hunts in range(max_hunts):
        arm_joints, gripper = sample_hunt_pose(rng)
        for name, value in arm_joints.items():
            set_joint(model, data, name, value)
        set_joint(model, data, "gripper", gripper)
        mujoco.mj_forward(model, data)
        reading = perception.look()
        if reading.complete:
            return _miss(cube, target, reading, hunts)
    return None


def _miss(cube: CubePose, target: CubePose, reading, hunts: int) -> LocalizationError:
    seen = reading.cube
    target_x, target_y = reading.target.xy
    return LocalizationError(
        cube_xy_m=math.hypot(seen.x - cube.x, seen.y - cube.y),
        cube_z_m=abs(seen.z - cube.z),
        cube_yaw_rad=abs(fold_cube_symmetry(seen.yaw - cube.yaw)),
        target_xy_m=math.hypot(target_x - target.x, target_y - target.y),
        hunts=hunts,
    )


def fold_cube_symmetry(angle: float) -> float:
    """Fold an angle difference into the cube's own quarter-turn symmetry.

    A cube face is indistinguishable from the next one round, so a yaw that
    differs by 90 degrees is not an error of 90 degrees — it is the same pose.
    """
    quarter = math.pi / 2.0
    return (angle + quarter / 2.0) % quarter - quarter / 2.0


def summarize(errors: list[LocalizationError], attempted: int) -> ErrorSummary:
    if not errors:
        raise ValueError("no episode localized both the cube and the target")
    return ErrorSummary(
        episodes=attempted,
        localized=len(errors),
        cube_xy_median_m=float(np.median([e.cube_xy_m for e in errors])),
        cube_xy_p90_m=float(np.percentile([e.cube_xy_m for e in errors], 90)),
        cube_z_median_m=float(np.median([e.cube_z_m for e in errors])),
        cube_yaw_median_rad=float(np.median([e.cube_yaw_rad for e in errors])),
        target_xy_median_m=float(np.median([e.target_xy_m for e in errors])),
        mean_hunts=float(np.mean([e.hunts for e in errors])),
    )


def measure(
    build_scene,
    *,
    episodes: int,
    seed: int,
    model_sigmas: MiscalibrationModel = MiscalibrationModel(),
    detector=None,
) -> ErrorSummary:
    """Draw ``episodes`` overhead calibrations and report the error they produce.

    One camera error per episode, because that is how the rig varies: the
    extrinsics are solved at the start of a session and hold for it.
    """
    model, data = build_scene()
    errors: list[LocalizationError] = []
    perception = SimOverheadPerception(model, data, detector=detector)
    try:
        for index in range(episodes):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            perception.set_error(model_sigmas.sample(rng).overhead_camera_error)
            error = measure_episode(model, data, perception, rng)
            if error is not None:
                errors.append(error)
    finally:
        perception.close()
    return summarize(errors, episodes)
