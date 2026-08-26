# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Prepare an episode the way the rig does: look first, then plan on what was seen.

The difference from :func:`~pick_and_place.runtime.episodes.prepare_episode` is
one of *provenance*, not of arithmetic. There, the planner's belief is the truth
plus a drawn error — a number applied to a pose nothing ever measured. Here the
scene is set up, the overhead camera is rendered, the detector runs, and
whatever comes out is what the planner gets. The error is then an outcome of a
calibration that is slightly wrong, which is what it is on the rig.

Two things follow from that and are visible in the loop below.

**The plate has to be on the table before anything is localized**, so the drop
zone is placed here rather than by the caller afterwards. Its yaw is drawn here
too, for the same reason: the detector sees a rotated square or it does not.

**A look can fail.** The arm stands between the camera and the cube, or the cube
sits on the plate and breaks its square. On the rig the remedy is to pan through
search poses and look again; when even that does not work, the episode is
resampled. Both happen here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.workspace_bounds import is_cube_drop_allowed, sample_target_plate_yaw
from pick_and_place.plant.overhead import OverheadReading, SimOverheadPerception
from pick_and_place.runtime.episodes import Episode, EpisodeSamplingError, prepare_episode
from pick_and_place.scripted.episode_sampling import sample_cube, sample_hunt_pose, sample_target
from pick_and_place.sim.model import set_cube_pose, set_joint
from pick_and_place.sim.paper_target_marker import place_paper_target_marker
from pick_and_place.spec.workspace import CUBE_REST_Z, DROP_ZONE_HALF_SIZE

#: Search poses to try before giving up on a scene and resampling it. The rig
#: pans until it sees; a generator cannot pan forever, and a scene that stays
#: hidden after this many looks is usually one where the cube is sitting on the
#: plate rather than one where the arm happens to be in the way.
MAX_HUNT_POSES = 8

#: How hard to try to plan one localized scene before drawing another. Small,
#: because the cube and plate are pinned to what was measured, so the planner
#: can only vary the start and end pose — if that does not find a clean
#: trajectory in a few goes, a different scene is the cheaper next move.
PLAN_ATTEMPTS_PER_SCENE = 8


@dataclass(frozen=True)
class LocalizedEpisode:
    """A prepared episode, plus what the look that produced it cost and saw."""

    episode: Episode
    target_plate_yaw: float
    hunts: int
    reading: OverheadReading


def localize(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: SimOverheadPerception,
    rng: np.random.Generator,
    *,
    max_hunts: int = MAX_HUNT_POSES,
) -> tuple[OverheadReading, int] | None:
    """Pan through search poses until the cube and the plate are both visible.

    Returns ``None`` when they never both were. The arm is left wherever the
    successful look put it, which is exactly where the rig would be: the episode
    starts from the pose it was localized from.
    """
    perception.reset()
    for hunts in range(max_hunts):
        arm_joints, gripper = sample_hunt_pose(rng)
        for name, value in arm_joints.items():
            set_joint(model, data, name, value)
        set_joint(model, data, "gripper", gripper)
        mujoco.mj_forward(model, data)
        reading = perception.look()
        if reading.complete:
            return reading, hunts
    return None


def prepare_localized_episode(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: SimOverheadPerception,
    *,
    source: CubePose | None = None,
    target: CubePose | None = None,
    target_sampler: Callable[[np.random.Generator], CubePose] | None = None,
    ground_truth_drop_target: bool = False,
    max_attempts: int = 20,
    plan_attempts: int = PLAN_ATTEMPTS_PER_SCENE,
    max_hunts: int = MAX_HUNT_POSES,
    verbose: bool = False,
    **kwargs: Any,
) -> LocalizedEpisode:
    """Sample a scene, localize it from above, and plan against what was localized.

    Every attempt draws a whole scene — cube, plate, plate yaw — because a scene
    that could not be seen is not one to retry with a different arm pose; the
    hunt already did that.

    The planner gets a *small* budget per scene. Its own resampling loop cannot
    help here — the cube and plate are pinned to the poses the belief was
    measured from, so all it can vary is the start and end pose — and left
    unbounded it will spin on an unplannable scene forever rather than let the
    outer loop draw a better one.
    """
    for attempt in range(1, max_attempts + 1):
        ep_source = source if source is not None else sample_cube(rng)
        ep_target = target if target is not None else (target_sampler or sample_target)(rng)
        plate_yaw = sample_target_plate_yaw(
            rng, ep_target.x, ep_target.y, half_size=DROP_ZONE_HALF_SIZE
        )

        set_cube_pose(model, data, ep_source)
        place_paper_target_marker(
            model,
            (ep_target.x, ep_target.y),
            plate_yaw,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(ep_target.x, ep_target.y),
            alpha=1.0,
        )

        looked = localize(model, data, perception, rng, max_hunts=max_hunts)
        if looked is None:
            if verbose:
                print(f"attempt {attempt}: could not see the cube and the plate together")
            continue
        reading, hunts = looked

        try:
            episode = prepare_episode(
                rng,
                ep_source,
                ep_target,
                model=model,
                data=data,
                believed_source=reading.cube,
                # The plate is still localized -- a scene neither camera can see
                # is still rejected, and the cube belief stays honest because the
                # descent servo is what corrects the pickup. Only the *drop* is
                # planned against truth, because nothing corrects the drop: the
                # overhead estimate's error lands directly in placement error.
                believed_target=(
                    ep_target
                    if ground_truth_drop_target
                    else CubePose(
                        x=reading.target.xy[0], y=reading.target.xy[1], z=CUBE_REST_Z
                    )
                ),
                max_attempts=plan_attempts,
                verbose=verbose,
                **kwargs,
            )
        except EpisodeSamplingError as exc:
            if verbose:
                print(f"attempt {attempt}: {exc}")
            continue
        return LocalizedEpisode(
            episode=episode, target_plate_yaw=plate_yaw, hunts=hunts, reading=reading
        )

    raise EpisodeSamplingError(
        f"no scene both localized and planned in {max_attempts} attempts"
    )
