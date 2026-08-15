# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Re-derive the rest of a trajectory from where the arm actually is.

Which phase boundaries get a checkpoint is the controller's policy and lives in
:mod:`pick_and_place.scripted.checkpoint`. Carrying one out is this module's job,
and it needs two capabilities the controller does not have: a compiled model to
map motor readback into, and live physics to preflight the candidates against.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import real_frame_to_sim
from pick_and_place.runtime.preflight import (
    preflight,
    preflight_collision_is_unexpected,
    save_failed_preflight_trajectory,
    write_failed_trajectory_note,
)
from pick_and_place.scripted.replan import replan_remaining_candidates
from pick_and_place.sim.collisions import is_unexpected, jaw_floor_clearance, jaw_geom_ids
from pick_and_place.sim.model import set_joint

#: Below this the measured pose has the jaws essentially on the floor, which is
#: worth saying out loud before replanning from it.
JAW_CLEARANCE_WARN_M = 0.005


def measured_sim_state(
    model: mujoco.MjModel,
    actual: np.ndarray,
    joint_offsets_deg: dict[str, float] | None,
) -> tuple[dict[str, float], float]:
    """Map motor readback into the sim frame, reporting jaws that read below the floor.

    The warning is not a failure: the replanner is about to be handed this state
    either way, and a small negative clearance is ordinary near contact. It is
    printed because it explains a replan that then fails preflight.
    """
    measured_joints, measured_gripper = real_frame_to_sim(actual, joint_offsets_deg)
    shadow = mujoco.MjData(model)
    for name, value in measured_joints.items():
        set_joint(model, shadow, name, value)
    set_joint(model, shadow, "gripper", measured_gripper)
    mujoco.mj_forward(model, shadow)
    clearance = jaw_floor_clearance(model, shadow, jaw_geom_ids(model))
    if clearance < JAW_CLEARANCE_WARN_M:
        print(f"Measured sim jaw clearance before replan: {clearance * 1000:+.1f} mm")
    return measured_joints, measured_gripper


def replan_from_checkpoint(
    model: mujoco.MjModel,
    *,
    kinematics,
    actuator_id: dict[str, int],
    robot_geom_ids,
    env_geom_ids,
    measured_joints: dict[str, float],
    measured_gripper: float,
    completed_phase_name: str | None,
    source: CubePose,
    target: CubePose,
    grasp,
    end_joints: dict[str, float],
    end_gripper: float,
    free_grasp: bool,
    failed_trajectory_dir=None,
    verbose: bool = True,
):
    """The first replan candidate that survives preflight, or ``None`` to restart.

    Candidates come in preference order and are vetted under live physics; the
    first clean one wins. When none is clean the episode cannot continue from
    here, and ``failed_trajectory_dir`` (if set) receives the last rejected
    trajectory and a note saying why, which is what makes an abort diagnosable
    after the fact.

    ``verbose`` gates the narration only. A sharded recording run has dozens of
    these going at once and wants them silent; the decision it reports is the
    same either way.
    """
    candidate = None
    rejected = None
    rejected_detail = None
    rejected_unexpected: list[tuple[float, str, str]] = []
    for index, replan_traj in enumerate(
        replan_remaining_candidates(
            kinematics,
            measured_joints,
            measured_gripper,
            completed_phase_name,
            source,
            target,
            grasp,
            end_joints,
            end_gripper,
            free_grasp=free_grasp,
        ),
        start=1,
    ):
        if failed_trajectory_dir is not None:
            detail_events = preflight(
                model, replan_traj, actuator_id, robot_geom_ids, env_geom_ids, detailed=True
            )
            unexpected_detail = [
                event for event in detail_events if preflight_collision_is_unexpected(event)
            ]
            unexpected = [(e.time, e.geom1, e.geom2) for e in unexpected_detail]
        else:
            events = preflight(model, replan_traj, actuator_id, robot_geom_ids, env_geom_ids)
            unexpected_detail = None
            unexpected = [(t, n1, n2) for t, n1, n2 in events if is_unexpected(n1, n2)]
        if not unexpected:
            candidate = replan_traj
            if index > 1 and verbose:
                print(f"Selected replan candidate {index} after preflight rejections.")
            break
        rejected = replan_traj
        rejected_detail = unexpected_detail
        rejected_unexpected = unexpected
        if replan_traj.carry is not None and verbose:
            print(
                f"  rejected replan candidate {index}: "
                f"carry={replan_traj.carry.mode} collision t={unexpected[0][0]:.3f}s "
                f"{unexpected[0][1]} ↔ {unexpected[0][2]}"
            )

    if candidate is not None:
        return candidate

    if rejected is None:
        if verbose:
            print("Error: No feasible plan from current state. Aborting episode.")
        reason = f"no feasible replan after {completed_phase_name}"
    else:
        if verbose:
            print("Error: All replan candidates failed preflight. Aborting episode.")
            for t, n1, n2 in rejected_unexpected:
                print(f"  collision t={t:.3f}s {n1} ↔ {n2}")
        reason = f"all replan candidates failed preflight after {completed_phase_name}"
        if failed_trajectory_dir is not None and rejected_detail is not None:
            path = failed_trajectory_dir / (
                f"replan_after_{completed_phase_name or 'start'}_failed.npz"
            )
            save_failed_preflight_trajectory(
                path, model, rejected, actuator_id, rejected_detail
            )
            if verbose:
                print(f"saved rejected replan trajectory: {path}")
    write_failed_trajectory_note(failed_trajectory_dir, reason, source=source, target=target)
    return None
