# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.sim.model import build_model
from pick_and_place.spec.workspace import CUBE_HALF_SIZE
from pick_and_place.core.geometry import CubePose
from pick_and_place.sim.derive_kinematics import derive_kinematics
from pick_and_place.planning.trajectory import (
    DROP_DWELL_DURATION,
    ReleasePhase,
    replan_remaining_candidates,
)
from pick_and_place.spec.robot import GRIPPER_GRASP, GRIPPER_OPEN, NEUTRAL_ARM_JOINTS


def test_release_lifts_from_locked_predrop_before_using_readback():
    predrop = dict(NEUTRAL_ARM_JOINTS)
    postdrop = {name: value + 0.1 for name, value in predrop.items()}
    phase = ReleasePhase(predrop, postdrop)

    start = phase.evaluate(0.0)
    before_drop = phase.evaluate(DROP_DWELL_DURATION)
    end = phase.evaluate(phase.duration)

    assert start.joints == predrop
    assert start.gripper == GRIPPER_GRASP
    assert before_drop.joints == predrop
    assert before_drop.gripper == GRIPPER_GRASP
    np.testing.assert_allclose(list(end.joints.values()), list(postdrop.values()))
    assert end.gripper == GRIPPER_OPEN


def test_replan_after_release_retreats_from_elevated_readback():
    source = CubePose(x=0.20, y=-0.12, z=CUBE_HALF_SIZE)
    target = CubePose(x=0.20, y=-0.05, z=CUBE_HALF_SIZE)
    model, _ = build_model(source)
    kinematics = derive_kinematics(model)
    measured = {name: value + 0.05 for name, value in NEUTRAL_ARM_JOINTS.items()}

    trajectories = list(
        replan_remaining_candidates(
            kinematics,
            measured,
            GRIPPER_OPEN,
            "release",
            source,
            target,
            None,
            NEUTRAL_ARM_JOINTS,
            0.0,
        )
    )

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert [phase.name for phase in trajectory.phases] == ["retreat"]
    assert trajectory.evaluate(0.0).joints == measured
