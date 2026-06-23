# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.episodes import prepare_episode
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.trajectory import RECOVERY_DROP_CUBE_CENTER_Z, RECOVERY_LIFT_CUBE_Z


def test_free_grasp_recovery_lifts_vertically_before_carry():
    source = CubePose(
        x=0.2652089160166924,
        y=0.0743065789637258,
        z=CUBE_HALF_SIZE,
        yaw=1.4937383406595242,
    )
    target = CubePose(
        x=0.32446189743377685,
        y=-0.11795889453162557,
        z=CUBE_HALF_SIZE,
        yaw=-1.8509609891758714,
    )

    episode = prepare_episode(
        np.random.default_rng(0),
        source,
        target,
        max_attempts=1,
        free_grasp=True,
    )

    assert [phase.name for phase in episode.trajectory.phases] == [
        "approach",
        "descent",
        "grasp",
        "recovery_lift",
        "carry",
        "release",
        "retreat",
    ]
    grasp = episode.trajectory.grasp
    assert grasp is not None
    assert episode.trajectory.carry is not None
    assert episode.trajectory.carry.drop_position[2] == RECOVERY_DROP_CUBE_CENTER_Z
    release = next(phase for phase in episode.trajectory.phases if phase.name == "release")
    assert release.pre_release_delay == 0.0
    np.testing.assert_allclose(
        grasp.lift_matrix[:2, 3],
        grasp.grasp_matrix[:2, 3],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        grasp.lift_matrix[2, 3] - grasp.grasp_matrix[2, 3],
        RECOVERY_LIFT_CUBE_Z - source.z,
        atol=1e-12,
    )
