# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.episodes import prepare_episode
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.trajectory import RECOVERY_DROP_CUBE_CENTER_Z, RECOVERY_LIFT_CUBE_Z


def test_free_grasp_recovery_lifts_vertically_before_carry():
    source = CubePose(
        x=0.42131095399862234,
        y=0.061409662483782095,
        z=CUBE_HALF_SIZE,
        yaw=1.8831453470115125,
    )
    target = CubePose(
        x=0.14152933877982182,
        y=0.07890637263598481,
        z=CUBE_HALF_SIZE,
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
        "drop_descent",
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
