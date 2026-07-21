# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np
import pytest

from pick_and_place.episode_loop import episode_loop
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.physical_collection import (
    CameraDriftError,
    recover_cube,
    reject_camera_drift,
    wait_for_target_movement,
)


def pose(x: float, y: float) -> CubePose:
    return CubePose(x, y, CUBE_HALF_SIZE)


def test_v2_cooldown_cadence_counts_only_completed_episodes():
    cooldowns = []
    loop = episode_loop(target=3, rest_every=2, cooldown=lambda: cooldowns.append("rest"))

    failed = next(loop)
    assert failed.index == 1
    first = next(loop)
    first.complete()
    second = next(loop)
    second.complete()
    third = next(loop)

    assert third.index == 3
    assert cooldowns == ["rest"]


def test_v2_camera_drift_rejects_translation_or_rotation_over_limit():
    position = np.zeros(3)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    drift = reject_camera_drift(
        position,
        quaternion,
        np.array([0.009, 0.0, 0.0]),
        quaternion,
        max_translation_mm=10.0,
        max_rotation_deg=2.0,
    )
    assert drift == pytest.approx((9.0, 0.0))

    with pytest.raises(CameraDriftError, match="11.0mm"):
        reject_camera_drift(
            position,
            quaternion,
            np.array([0.011, 0.0, 0.0]),
            quaternion,
            max_translation_mm=10.0,
            max_rotation_deg=2.0,
        )

    angle = np.radians(3.0)
    rotated = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    with pytest.raises(CameraDriftError, match="3.00deg"):
        reject_camera_drift(
            position,
            quaternion,
            position,
            rotated,
            max_translation_mm=10.0,
            max_rotation_deg=2.0,
        )


def test_v2_target_movement_waits_for_visibility_distance_and_rest_duration():
    now = [0.0]
    detections = iter((None, pose(0.01, 0.0), pose(0.04, 0.0)))
    alerts = []

    moved = wait_for_target_movement(
        pose(0.0, 0.0),
        minimum_distance=0.03,
        minimum_rest_until=5.0,
        detect_target=lambda: next(detections),
        alert=alerts.append,
        alert_min_seconds=1.0,
        alert_max_seconds=2.0,
        poll_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )

    assert moved == pose(0.04, 0.0)
    assert now[0] == 5.0
    assert any("not visible" in message for message in alerts)
    assert any("moved only" in message for message in alerts)


def test_v2_cube_recovery_retries_failed_and_unsafe_relocations():
    relocation_results = iter((False, True, True))
    located = iter((pose(0.0, 0.0), pose(0.2, 0.0)))
    attempts = []

    recovered = recover_cube(
        max_attempts=3,
        relocate=lambda attempt: attempts.append(attempt) or next(relocation_results),
        locate=lambda: next(located),
        is_allowed=lambda x, _y: x >= 0.1,
    )

    assert recovered == pose(0.2, 0.0)
    assert attempts == [1, 2, 3]
