# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pick_and_place.geometry import CubePose
from pick_and_place.visual_servo import (
    DESCENT_SERVO_BACKUP_DURATION,
    DESCENT_SERVO_STABLE_FRAMES,
    DescentServoConvergence,
    DescentServoRetryState,
)


def test_descent_servo_convergence_requires_consecutive_stable_targets():
    tracker = DescentServoConvergence()
    source = CubePose(x=0.2, y=0.1, z=0.015, roll=0.0, pitch=0.0, yaw=0.2)

    for _ in range(DESCENT_SERVO_STABLE_FRAMES - 1):
        tracker.observe(source)

    assert not tracker.is_stable()

    tracker.observe(source)

    assert tracker.is_stable()

    tracker.observe(CubePose(x=0.21, y=0.1, z=0.015, roll=0.0, pitch=0.0, yaw=0.2))

    assert not tracker.is_stable()


def test_descent_servo_retry_backs_up_to_pregrasp_before_retrying():
    retry = DescentServoRetryState(max_retries=1, backup_duration=DESCENT_SERVO_BACKUP_DURATION)
    descent_duration = 1.0

    retry.start_backup(1.0)

    assert retry.is_backing_up()
    assert retry.command_phase_t(1.0, descent_duration) == descent_duration
    assert retry.command_phase_t(
        1.0 + DESCENT_SERVO_BACKUP_DURATION,
        descent_duration,
    ) == 0.0
    assert retry.backup_complete(1.0 + DESCENT_SERVO_BACKUP_DURATION)

    retry.finish_backup()

    assert not retry.is_backing_up()
    assert not retry.can_retry()
