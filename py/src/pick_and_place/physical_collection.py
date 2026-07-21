# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Testable unattended-collection behavior for the physical runner."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from pick_and_place.cam_align_solve import pose_delta_mm_deg
from pick_and_place.geometry import CubePose


class CameraDriftError(RuntimeError):
    """The overhead camera moved too far from its startup calibration."""


def reject_camera_drift(
    startup_position: np.ndarray,
    startup_quaternion: np.ndarray,
    check_position: np.ndarray,
    check_quaternion: np.ndarray,
    *,
    max_translation_mm: float,
    max_rotation_deg: float,
) -> tuple[float, float]:
    """Return measured drift or raise when either configured limit is exceeded."""
    drift_mm, drift_deg = pose_delta_mm_deg(
        startup_position,
        startup_quaternion,
        check_position,
        check_quaternion,
    )
    if drift_mm > max_translation_mm or drift_deg > max_rotation_deg:
        raise CameraDriftError(
            f"overhead camera drifted {drift_mm:.1f}mm / {drift_deg:.2f}deg "
            f"(limits {max_translation_mm:.1f}mm / {max_rotation_deg:.2f}deg)"
        )
    return drift_mm, drift_deg


def target_distance(a: CubePose, b: CubePose) -> float:
    """Return planar distance between two target poses."""
    return float(np.hypot(a.x - b.x, a.y - b.y))


def wait_for_target_movement(
    reference: CubePose | None,
    *,
    minimum_distance: float,
    minimum_rest_until: float,
    detect_target: Callable[[], CubePose | None],
    alert: Callable[[str], None],
    should_continue: Callable[[], bool] = lambda: True,
    alert_min_seconds: float = 10.0,
    alert_max_seconds: float = 120.0,
    poll_seconds: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> CubePose | None:
    """Wait through a cooldown and require a visible, sufficiently moved target."""
    if reference is None or minimum_distance <= 0.0:
        remaining = minimum_rest_until - clock()
        if remaining > 0.0:
            sleep(remaining)
        return reference

    alert("Please move the target plate to a substantially different position.")
    alert_interval = alert_min_seconds
    next_alert_at = clock()
    while should_continue():
        target = detect_target()
        now = clock()
        if target is not None and target_distance(reference, target) >= minimum_distance:
            remaining = minimum_rest_until - now
            if remaining > 0.0:
                sleep(remaining)
            return target
        if now >= next_alert_at:
            if target is None:
                alert("Target plate is not visible. Move it into view before continuing.")
            else:
                moved_cm = target_distance(reference, target) * 100.0
                required_cm = minimum_distance * 100.0
                alert(
                    f"Target plate moved only {moved_cm:.1f}cm; "
                    f"move it at least {required_cm:.1f}cm."
                )
            next_alert_at = now + alert_interval
            alert_interval = min(alert_interval * 2.0, alert_max_seconds)
        sleep(poll_seconds)
    return None


def recover_cube(
    *,
    max_attempts: int,
    relocate: Callable[[int], bool],
    locate: Callable[[], CubePose | None],
    is_allowed: Callable[[float, float], bool],
) -> CubePose | None:
    """Retry unrecorded relocation until its observed result is pickup-safe."""
    for attempt in range(1, max_attempts + 1):
        if not relocate(attempt):
            continue
        recovered = locate()
        if recovered is not None and is_allowed(recovered.x, recovered.y):
            return recovered
    return None
