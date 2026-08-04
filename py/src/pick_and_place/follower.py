# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Construct the lerobot devices for a physical SO-101 leader/follower pair.

Device construction only. The joint conversions a hardware run needs live in
:mod:`pick_and_place.joint_frames`, which imports nothing heavy, so code that
merely speaks the real frame does not drag lerobot in.
"""

from __future__ import annotations

from typing import Any


def make_so101_follower(
    port: str,
    robot_id: str,
    *,
    calibration_dir: str | None = None,
    max_relative_target: float | None = None,
    disable_torque_on_disconnect: bool = True,
) -> Any:
    """Construct a lerobot ``SO101Follower``.

    ``use_degrees=True`` makes the follower report and accept the arm joints in
    degrees (gripper as a 0-100 position), which is the real frame
    :mod:`pick_and_place.joint_frames` converts to.
    """
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig

    return make_robot_from_config(
        SO101FollowerConfig(
            port=port,
            id=robot_id,
            calibration_dir=calibration_dir,
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=disable_torque_on_disconnect,
            use_degrees=True,
        )
    )


def make_so101_leader(
    port: str,
    robot_id: str,
    *,
    calibration_dir: str | None = None,
) -> Any:
    """Construct a lerobot ``SO101Leader``."""
    try:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        return SO101Leader(SO101LeaderConfig(port=port, id=robot_id, calibration_dir=calibration_dir))
    except ModuleNotFoundError:
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig

        leader_cfg = SO101LeaderConfig(port=port, id=robot_id, calibration_dir=calibration_dir)
        return make_teleoperator_from_config(leader_cfg)
