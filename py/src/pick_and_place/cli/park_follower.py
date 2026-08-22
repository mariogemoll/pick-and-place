# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Park the physical SO-101 at REST and release its torque.

``pap run-policy-real`` parks the arm in a ``finally`` block on every ordinary
exit, but a segfault in a native extension -- OpenCV, the AprilTag detector,
torch -- takes the interpreter down before that block runs. The servos hold
torque in their own registers, so the arm is left energized wherever it stood,
with no process left to bring it down. This is the recovery: connect, ramp
NEUTRAL -> REST, release.

It is also the safe way to stand the rig down by hand, since dropping torque
from a raised pose lets the arm fall under gravity. Ramping to REST first puts
it somewhere it can be released from.
"""

from __future__ import annotations

import argparse

import numpy as np

from pick_and_place.cli.rig import add_follower_arguments, add_max_joint_speed_argument
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.joint_frames import follower_clamp_limits, sim_frame_to_real
from pick_and_place.hardware.follower import make_so101_follower
from pick_and_place.runtime.ramp import ramp_follower
from pick_and_place.sim.derive_kinematics import derive_kinematics
from pick_and_place.sim.scene import build_scene
from pick_and_place.spec.robot import (
    NEUTRAL_ARM_JOINTS,
    NEUTRAL_GRIPPER,
    REST_ARM_JOINTS,
    REST_GRIPPER,
)


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the parking command."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_follower_arguments(parser)
    add_max_joint_speed_argument(
        parser,
        default=0.0,
        extra_help=". Uncapped, the parking ramp runs at its own pace; pass e.g. 3 to crawl",
    )
    parser.add_argument(
        "--skip-neutral",
        action="store_true",
        help=(
            "ramp straight to REST instead of via NEUTRAL. Only for an arm already low "
            "and clear -- the NEUTRAL waypoint is what keeps a raised arm from sweeping "
            "through the workspace on its way down"
        ),
    )
    parser.add_argument(
        "--no-release",
        action="store_true",
        help="park at REST but leave torque enabled, so the arm holds instead of going limp",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Park the arm."""
    model = build_scene(include_environment=True).compile()
    kinematics = derive_kinematics(model)
    clamp_low, clamp_high = follower_clamp_limits(kinematics)
    clip_warned: set[str] = set()

    neutral_real = sim_frame_to_real(NEUTRAL_ARM_JOINTS, NEUTRAL_GRIPPER)
    rest_real = sim_frame_to_real(REST_ARM_JOINTS, REST_GRIPPER)

    speed = args.max_joint_speed if args.max_joint_speed > 0 else None
    print(
        f"Parking at {'uncapped speed' if speed is None else f'{speed:g} deg/s'}"
        f"{' (straight to REST)' if args.skip_neutral else ' via NEUTRAL'}."
    )

    print(f"Connecting to the follower on {args.follower_port}...")
    # Torque stays on through a plain disconnect so the arm holds rather than
    # going limp; it is released deliberately at REST below.
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    parked = False
    try:
        # The arm may have been left holding a pose by a crashed run, or may
        # already be limp; either way torque has to be on to ramp it.
        follower.bus.enable_torque()
        waypoints: list[tuple[str, np.ndarray]] = []
        if not args.skip_neutral:
            waypoints.append(("NEUTRAL", neutral_real))
        waypoints.append(("REST", rest_real))
        for name, target in waypoints:
            print(f"Ramping to {name}...")
            ramp_follower(
                follower,
                target,
                clamp_low,
                clamp_high,
                clip_warned,
                max_joint_speed=speed,
            )
        parked = True
    except Exception as exc:  # noqa: BLE001 - best-effort park before release
        print(f"Warning: could not park the arm: {exc}")

    if parked and not args.no_release:
        print("At REST — releasing torque.")
        try:
            follower.bus.disable_torque()
        except Exception as exc:  # noqa: BLE001 - best-effort torque release
            print(f"Warning: could not release torque: {exc}")
    elif parked:
        print("At REST — leaving torque enabled (--no-release).")
    else:
        # Releasing an arm that never reached REST would drop it from wherever
        # it stalled, so it is left holding for the operator to deal with.
        print("Not at REST — leaving torque enabled rather than dropping the arm.")

    print("Disconnecting hardware...")
    follower.disconnect()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
