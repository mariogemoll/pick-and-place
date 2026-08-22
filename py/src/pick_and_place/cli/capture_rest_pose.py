# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import argparse
import math
from pathlib import Path
import numpy as np

from pick_and_place.cli.rig import add_follower_arguments
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.joint_frames import action_to_joints, real_frame_to_sim
from pick_and_place.hardware.follower import make_so101_follower
from pick_and_place.spec import robot as robot_spec

def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the rest-pose capture."""
    parser = SuggestingArgumentParser(description="Capture the current robot pose as the rest position.")
    add_follower_arguments(parser)
    parser.add_argument("--update", action="store_true", help="Automatically update py/src/pick_and_place/scripted/trajectory.py")
    return parser


def run(args: argparse.Namespace) -> None:
    """Read the pose and print it."""
    print(f"Connecting to follower on {args.follower_port}...")
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()
    
    print("Reading current pose...")
    obs = follower.get_observation()
    real_joints = action_to_joints(obs, np.zeros(6))

    arm_rad, gripper_rad = real_frame_to_sim(real_joints)
    
    print("\nCaptured Rest Pose (radians):")
    print("REST_ARM_JOINTS = {")
    for name, val in arm_rad.items():
        print(f"    \"{name}\": math.radians({math.degrees(val)}),")
    print("}")
    
    pos = real_joints[5]
    print(f"REST_GRIPPER = math.radians(({pos:.1f} - 2.3) / 96.2 * 130 - 10)")
    
    if args.update:
        # The module that declares them, asked for its own file rather than
        # spelled out: these constants have already outlived one path.
        rest_pose_path = Path(robot_spec.__file__)
        content = rest_pose_path.read_text()
        
        import re
        
        # Replace REST_ARM_JOINTS
        new_arm_joints = "REST_ARM_JOINTS: dict[str, float] = {\n"
        for name, val in arm_rad.items():
            new_arm_joints += f"    \"{name}\": math.radians({math.degrees(val)}),\n"
        new_arm_joints += "}"
        
        content = re.sub(r"REST_ARM_JOINTS: dict\[str, float\] = \{.*?\}", new_arm_joints, content, flags=re.DOTALL)
        
        # Replace REST_GRIPPER
        # We look for the literal assignment to REST_GRIPPER on its own line.
        new_gripper = f"REST_GRIPPER = math.radians(({pos:.1f} - 2.3) / 96.2 * 130 - 10)"
        content = re.sub(r"REST_GRIPPER = .*", new_gripper, content)
        
        rest_pose_path.write_text(content)
        print(f"\nUpdated {rest_pose_path}")

    follower.disconnect()


def main() -> None:
    run(build_parser().parse_args())

if __name__ == "__main__":
    main()
