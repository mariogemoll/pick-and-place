#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import argparse
import math
import sys
from pathlib import Path
import numpy as np

# Add py/src to path so we can import pick_and_place
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pick_and_place.follower import (
    action_to_joints,
    make_so101_follower,
    real_frame_to_sim,
)

def main():
    parser = argparse.ArgumentParser(description="Capture the current robot pose as the rest position.")
    parser.add_argument("--port", required=True, help="Serial port of the SO-101 follower")
    parser.add_argument("--id", default="folly", help="Follower ID (default: folly)")
    parser.add_argument("--update", action="store_true", help="Automatically update py/src/pick_and_place/trajectory.py")
    args = parser.parse_args()

    print(f"Connecting to follower on {args.port}...")
    follower = make_so101_follower(args.port, args.id, disable_torque_on_disconnect=False)
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
        trajectory_path = Path(__file__).resolve().parents[1] / "src" / "pick_and_place" / "trajectory.py"
        content = trajectory_path.read_text()
        
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
        
        trajectory_path.write_text(content)
        print(f"\nUpdated {trajectory_path}")

    follower.disconnect()

if __name__ == "__main__":
    main()
