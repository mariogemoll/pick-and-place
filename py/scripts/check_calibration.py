# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import argparse
import json
from pathlib import Path

# Feetech STS3215 resolution: 4096 steps per 360 degrees
STEPS_PER_DEGREE = 4096 / 360.0

def steps_to_degrees(steps: int) -> float:
    return steps / STEPS_PER_DEGREE

def get_lerobot_calibration_path(robot_type: str, robot_id: str) -> Path:
    base = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    return base / robot_type / f"{robot_id}.json"

def analyze_calibration(path: Path, name: str) -> dict:
    if not path.exists():
        print(f"❌ Error: Could not find {name} calibration at {path}")
        return {}
    
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def check_joint(joint_name: str, leader_data: dict, follower_data: dict):
    if joint_name not in leader_data or joint_name not in follower_data:
        return
    
    ld = leader_data[joint_name]
    fd = follower_data[joint_name]

    print(f"\n--- {joint_name.upper()} ---")

    # 1. Check Range Span
    l_span = ld['range_max'] - ld['range_min']
    f_span = fd['range_max'] - fd['range_min']
    l_span_deg = steps_to_degrees(l_span)
    f_span_deg = steps_to_degrees(f_span)

    print(f"Leader range:   {ld['range_min']} to {ld['range_max']} ({l_span_deg:.1f}° span)")
    print(f"Follower range: {fd['range_min']} to {fd['range_max']} ({f_span_deg:.1f}° span)")

    if joint_name == "wrist_roll":
        if l_span != 4095 or f_span != 4095:
            print("⚠️  WARNING: wrist_roll is a continuous motor but range is not 0-4095!")
    else:
        if l_span_deg < 90 or f_span_deg < 90:
            print(f"⚠️  WARNING: Unusually small range of motion (< 90°). Did you move it fully?")
        if abs(l_span_deg - f_span_deg) > 45:
            print(f"⚠️  WARNING: Leader and follower ranges differ by more than 45°!")

    # 2. Check Homing Offset
    l_home = ld['homing_offset']
    f_home = fd['homing_offset']
    print(f"Leader Homing Offset:   {l_home} ({steps_to_degrees(l_home):.1f}°)")
    print(f"Follower Homing Offset: {f_home} ({steps_to_degrees(f_home):.1f}°)")

    diff_home_deg = steps_to_degrees(abs(l_home - f_home))
    if diff_home_deg > 90:
        print(f"ℹ️  NOTE: Homing offsets differ by {diff_home_deg:.1f}°. This is normal if the servo horns were attached at different angles during assembly, but if they were built identically, one might have been zeroed in the wrong physical pose.")


def main():
    parser = argparse.ArgumentParser(description="Check and compare LeRobot calibrations for SO101")
    parser.add_argument("--leader-id", default="liddy", help="Leader ID in lerobot cache")
    parser.add_argument("--follower-id", default="folly", help="Follower ID in lerobot cache")
    args = parser.parse_args()

    leader_path = get_lerobot_calibration_path("teleoperators/so_leader", args.leader_id)
    follower_path = get_lerobot_calibration_path("robots/so_follower", args.follower_id)

    print(f"Loading Leader: {leader_path}")
    leader_data = analyze_calibration(leader_path, "Leader")
    print(f"Loading Follower: {follower_path}")
    follower_data = analyze_calibration(follower_path, "Follower")

    if not leader_data or not follower_data:
        return

    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    for joint in joints:
        check_joint(joint, leader_data, follower_data)

if __name__ == "__main__":
    main()
