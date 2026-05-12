#!/usr/bin/env python3
"""
Record the current robot end-effector pose as the ground-truth grasp point.

Usage:
    # Move the robot manually to the desired grasp position, then run:
    python full_system_pipeline/record_grasp_point.py --out runs/trial_01/gt_pose.json

    # Or just print to screen without saving:
    python full_system_pipeline/record_grasp_point.py
"""

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundation_model.servo_pipeline_sam3 import RobotController

ROBOT_IP = "192.168.1.241"

def main():
    p = argparse.ArgumentParser(description="Record GT grasp pose from current EE position")
    p.add_argument("--ip",  default=ROBOT_IP, help="xArm IP (default: %(default)s)")
    p.add_argument("--out", default=None,     help="Path to save JSON, e.g. runs/trial_01/gt_pose.json")
    args = p.parse_args()

    print(f"Connecting to robot at {args.ip} ...")
    robot = RobotController(args.ip)
    ok = robot.connect()
    if not ok:
        print("ERROR: could not connect. Is the robot powered on and in PC mode?")
        sys.exit(1)

    time.sleep(0.5)
    pos = robot._get_pos()
    robot.stop()

    if pos is None:
        print("ERROR: get_position failed — check robot status.")
        sys.exit(1)

    labels = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
    print("\nCurrent EE pose (grasp point):")
    for label, val in zip(labels, pos):
        print(f"  {label:12s}: {val:.3f}")

    payload = {"gt_pose": pos, "pose_labels": labels}

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to: {args.out}")
    else:
        print("\n(use --out <path> to save)")

if __name__ == "__main__":
    main()
