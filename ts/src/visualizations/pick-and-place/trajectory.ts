// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  ARM_JOINT_NAMES,
  type ArmJointName,
  type So101Kinematics
} from '../../ik/kinematics';
import { solveSimplePregraspIk } from '../../ik/simple-ik';
import {
  type CubeFace,
  type CubePose
} from '../pregrasp-pose-shared/body-factories';
import { createSimplePregraspMatrix } from '../simple-pregrasp-pose/pose';

// 2.5 cm above the simple pregrasp position along world Z.
const HOVER_Z_OFFSET = 0.025;

// Gripper joint angle at hover pregrasp: 40° open.
export const GRIPPER_OPEN = 40 * (Math.PI / 180);

export type JointAngles = Record<ArmJointName, number>;

export interface TrajectoryFrame {
  joints: JointAngles;
  gripper: number;
}

export const NEUTRAL_FRAME: TrajectoryFrame = {
  joints: {
    shoulder_pan: 0,
    shoulder_lift: 0,
    elbow_flex: 0,
    wrist_flex: 0,
    wrist_roll: 0
  },
  gripper: 0
};

// Duration of stage 1 – neutral → hover pregrasp above source cube.
const STAGE1_DURATION = 2.0;

// The four vertical faces tried in order of how naturally the robot approaches
// a cube that is roughly in the +x direction from the pan axis.
const VERTICAL_FACES: CubeFace[] = ['+x', '-x', '+y', '-y'];

// Simple-pregrasp matrix for `face` shifted up by HOVER_Z_OFFSET, or null if
// the face is not vertical for this pose.
function hoverMatrix(face: CubeFace, sourcePose: CubePose): THREE.Matrix4 | null {
  const pregrasp = createSimplePregraspMatrix(face, sourcePose);
  if (!pregrasp) { return null; }
  const pos = new THREE.Vector3().setFromMatrixPosition(pregrasp);
  pregrasp.setPosition(pos.x, pos.y, pos.z + HOVER_Z_OFFSET);
  return pregrasp;
}

function smoothstep(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

function lerpJoints(a: JointAngles, b: JointAngles, alpha: number): JointAngles {
  const out = {} as JointAngles;
  for (const name of ARM_JOINT_NAMES) {
    out[name] = a[name] + (b[name] - a[name]) * alpha;
  }
  return out;
}

export interface Trajectory {
  duration: number;
  evaluate(t: number): TrajectoryFrame;
}

export function computeTrajectory(
  k: So101Kinematics,
  sourcePose: CubePose
): Trajectory | null {
  let hoverJoints: JointAngles | null = null;
  for (const face of VERTICAL_FACES) {
    const matrix = hoverMatrix(face, sourcePose);
    if (!matrix) { continue; }
    const result = solveSimplePregraspIk(k, matrix);
    if (result.type === 'unreachable') { continue; }
    hoverJoints = (result.branches.find(b => b.elbow === 'up') ?? result.branches[0]).joints;
    break;
  }
  if (hoverJoints === null) { return null; }

  const endJoints = hoverJoints;
  return {
    duration: STAGE1_DURATION,
    evaluate(t: number): TrajectoryFrame {
      const alpha = smoothstep(t / STAGE1_DURATION);
      return {
        joints: lerpJoints(NEUTRAL_FRAME.joints, endJoints, alpha),
        gripper: NEUTRAL_FRAME.gripper + (GRIPPER_OPEN - NEUTRAL_FRAME.gripper) * alpha
      };
    }
  };
}
