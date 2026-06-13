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

// 1 cm above the simple pregrasp position along world Z.
const HOVER_Z_OFFSET = 0.01;

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
// Duration of stage 2 – hover pregrasp → pregrasp at source cube center.
const STAGE2_DURATION = 1.0;

// The four vertical faces tried in order of how naturally the robot approaches
// a cube that is roughly in the +x direction from the pan axis.
const VERTICAL_FACES: CubeFace[] = ['+x', '-x', '+y', '-y'];

// Simple-pregrasp matrix for `face` shifted up along world Z, or null if the
// face is not vertical for this pose.
function pregraspMatrix(
  face: CubeFace,
  sourcePose: CubePose,
  zOffset = 0
): THREE.Matrix4 | null {
  const pregrasp = createSimplePregraspMatrix(face, sourcePose);
  if (!pregrasp) { return null; }
  const pos = new THREE.Vector3().setFromMatrixPosition(pregrasp);
  pregrasp.setPosition(pos.x, pos.y, pos.z + zOffset);
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
  let pregraspJoints: JointAngles | null = null;
  let selectedFace: CubeFace | null = null;
  let selectedElbow: 'up' | 'down' | null = null;
  for (const face of VERTICAL_FACES) {
    const hover = pregraspMatrix(face, sourcePose, HOVER_Z_OFFSET);
    const pregrasp = pregraspMatrix(face, sourcePose);
    if (!hover || !pregrasp) { continue; }
    const hoverResult = solveSimplePregraspIk(k, hover);
    const pregraspResult = solveSimplePregraspIk(k, pregrasp);
    if (hoverResult.type === 'unreachable' || pregraspResult.type === 'unreachable') {
      continue;
    }
    const hoverElbows = new Set(hoverResult.branches.map(branch => branch.elbow));
    const branch = pregraspResult.branches.find(
      candidate => candidate.elbow === 'up' && hoverElbows.has(candidate.elbow)
    ) ?? pregraspResult.branches.find(candidate => hoverElbows.has(candidate.elbow));
    if (!branch) { continue; }
    const hoverBranch = hoverResult.branches.find(
      candidate => candidate.elbow === branch.elbow
    );
    if (!hoverBranch) { continue; }
    hoverJoints = hoverBranch.joints;
    pregraspJoints = branch.joints;
    selectedFace = face;
    selectedElbow = branch.elbow;
    break;
  }
  if (
    hoverJoints === null ||
    pregraspJoints === null ||
    selectedFace === null ||
    selectedElbow === null
  ) {
    return null;
  }

  const stage1EndJoints = hoverJoints;
  const stage2EndJoints = pregraspJoints;
  const duration = STAGE1_DURATION + STAGE2_DURATION;
  return {
    duration,
    evaluate(t: number): TrajectoryFrame {
      if (t >= STAGE1_DURATION) {
        const alpha = smoothstep((t - STAGE1_DURATION) / STAGE2_DURATION);
        const matrix = pregraspMatrix(
          selectedFace,
          sourcePose,
          HOVER_Z_OFFSET * (1 - alpha)
        );
        const result = matrix ? solveSimplePregraspIk(k, matrix) : null;
        const branch = result?.type === 'success'
          ? result.branches.find(candidate => candidate.elbow === selectedElbow)
          : undefined;
        return {
          joints: branch?.joints ??
            lerpJoints(stage1EndJoints, stage2EndJoints, alpha),
          gripper: GRIPPER_OPEN
        };
      }
      const alpha = smoothstep(t / STAGE1_DURATION);
      return {
        joints: lerpJoints(NEUTRAL_FRAME.joints, stage1EndJoints, alpha),
        gripper: NEUTRAL_FRAME.gripper + (GRIPPER_OPEN - NEUTRAL_FRAME.gripper) * alpha
      };
    }
  };
}
