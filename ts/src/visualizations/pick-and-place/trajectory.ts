// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  ARM_JOINT_NAMES,
  type ArmJointName,
  type So101Kinematics
} from '../../ik/kinematics';
import { solveSimplePregraspIk } from '../../ik/simple-ik';
import { SAFETY_MARGIN } from '../pregrasp-pose-shared/bodies';
import {
  createWorldFromCubeContactMatrix,
  type CubeFace,
  type CubePose
} from '../pregrasp-pose-shared/body-factories';
import { createSimplePregraspMatrix } from '../simple-pregrasp-pose/pose';

// 1 cm above the simple pregrasp position along world Z.
const HOVER_Z_OFFSET = 0.01;

// Gripper joint angle at hover pregrasp: 40° open.
export const GRIPPER_OPEN = 40 * (Math.PI / 180);
// Gripper joint angle once the jaws have pinched the cube. Geometrically
// estimated: at this angle the moving jaw's fingertips reach the cube's far
// face after the cube has slid flush against the fixed jaw. (Faked, not solved
// from a physics engine – fine-tune visually if the pinch looks off.)
export const GRIPPER_CLOSED = 10.5 * (Math.PI / 180);
// Gripper angle at which the cube starts being shoved toward the fixed jaw.
// The true geometric contact is a bit later (~14°), but starting the slide
// sooner reads better: the cube eases over rather than snapping flush at the
// very end of the close.
const GRIPPER_CONTACT = 20 * (Math.PI / 180);

export type JointAngles = Record<ArmJointName, number>;

export interface RobotPose {
  joints: JointAngles;
  gripper: number;
}

export interface TrajectoryFrame extends RobotPose {
  sourceCube: CubePose;
}

export const NEUTRAL_FRAME: RobotPose = {
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
// Duration of stage 3 – close the gripper, pushing the cube against the fixed
// jaw.
const STAGE3_DURATION = 1.0;

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
  const stage2End = STAGE1_DURATION + STAGE2_DURATION;
  const duration = stage2End + STAGE3_DURATION;

  // Direction pointing from the grasped face into the cube. The cube is pushed
  // the opposite way (toward the fixed jaw) as the gripper closes.
  const inwardNormal = new THREE.Vector3(0, 0, 1).transformDirection(
    createWorldFromCubeContactMatrix(selectedFace, sourcePose)
  );

  return {
    duration,
    evaluate(t: number): TrajectoryFrame {
      if (t >= stage2End) {
        // Stage 3: hold the arm at pregrasp and close the gripper. Once the
        // moving jaw reaches the cube it shoves it flush against the fixed jaw.
        const alpha = smoothstep((t - stage2End) / STAGE3_DURATION);
        const gripper = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * alpha;
        const push = gripper < GRIPPER_CONTACT
          ? SAFETY_MARGIN *
            (GRIPPER_CONTACT - gripper) / (GRIPPER_CONTACT - GRIPPER_CLOSED)
          : 0;
        return {
          joints: stage2EndJoints,
          gripper,
          sourceCube: pushedCube(sourcePose, inwardNormal, push)
        };
      }
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
          gripper: GRIPPER_OPEN,
          sourceCube: sourcePose
        };
      }
      const alpha = smoothstep(t / STAGE1_DURATION);
      return {
        joints: lerpJoints(NEUTRAL_FRAME.joints, stage1EndJoints, alpha),
        gripper: NEUTRAL_FRAME.gripper + (GRIPPER_OPEN - NEUTRAL_FRAME.gripper) * alpha,
        sourceCube: sourcePose
      };
    }
  };
}

// Source cube translated `push` metres toward the fixed jaw (opposite the
// face's inward normal). Faces are vertical, so this only moves x/y.
function pushedCube(
  pose: CubePose,
  inwardNormal: THREE.Vector3,
  push: number
): CubePose {
  if (push === 0) { return pose; }
  return {
    ...pose,
    x: pose.x - inwardNormal.x * push,
    y: pose.y - inwardNormal.y * push,
    z: pose.z - inwardNormal.z * push
  };
}
