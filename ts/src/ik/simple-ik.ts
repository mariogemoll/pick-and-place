// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  GRIPPER_TARGET_POSITION
} from '../visualizations/grasp-pose-shared/body-factories';
import { type ArmJointName, type So101Kinematics } from './kinematics';

// Closed-form IK for the *simple* grasp pose (see
// src/visualizations/simple-grasp-pose). That pose keeps the gripper
// vertical: the wrist-roll axis (gripper z) points straight up, so the tool
// hangs straight down and the jaws close horizontally onto a vertical cube
// face. Because the approach direction is fixed (down), the decomposition is
// unambiguous: only the two elbow branches of a planar 2R arm remain.

export interface SimpleIkBranch {
  joints: Record<ArmJointName, number>;
  elbow: 'up' | 'down';
}

export type SimpleIkResult =
  | { type: 'success'; branches: SimpleIkBranch[] }
  | { type: 'unreachable'; reason: string };

function normalizeAngle(angle: number): number {
  let result = angle % (2 * Math.PI);
  if (result > Math.PI) { result -= 2 * Math.PI; }
  if (result <= -Math.PI) { result += 2 * Math.PI; }
  return result;
}

// Planar two-link IK: place the wrist at (targetRadial, targetHeight) relative
// to the shoulder pivot. Returns the elbow-up and elbow-down branches as
// (shoulder geometric angle, elbow geometric angle) pairs, or null if the point
// lies outside the arm's annulus.
function solve2R(
  l1: number,
  l2: number,
  targetRadial: number,
  targetHeight: number
): { up: [number, number]; down: [number, number] } | null {
  const r2 = targetRadial * targetRadial + targetHeight * targetHeight;
  const r = Math.sqrt(r2);
  if (r > l1 + l2 || r < Math.abs(l1 - l2)) {
    return null;
  }

  const cos2 = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2);
  const elbowGeom = Math.acos(THREE.MathUtils.clamp(cos2, -1, 1));

  const phi = Math.atan2(targetHeight, targetRadial);
  const cosAlpha = (l1 * l1 + r2 - l2 * l2) / (2 * l1 * r);
  const alpha = Math.acos(THREE.MathUtils.clamp(cosAlpha, -1, 1));

  return {
    up: [phi + alpha, -elbowGeom],
    down: [phi - alpha, elbowGeom]
  };
}

export function solveSimpleGraspIk(
  k: So101Kinematics,
  worldFromGripper: THREE.Matrix4
): SimpleIkResult {
  const gripperX = new THREE.Vector3(1, 0, 0)
    .transformDirection(worldFromGripper).normalize();
  const gripperZ = new THREE.Vector3(0, 0, 1)
    .transformDirection(worldFromGripper).normalize();

  // IK position target: the contact point projected onto the roll axis. It sits
  // on the gripper z-axis below the gripper origin.
  const target = GRIPPER_TARGET_POSITION.clone().applyMatrix4(worldFromGripper);

  // Approach direction: the tool points from the wrist down the gripper -z axis
  // to the target (straight down for the simple pose).
  const approach = gripperZ.clone().negate();
  // Jaw closing direction = face normal that the wrist roll must align with.
  const closing = gripperX.clone();

  // shoulder_pan selects the radial plane through the pan axis and the target.
  const dx = target.x - k.panAxis.x;
  const dy = target.y - k.panAxis.y;
  if (Math.hypot(dx, dy) < 1e-4) {
    return { type: 'unreachable', reason: 'Target is on the pan axis' };
  }
  const azimuth = Math.atan2(dy, dx);
  const shoulderPan = -azimuth;
  const radialDir = new THREE.Vector3(Math.cos(azimuth), Math.sin(azimuth), 0);
  const planeNormal = new THREE.Vector3(-Math.sin(azimuth), Math.cos(azimuth), 0);

  // The wrist pivot sits one tool length back from the target along -approach.
  const wrist = target.clone().sub(approach.clone().multiplyScalar(k.toolLength));
  const targetRadial =
    (wrist.x - k.panAxis.x) * radialDir.x +
    (wrist.y - k.panAxis.y) * radialDir.y -
    k.shoulderLift.radial;
  const targetHeight = wrist.z - k.shoulderLift.height;

  const solutions = solve2R(
    k.upperArm.length, k.lowerArm.length, targetRadial, targetHeight
  );
  if (!solutions) {
    return { type: 'unreachable', reason: 'Arm cannot reach the target' };
  }

  // Geometric link rest angles in the radial-height plane.
  const upperRest = Math.atan2(k.upperArm.height, k.upperArm.radial);
  const lowerRest = Math.atan2(k.lowerArm.height, k.lowerArm.radial);
  const elbowRest = lowerRest - upperRest;
  // Absolute pitch of the tool in the radial-height plane (straight down here).
  const toolPitch = Math.atan2(approach.z, approach.dot(radialDir));

  // wrist_roll aligns the jaw closing direction with the face normal.
  const zeroRollX = new THREE.Vector3()
    .crossVectors(approach, planeNormal).normalize();
  const zeroRollY = planeNormal;
  const rollAngle =
    Math.atan2(closing.dot(zeroRollY), closing.dot(zeroRollX)) -
    k.wristRollZeroTwist;

  const branches: SimpleIkBranch[] = [];
  for (const [elbow, [shoulderGeom, elbowGeom]] of
    [['up', solutions.up], ['down', solutions.down]] as const) {
    const shoulderLift = upperRest - shoulderGeom;
    const elbowFlex = elbowRest - elbowGeom;
    const wristFlex = -shoulderLift - elbowFlex - toolPitch;

    const joints: Record<ArmJointName, number> = {
      shoulder_pan: normalizeAngle(shoulderPan),
      shoulder_lift: normalizeAngle(shoulderLift),
      elbow_flex: normalizeAngle(elbowFlex),
      wrist_flex: normalizeAngle(wristFlex),
      wrist_roll: normalizeAngle(rollAngle)
    };

    let withinLimits = true;
    for (const [name, value] of Object.entries(joints) as [ArmJointName, number][]) {
      const limit = k.jointLimits[name];
      if (value < limit.min || value > limit.max) {
        withinLimits = false;
        break;
      }
    }
    if (withinLimits) {
      branches.push({ joints, elbow });
    }
  }

  if (branches.length === 0) {
    return { type: 'unreachable', reason: 'Solutions exceed joint limits' };
  }
  return { type: 'success', branches };
}
