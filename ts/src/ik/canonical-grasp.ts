// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  createGraspMatrix,
  createPregraspMatrix,
  PREGRASP_DISTANCE
} from '../visualizations/canonical-grasp/pose';
import {
  type CubeFace,
  type CubePose
} from '../visualizations/grasp-pose-shared/body-factories';
import { type ArmJointName, type So101Kinematics } from './kinematics';
import {
  type SimpleIkBranch,
  solveSimpleGraspIk
} from './simple-ik';

export interface CanonicalGraspChoice {
  face: Extract<CubeFace, '+x' | '-x' | '+y' | '-y'>;
  elbow: SimpleIkBranch['elbow'];
  pitch: number;
  rollOffset: number;
  closingAzimuth: number;
  cameraOutward: number;
  hoverJoints: Record<ArmJointName, number>;
  graspJoints: Record<ArmJointName, number>;
  liftJoints: Record<ArmJointName, number>;
  hoverMatrix: THREE.Matrix4;
  graspMatrix: THREE.Matrix4;
  liftMatrix: THREE.Matrix4;
  inwardNormal: THREE.Vector3;
}

const MIN_CANONICAL_GRASP_RADIUS = 0.11;
export const MAX_CANONICAL_GRASP_RADIUS = 0.426;
export const MIN_CANONICAL_AZIMUTH = (-100 * Math.PI) / 180;
export const MAX_CANONICAL_AZIMUTH = (100 * Math.PI) / 180;
export const CANONICAL_GRASP_Z_OFFSET = 0.005;
export const RECOVERY_LIFT_CUBE_Z = 0.08;

const N_DESCENT_CHECKS = 8;
const HORIZONTAL_GRASP_RADIUS = 0.36;
const SQUARE_TOP_DOWN_PITCH = Math.PI / 2;

const toRad = (deg: number): number => (deg * Math.PI) / 180;

const CANONICAL_PITCHES = [
  SQUARE_TOP_DOWN_PITCH,
  ...Array.from({ length: 81 }, (_, index) => 10 + index * 2)
    .filter(deg => deg !== 90)
    .sort((a, b) => Math.abs(a - 90) - Math.abs(b - 90))
    .map(toRad)
] as const;

const OUTER_HORIZONTAL_PITCHES = Array.from(
  { length: 26 },
  (_, index) => 10 + index * 2
).sort((a, b) => Math.abs(a - 16) - Math.abs(b - 16) || a - b).map(toRad);

const CANONICAL_ROLL_OFFSETS = [0, -10, 10, -20, 20, -30, 30, -45, 45]
  .map(toRad);

function canonicalPitchOrder(radius: number): readonly number[] {
  if (radius <= HORIZONTAL_GRASP_RADIUS) { return CANONICAL_PITCHES; }
  return [
    ...OUTER_HORIZONTAL_PITCHES,
    ...CANONICAL_PITCHES.filter(pitch => !OUTER_HORIZONTAL_PITCHES.includes(pitch))
  ];
}

function normalizeAngle(angle: number): number {
  let result = angle % (2 * Math.PI);
  if (result > Math.PI) { result -= 2 * Math.PI; }
  if (result <= -Math.PI) { result += 2 * Math.PI; }
  return result;
}

function squareToCubeFace(nominal: number, cubeYaw: number): number {
  const quarter = Math.PI / 2;
  return cubeYaw + Math.round((nominal - cubeYaw) / quarter) * quarter;
}

function faceFromClosing(
  closingAzimuth: number,
  cubeYaw: number
): CanonicalGraspChoice['face'] {
  const local = normalizeAngle(closingAzimuth - cubeYaw);
  const index = ((Math.round(local / (Math.PI / 2)) % 4) + 4) % 4;
  return ['+x', '+y', '-x', '-y'][index] as CanonicalGraspChoice['face'];
}

export function canonicalApproachVector(
  radialAzimuth: number,
  pitch: number
): THREE.Vector3 {
  const horizontal = Math.cos(pitch);
  return new THREE.Vector3(
    Math.cos(radialAzimuth) * horizontal,
    Math.sin(radialAzimuth) * horizontal,
    -Math.sin(pitch)
  );
}

function rollGraspAboutToolAxis(
  grasp: THREE.Matrix4,
  rollOffset: number
): THREE.Matrix4 {
  if (rollOffset === 0) { return grasp; }
  return grasp.clone().multiply(new THREE.Matrix4().makeRotationZ(rollOffset));
}

function branchesFor(
  k: So101Kinematics,
  matrix: THREE.Matrix4
): SimpleIkBranch[] {
  const result = solveSimpleGraspIk(k, matrix);
  return result.type === 'success' ? result.branches : [];
}

function branchByElbow(
  branches: SimpleIkBranch[],
  elbow: SimpleIkBranch['elbow']
): SimpleIkBranch | undefined {
  return branches.find(branch => branch.elbow === elbow);
}

function hasElbowBranch(
  k: So101Kinematics,
  matrix: THREE.Matrix4,
  elbow: SimpleIkBranch['elbow']
): boolean {
  return branchByElbow(branchesFor(k, matrix), elbow) !== undefined;
}

function withPosition(matrix: THREE.Matrix4, position: THREE.Vector3): THREE.Matrix4 {
  const out = matrix.clone();
  out.setPosition(position);
  return out;
}

export function* canonicalGraspCandidates(
  k: So101Kinematics,
  source: CubePose
): Generator<CanonicalGraspChoice> {
  const radius = Math.hypot(source.x - k.panAxis.x, source.y - k.panAxis.y);
  if (
    radius < MIN_CANONICAL_GRASP_RADIUS - 1e-6 ||
    radius > MAX_CANONICAL_GRASP_RADIUS + 1e-6
  ) {
    return;
  }
  const azimuth = Math.atan2(source.y - k.panAxis.y, source.x - k.panAxis.x);
  if (
    azimuth < MIN_CANONICAL_AZIMUTH - 1e-6 ||
    azimuth > MAX_CANONICAL_AZIMUTH + 1e-6
  ) {
    return;
  }

  const closings = [azimuth + Math.PI / 2, azimuth - Math.PI / 2]
    .map(nominal => squareToCubeFace(nominal, source.yaw));
  const radial = new THREE.Vector3(Math.cos(azimuth), Math.sin(azimuth), 0);
  const pendingInward: [number, number, CanonicalGraspChoice][] = [];
  let firstReachablePitch: number | null = null;

  for (const pitch of canonicalPitchOrder(radius)) {
    const approach = canonicalApproachVector(azimuth, pitch);
    const pitchCandidates: [number, number, CanonicalGraspChoice][] = [];
    for (const closing of closings) {
      const baseGrasp = createGraspMatrix(source, closing, approach);
      const basePosition = new THREE.Vector3().setFromMatrixPosition(baseGrasp);
      const unrolledGrasp = withPosition(
        baseGrasp,
        basePosition.add(new THREE.Vector3(0, 0, CANONICAL_GRASP_Z_OFFSET))
      );
      const face = faceFromClosing(closing, source.yaw);
      const inwardNormal = new THREE.Vector3()
        .setFromMatrixColumn(unrolledGrasp, 0)
        .normalize();

      for (const rollOffset of CANONICAL_ROLL_OFFSETS) {
        const grasp = rollGraspAboutToolAxis(unrolledGrasp, rollOffset);
        const hover = createPregraspMatrix(grasp, approach, PREGRASP_DISTANCE);
        const graspPosition = new THREE.Vector3().setFromMatrixPosition(grasp);
        const lift = withPosition(
          grasp,
          graspPosition.clone().add(
            new THREE.Vector3(0, 0, Math.max(0, RECOVERY_LIFT_CUBE_Z - source.z))
          )
        );
        const graspBranches = branchesFor(k, grasp);
        const hoverBranches = branchesFor(k, hover);
        const liftBranches = branchesFor(k, lift);
        if (
          graspBranches.length === 0 ||
          hoverBranches.length === 0 ||
          liftBranches.length === 0
        ) {
          continue;
        }

        const cameraOutward = new THREE.Vector3()
          .setFromMatrixColumn(grasp, 1)
          .dot(radial);
        for (const elbow of ['up', 'down'] as const) {
          const graspBranch = branchByElbow(graspBranches, elbow);
          const hoverBranch = branchByElbow(hoverBranches, elbow);
          const liftBranch = branchByElbow(liftBranches, elbow);
          if (!graspBranch || !hoverBranch || !liftBranch) { continue; }

          let descentOk = true;
          for (let i = 1; i < N_DESCENT_CHECKS; i++) {
            const checkPosition = graspPosition.clone().addScaledVector(
              approach,
              -PREGRASP_DISTANCE * (1 - i / N_DESCENT_CHECKS)
            );
            if (!hasElbowBranch(k, withPosition(grasp, checkPosition), elbow)) {
              descentOk = false;
              break;
            }
          }
          if (!descentOk) { continue; }

          let liftOk = true;
          const liftPosition = new THREE.Vector3().setFromMatrixPosition(lift);
          for (let i = 1; i < N_DESCENT_CHECKS; i++) {
            const checkPosition = graspPosition.clone().lerp(
              liftPosition,
              i / N_DESCENT_CHECKS
            );
            if (!hasElbowBranch(k, withPosition(grasp, checkPosition), elbow)) {
              liftOk = false;
              break;
            }
          }
          if (!liftOk) { continue; }

          pitchCandidates.push([
            cameraOutward,
            Math.abs(rollOffset),
            {
              face,
              elbow,
              pitch,
              rollOffset,
              closingAzimuth: closing,
              cameraOutward,
              hoverJoints: hoverBranch.joints,
              graspJoints: graspBranch.joints,
              liftJoints: liftBranch.joints,
              hoverMatrix: hover,
              graspMatrix: grasp,
              liftMatrix: lift,
              inwardNormal
            }
          ]);
        }
      }
    }

    if (pitchCandidates.length === 0) { continue; }
    firstReachablePitch ??= pitch;
    pitchCandidates.sort((a, b) =>
      Number(a[1] > 0) - Number(b[1] > 0) ||
      Number(a[0] <= 0) - Number(b[0] <= 0) ||
      (a[2].elbow === 'up' ? 0 : 1) - (b[2].elbow === 'up' ? 0 : 1) ||
      a[1] - b[1] ||
      b[0] - a[0]
    );

    const outward = pitchCandidates.filter(candidate => candidate[0] > 0);
    if (outward.length > 0) {
      for (const [, , candidate] of outward) { yield candidate; }
      for (const [, , candidate] of pendingInward) { yield candidate; }
      return;
    }
    pendingInward.push(...pitchCandidates);
  }

  if (firstReachablePitch !== null) {
    pendingInward.sort((a, b) =>
      Number(a[1] > 0) - Number(b[1] > 0) ||
      (a[2].elbow === 'up' ? 0 : 1) - (b[2].elbow === 'up' ? 0 : 1) ||
      a[1] - b[1] ||
      b[0] - a[0]
    );
    for (const [, , candidate] of pendingInward) { yield candidate; }
  }
}

export function selectCanonicalGrasp(
  k: So101Kinematics,
  source: CubePose
): CanonicalGraspChoice | null {
  const result = canonicalGraspCandidates(k, source).next();
  return result.done === true ? null : result.value;
}
