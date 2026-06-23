// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  CUBE_HALF_SIZE,
  DEFAULT_CUBE_POSE,
  GRIPPER_TARGET_POSITION
} from '../visualizations/grasp-pose-shared/body-factories';
import { createSimpleGraspMatrix } from '../visualizations/simple-grasp-pose/pose';
import { type So101Kinematics } from './kinematics';
import { solveSimpleGraspIk } from './simple-ik';

// Cube-center z where the gripper tip (IK contact) is exactly 1 cm above the
// top of a ground cube: top-of-ground-cube (2×half) + 1 cm clearance.
export const CUBE_Z_1CM_OVER_GROUND_TOP = CUBE_HALF_SIZE * 2 + 0.010;

// Reachable workspace for the *simple* vertical grasp with a cube resting on
// the ground. Because the cube is on the floor and the pose keeps the gripper
// vertical, the IK target (the jaw contact projected onto the roll axis) sits at
// a fixed height, with the wrist straight above it. The reachable set therefore
// collapses to a closed-form annular SECTOR around the shoulder_pan axis:
//
//   * radial band  — set purely by the planar 2R reach plus the
//     shoulder_lift / elbow_flex / wrist_flex limits. The wrist height is
//     constant, so this band is azimuth-independent.
//   * azimuth band — exactly the shoulder_pan range (shoulder_pan = -azimuth).
//
// wrist_roll never enters the region boundary: rollAngle = (faceNormalAzimuth -
// targetAzimuth) - twist, and its forbidden arc (the gap past the asymmetric
// roll limits) is narrower than the 90 deg spacing of a cube's four vertical
// faces. So at most one face is ever roll-blocked and at least three remain
// graspable -- hence for *any* cube yaw at least one face solves, everywhere in
// the sector. `anyYawReachable` records that this holds for the loaded model.

export interface SimpleWorkspaceSector {
  // World location of the vertical shoulder_pan axis.
  panAxis: THREE.Vector2;
  // Horizontal distance from the pan axis to the IK target, in metres. The IK
  // target is the cube-center offset outward along the grasped face normal by
  // `faceOffset`, so the graspable cube-center band is this band +/- faceOffset.
  radial: { min: number; max: number };
  // World azimuth of the target, measured from the pan axis, in radians.
  azimuth: { min: number; max: number };
  // Height of the IK target above the floor, in metres (constant: cube on
  // ground, vertical pose).
  targetHeight: number;
  // Radial distance from the cube center to the IK target along the grasped
  // face normal, in metres.
  faceOffset: number;
  // True when every cube yaw has at least one graspable face throughout the
  // sector (the wrist_roll argument above).
  anyYawReachable: boolean;
}

// Derive the IK target height and face offset for a cube whose center is at
// `cubeCenterZ`. The target height equals the cube-center z (face-center z for
// a vertical face). The face offset (cube-center → IK target, horizontal) is
// z-independent but is computed from the grasp geometry to stay in sync.
function deriveSampleGeometry(cubeCenterZ: number): { height: number; faceOffset: number } {
  const samplePose = { ...DEFAULT_CUBE_POSE, z: cubeCenterZ };
  const worldFromGripper = createSimpleGraspMatrix('+x', samplePose);
  if (worldFromGripper === undefined) {
    throw new Error(`Simple grasp matrix undefined at cubeCenterZ=${cubeCenterZ}`);
  }
  const sampleTarget = GRIPPER_TARGET_POSITION.clone().applyMatrix4(worldFromGripper);
  return {
    height: sampleTarget.z,
    faceOffset: Math.hypot(sampleTarget.x - samplePose.x, sampleTarget.y - samplePose.y)
  };
}

// Build the world-from-gripper matrix for a vertical grasp whose IK target lands
// at `target`, with the jaws closing along the horizontal `closingAzimuth`.
function verticalGraspMatrix(
  target: THREE.Vector3,
  closingAzimuth: number
): THREE.Matrix4 {
  const x = new THREE.Vector3(Math.cos(closingAzimuth), Math.sin(closingAzimuth), 0);
  const z = new THREE.Vector3(0, 0, 1);
  const y = new THREE.Vector3().crossVectors(z, x);
  const matrix = new THREE.Matrix4().makeBasis(x, y, z);
  // target = origin + R * GRIPPER_TARGET_POSITION, and R maps (0,0,z) -> (0,0,z),
  // so the gripper origin is simply the target lifted by -GRIPPER_TARGET z.
  matrix.setPosition(
    target.x,
    target.y,
    target.z - GRIPPER_TARGET_POSITION.z
  );
  return matrix;
}

// Does a vertical grasp at horizontal distance `radial` along azimuth 0 solve?
// Closing along the radial keeps wrist_roll near zero and pan at zero, isolating
// the radial reach + flex limits.
function radialReachable(
  k: So101Kinematics,
  radial: number,
  targetHeight: number
): boolean {
  const target = new THREE.Vector3(
    k.panAxis.x + radial,
    k.panAxis.y,
    targetHeight
  );
  return solveSimpleGraspIk(k, verticalGraspMatrix(target, 0)).type === 'success';
}

// Bisect the radial reach boundary between a known-reachable and a known-
// unreachable distance.
function bisectRadial(
  k: So101Kinematics,
  reachable: number,
  unreachable: number,
  targetHeight: number
): number {
  let lo = reachable;
  let hi = unreachable;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (radialReachable(k, mid, targetHeight)) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

export function computeSimpleWorkspace(k: So101Kinematics): SimpleWorkspaceSector {
  return computeSimpleWorkspaceForCubeZ(k, DEFAULT_CUBE_POSE.z);
}

// Compute the simple-grasp workspace for a cube whose center sits at
// `cubeCenterZ`. Everything else is identical to `computeSimpleWorkspace`.
export function computeSimpleWorkspaceForCubeZ(
  k: So101Kinematics,
  cubeCenterZ: number
): SimpleWorkspaceSector {
  const { height, faceOffset } = deriveSampleGeometry(cubeCenterZ);

  const step = 0.002;
  let firstHit = NaN;
  let lastHit = NaN;
  for (let r = 0.01; r <= 0.40 + 1e-9; r += step) {
    if (radialReachable(k, r, height)) {
      if (Number.isNaN(firstHit)) { firstHit = r; }
      lastHit = r;
    }
  }
  if (Number.isNaN(firstHit)) {
    throw new Error('No reachable radial distance found for the simple workspace');
  }
  const radialMin = bisectRadial(k, firstHit, firstHit - step, height);
  const radialMax = bisectRadial(k, lastHit, lastHit + step, height);

  const pan = k.jointLimits.shoulder_pan;
  const azimuth = { min: -pan.max, max: -pan.min };
  const roll = k.jointLimits.wrist_roll;
  const forbiddenArc = 2 * Math.PI - (roll.max - roll.min);
  const anyYawReachable = forbiddenArc < Math.PI / 2;

  return {
    panAxis: k.panAxis.clone(),
    radial: { min: radialMin, max: radialMax },
    azimuth,
    targetHeight: height,
    faceOffset,
    anyYawReachable
  };
}

// Maximum horizontal (XY) reach of the gripper contact point at a fixed
// `targetHeight`, for any joint configuration (not restricted to the grasp
// pose). For each (shoulder_lift, elbow_flex) pair, wrist_flex is solved
// analytically (two branches from the sin inverse) so there is no wrist_flex
// scan — the result is exact within the (lift × flex) grid resolution.
export function computeArmWorkspaceAtHeight(
  k: So101Kinematics,
  targetHeight: number
): SimpleWorkspaceSector {
  const pan = k.jointLimits.shoulder_pan;
  const azimuth = { min: -pan.max, max: -pan.min };
  const roll = k.jointLimits.wrist_roll;
  const forbiddenArc = 2 * Math.PI - (roll.max - roll.min);
  const anyYawReachable = forbiddenArc < Math.PI / 2;

  const upperRest = Math.atan2(k.upperArm.height, k.upperArm.radial);
  const lowerRest = Math.atan2(k.lowerArm.height, k.lowerArm.radial);
  const { min: liftMin, max: liftMax } = k.jointLimits.shoulder_lift;
  const { min: flexMin, max: flexMax } = k.jointLimits.elbow_flex;
  const { min: wflexMin, max: wflexMax } = k.jointLimits.wrist_flex;

  const STEPS = 300;
  let radialMin = Infinity;
  let radialMax = -Infinity;

  for (let i = 0; i <= STEPS; i++) {
    const lift = liftMin + (i / STEPS) * (liftMax - liftMin);
    const theta1 = upperRest - lift;
    const wr1 = k.shoulderLift.radial + k.upperArm.length * Math.cos(theta1);
    const wh1 = k.shoulderLift.height + k.upperArm.length * Math.sin(theta1);
    for (let j = 0; j <= STEPS; j++) {
      const flex = flexMin + (j / STEPS) * (flexMax - flexMin);
      const theta2 = lowerRest - lift - flex;
      const wr = wr1 + k.lowerArm.length * Math.cos(theta2);
      const wh = wh1 + k.lowerArm.length * Math.sin(theta2);

      // Analytically solve for the tool pitch that places the tip at targetHeight.
      const sinArg = (targetHeight - wh) / k.toolLength;
      if (Math.abs(sinArg) > 1) { continue; }
      const asinVal = Math.asin(sinArg);
      for (const toolPitch of [asinVal, Math.PI - asinVal]) {
        const wflex = -(lift + flex + toolPitch);
        if (wflex < wflexMin || wflex > wflexMax) { continue; }
        const targetR = wr + k.toolLength * Math.cos(toolPitch);
        if (targetR > radialMax) { radialMax = targetR; }
        if (targetR < radialMin) { radialMin = targetR; }
      }
    }
  }

  if (!isFinite(radialMax)) {
    throw new Error(`No configuration reaches height ${targetHeight} m`);
  }

  return {
    panAxis: k.panAxis.clone(),
    radial: { min: Math.max(0, radialMin), max: radialMax },
    azimuth,
    targetHeight,
    faceOffset: 0,
    anyYawReachable
  };
}

// Maximum horizontal (XY) reach of the arm at any joint configuration,
// projected vertically onto the floor. Uses planar-arm FK (shoulder_lift ×
// elbow_flex × wrist_flex) — not restricted to the simple grasp pose.
// wrist_roll is omitted since it only spins the gripper without moving it.
// `targetHeight` is NaN; the sector spans from the minimum to the maximum
// radial reach subject to the arm staying above the floor.
export function computeGlobalXyWorkspace(k: So101Kinematics): SimpleWorkspaceSector {
  const { faceOffset } = deriveSampleGeometry(DEFAULT_CUBE_POSE.z);
  const pan = k.jointLimits.shoulder_pan;
  const azimuth = { min: -pan.max, max: -pan.min };
  const roll = k.jointLimits.wrist_roll;
  const forbiddenArc = 2 * Math.PI - (roll.max - roll.min);
  const anyYawReachable = forbiddenArc < Math.PI / 2;

  // Rest angles of the two links (measured from the radial axis in the
  // radial-height plane, as used by the 2R solver).
  const upperRest = Math.atan2(k.upperArm.height, k.upperArm.radial);
  const lowerRest = Math.atan2(k.lowerArm.height, k.lowerArm.radial);

  const { min: liftMin, max: liftMax } = k.jointLimits.shoulder_lift;
  const { min: flexMin, max: flexMax } = k.jointLimits.elbow_flex;
  const { min: wflexMin, max: wflexMax } = k.jointLimits.wrist_flex;

  const STEPS = 80;
  let radialMin = Infinity;
  let radialMax = -Infinity;

  for (let i = 0; i <= STEPS; i++) {
    const lift = liftMin + (i / STEPS) * (liftMax - liftMin);
    const theta1 = upperRest - lift; // upper-arm geometric angle from radial
    const wristR1 = k.shoulderLift.radial + k.upperArm.length * Math.cos(theta1);
    const wristH1 = k.shoulderLift.height + k.upperArm.length * Math.sin(theta1);
    for (let j = 0; j <= STEPS; j++) {
      const flex = flexMin + (j / STEPS) * (flexMax - flexMin);
      const theta2 = lowerRest - lift - flex; // lower-arm absolute angle
      const wristR = wristR1 + k.lowerArm.length * Math.cos(theta2);
      const wristH = wristH1 + k.lowerArm.length * Math.sin(theta2);
      for (let l = 0; l <= STEPS; l++) {
        const wflex = wflexMin + (l / STEPS) * (wflexMax - wflexMin);
        // toolPitch: pitch of the tool (from wrist to target) in the
        // radial-height plane; derived from wristFlex = -lift-flex-toolPitch.
        const toolPitch = -(lift + flex + wflex);
        const targetH = wristH + k.toolLength * Math.sin(toolPitch);
        if (targetH < 0) { continue; } // gripper underground
        const targetR = wristR + k.toolLength * Math.cos(toolPitch);
        if (targetR > radialMax) { radialMax = targetR; }
        if (targetR < radialMin) { radialMin = targetR; }
      }
    }
  }

  if (!isFinite(radialMax)) {
    throw new Error('No reachable position found for the global workspace scan');
  }

  return {
    panAxis: k.panAxis.clone(),
    radial: { min: Math.max(0, radialMin), max: radialMax },
    azimuth,
    targetHeight: NaN,
    faceOffset,
    anyYawReachable
  };
}

// The cube-center radial band graspable for *any* yaw: the target band shrunk by
// the face offset on both ends (the worst-case grasped face can sit a face
// offset either side of the center).
export function anyYawCubeCenterBand(
  sector: SimpleWorkspaceSector
): { min: number; max: number } {
  return {
    min: sector.radial.min + sector.faceOffset,
    max: sector.radial.max - sector.faceOffset
  };
}

// Axis-aligned world bounding box (metres) of the any-yaw cube-center sector.
// Used to bound the X/Y placement sliders to the usable workspace.
export function sectorBoundingBox(
  sector: SimpleWorkspaceSector
): { x: { min: number; max: number }; y: { min: number; max: number } } {
  const band = anyYawCubeCenterBand(sector);
  const { min: azMin, max: azMax } = sector.azimuth;
  // Angles where cos/sin reach extrema, kept only when inside the swept sector.
  const angles = [azMin, azMax];
  for (const a of [0, Math.PI / 2, Math.PI, -Math.PI / 2, -Math.PI]) {
    if (a >= azMin && a <= azMax) { angles.push(a); }
  }
  const xs: number[] = [];
  const ys: number[] = [];
  for (const radius of [band.min, band.max]) {
    for (const a of angles) {
      xs.push(sector.panAxis.x + radius * Math.cos(a));
      ys.push(sector.panAxis.y + radius * Math.sin(a));
    }
  }
  return {
    x: { min: Math.min(...xs), max: Math.max(...xs) },
    y: { min: Math.min(...ys), max: Math.max(...ys) }
  };
}
