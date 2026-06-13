// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import type * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { bodyWorldTransform } from '../../ik/fk';
import { deriveSo101Kinematics } from '../../ik/kinematics';
import {
  anyYawCubeCenterBand,
  computeSimpleWorkspaceForCubeZ,
  CUBE_Z_1CM_OVER_GROUND_TOP
} from '../../ik/workspace';
import type { WebModel } from '../../web-model';
import {
  CUBE_HALF_SIZE,
  type CubePose,
  DEFAULT_CUBE_POSE,
  GRIPPER_TARGET_POSITION } from '../pregrasp-pose-shared/body-factories';
import {
  computeTrajectory,
  GRIPPER_CLOSED,
  GRIPPER_OPEN,
  NEUTRAL_FRAME
} from './trajectory';

const model = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../../public/so101.json', import.meta.url)),
    'utf8'
  )
) as WebModel;

const kinematics = deriveSo101Kinematics(model);
const sourcePose = { ...DEFAULT_CUBE_POSE, x: 0.2, y: -0.08 };
const targetPose = { ...DEFAULT_CUBE_POSE, x: 0.2, y: 0.08 };

function gripperTarget(joints: Record<string, number>): THREE.Vector3 {
  return GRIPPER_TARGET_POSITION.clone().applyMatrix4(
    bodyWorldTransform(model, joints, 'gripper')
  );
}

describe('pick-and-place trajectory', () => {
  it('lowers vertically from hover to pregrasp while keeping the gripper open', () => {
    const trajectory = computeTrajectory(kinematics, sourcePose, targetPose);
    if (!trajectory) { throw new Error('expected source pose to be reachable'); }

    const start = trajectory.evaluate(0);
    const hover = trajectory.evaluate(2);
    const halfwayDown = trajectory.evaluate(2.5);
    const pregrasp = trajectory.evaluate(3);
    const hoverTarget = gripperTarget(hover.joints);
    const halfwayTarget = gripperTarget(halfwayDown.joints);
    const pregraspTarget = gripperTarget(pregrasp.joints);

    expect(start).toEqual({ ...NEUTRAL_FRAME, sourceCube: sourcePose });
    expect(trajectory.duration).toBe(6);
    expect(hover.gripper).toBeCloseTo(GRIPPER_OPEN);
    expect(halfwayDown.gripper).toBeCloseTo(GRIPPER_OPEN);
    expect(pregrasp.gripper).toBeCloseTo(GRIPPER_OPEN);
    expect(hoverTarget.x).toBeCloseTo(pregraspTarget.x, 3);
    expect(hoverTarget.y).toBeCloseTo(pregraspTarget.y, 3);
    // Source hover puts the tip contact 1 cm above the cube top (z = 4 cm),
    // i.e. 2.5 cm above the grasp's face-center contact.
    expect(hoverTarget.z - pregraspTarget.z).toBeCloseTo(0.025, 3);
    expect(halfwayTarget.z).toBeCloseTo((hoverTarget.z + pregraspTarget.z) / 2, 3);
  });

  it('closes the gripper and pushes the cube flush against the fixed jaw', () => {
    const trajectory = computeTrajectory(kinematics, sourcePose, targetPose);
    if (!trajectory) { throw new Error('expected source pose to be reachable'); }

    const pregrasp = trajectory.evaluate(3);
    // The close runs over stage 3 and completes at t = 4, where stage 4 (the
    // carry) takes over – so sample the fully-closed frame just before then.
    const closed = trajectory.evaluate(4 - 1e-6);

    // Gripper goes from open to closed across stage 3.
    expect(pregrasp.gripper).toBeCloseTo(GRIPPER_OPEN);
    expect(closed.gripper).toBeCloseTo(GRIPPER_CLOSED);

    // The arm stays put while the gripper closes.
    expect(closed.joints).toEqual(pregrasp.joints);

    // The cube starts at its source pose and is shoved toward the fixed jaw by
    // the safety margin. The grasp face is chosen to make the whole carry
    // feasible, so the slide is along x but its sign depends on that choice.
    expect(pregrasp.sourceCube).toEqual(sourcePose);
    const slide = Math.hypot(
      closed.sourceCube.x - sourcePose.x,
      closed.sourceCube.y - sourcePose.y
    );
    expect(slide).toBeCloseTo(0.01, 3);
    expect(Math.abs(closed.sourceCube.x - sourcePose.x)).toBeCloseTo(0.01, 3);
    expect(closed.sourceCube.z).toBeCloseTo(CUBE_HALF_SIZE);
  });

  it('carries the cube up first, then over to the drop hover above the target', () => {
    const trajectory = computeTrajectory(kinematics, sourcePose, targetPose);
    if (!trajectory) { throw new Error('expected source pose to be reachable'); }

    const grasped = trajectory.evaluate(4);
    const liftingEarly = trajectory.evaluate(4.3);
    const dropped = trajectory.evaluate(trajectory.duration);

    // The gripper stays closed for the whole carry.
    expect(grasped.gripper).toBeCloseTo(GRIPPER_CLOSED);
    expect(liftingEarly.gripper).toBeCloseTo(GRIPPER_CLOSED);
    expect(dropped.gripper).toBeCloseTo(GRIPPER_CLOSED);

    // Early in the carry the cube rises before reaching its cruise height.
    expect(liftingEarly.sourceCube.z).toBeGreaterThan(grasped.sourceCube.z + 0.002);

    // It ends one hover (0.5 cm) above the target, sitting over the target x/y.
    expect(dropped.sourceCube.z).toBeCloseTo(CUBE_HALF_SIZE + 0.005, 3);
    expect(dropped.sourceCube.x).toBeCloseTo(targetPose.x, 3);
    expect(dropped.sourceCube.y).toBeCloseTo(targetPose.y, 3);
  });

  it('carries without a discontinuous wrist flip', () => {
    // The grasp face/elbow is validated across the whole carry, so the arm
    // never has to drive a joint past its limit mid-move (which the per-frame
    // IK would resolve by silently lerping joints, whipping the gripper). Joint
    // angles should therefore stay continuous frame to frame across the carry.
    const trajectory = computeTrajectory(kinematics, sourcePose, targetPose);
    if (!trajectory) { throw new Error('expected source pose to be reachable'); }

    const maxStep = (15 * Math.PI) / 180;
    let prev = trajectory.evaluate(4).joints;
    for (let t = 4.05; t <= trajectory.duration + 1e-9; t += 0.05) {
      const joints = trajectory.evaluate(t).joints;
      expect(Math.abs(joints.wrist_roll - prev.wrist_roll)).toBeLessThan(maxStep);
      expect(Math.abs(joints.shoulder_pan - prev.shoulder_pan)).toBeLessThan(maxStep);
      prev = joints;
    }
  });

  it('has vertical endpoint directions and is C2 at every carry waypoint', () => {
    const trajectory = computeTrajectory(kinematics, sourcePose, targetPose);
    if (!trajectory) { throw new Error('expected source pose to be reachable'); }

    const carryStart = 4;
    const atPhase = (phase: number): CubePose =>
      trajectory.evaluate(carryStart + phase * 2).sourceCube;
    const horizontalDistance = (a: CubePose, b: CubePose): number =>
      Math.hypot(a.x - b.x, a.y - b.y);
    const components = (pose: CubePose): number[] => [pose.x, pose.y, pose.z];
    const derivative = (
      before: CubePose, after: CubePose, interval: number
    ): number[] => components(after).map(
      (value, i) => (value - components(before)[i]) / interval
    );
    const secondDerivative = (
      a: CubePose, b: CubePose, c: CubePose, interval: number
    ): number[] => components(a).map(
      (value, i) =>
        (value - 2 * components(b)[i] + components(c)[i]) / interval ** 2
    );
    const vectorDistance = (a: number[], b: number[]): number =>
      Math.hypot(...a.map((value, i) => value - b[i]));
    const magnitude = (v: number[]): number => Math.hypot(...v);

    // The endpoint tangent directions are strictly vertical.
    const endpointH = 0.01;
    const start = atPhase(0);
    const justAfterStart = atPhase(endpointH);
    const justBeforeEnd = atPhase(1 - endpointH);
    const end = atPhase(1);
    expect(horizontalDistance(start, justAfterStart)).toBeLessThan(
      (justAfterStart.z - start.z) * 0.01
    );
    expect(justAfterStart.z).toBeGreaterThan(start.z);
    expect(horizontalDistance(justBeforeEnd, end)).toBeLessThan(
      (justBeforeEnd.z - end.z) * 0.01
    );
    expect(end.z).toBeLessThan(justBeforeEnd.z);

    // Each internal waypoint keeps moving, while velocity and acceleration
    // match on both sides of the join after arc-length timing is applied.
    const h = 0.001;
    const waypointPhases = trajectory.carryProfile().filter(
      point => point.waypoint === 'Cruise start' ||
        point.waypoint === 'Cruise end'
    ).map(point => point.phase);
    for (const waypoint of waypointPhases) {
      const center = atPhase(waypoint);
      const leftVelocity = derivative(atPhase(waypoint - h), center, h);
      const rightVelocity = derivative(center, atPhase(waypoint + h), h);
      const leftAcceleration = secondDerivative(
        center, atPhase(waypoint - h), atPhase(waypoint - 2 * h), h
      );
      const rightAcceleration = secondDerivative(
        atPhase(waypoint + 2 * h), atPhase(waypoint + h), center, h
      );
      expect(magnitude(leftVelocity)).toBeGreaterThan(0.005);
      expect(magnitude(rightVelocity)).toBeGreaterThan(0.005);
      expect(vectorDistance(leftVelocity, rightVelocity)).toBeLessThan(1e-4);
      expect(vectorDistance(leftAcceleration, rightAcceleration)).toBeLessThan(0.03);
    }

    expect(trajectory.carryProfile().filter(
      point => point.waypoint !== undefined
    ).map(point => point.waypoint)).toEqual([
      'Pick', 'Cruise start', 'Cruise end', 'Place'
    ]);
  });

  it('eases only at the carry endpoints and keeps an even middle speed', () => {
    const trajectory = computeTrajectory(kinematics, sourcePose, targetPose);
    if (!trajectory) { throw new Error('expected source pose to be reachable'); }

    const atPhase = (phase: number): CubePose =>
      trajectory.evaluate(4 + phase * 2).sourceCube;
    const distance = (a: CubePose, b: CubePose): number =>
      Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
    const step = 0.02;
    const speeds = [0.35, 0.45, 0.55, 0.65].map(
      phase => distance(atPhase(phase), atPhase(phase + step)) / step
    );
    expect(Math.max(...speeds) / Math.min(...speeds)).toBeLessThan(1.03);
    expect(distance(atPhase(0), atPhase(step))).toBeLessThan(
      distance(atPhase(0.49), atPhase(0.49 + step)) * 0.1
    );
    expect(distance(atPhase(1 - step), atPhase(1))).toBeLessThan(
      distance(atPhase(0.49), atPhase(0.49 + step)) * 0.1
    );
  });

  it('carries along an arc, keeping the cube inside the annular sector', () => {
    // Source and target sit at a high radius on opposite sides of the sector.
    // The straight chord between them bows inward through the hole, but the
    // polar (radius/azimuth) carry sweeps an arc that never leaves the band.
    const workspace = computeSimpleWorkspaceForCubeZ(
      kinematics, CUBE_Z_1CM_OVER_GROUND_TOP
    );
    const band = anyYawCubeCenterBand(workspace);
    const ax = workspace.panAxis.x;
    const ay = workspace.panAxis.y;
    const radius = band.min + 0.85 * (band.max - band.min);
    const azimuth = (75 * Math.PI) / 180;
    const source = {
      ...DEFAULT_CUBE_POSE,
      x: ax + radius * Math.cos(-azimuth),
      y: ay + radius * Math.sin(-azimuth)
    };
    const target = {
      ...DEFAULT_CUBE_POSE,
      x: ax + radius * Math.cos(azimuth),
      y: ay + radius * Math.sin(azimuth)
    };

    // The naive straight-line carry would pass this close to the pan axis...
    const chordMidRadius = Math.hypot(
      (source.x + target.x) / 2 - ax, (source.y + target.y) / 2 - ay
    );
    expect(chordMidRadius).toBeLessThan(band.min);

    const trajectory = computeTrajectory(kinematics, source, target);
    if (!trajectory) { throw new Error('expected this pose pair to be reachable'); }

    // ...yet every sampled cube centre across the carry stays within the band.
    for (let t = 4; t <= trajectory.duration + 1e-9; t += 0.05) {
      const cube = trajectory.evaluate(t).sourceCube;
      const r = Math.hypot(cube.x - ax, cube.y - ay);
      expect(r).toBeGreaterThanOrEqual(band.min - 1e-6);
      expect(r).toBeLessThanOrEqual(band.max + 1e-6);
    }
  });
});
