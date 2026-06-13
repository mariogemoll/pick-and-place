// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  type CubeFace,
  type CubePose,
  DEFAULT_CUBE_POSE
} from '../visualizations/pregrasp-pose-shared/body-factories';
import { createSimplePregraspMatrix } from '../visualizations/simple-pregrasp-pose/pose';
import type { WebModel } from '../web-model';
import { bodyWorldTransform } from './fk';
import { deriveSo101Kinematics } from './kinematics';
import { solveSimplePregraspIk } from './simple-ik';

const model = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../public/so101.json', import.meta.url)),
    'utf8'
  )
) as WebModel;

const k = deriveSo101Kinematics(model);
const SIDE_FACES: CubeFace[] = ['+x', '-x', '+y', '-y'];

function gripperPose(joints: Record<string, number>): {
  position: THREE.Vector3;
  x: THREE.Vector3;
  z: THREE.Vector3;
} {
  const world = bodyWorldTransform(model, joints, 'gripper');
  return {
    position: new THREE.Vector3().setFromMatrixPosition(world),
    x: new THREE.Vector3(1, 0, 0).transformDirection(world).normalize(),
    z: new THREE.Vector3(0, 0, 1).transformDirection(world).normalize()
  };
}

describe('solveSimplePregraspIk', () => {
  // Every branch the solver returns must reproduce the requested gripper pose
  // when fed back through forward kinematics. Sweep a grid of side grasps; some
  // are unreachable (notably the far +x face, limited by wrist roll) and are
  // skipped — we only assert correctness of the solutions actually returned.
  it('returns branches that round-trip through forward kinematics', () => {
    let checked = 0;
    for (const face of SIDE_FACES) {
      for (let x = 0.12; x <= 0.32; x += 0.02) {
        for (let y = -0.15; y <= 0.15; y += 0.05) {
          const pose: CubePose = { ...DEFAULT_CUBE_POSE, x, y };
          const worldFromGripper = createSimplePregraspMatrix(face, pose);
          if (!worldFromGripper) { continue; }
          const result = solveSimplePregraspIk(k, worldFromGripper);
          if (result.type !== 'success') { continue; }

          const want = {
            position: new THREE.Vector3().setFromMatrixPosition(worldFromGripper),
            x: new THREE.Vector3(1, 0, 0)
              .transformDirection(worldFromGripper).normalize(),
            z: new THREE.Vector3(0, 0, 1)
              .transformDirection(worldFromGripper).normalize()
          };
          for (const branch of result.branches) {
            const got = gripperPose(branch.joints);
            // 0.5 mm absorbs the zeroed 0.15/0.18 mm offsets (see plan doc).
            expect(got.position.distanceTo(want.position)).toBeLessThan(5e-4);
            expect(got.x.distanceTo(want.x)).toBeLessThan(2e-3);
            expect(got.z.distanceTo(want.z)).toBeLessThan(2e-3);
            checked++;
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(100);
  });

  // The near (robot-facing) faces are the natural pick-and-place case and are
  // comfortably reachable.
  const reachableCases: { face: CubeFace; x: number; y: number }[] = [
    { face: '-x', x: 0.2, y: 0 },
    { face: '-x', x: 0.18, y: -0.08 },
    { face: '+y', x: 0.2, y: 0.06 },
    { face: '-y', x: 0.2, y: -0.06 }
  ];
  it.each(reachableCases)('finds a solution for the $face face at ($x, $y)', ({ face, x, y }) => {
    const worldFromGripper = createSimplePregraspMatrix(face, {
      ...DEFAULT_CUBE_POSE, x, y
    });
    if (!worldFromGripper) { throw new Error('expected a vertical pose'); }
    const result = solveSimplePregraspIk(k, worldFromGripper);
    expect(result.type).toBe('success');
  });

  it('keeps the wrist-roll axis (gripper z) vertical', () => {
    const worldFromGripper = createSimplePregraspMatrix('-x', {
      ...DEFAULT_CUBE_POSE, x: 0.2, y: 0
    });
    if (!worldFromGripper) { throw new Error('expected a vertical pose'); }
    const result = solveSimplePregraspIk(k, worldFromGripper);
    if (result.type !== 'success') { throw new Error('expected success'); }
    for (const branch of result.branches) {
      expect(Math.abs(gripperPose(branch.joints).z.z)).toBeGreaterThan(1 - 1e-3);
    }
  });

  it('reports unreachable for a far-away target', () => {
    const worldFromGripper = createSimplePregraspMatrix('-x', {
      ...DEFAULT_CUBE_POSE, x: 1.5
    });
    if (!worldFromGripper) { throw new Error('expected a vertical pose'); }
    expect(solveSimplePregraspIk(k, worldFromGripper).type).toBe('unreachable');
  });
});
