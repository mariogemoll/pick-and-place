// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { SAFETY_MARGIN } from '../grasp-pose-shared/bodies';
import {
  createGripperFromContactMatrix,
  createWorldFromCubeContactMatrix,
  type CubeFace,
  DEFAULT_CUBE_POSE } from '../grasp-pose-shared/body-factories';
import { createSimpleGraspMatrix } from './pose';

const SIDE_FACES: CubeFace[] = ['+x', '-x', '+y', '-y'];

describe('simple grasp pose', () => {
  it.each(SIDE_FACES)('keeps the gripper vertical for %s', face => {
    const matrix = createSimpleGraspMatrix(face, DEFAULT_CUBE_POSE);
    expect(matrix).toBeDefined();
    if (!matrix) { throw new Error('Expected a valid simple grasp pose'); }

    const gripperUp = new THREE.Vector3(0, 0, 1).transformDirection(matrix);
    expect(gripperUp.distanceTo(new THREE.Vector3(0, 0, 1))).toBeLessThan(1e-12);
  });

  it.each(SIDE_FACES)('places the jaw 1 cm away from %s', face => {
    const matrix = createSimpleGraspMatrix(face, DEFAULT_CUBE_POSE);
    if (!matrix) { throw new Error('Expected a valid simple grasp pose'); }
    const jawContact = new THREE.Vector3().setFromMatrixPosition(
      matrix.clone().multiply(createGripperFromContactMatrix())
    );
    const cubeContact = new THREE.Vector3().setFromMatrixPosition(
      createWorldFromCubeContactMatrix(face, DEFAULT_CUBE_POSE)
    );

    expect(jawContact.distanceTo(cubeContact)).toBeCloseTo(SAFETY_MARGIN);
  });

  it('allows yaw around the vertical axis', () => {
    const matrix = createSimpleGraspMatrix('+x', {
      ...DEFAULT_CUBE_POSE,
      yaw: Math.PI / 3
    });

    expect(matrix).toBeDefined();
  });

  it('has no solution when the selected face is tilted', () => {
    expect(createSimpleGraspMatrix('+x', {
      ...DEFAULT_CUBE_POSE,
      pitch: 0.1
    })).toBeUndefined();
    expect(createSimpleGraspMatrix('+y', {
      ...DEFAULT_CUBE_POSE,
      roll: 0.1
    })).toBeUndefined();
  });
});
