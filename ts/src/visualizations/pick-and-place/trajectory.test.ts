// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import type * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { bodyWorldTransform } from '../../ik/fk';
import { deriveSo101Kinematics } from '../../ik/kinematics';
import type { WebModel } from '../../web-model';
import {
  DEFAULT_CUBE_POSE,
  GRIPPER_TARGET_POSITION
} from '../pregrasp-pose-shared/body-factories';
import { computeTrajectory, GRIPPER_OPEN, NEUTRAL_FRAME } from './trajectory';

const model = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../../public/so101.json', import.meta.url)),
    'utf8'
  )
) as WebModel;

const kinematics = deriveSo101Kinematics(model);
const sourcePose = { ...DEFAULT_CUBE_POSE, x: 0.2, y: -0.08 };

function gripperTarget(joints: Record<string, number>): THREE.Vector3 {
  return GRIPPER_TARGET_POSITION.clone().applyMatrix4(
    bodyWorldTransform(model, joints, 'gripper')
  );
}

describe('pick-and-place trajectory', () => {
  it('lowers vertically from hover to pregrasp while keeping the gripper open', () => {
    const trajectory = computeTrajectory(kinematics, sourcePose);
    if (!trajectory) { throw new Error('expected source pose to be reachable'); }

    const start = trajectory.evaluate(0);
    const hover = trajectory.evaluate(2);
    const halfwayDown = trajectory.evaluate(2.5);
    const pregrasp = trajectory.evaluate(trajectory.duration);
    const hoverTarget = gripperTarget(hover.joints);
    const halfwayTarget = gripperTarget(halfwayDown.joints);
    const pregraspTarget = gripperTarget(pregrasp.joints);

    expect(start).toEqual(NEUTRAL_FRAME);
    expect(trajectory.duration).toBe(3);
    expect(hover.gripper).toBeCloseTo(GRIPPER_OPEN);
    expect(halfwayDown.gripper).toBeCloseTo(GRIPPER_OPEN);
    expect(pregrasp.gripper).toBeCloseTo(GRIPPER_OPEN);
    expect(hoverTarget.x).toBeCloseTo(pregraspTarget.x, 3);
    expect(hoverTarget.y).toBeCloseTo(pregraspTarget.y, 3);
    expect(hoverTarget.z - pregraspTarget.z).toBeCloseTo(0.01, 3);
    expect(halfwayTarget.z).toBeCloseTo((hoverTarget.z + pregraspTarget.z) / 2, 3);
  });
});
