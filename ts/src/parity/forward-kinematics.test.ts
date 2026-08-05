// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, it } from 'vitest';

import { bodyWorldTransform } from '../ik/fk';
import { GRIPPER_TARGET_POSITION } from '../visualizations/grasp-pose-shared/body-factories';
import {
  expectVectorClose,
  type FixtureJoints,
  loadFixture,
  webModel
} from './fixtures';

interface FkCase {
  label: string;
  joints: FixtureJoints;
  tip: [number, number, number];
}

const fixture = loadFixture('forward_kinematics.json') as { cases: FkCase[] };

// Python solves the planar chain in closed form; TypeScript walks the model's
// body tree. Two derivations that share nothing but the arm they describe, so
// the agreement is worth more than a round trip through the IK — but the closed
// form is an approximation of the tree, so allow the sub-millimetre gap the
// Python side documents rather than the fixture tolerance.
const TIP_TOLERANCE_M = 5e-4;

describe('forward kinematics', () => {
  it.each(fixture.cases)('places the tip where Python does for $label', testCase => {
    const world = bodyWorldTransform(webModel, testCase.joints, 'gripper');
    const tip = GRIPPER_TARGET_POSITION.clone().applyMatrix4(world);
    expectVectorClose(
      tip,
      testCase.tip,
      testCase.label,
      TIP_TOLERANCE_M
    );
  });
});
