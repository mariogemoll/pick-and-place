// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { ARM_JOINT_NAMES } from '../ik/kinematics';
import { expectClose, kinematics, loadFixture } from './fixtures';

interface KinematicsFixture {
  kinematics: {
    panAxis: [number, number];
    shoulderLift: { radial: number; height: number };
    upperArm: { radial: number; height: number; length: number };
    lowerArm: { radial: number; height: number; length: number };
    toolLength: number;
    wristRollZeroTwist: number;
    jointLimits: Record<string, { min: number; max: number }>;
  };
}

const expected = (loadFixture('kinematics.json') as KinematicsFixture).kinematics;

// Python measures the arm off the compiled MuJoCo model; TypeScript measures it
// off the manifest that same pipeline exports. Everything else in these fixtures
// is built on these numbers, so this is the first thing to check and the first
// thing to suspect when the rest disagrees.
describe('derived SO-101 kinematics', () => {
  it('matches the Python measurement of the pan axis and shoulder', () => {
    expectClose(kinematics.panAxis.x, expected.panAxis[0], 'panAxis.x');
    expectClose(kinematics.panAxis.y, expected.panAxis[1], 'panAxis.y');
    expectClose(kinematics.shoulderLift.radial, expected.shoulderLift.radial, 'lift.radial');
    expectClose(kinematics.shoulderLift.height, expected.shoulderLift.height, 'lift.height');
  });

  it('matches the Python segment lengths and wrist twist', () => {
    for (const name of ['upperArm', 'lowerArm'] as const) {
      expectClose(kinematics[name].radial, expected[name].radial, `${name}.radial`);
      expectClose(kinematics[name].height, expected[name].height, `${name}.height`);
      expectClose(kinematics[name].length, expected[name].length, `${name}.length`);
    }
    expectClose(kinematics.toolLength, expected.toolLength, 'toolLength');
    expectClose(
      kinematics.wristRollZeroTwist, expected.wristRollZeroTwist, 'wristRollZeroTwist'
    );
  });

  it('matches the Python joint limits', () => {
    expect(Object.keys(expected.jointLimits).sort()).toEqual([...ARM_JOINT_NAMES].sort());
    for (const name of ARM_JOINT_NAMES) {
      expectClose(kinematics.jointLimits[name].min, expected.jointLimits[name].min, `${name}.min`);
      expectClose(kinematics.jointLimits[name].max, expected.jointLimits[name].max, `${name}.max`);
    }
  });
});
