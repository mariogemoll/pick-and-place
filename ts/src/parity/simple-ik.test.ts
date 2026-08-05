// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { solveSimpleGraspIk } from '../ik/simple-ik';
import {
  expectJointsClose,
  type FixtureJoints,
  kinematics,
  loadFixture,
  matrixFromFixture
} from './fixtures';

interface IkBranch {
  elbow: 'up' | 'down';
  joints: FixtureJoints;
}

interface IkCase {
  label: string;
  worldFromGripper: number[];
  branches: IkBranch[];
}

const fixture = loadFixture('simple_ik.json') as { cases: IkCase[] };

// Both languages solve the same closed form, so they must agree on the branches
// *and* on which poses have none. The empty-branch cases are the ones worth
// having: an unreachable pose that still returns joints is a pose the arm will
// be commanded into.
describe('solveSimpleGraspIk', () => {
  it.each(fixture.cases)('matches Python for $label', testCase => {
    const result = solveSimpleGraspIk(kinematics, matrixFromFixture(testCase.worldFromGripper));

    if (testCase.branches.length === 0) {
      expect(result.type, `${testCase.label} should be unreachable`).toBe('unreachable');
      return;
    }
    expect(result.type, `${testCase.label} should solve`).toBe('success');
    if (result.type !== 'success') { return; }

    expect(result.branches.map(branch => branch.elbow))
      .toEqual(testCase.branches.map(branch => branch.elbow));
    for (const [index, branch] of result.branches.entries()) {
      expectJointsClose(
        branch.joints, testCase.branches[index].joints, `${testCase.label}[${branch.elbow}]`
      );
    }
  });
});
