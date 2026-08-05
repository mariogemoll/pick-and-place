// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { canonicalGraspCandidates, selectCanonicalGrasp } from '../ik/canonical-grasp';
import {
  expectClose,
  expectJointsClose,
  expectMatrixClose,
  expectVectorClose,
  type FixtureJoints,
  type FixturePose,
  kinematics,
  loadFixture
} from './fixtures';

interface CandidateSummary {
  face: string;
  elbow: string;
  pitch: number;
  rollOffset: number;
}

interface SelectedGrasp extends CandidateSummary {
  closingAzimuth: number;
  cameraOutward: number;
  inwardNormal: [number, number, number];
  hoverJoints: FixtureJoints;
  graspJoints: FixtureJoints;
  liftJoints: FixtureJoints;
  hoverMatrix: number[];
  graspMatrix: number[];
  liftMatrix: number[];
}

interface GraspCase {
  pose: FixturePose;
  candidatePrefix: CandidateSummary[];
  selected: SelectedGrasp | null;
}

const fixture = loadFixture('grasp.json') as { cases: GraspCase[] };

function summarize(candidate: CandidateSummary): CandidateSummary {
  return {
    face: candidate.face,
    elbow: candidate.elbow,
    pitch: candidate.pitch,
    rollOffset: candidate.rollOffset
  };
}

// The grasp search is an ordering, not a predicate: the planner takes the first
// candidate that survives, so a port that finds the same grasps in a different
// order picks a different grasp on the next cube. Both the winner and the head
// of the stream have to match.
describe('canonical grasp selection', () => {
  it.each(fixture.cases)('matches Python at ($pose.x, $pose.y)', testCase => {
    const choice = selectCanonicalGrasp(kinematics, testCase.pose);

    if (testCase.selected === null) {
      expect(choice, 'no grasp exists here').toBeNull();
      return;
    }
    expect(choice).not.toBeNull();
    if (choice === null) { return; }

    const expected = testCase.selected;
    expect(choice.face).toBe(expected.face);
    expect(choice.elbow).toBe(expected.elbow);
    expectClose(choice.pitch, expected.pitch, 'pitch');
    expectClose(choice.rollOffset, expected.rollOffset, 'rollOffset');
    expectClose(choice.closingAzimuth, expected.closingAzimuth, 'closingAzimuth');
    expectClose(choice.cameraOutward, expected.cameraOutward, 'cameraOutward');
    expectVectorClose(choice.inwardNormal, expected.inwardNormal, 'inwardNormal');
    expectJointsClose(choice.hoverJoints, expected.hoverJoints, 'hoverJoints');
    expectJointsClose(choice.graspJoints, expected.graspJoints, 'graspJoints');
    expectJointsClose(choice.liftJoints, expected.liftJoints, 'liftJoints');
    expectMatrixClose(choice.hoverMatrix, expected.hoverMatrix, 'hoverMatrix');
    expectMatrixClose(choice.graspMatrix, expected.graspMatrix, 'graspMatrix');
    expectMatrixClose(choice.liftMatrix, expected.liftMatrix, 'liftMatrix');
  });

  it.each(fixture.cases)('offers the same candidates at ($pose.x, $pose.y)', testCase => {
    const prefix: CandidateSummary[] = [];
    for (const candidate of canonicalGraspCandidates(kinematics, testCase.pose)) {
      if (prefix.length >= testCase.candidatePrefix.length) { break; }
      prefix.push(summarize(candidate));
    }

    expect(prefix.map(candidate => `${candidate.face} ${candidate.elbow}`))
      .toEqual(testCase.candidatePrefix.map(c => `${c.face} ${c.elbow}`));
    for (const [index, candidate] of prefix.entries()) {
      const expected = testCase.candidatePrefix[index];
      expectClose(candidate.pitch, expected.pitch, `[${index}].pitch`);
      expectClose(candidate.rollOffset, expected.rollOffset, `[${index}].rollOffset`);
    }
  });
});
