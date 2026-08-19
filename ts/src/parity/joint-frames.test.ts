// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, it } from 'vitest';

import { gripperAngleToPosition, gripperPositionToAngle, simFrameToReal } from '../joint-frames';
import { expectClose, loadFixture } from './fixtures';

interface JointFramesFixture {
  gripperAngleToPosition: { angleRad: number; position: number }[];
  gripperPositionToAngle: { position: number; angleRad: number }[];
  simFrameToReal: { simJoints: number[]; realJoints: number[] }[];
}

const fixture = loadFixture('joint_frames.json') as JointFramesFixture;

// Every action the browser policy page issues crosses this boundary, and so
// does every observation it packs. A drift here would not throw: it would
// quietly hand the policy joints in the wrong units.
describe('sim-frame/real-frame joint conversion', () => {
  it('matches Python on the gripper angle to position map', () => {
    for (const sample of fixture.gripperAngleToPosition) {
      expectClose(
        gripperAngleToPosition(sample.angleRad),
        sample.position,
        `gripperAngleToPosition(${sample.angleRad})`
      );
    }
  });

  it('matches Python on the gripper position to angle map', () => {
    for (const sample of fixture.gripperPositionToAngle) {
      expectClose(
        gripperPositionToAngle(sample.position),
        sample.angleRad,
        `gripperPositionToAngle(${sample.position})`
      );
    }
  });

  it('matches Python on whole sim-frame joint vectors', () => {
    for (const sample of fixture.simFrameToReal) {
      const actual = simFrameToReal(sample.simJoints);
      for (const [index, expected] of sample.realJoints.entries()) {
        const label = `simFrameToReal[${index}] of ${JSON.stringify(sample.simJoints)}`;
        expectClose(actual[index], expected, label);
      }
    }
  });
});
