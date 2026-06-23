// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  CUBE_HALF_SIZE,
  type CubePose,
  DEFAULT_CUBE_POSE
} from '../visualizations/grasp-pose-shared/body-factories';
import type { WebModel } from '../web-model';
import { selectCanonicalGrasp } from './canonical-grasp';
import { deriveSo101Kinematics } from './kinematics';

const model = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../public/so101.json', import.meta.url)),
    'utf8'
  )
) as WebModel;

const k = deriveSo101Kinematics(model);

interface OracleCase {
  pose: CubePose;
  face: string;
  elbow: string;
  closingDeg: number;
  pitchDeg: number;
  rollOffsetDeg: number;
  jointsDeg: {
    shoulder_pan: number;
    shoulder_lift: number;
    elbow_flex: number;
    wrist_flex: number;
    wrist_roll: number;
  };
}

const PYTHON_ORACLE_CASES: OracleCase[] = [
  {
    pose: {
      ...DEFAULT_CUBE_POSE,
      x: 0.001443059,
      y: -0.212061945,
      z: CUBE_HALF_SIZE,
      yaw: (-55 * Math.PI) / 180
    },
    face: '-x',
    elbow: 'up',
    closingDeg: -235,
    pitchDeg: 86,
    rollOffsetDeg: 0,
    jointsDeg: {
      shoulder_pan: 96.948952,
      shoulder_lift: 16.169416,
      elbow_flex: 4.702799,
      wrist_flex: 65.133437,
      wrist_roll: -135.184826
    }
  },
  {
    pose: {
      ...DEFAULT_CUBE_POSE,
      x: -0.016847882,
      y: 0.315795010,
      z: CUBE_HALF_SIZE,
      yaw: (100 * Math.PI) / 180
    },
    face: '-y',
    elbow: 'up',
    closingDeg: 10,
    pitchDeg: 66,
    rollOffsetDeg: 0,
    jointsDeg: {
      shoulder_pan: -103.056050,
      shoulder_lift: 39.448138,
      elbow_flex: -25.986988,
      wrist_flex: 52.569133,
      wrist_roll: -90.003803
    }
  },
  {
    pose: {
      ...DEFAULT_CUBE_POSE,
      x: 0.394753108,
      y: 0.234090816,
      z: CUBE_HALF_SIZE,
      yaw: (33.333333 * Math.PI) / 180
    },
    face: '-y',
    elbow: 'up',
    closingDeg: -56.666667,
    pitchDeg: 16,
    rollOffsetDeg: 0,
    jointsDeg: {
      shoulder_pan: -35.634686,
      shoulder_lift: 82.160374,
      elbow_flex: -59.494260,
      wrist_flex: -6.653861,
      wrist_roll: -87.845997
    }
  }
];

function deg(rad: number): number {
  return (rad * 180) / Math.PI;
}

function angleDeltaDeg(a: number, b: number): number {
  let delta = (b - a) % 360;
  if (delta > 180) { delta -= 360; }
  if (delta <= -180) { delta += 360; }
  return delta;
}

describe('selectCanonicalGrasp', () => {
  it.each(PYTHON_ORACLE_CASES)('matches the Python oracle for %#', expected => {
    const choice = selectCanonicalGrasp(k, expected.pose);

    expect(choice).not.toBeNull();
    if (choice === null) { return; }
    expect(choice.face).toBe(expected.face);
    expect(choice.elbow).toBe(expected.elbow);
    expect(deg(choice.pitch)).toBeCloseTo(expected.pitchDeg, 6);
    expect(deg(choice.rollOffset)).toBeCloseTo(expected.rollOffsetDeg, 6);
    expect(angleDeltaDeg(deg(choice.closingAzimuth), expected.closingDeg))
      .toBeCloseTo(0, 6);
    for (const [name, value] of Object.entries(expected.jointsDeg)) {
      expect(Math.abs(
        deg(choice.graspJoints[name as keyof typeof expected.jointsDeg]) - value
      )).toBeLessThanOrEqual(1e-3);
    }
  });
});
