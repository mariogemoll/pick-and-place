// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import type { WebModel } from '../web-model';
import { deriveSo101Kinematics } from './kinematics';

const model = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../public/so101.json', import.meta.url)),
    'utf8'
  )
) as WebModel;

describe('deriveSo101Kinematics', () => {
  const k = deriveSo101Kinematics(model);

  it('locates the pan axis under the shoulder', () => {
    expect(k.panAxis.x).toBeCloseTo(0.03884, 4);
    expect(k.panAxis.y).toBeCloseTo(0, 4);
  });

  it('places the shoulder_lift pivot in the radial plane', () => {
    expect(k.shoulderLift.radial).toBeCloseTo(0.0304, 4);
    expect(k.shoulderLift.height).toBeCloseTo(0.1166, 4);
  });

  it('derives the upper-arm link', () => {
    expect(k.upperArm.radial).toBeCloseTo(0.028, 4);
    expect(k.upperArm.height).toBeCloseTo(0.1126, 4);
    expect(k.upperArm.length).toBeCloseTo(0.116, 4);
  });

  it('derives the lower-arm link', () => {
    expect(k.lowerArm.radial).toBeCloseTo(0.1349, 4);
    expect(k.lowerArm.height).toBeCloseTo(0.0052, 4);
    expect(k.lowerArm.length).toBeCloseTo(0.135, 4);
  });

  it('derives the tool length to the gripper target', () => {
    expect(k.toolLength).toBeCloseTo(0.1605, 4);
  });

  it('reads joint limits straight from the model ranges', () => {
    expect(k.jointLimits.shoulder_pan.min).toBeCloseTo(-1.91986, 4);
    expect(k.jointLimits.shoulder_pan.max).toBeCloseTo(1.91986, 4);
    expect(k.jointLimits.elbow_flex.min).toBeCloseTo(-1.69, 4);
    expect(k.jointLimits.elbow_flex.max).toBeCloseTo(1.69, 4);
    expect(k.jointLimits.wrist_roll.min).toBeCloseTo(-2.74385, 4);
    expect(k.jointLimits.wrist_roll.max).toBeCloseTo(2.84121, 4);
  });

  it('matches the link lengths implied by the limit-derived constants', () => {
    // The planar segments must close on their own radial/height components.
    for (const segment of [k.upperArm, k.lowerArm]) {
      expect(segment.length)
        .toBeCloseTo(Math.hypot(segment.radial, segment.height), 12);
    }
  });
});
