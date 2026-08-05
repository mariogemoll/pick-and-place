// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, it } from 'vitest';

import { smoothstep, timedArcFraction } from '../visualizations/pick-and-place/trajectory';
import { expectClose, loadFixture } from './fixtures';

interface EasingFixture {
  smoothstep: { t: number; value: number }[];
  timedArcFraction: { phase: number; value: number }[];
}

const fixture = loadFixture('easing.json') as EasingFixture;

// The two trajectories have diverged on purpose — Python plans the physical
// eight-phase motion from a canonical grasp, TypeScript animates the five-stage
// illustrative one from a vertical grasp — but both still shape every move with
// these curves, including the clamping outside [0, 1].
describe('shared easing curves', () => {
  it('matches Python smoothstep', () => {
    for (const sample of fixture.smoothstep) {
      expectClose(smoothstep(sample.t), sample.value, `smoothstep(${sample.t})`);
    }
  });

  it('matches Python timed arc fraction', () => {
    for (const sample of fixture.timedArcFraction) {
      expectClose(
        timedArcFraction(sample.phase), sample.value, `timedArcFraction(${sample.phase})`
      );
    }
  });
});
