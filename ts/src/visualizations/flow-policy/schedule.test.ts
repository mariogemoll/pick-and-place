// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { buildSchedule, momentAt, scheduleDuration } from './schedule';
import type { FlowTrace } from './trace';

const SAMPLE = 0.5;
const FLOW = 1;
// Eight ticks per horizon in the test trace, so this beat runs them at 10 Hz.
const EXECUTE = 0.8;
const HOLD = 0.25;
const CYCLE = SAMPLE + FLOW + HOLD + EXECUTE + HOLD;

// Only the fields the schedule reads; the arrays are irrelevant to timing.
function trace(overrides: Partial<FlowTrace> = {}): FlowTrace {
  return {
    fps: 10,
    actSteps: 8,
    targetX: 0,
    targetY: 0,
    frames: 25,
    nq: 13,
    joints: 6,
    steps: 16,
    eulerSteps: 10,
    chunks: 3,
    qpos: new Float32Array(),
    chunkTicks: new Uint32Array([0, 8, 16]),
    path: new Float32Array(),
    commands: new Float32Array(),
    ...overrides
  };
}

const options = {
  sampleSeconds: SAMPLE, flowSeconds: FLOW, executeSeconds: EXECUTE, holdSeconds: HOLD
};

describe('buildSchedule', () => {
  it('runs the same beats for every horizon, in order', () => {
    const schedule = buildSchedule(trace(), options);
    const phases = ['sample', 'flow', 'flow', 'execute', 'execute'];
    const beats = ['hold', 'run', 'hold', 'run', 'rest'];

    expect(schedule.map(segment => segment.phase))
      .toEqual([...phases, ...phases, ...phases]);
    expect(schedule.map(segment => segment.beat))
      .toEqual([...beats, ...beats, ...beats]);
    expect(schedule.map(segment => segment.chunk))
      .toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
  });

  it('advances the replay only while executing, never on a hold', () => {
    const schedule = buildSchedule(trace(), options);
    for (const segment of schedule) {
      if (segment.phase === 'execute' && segment.beat === 'run') {
        expect(segment.endTick).toBeGreaterThan(segment.startTick);
      } else {
        expect(segment.endTick).toBe(segment.startTick);
      }
    }
  });

  it('runs the last horizon out to the end of the recording', () => {
    const schedule = buildSchedule(trace(), options);
    const last = schedule
      .filter(segment => segment.phase === 'execute' && segment.beat === 'run').pop();
    expect(last?.startTick).toBe(16);
    expect(last?.endTick).toBe(24);
  });

  it('spends a second on each of the flow and execute beats by default', () => {
    expect(buildSchedule(trace()).slice(0, 5).map(segment => segment.duration))
      .toEqual([0.5, 1, 0.5, 1, 0.5]);
  });

  it('keeps the same cycle whatever the control rate', () => {
    for (const fps of [5, 10, 20]) {
      // Three horizons of 0.5 + 1 + 0.5 + 1 + 0.5 seconds.
      expect(scheduleDuration(buildSchedule(trace({ fps })))).toBeCloseTo(3 * 3.5);
    }
  });

  it('lays segments end to end with no gaps', () => {
    const schedule = buildSchedule(trace(), options);
    for (let index = 1; index < schedule.length; index++) {
      expect(schedule[index].start).toBeCloseTo(
        schedule[index - 1].start + schedule[index - 1].duration
      );
    }
    expect(scheduleDuration(schedule)).toBeCloseTo(3 * CYCLE);
  });
});

describe('buildSchedule, continuous', () => {
  const continuous = { ...options, mode: 'continuous' } as const;

  it('gives each horizon a single execute beat and nothing else', () => {
    const schedule = buildSchedule(trace(), continuous);
    expect(schedule.map(segment => segment.phase)).toEqual(['execute', 'execute', 'execute']);
    expect(schedule.map(segment => segment.beat)).toEqual(['run', 'run', 'run']);
  });

  it('runs at the rate the rollout was recorded at', () => {
    // Eight ticks per horizon at 10 Hz, so 0.8 seconds each.
    const schedule = buildSchedule(trace(), continuous);
    expect(schedule.map(segment => segment.duration)).toEqual([0.8, 0.8, 0.8]);
    expect(scheduleDuration(buildSchedule(trace({ fps: 20 }), continuous))).toBeCloseTo(1.2);
  });

  it('covers the same ticks as the stepped schedule', () => {
    const schedule = buildSchedule(trace(), continuous);
    expect(schedule[0].startTick).toBe(0);
    expect(schedule[schedule.length - 1].endTick).toBe(24);
  });

  it('shows every horizon finished, with the arm moving throughout', () => {
    const schedule = buildSchedule(trace(), continuous);
    const moment = momentAt(schedule, 0.4);
    expect(moment).toMatchObject({ chunk: 0, phase: 'execute', beat: 'run' });
    expect(moment.tickFloat).toBeCloseTo(4);
  });
});

describe('momentAt', () => {
  const schedule = buildSchedule(trace(), options);

  it('starts on the first sample with the arm at tick zero', () => {
    const moment = momentAt(schedule, 0);
    expect(moment).toMatchObject({ chunk: 0, phase: 'sample', tickFloat: 0 });
  });

  it('holds the arm still while sampling and integrating', () => {
    expect(momentAt(schedule, 0.25).tickFloat).toBe(0);
    expect(momentAt(schedule, SAMPLE + 0.5)).toMatchObject({ phase: 'flow', tickFloat: 0 });
  });

  it('sits on the finished horizon through the hold after the flow', () => {
    expect(momentAt(schedule, SAMPLE + FLOW + HOLD / 2)).toMatchObject({
      chunk: 0, phase: 'flow', beat: 'hold', progress: 1, tickFloat: 0
    });
  });

  it('rests on the scrolled-out horizon before drawing the next one', () => {
    const restStart = SAMPLE + FLOW + HOLD + EXECUTE;
    // The arm stays where the horizon left it, and the horizon stays fully
    // executed, which is what puts the strip where the next cycle begins.
    expect(momentAt(schedule, restStart + HOLD / 2)).toMatchObject({
      chunk: 0, phase: 'execute', beat: 'rest', progress: 1, tickFloat: 8
    });
    expect(momentAt(schedule, restStart + HOLD + 0.01)).toMatchObject({
      chunk: 1, phase: 'sample', tickFloat: 8
    });
  });

  it('walks the arm across the horizon during execute', () => {
    const executeStart = SAMPLE + FLOW + HOLD;
    expect(momentAt(schedule, executeStart).tickFloat).toBeCloseTo(0);
    expect(momentAt(schedule, executeStart + 0.4).tickFloat).toBeCloseTo(4);
    expect(momentAt(schedule, executeStart + 0.79).tickFloat).toBeCloseTo(7.9);
  });

  it('reports progress through the integration', () => {
    expect(momentAt(schedule, SAMPLE + FLOW / 4).progress).toBeCloseTo(0.25);
    expect(momentAt(schedule, SAMPLE + FLOW / 2).progress).toBeCloseTo(0.5);
    // The noise draw is instant, so its beat is finished from the first frame.
    expect(momentAt(schedule, SAMPLE / 2).progress).toBe(1);
  });

  it('moves on to the next horizon after one finishes executing', () => {
    const secondChunk = CYCLE;
    expect(momentAt(schedule, secondChunk + 0.01)).toMatchObject({ chunk: 1, phase: 'sample' });
  });

  it('clamps past the end instead of running off the schedule', () => {
    const moment = momentAt(schedule, 999);
    expect(moment.chunk).toBe(2);
    expect(moment.phase).toBe('execute');
    expect(moment.beat).toBe('rest');
    expect(moment.tickFloat).toBe(24);
  });
});
