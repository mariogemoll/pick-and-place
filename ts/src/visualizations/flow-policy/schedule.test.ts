// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { describe, expect, it } from 'vitest';

import { buildSchedule, momentAt, scheduleDuration } from './schedule';
import type { FlowTrace } from './trace';

const SAMPLE = 0.5;
const FLOW = 1;

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

const options = { sampleSeconds: SAMPLE, flowSeconds: FLOW };

describe('buildSchedule', () => {
  it('gives every horizon a sample, flow and execute beat in order', () => {
    const schedule = buildSchedule(trace(), options);
    expect(schedule.map(segment => segment.phase)).toEqual([
      'sample', 'flow', 'execute',
      'sample', 'flow', 'execute',
      'sample', 'flow', 'execute'
    ]);
    expect(schedule.map(segment => segment.chunk)).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2]);
  });

  it('advances the replay only during execute', () => {
    const schedule = buildSchedule(trace(), options);
    for (const segment of schedule) {
      if (segment.phase === 'execute') {
        expect(segment.endTick).toBeGreaterThan(segment.startTick);
      } else {
        expect(segment.endTick).toBe(segment.startTick);
      }
    }
  });

  it('runs the last horizon out to the end of the recording', () => {
    const schedule = buildSchedule(trace(), options);
    const last = schedule[schedule.length - 1];
    expect(last.startTick).toBe(16);
    expect(last.endTick).toBe(24);
  });

  it('integrates faster than it executes, and barely pauses to sample', () => {
    // With no overrides the durations are derived from the trace, so this holds
    // whatever the control rate and chunk size are.
    const schedule = buildSchedule(trace());
    const [sample, flow, execute] = schedule;

    expect(flow.duration).toBeLessThan(execute.duration);
    expect(sample.duration).toBeLessThan(flow.duration / 2);
  });

  it('scales the beats with the control rate', () => {
    const slow = buildSchedule(trace({ fps: 5 }));
    const fast = buildSchedule(trace({ fps: 20 }));

    // Halving the rate doubles the execution window, and the flow beat follows.
    expect(slow[1].duration).toBeCloseTo(fast[1].duration * 4);
    expect(slow[1].duration).toBeLessThan(slow[2].duration);
    expect(fast[1].duration).toBeLessThan(fast[2].duration);
  });

  it('lays segments end to end with no gaps', () => {
    const schedule = buildSchedule(trace(), options);
    for (let index = 1; index < schedule.length; index++) {
      expect(schedule[index].start).toBeCloseTo(
        schedule[index - 1].start + schedule[index - 1].duration
      );
    }
    // Three horizons: 3 * (0.5 + 1) beats, plus 24 ticks of execution at 10 Hz.
    expect(scheduleDuration(schedule)).toBeCloseTo(3 * (SAMPLE + FLOW) + 2.4);
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

  it('walks the arm across the horizon during execute', () => {
    const executeStart = SAMPLE + FLOW;
    expect(momentAt(schedule, executeStart).tickFloat).toBeCloseTo(0);
    expect(momentAt(schedule, executeStart + 0.4).tickFloat).toBeCloseTo(4);
    expect(momentAt(schedule, executeStart + 0.79).tickFloat).toBeCloseTo(7.9);
  });

  it('reports progress through the current phase', () => {
    expect(momentAt(schedule, SAMPLE / 2).progress).toBeCloseTo(0.5);
    expect(momentAt(schedule, SAMPLE + FLOW / 4).progress).toBeCloseTo(0.25);
  });

  it('moves on to the next horizon after one finishes executing', () => {
    const secondChunk = SAMPLE + FLOW + 0.8;
    expect(momentAt(schedule, secondChunk + 0.01)).toMatchObject({ chunk: 1, phase: 'sample' });
  });

  it('clamps past the end instead of running off the schedule', () => {
    const moment = momentAt(schedule, 999);
    expect(moment.chunk).toBe(2);
    expect(moment.phase).toBe('execute');
    expect(moment.tickFloat).toBe(24);
  });
});
