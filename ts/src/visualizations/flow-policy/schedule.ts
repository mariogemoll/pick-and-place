// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// The rollout replayed one horizon at a time, in the order the policy actually
// works: draw noise, integrate it into a horizon, then execute that horizon's
// actions while the world moves.
//
// The noise draw gets its own beat because it is its own step, but only just
// long enough to register: it is where the flow starts, at t = 0, so anything
// more than a glimpse reads as a stall before the integration.
//
// This is deliberately not wall-clock faithful. The real sampler runs between
// two control ticks, far too fast to watch, so sampling and integration get
// their own beats with the arm held still, and only the execute phase advances
// the replay. Playback time is therefore its own clock, not episode time.

import type { FlowTrace } from './trace';

export type Phase = 'sample' | 'flow' | 'execute';

export interface Segment {
  chunk: number;
  phase: Phase;
  /** Seconds from the start of the rollout. */
  start: number;
  duration: number;
  /** Replay tick at the start and end of the segment. */
  startTick: number;
  endTick: number;
}

export interface Moment {
  chunk: number;
  phase: Phase;
  /** 0 to 1 within the current phase. */
  progress: number;
  /** Where the arm stands, in fractional replay ticks. */
  tickFloat: number;
}

export interface ScheduleOptions {
  /** Beat spent holding the fresh noise draw before integrating. */
  sampleSeconds?: number;
  /** Beat spent integrating noise into the horizon. */
  flowSeconds?: number;
}

// Just long enough for the noise to register as its own step.
const DEFAULT_SAMPLE_SECONDS = 0.12;

// The integration is drawn as a fraction of the window its horizon is then
// executed over, so generating a chunk always visibly outruns running it --
// which is the true relationship, the real sampler finishing between two ticks.
const FLOW_FRACTION_OF_EXECUTION = 0.55;

export function buildSchedule(trace: FlowTrace, options: ScheduleOptions = {}): Segment[] {
  const sampleSeconds = options.sampleSeconds ?? DEFAULT_SAMPLE_SECONDS;
  const executionWindow = trace.actSteps / trace.fps;
  const flowSeconds = options.flowSeconds ?? FLOW_FRACTION_OF_EXECUTION * executionWindow;

  const segments: Segment[] = [];
  let start = 0;
  for (let chunk = 0; chunk < trace.chunks; chunk++) {
    const startTick = trace.chunkTicks[chunk];
    // A horizon runs until the next one is generated; the last runs to the end
    // of what was recorded, which is where the episode terminated.
    const endTick = chunk + 1 < trace.chunks ? trace.chunkTicks[chunk + 1] : trace.frames - 1;

    for (const [phase, duration] of [
      ['sample', sampleSeconds],
      ['flow', flowSeconds],
      ['execute', Math.max(endTick - startTick, 0) / trace.fps]
    ] as [Phase, number][]) {
      segments.push({
        chunk,
        phase,
        start,
        duration,
        startTick,
        endTick: phase === 'execute' ? endTick : startTick
      });
      start += duration;
    }
  }
  return segments;
}

export function scheduleDuration(schedule: Segment[]): number {
  if (schedule.length === 0) { return 0; }
  const last = schedule[schedule.length - 1];
  return last.start + last.duration;
}

/** Where playback stands at `seconds`, clamped to the ends of the schedule. */
export function momentAt(schedule: Segment[], seconds: number): Moment {
  if (schedule.length === 0) {
    return { chunk: -1, phase: 'sample', progress: 0, tickFloat: 0 };
  }
  let index = 0;
  while (
    index + 1 < schedule.length &&
    seconds >= schedule[index].start + schedule[index].duration
  ) {
    index++;
  }
  const segment = schedule[index];
  const elapsed = seconds - segment.start;
  const progress = segment.duration > 0
    ? Math.min(Math.max(elapsed / segment.duration, 0), 1)
    : 1;
  return {
    chunk: segment.chunk,
    phase: segment.phase,
    progress,
    tickFloat: segment.startTick + (segment.endTick - segment.startTick) * progress
  };
}
