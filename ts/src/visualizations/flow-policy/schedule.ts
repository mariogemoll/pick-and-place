// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// The rollout replayed one horizon at a time, in the order the policy actually
// works: draw noise, integrate it into a horizon, then execute that horizon's
// actions while the world moves.
//
// Stepped, every horizon gets the same three-and-a-half-second cycle: half a
// second holding the fresh noise, a second of integration, half a second frozen
// on the horizon it produced, a second of execution, and half a second of rest
// on what that left behind before the next draw. The rhythm is the same for
// every horizon, and the holds give each result a beat to be looked at before
// the next step overwrites it.
//
// Executing a horizon scrolls it off to the left, so it needs no beat to be
// cleared away: by the end of the execute phase the panel is already showing
// nothing but the two commands that scrolled into the past, which is exactly
// where the next horizon's cycle begins.
//
// Continuous drops all of that: each horizon is shown finished and executed
// straight away, at the rate it was recorded at, so the rollout runs as the
// policy actually ran it.
//
// This is deliberately not wall-clock faithful. The real sampler runs between
// two control ticks, far too fast to watch, so sampling and integration get
// their own beats with the arm held still, and only the execute phase advances
// the replay. Playback time is therefore its own clock, not episode time.

import { chunkSpan, type FlowTrace } from './trace';

export type Phase = 'sample' | 'flow' | 'execute';

/**
 * What a beat does with its phase: `run` produces the phase's result, `hold`
 * sits on it, and `rest` is the pause on what the phase left behind.
 */
export type Beat = 'run' | 'hold' | 'rest';

/**
 * `stepped` walks the cycle beat by beat; `continuous` shows each horizon
 * finished and executes it at the recorded control rate.
 */
export type ScheduleMode = 'stepped' | 'continuous';

export interface Segment {
  chunk: number;
  phase: Phase;
  beat: Beat;
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
  beat: Beat;
  /** 0 to 1 within the current phase; 1 once the phase has produced its result. */
  progress: number;
  /** Where the arm stands, in fractional replay ticks. */
  tickFloat: number;
}

export interface ScheduleOptions {
  mode?: ScheduleMode;
  /** Beat spent holding the fresh noise draw before integrating. */
  sampleSeconds?: number;
  /** Beat spent integrating noise into the horizon. */
  flowSeconds?: number;
  /** Beat spent walking the arm through the horizon's actions. */
  executeSeconds?: number;
  /**
   * The half-beat: holding the noise draw, holding the finished horizon, and
   * resting on what executing it scrolled into the past.
   */
  holdSeconds?: number;
}

const DEFAULT_BEAT_SECONDS = 1;
const DEFAULT_HOLD_SECONDS = 0.5;

export function buildSchedule(trace: FlowTrace, options: ScheduleOptions = {}): Segment[] {
  const holdSeconds = options.holdSeconds ?? DEFAULT_HOLD_SECONDS;
  // The noise draw is instant, so its beat is a hold on the draw from the start.
  const sampleSeconds = options.sampleSeconds ?? holdSeconds;
  const flowSeconds = options.flowSeconds ?? DEFAULT_BEAT_SECONDS;
  const executeSeconds = options.executeSeconds ?? DEFAULT_BEAT_SECONDS;

  const segments: Segment[] = [];
  let start = 0;
  const push = (
    chunk: number, phase: Phase, beat: Beat, duration: number,
    startTick: number, endTick: number
  ): void => {
    segments.push({ chunk, phase, beat, start, duration, startTick, endTick });
    start += duration;
  };

  for (let chunk = 0; chunk < trace.chunks; chunk++) {
    const [startTick, endTick] = chunkSpan(trace, chunk);

    if ((options.mode ?? 'stepped') === 'continuous') {
      push(chunk, 'execute', 'run', Math.max(endTick - startTick, 0) / trace.fps,
        startTick, endTick);
      continue;
    }

    push(chunk, 'sample', 'hold', sampleSeconds, startTick, startTick);
    push(chunk, 'flow', 'run', flowSeconds, startTick, startTick);
    push(chunk, 'flow', 'hold', holdSeconds, startTick, startTick);
    push(chunk, 'execute', 'run', executeSeconds, startTick, endTick);
    // A beat on the scrolled-out result, which is the two commands the next
    // horizon will be predicted from.
    push(chunk, 'execute', 'rest', holdSeconds, endTick, endTick);
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
    return { chunk: -1, phase: 'sample', beat: 'rest', progress: 0, tickFloat: 0 };
  }

  let index = 0;
  while (
    index + 1 < schedule.length &&
    seconds >= schedule[index].start + schedule[index].duration
  ) {
    index++;
  }
  const segment = schedule[index];
  const elapsed = segment.duration > 0
    ? Math.min(Math.max((seconds - segment.start) / segment.duration, 0), 1)
    : 1;
  // Only a `run` beat is still working towards its result; the others have it.
  const progress = segment.beat === 'run' ? elapsed : 1;
  return {
    chunk: segment.chunk,
    phase: segment.phase,
    beat: segment.beat,
    progress,
    tickFloat: segment.startTick + (segment.endTick - segment.startTick) * progress
  };
}
