// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { existsSync, readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import {
  chunkAt,
  chunkSpan,
  executedSteps,
  executedTail,
  type FlowTrace,
  frame,
  parseFlowTrace,
  pathState
} from './trace';

const FRAMES = 5;
const NQ = 13;
const JOINTS = 6;
const STEPS = 16;
const ACT_STEPS = 8;
const EULER = 10;
const CHUNKS = 2;

// Mirrors what flow_trace_recording.encode writes, so the parser is checked
// against the layout rather than against itself.
function buildBuffer(): ArrayBuffer {
  const states = EULER + 1;
  const qposLength = FRAMES * NQ;
  const pathLength = CHUNKS * states * STEPS * JOINTS;
  const commandsLength = CHUNKS * STEPS * JOINTS;
  const bytes = 48 + 4 * (qposLength + CHUNKS + pathLength + commandsLength);
  const buffer = new ArrayBuffer(bytes);
  const view = new DataView(buffer);

  new Uint8Array(buffer, 0, 4).set(new TextEncoder().encode('PPFT'));
  let offset = 4;
  for (const value of [1, 10, FRAMES, NQ, JOINTS, STEPS, ACT_STEPS, EULER, CHUNKS]) {
    view.setUint32(offset, value, true);
    offset += 4;
  }
  view.setFloat32(offset, 0.25, true); offset += 4;
  view.setFloat32(offset, -0.125, true); offset += 4;

  const write = (length: number, at: (index: number) => number): void => {
    for (let index = 0; index < length; index++) {
      view.setFloat32(offset, at(index), true);
      offset += 4;
    }
  };
  write(qposLength, index => index);
  for (let index = 0; index < CHUNKS; index++) {
    view.setUint32(offset, index * ACT_STEPS, true);
    offset += 4;
  }
  write(pathLength, index => index * 0.5);
  write(commandsLength, index => -index);

  return buffer;
}

function parsed(): FlowTrace {
  return parseFlowTrace(buildBuffer());
}

describe('parseFlowTrace', () => {
  it('reads the header', () => {
    const trace = parsed();
    expect(trace.fps).toBe(10);
    expect(trace.frames).toBe(FRAMES);
    expect(trace.joints).toBe(JOINTS);
    expect(trace.steps).toBe(STEPS);
    expect(trace.actSteps).toBe(ACT_STEPS);
    expect(trace.eulerSteps).toBe(EULER);
    expect(trace.chunks).toBe(CHUNKS);
    expect(trace.targetX).toBeCloseTo(0.25);
    expect(trace.targetY).toBeCloseTo(-0.125);
  });

  it('consumes exactly the whole payload', () => {
    // The parser throws on any leftover or shortfall, so reaching here is the
    // assertion; this pins the arrays it produced to the declared shapes.
    const trace = parsed();
    expect(trace.qpos.length).toBe(FRAMES * NQ);
    expect(trace.chunkTicks.length).toBe(CHUNKS);
    expect(trace.path.length).toBe(CHUNKS * (EULER + 1) * STEPS * JOINTS);
    expect(trace.commands.length).toBe(CHUNKS * STEPS * JOINTS);
  });

  it('rejects a foreign magic header', () => {
    const buffer = buildBuffer();
    new Uint8Array(buffer, 0, 4).set(new TextEncoder().encode('XXXX'));
    expect(() => parseFlowTrace(buffer)).toThrow(/magic/);
  });

  it('rejects a truncated payload', () => {
    expect(() => parseFlowTrace(buildBuffer().slice(0, 200))).toThrow();
  });
});

describe('accessors', () => {
  it('slices a replay frame and clamps past the ends', () => {
    const trace = parsed();
    expect([...frame(trace, 0)]).toEqual([...Array(NQ).keys()]);
    expect(frame(trace, 99)).toEqual(frame(trace, FRAMES - 1));
    expect(frame(trace, -3)).toEqual(frame(trace, 0));
  });

  it('slices one integration state per chunk', () => {
    const trace = parsed();
    const stride = STEPS * JOINTS;
    expect(pathState(trace, 0, 0).length).toBe(stride);
    // Chunk 0 state 0 starts the array; chunk 1 starts a whole chunk in.
    expect(pathState(trace, 0, 0)[0]).toBe(0);
    expect(pathState(trace, 0, 1)[0]).toBe(stride * 0.5);
    expect(pathState(trace, 1, 0)[0]).toBe((EULER + 1) * stride * 0.5);
  });

  it('finds the horizon in flight at a tick', () => {
    const trace = parsed();
    expect(chunkAt(trace, 0)).toBe(0);
    expect(chunkAt(trace, ACT_STEPS - 1)).toBe(0);
    expect(chunkAt(trace, ACT_STEPS)).toBe(1);
    expect(chunkAt(trace, 999)).toBe(CHUNKS - 1);
  });
});

// A small trace whose path is filled with its own indices, so a slice can be
// checked against the offset it should have come from.
const SMALL_STEPS = 4;
const SMALL_JOINTS = 2;
const SMALL_STRIDE = SMALL_STEPS * SMALL_JOINTS;

function small(overrides: Partial<FlowTrace> = {}): FlowTrace {
  const states = 2;
  const chunks = 2;
  return {
    fps: 10,
    actSteps: 2,
    targetX: 0,
    targetY: 0,
    frames: 5,
    nq: 13,
    joints: SMALL_JOINTS,
    steps: SMALL_STEPS,
    eulerSteps: states - 1,
    chunks,
    qpos: new Float32Array(),
    chunkTicks: new Uint32Array([0, 2]),
    path: new Float32Array(chunks * states * SMALL_STRIDE).map((_, index) => index),
    commands: new Float32Array(),
    ...overrides
  };
}

describe('the executed part of a horizon', () => {
  it('runs a horizon until the next one replaces it', () => {
    expect(chunkSpan(small(), 0)).toEqual([0, 2]);
  });

  it('runs the last horizon out to the end of the recording', () => {
    expect(chunkSpan(small(), 1)).toEqual([2, 4]);
  });

  it('executes every step it hands out', () => {
    expect(executedSteps(small(), 0)).toBe(2);
    expect(executedSteps(small(), 1)).toBe(2);
  });

  it('never counts more steps executed than the horizon hands out', () => {
    // A horizon that stayed in flight far longer than it had steps for.
    expect(executedSteps(small({ chunkTicks: new Uint32Array([0, 99]) }), 0)).toBe(2);
  });

  it('counts nothing for a horizon the recording ended on', () => {
    expect(executedSteps(small({ frames: 3, chunkTicks: new Uint32Array([0, 2]) }), 1)).toBe(0);
  });

  it('takes the tail from the finished sample, not from the noise', () => {
    const trace = small();
    // Chunk 0's finished sample is the second of its two states.
    expect([...executedTail(trace, 0, 2)]).toEqual([...pathState(trace, 0, 1).slice(0, 4)]);
    expect([...executedTail(trace, 0, 2)]).toEqual([8, 9, 10, 11]);
  });

  it('takes the tail from the end of what was executed', () => {
    // One step executed, so the tail is that step alone however many are asked
    // for -- and it is the first step of the horizon, not the last.
    const trace = small({ chunkTicks: new Uint32Array([0, 1]) });
    expect([...executedTail(trace, 0, 2)]).toEqual([8, 9]);
  });

  it('has no tail before the first horizon', () => {
    expect(executedTail(small(), -1, 2).length).toBe(0);
  });
});

// The recordings are generated, not committed, so this only runs where one has
// been written. It is the check that the two implementations of the format
// actually agree.
const REAL = 'public/flow-traces/dppo-train-000000.bin';
describe.skipIf(!existsSync(REAL))('a real recording', () => {
  it('parses with the shapes the recorder documented', () => {
    const file = readFileSync(REAL);
    const trace = parseFlowTrace(
      file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)
    );
    expect(trace.fps).toBe(10);
    expect(trace.joints).toBe(6);
    expect(trace.steps).toBe(16);
    expect(trace.actSteps).toBe(8);
    expect(trace.eulerSteps).toBe(10);
    expect(trace.nq).toBe(13);

    // Horizons are generated every actSteps ticks, starting at tick 0.
    expect([...trace.chunkTicks]).toEqual(
      [...Array(trace.chunks).keys()].map(index => index * trace.actSteps)
    );

    // The path starts as a standard normal draw and lands in [-1, 1].
    const noise = pathState(trace, 0, 0);
    const final = pathState(trace, 0, trace.eulerSteps);
    expect(Math.max(...noise.map(Math.abs))).toBeGreaterThan(1.5);
    expect(Math.max(...final.map(Math.abs))).toBeLessThan(1.2);
  });
});
