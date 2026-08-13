// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Parser for the "PPFT" flow trace format written by
// py/src/pick_and_place/analysis/flow_trace_recording.py.
//
// One recording is a single policy rollout: the sim state to replay the arm
// from, plus, for every horizon the policy generated, the whole path the
// sampler integrated to produce it -- the Gaussian noise it started from, one
// state per Euler step, and the sample at t = 1.
//
// `path` is in the model's normalized action space, where integration actually
// happens: it starts as a standard normal draw and only lands inside [-1, 1]
// at t = 1. Only `commands` is in degrees.

const MAGIC = 'PPFT';
const HEADER_BYTES = 4 + 9 * 4 + 2 * 4;

export interface FlowTrace {
  fps: number;
  /** Steps of each horizon actually executed before the next is generated. */
  actSteps: number;
  targetX: number;
  targetY: number;
  /** Policy ticks in the rollout. */
  frames: number;
  /** Floats per replay frame: 6 joint angles + 7 cube pose. */
  nq: number;
  /** Action dimensions per predicted step. */
  joints: number;
  /** Predicted steps per horizon. */
  steps: number;
  /** Euler steps per horizon; the path holds one more state than this. */
  eulerSteps: number;
  chunks: number;
  /** (frames * nq) sim replay state, one row per tick. */
  qpos: Float32Array;
  /** (chunks) the tick each horizon was generated on. */
  chunkTicks: Uint32Array;
  /** (chunks * (eulerSteps + 1) * steps * joints) normalized action space. */
  path: Float32Array;
  /** (chunks * steps * joints) degrees. */
  commands: Float32Array;
}

export function parseFlowTrace(buffer: ArrayBuffer): FlowTrace {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== MAGIC) {
    throw new Error(`Unexpected flow trace magic header: ${magic}`);
  }

  let offset = 4;
  const readU32 = (): number => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readF32 = (): number => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };

  const version = readU32();
  if (version !== 1) {
    throw new Error(`Unsupported flow trace version: ${version}`);
  }
  const fps = readU32();
  const frames = readU32();
  const nq = readU32();
  const joints = readU32();
  const steps = readU32();
  const actSteps = readU32();
  const eulerSteps = readU32();
  const chunks = readU32();
  const targetX = readF32();
  const targetY = readF32();

  if (offset !== HEADER_BYTES) {
    throw new Error(`Flow trace header size mismatch: ${offset} !== ${HEADER_BYTES}`);
  }

  const take = <T>(
    make: (buffer: ArrayBuffer, offset: number, length: number) => T,
    length: number
  ): T => {
    const values = make(buffer, offset, length);
    offset += length * 4;
    return values;
  };

  const qpos = take(
    (b, o, l) => new Float32Array(b, o, l), frames * nq
  );
  const chunkTicks = take(
    (b, o, l) => new Uint32Array(b, o, l), chunks
  );
  const path = take(
    (b, o, l) => new Float32Array(b, o, l), chunks * (eulerSteps + 1) * steps * joints
  );
  const commands = take(
    (b, o, l) => new Float32Array(b, o, l), chunks * steps * joints
  );

  if (offset !== buffer.byteLength) {
    throw new Error(
      `Flow trace payload size mismatch: consumed ${offset} of ${buffer.byteLength} bytes`
    );
  }

  return {
    fps, actSteps, targetX, targetY, frames, nq, joints, steps, eulerSteps, chunks,
    qpos, chunkTicks, path, commands
  };
}

/** The replay frame at tick `index`, as a view into the recording. */
export function frame(trace: FlowTrace, index: number): Float32Array {
  const clamped = Math.min(Math.max(index, 0), trace.frames - 1);
  return trace.qpos.subarray(clamped * trace.nq, (clamped + 1) * trace.nq);
}

/**
 * One state of one horizon's integration path.
 *
 * `chunk` selects the horizon, `state` walks from 0 (the noise draw) to
 * `eulerSteps` (the finished sample). The returned view is `steps * joints`
 * long, in row-major predicted-step order.
 */
export function pathState(trace: FlowTrace, chunk: number, state: number): Float32Array {
  const stride = trace.steps * trace.joints;
  const perChunk = (trace.eulerSteps + 1) * stride;
  const start = chunk * perChunk + state * stride;
  return trace.path.subarray(start, start + stride);
}

/** The index of the horizon in flight at `tick`, or -1 before the first. */
export function chunkAt(trace: FlowTrace, tick: number): number {
  let found = -1;
  for (let index = 0; index < trace.chunks; index++) {
    if (trace.chunkTicks[index] <= tick) { found = index; } else { break; }
  }
  return found;
}
