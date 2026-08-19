// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// The state flow policy, run in the browser through onnxruntime-web.
//
// A port of `StateFlowPolicy` in py/scripts/run_flow_policy_sim.py. The policy
// works in chunks: pack the last two observations, integrate a noise draw into
// a sixteen-step horizon, execute the first eight of those steps, repeat.
//
// The Euler integration is not here. It lives inside the ONNX graph, unrolled,
// so there is no second integrator on this side that could drift from the one
// `flow_matching.generate` runs. What is left is the contract around it: how an
// observation is packed, how it is normalized, and how many of the predicted
// steps get executed before the next horizon is drawn.

import * as ort from 'onnxruntime-web/wasm';

import type { CubePose } from './environment';

export interface FlowPolicyManifest {
  format: 'pick-and-place-flow-policy';
  version: number;
  observationSteps: number;
  observationDim: number;
  observationNames: string[];
  predictionSteps: number;
  endpointDim: number;
  endpointSemantics: string;
  policyHz: number;
  actSteps: number;
  integrationSteps: number;
  model: string;
  precision: string;
  checkpointSha256: string;
  normalization: {
    observation_min: number[];
    observation_max: number[];
    endpoint_min: number[];
    endpoint_max: number[];
  };
}

export interface FlowPolicy {
  readonly manifest: FlowPolicyManifest;
  reset(): void;
  /** The next real-frame action, drawing a fresh horizon when the queue empties. */
  act(state: ArrayLike<number>, cube: CubePose, targetXy: [number, number]): Promise<Float32Array>;
  /** Fraction of the last horizon's values that fell outside [-1, 1] before clipping. */
  clippedFraction(): number;
  destroy(): Promise<void>;
}

/**
 * The first two columns of the rotation matrix a wxyz quaternion describes,
 * column by column: `[r00, r10, r20, r01, r11, r21]`.
 *
 * Deliberately not the row-major flattening of the first two columns; the
 * training export packs it this way and the model reads it positionally.
 */
export function quatWxyzToRotation6d(
  quaternion: [number, number, number, number]
): number[] {
  const [w, x, y, z] = quaternion;
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y + z * w),
    2 * (x * z - y * w),
    2 * (x * y - z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z + x * w)
  ];
}

/** Map each dimension onto [-1, 1], leaving a degenerate one at zero. */
export function normalize(
  values: ArrayLike<number>,
  minimum: number[],
  maximum: number[]
): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const span = maximum[i] - minimum[i];
    out[i] = span > 1e-6 ? (2 * (values[i] - minimum[i])) / span - 1 : 0;
  }
  return out;
}

/** Invert `normalize`. */
export function unnormalize(
  values: ArrayLike<number>,
  minimum: number[],
  maximum: number[]
): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const span = maximum[i] - minimum[i];
    out[i] = span > 1e-6 ? ((values[i] + 1) / 2) * span + minimum[i] : minimum[i];
  }
  return out;
}

/**
 * Pack state in the order the flow-policy export declares: six robot
 * coordinates, the cube's position, the first two columns of its rotation
 * matrix, and the target's planar position.
 */
export function packObservation(
  state: ArrayLike<number>,
  cube: CubePose,
  targetXy: [number, number]
): Float32Array {
  const packed = new Float32Array(17);
  packed.set(Float32Array.from(state).subarray(0, 6), 0);
  packed.set(cube.position, 6);
  packed.set(quatWxyzToRotation6d(cube.quaternion), 9);
  packed.set(targetXy, 15);
  return packed;
}

/** A seeded normal draw, so a rollout on this page can be repeated exactly. */
export function createNoiseSource(seed: number): () => number {
  let state = seed >>> 0 || 0x9e3779b9;
  const uniform = (): number => {
    // xorshift32: small, fast, and enough for a demonstration's noise.
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return (state + 1) / 4294967297;
  };
  return () => {
    // Box-Muller, taking one of the two draws.
    const u = uniform();
    const v = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

export interface FlowPolicyOptions {
  /** Seed for the built-in noise source; ignored when `noise` is given. */
  seed?: number;
  /**
   * Where the sampler's noise comes from. Supplying it is how a rollout gets
   * replayed against draws made somewhere else -- which is what the parity
   * check against Python needs.
   */
  noise?: () => number;
}

/** Fetch the policy's manifest and weights and build the runner from them. */
export async function loadFlowPolicy(
  baseUrl: string,
  options: FlowPolicyOptions = {}
): Promise<FlowPolicy> {
  const manifest = (await (await fetch(`${baseUrl}.json`)).json()) as FlowPolicyManifest;
  const modelUrl = `${baseUrl.replace(/[^/]*$/, '')}${manifest.model}`;
  return createFlowPolicy(manifest, modelUrl, options);
}

export async function createFlowPolicy(
  manifest: FlowPolicyManifest,
  model: string | Uint8Array,
  options: FlowPolicyOptions = {}
): Promise<FlowPolicy> {
  const sessionOptions = { executionProviders: ['wasm'] } as const;
  const session = typeof model === 'string'
    ? await ort.InferenceSession.create(model, sessionOptions)
    : await ort.InferenceSession.create(model, sessionOptions);

  const { observationSteps, observationDim, predictionSteps, endpointDim } = manifest;
  const bounds = manifest.normalization;
  const outputDim = predictionSteps * endpointDim;

  const makeNoise = (): (() => number) =>
    options.noise ?? createNoiseSource(options.seed ?? 0);

  let history: Float32Array[] = [];
  let queue: Float32Array[] = [];
  let clipped = 0;
  let noise = makeNoise();

  function reset(): void {
    history = [];
    queue = [];
    clipped = 0;
    noise = makeNoise();
  }

  async function drawHorizon(): Promise<void> {
    const flat = new Float32Array(observationSteps * observationDim);
    for (let step = 0; step < observationSteps; step += 1) {
      flat.set(history[step], step * observationDim);
    }
    const draw = new Float32Array(outputDim);
    for (let i = 0; i < outputDim; i += 1) {
      draw[i] = noise();
    }
    const outputs = await session.run({
      observations: new ort.Tensor('float32', flat, [1, flat.length]),
      noise: new ort.Tensor('float32', draw, [1, outputDim])
    });
    const generated = outputs.endpoint.data as Float32Array;

    let outside = 0;
    const clampedValues = new Float32Array(outputDim);
    for (let i = 0; i < outputDim; i += 1) {
      clampedValues[i] = Math.min(Math.max(generated[i], -1), 1);
      if (clampedValues[i] !== generated[i]) {
        outside += 1;
      }
    }
    clipped = outside / outputDim;

    for (let step = 0; step < manifest.actSteps; step += 1) {
      const slice = clampedValues.subarray(step * endpointDim, (step + 1) * endpointDim);
      queue.push(unnormalize(slice, bounds.endpoint_min, bounds.endpoint_max));
    }
  }

  async function act(
    state: ArrayLike<number>,
    cube: CubePose,
    targetXy: [number, number]
  ): Promise<Float32Array> {
    const current = packObservation(state, cube, targetXy);
    history.push(normalize(current, bounds.observation_min, bounds.observation_max));
    // At episode start the missing step is padded by repeating the first
    // observation, which is what the training export does.
    while (history.length < observationSteps) {
      history.unshift(history[0].slice());
    }
    if (history.length > observationSteps) {
      history = history.slice(history.length - observationSteps);
    }
    if (queue.length === 0) {
      await drawHorizon();
    }
    const next = queue[0];
    queue = queue.slice(1);
    return next;
  }

  return {
    manifest,
    reset,
    act,
    clippedFraction: () => clipped,
    destroy: async(): Promise<void> => {
      await session.release();
    }
  };
}
