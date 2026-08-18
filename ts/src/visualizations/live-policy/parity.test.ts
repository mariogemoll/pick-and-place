// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

// The browser policy stack, checked against a rollout Python actually ran.
//
// `py/scripts/export_policy_parity_fixture.py` drives the real `PolicySimEnv`
// and the real checkpoint for forty ticks, writing down the noise behind every
// horizon along with the observation, action and qpos at each one. Here the
// same noise goes through the TypeScript environment and the ONNX export, and
// the trajectory has to come back the same.
//
// This is the check that stands in for opening a browser: everything the page
// runs -- the compiled scene, the WebAssembly engine, the exported graph, the
// packing and normalization around it -- is exercised here, in Node, against
// numbers Python produced.
//
// Like the five tests that read `ts/public/so101.json`, the inputs are
// generated rather than committed: they need a checkpoint and a compiled scene.
// The suite skips itself when they are absent.

import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  createPolicyEnvironment,
  type PolicySceneManifest
} from './environment';
import {
  createFlowPolicy,
  type FlowPolicyManifest,
  packObservation
} from './flow-policy-runner';

function publicFile(name: string): string {
  return fileURLToPath(new URL(`../../../public/${name}`, import.meta.url));
}

interface ParityTick {
  packedObservation: number[];
  drewNoise: boolean;
  action: number[];
  qposBefore: number[];
  qposAfter: number[];
}

interface ParityFixture {
  scenarioId: string;
  setup: {
    cube: { position: number[]; quaternion: number[] };
    targetXy: number[];
    initialJointsReal: number[];
  };
  noiseDraws: number[][];
  ticks: ParityTick[];
}

const required = [
  'policy-parity.json',
  'policy-scene.json',
  'policy-scene.mjb',
  'flow-policy.json'
];
const available = required.every(name => existsSync(publicFile(name)));

// The check has two halves, because a closed loop cannot be held to one bound.
//
// **The first tick is the sharp one.** Both sides start from the same state and
// the same noise, so nothing has had a chance to drift and the only difference
// left is arithmetic. Python and the browser agree on the observation exactly,
// on the action to a few times 1e-5 degrees (the exported graph's disagreement
// with PyTorch, ~5e-7 in normalized action space, unnormalized into degrees),
// and on the resulting state to 2e-7. This is what catches a real defect: the
// unapplied tracking bias found during development sat at 2.8e-3 here, four
// orders of magnitude above the floor.
const FIRST_TICK_ACTION_TOLERANCE = 1e-4;
const FIRST_TICK_QPOS_TOLERANCE = 1e-5;

// **The whole rollout is the loose one, and deliberately so.** Every tick feeds
// the next, so a last-place difference in one action is amplified by four
// seconds of contact-rich physics. Over forty ticks the two trajectories part
// by about 0.04 degrees and 0.02 in qpos, which is a property of the task
// rather than of either implementation. Bounding it says the browser stays in
// the same episode Python ran; it does not say the two are identical, and no
// tolerance here could.
const ROLLOUT_OBSERVATION_TOLERANCE = 0.5;
const ROLLOUT_QPOS_TOLERANCE = 0.1;

describe.skipIf(!available)('browser policy stack against a Python rollout', () => {
  it('reproduces the recorded trajectory tick for tick', async() => {
    const fixture = JSON.parse(
      readFileSync(publicFile('policy-parity.json'), 'utf8')
    ) as ParityFixture;
    const sceneManifest = JSON.parse(
      readFileSync(publicFile('policy-scene.json'), 'utf8')
    ) as PolicySceneManifest;
    const policyManifest = JSON.parse(
      readFileSync(publicFile('flow-policy.json'), 'utf8')
    ) as FlowPolicyManifest;

    const environment = await createPolicyEnvironment(
      sceneManifest,
      new Uint8Array(readFileSync(publicFile(sceneManifest.model)))
    );

    // Hand the sampler exactly the draws Python used, in order. Anything else
    // and the two integrate different noise and nothing downstream compares.
    let drawIndex = 0;
    let cursor = 0;
    const noise = (): number => {
      const value = fixture.noiseDraws[drawIndex][cursor];
      cursor += 1;
      if (cursor === fixture.noiseDraws[drawIndex].length) {
        cursor = 0;
        drawIndex += 1;
      }
      return value;
    };

    const policy = await createFlowPolicy(
      policyManifest,
      new Uint8Array(readFileSync(publicFile(policyManifest.model))),
      { noise }
    );

    environment.reset({
      cube: {
        position: fixture.setup.cube.position as [number, number, number],
        quaternion: fixture.setup.cube.quaternion as [number, number, number, number]
      },
      targetXy: fixture.setup.targetXy as [number, number],
      initialJointsReal: fixture.setup.initialJointsReal
    });

    // Collected across the whole rollout rather than asserted tick by tick, so
    // a failure reports how far the two actually drifted apart instead of
    // stopping at whichever tick crossed the line first.
    const worst = { observation: 0, action: 0, qpos: 0 };
    const first = { observation: 0, action: 0, qpos: 0 };
    let tickIndex = 0;
    const record = (key: keyof typeof worst, actual: number, expected: number): void => {
      const difference = Math.abs(actual - expected);
      worst[key] = Math.max(worst[key], difference);
      if (tickIndex === 0) { first[key] = Math.max(first[key], difference); }
    };

    for (const tick of fixture.ticks) {
      const state = environment.observe();
      const cube = environment.cubePose();

      const packed = packObservation(state, cube, environment.targetXy());
      tick.packedObservation.forEach((expected, i) => {
        record('observation', packed[i], expected);
      });

      const action = await policy.act(state, cube, environment.targetXy());
      tick.action.forEach((expected, i) => { record('action', action[i], expected); });

      environment.step(action);

      const pose = environment.cubePose();
      const actual = [...environment.jointAnglesRad(), ...pose.position, ...pose.quaternion];
      tick.qposAfter.forEach((expected, i) => { record('qpos', actual[i], expected); });
      tickIndex += 1;
    }

    expect(fixture.ticks.length).toBeGreaterThan(20);

    expect(first.observation, 'first-tick observation difference').toBe(0);
    expect(first.action, 'first-tick action difference').toBeLessThan(FIRST_TICK_ACTION_TOLERANCE);
    expect(first.qpos, 'first-tick state difference').toBeLessThan(FIRST_TICK_QPOS_TOLERANCE);

    expect(worst.observation, 'rollout observation drift').toBeLessThan(
      ROLLOUT_OBSERVATION_TOLERANCE
    );
    expect(worst.qpos, 'rollout state drift').toBeLessThan(ROLLOUT_QPOS_TOLERANCE);

    await policy.destroy();
    environment.destroy();
  }, 120_000);
});
