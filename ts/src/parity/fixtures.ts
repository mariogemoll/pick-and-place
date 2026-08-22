// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

// Loading and comparison helpers for the cross-language parity fixtures in
// `fixtures/parity/`. Python writes them (`pap generate-parity-fixtures`)
// and checks itself against them (py/tests/test_parity.py); the tests beside
// this file are the TypeScript half of the same check.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import * as THREE from 'three';
import { expect } from 'vitest';

import { ARM_JOINT_NAMES, type ArmJointName, deriveSo101Kinematics } from '../ik/kinematics';
import type { WebModel } from '../web-model';

// Absolute tolerance on a fixture number. The fixtures carry twelve significant
// digits, so this admits only last-place arithmetic wobble between the two
// languages, not a difference anyone would have to reason about.
export const TOLERANCE = 1e-9;

export type FixtureJoints = Record<ArmJointName, number>;

export interface FixturePose {
  x: number;
  y: number;
  z: number;
  roll: number;
  pitch: number;
  yaw: number;
}

function repoFile(relativePath: string): string {
  return readFileSync(
    fileURLToPath(new URL(`../../../${relativePath}`, import.meta.url)),
    'utf8'
  );
}

export function loadFixture(name: string): unknown {
  return JSON.parse(repoFile(`fixtures/parity/${name}`));
}

// The exported robot manifest the TypeScript app runs on. Generated, not
// committed: see the note on generated fixtures in AGENTS.md.
export const webModel = JSON.parse(repoFile('ts/public/so101.json')) as WebModel;

export const kinematics = deriveSo101Kinematics(webModel);

// The fixtures store a 4x4 as sixteen row-major numbers; `fromArray` reads
// column-major, so transpose after loading.
export function matrixFromFixture(values: number[]): THREE.Matrix4 {
  return new THREE.Matrix4().fromArray(values).transpose();
}

export function expectClose(
  actual: number,
  expected: number,
  label: string,
  tolerance = TOLERANCE
): void {
  expect(Math.abs(actual - expected), `${label}: ${actual} != ${expected}`)
    .toBeLessThanOrEqual(tolerance);
}

export function expectVectorClose(
  actual: THREE.Vector3,
  expected: number[],
  label: string,
  tolerance = TOLERANCE
): void {
  expectClose(actual.x, expected[0], `${label}.x`, tolerance);
  expectClose(actual.y, expected[1], `${label}.y`, tolerance);
  expectClose(actual.z, expected[2], `${label}.z`, tolerance);
}

export function expectMatrixClose(
  actual: THREE.Matrix4,
  expected: number[],
  label: string
): void {
  const rowMajor = actual.clone().transpose().toArray();
  for (let i = 0; i < 16; i++) {
    expectClose(rowMajor[i], expected[i], `${label}[${i}]`);
  }
}

export function expectJointsClose(
  actual: Record<ArmJointName, number>,
  expected: FixtureJoints,
  label: string
): void {
  for (const name of ARM_JOINT_NAMES) {
    expectClose(actual[name], expected[name], `${label}.${name}`);
  }
}
