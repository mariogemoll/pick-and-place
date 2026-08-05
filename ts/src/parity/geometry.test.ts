// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  createGraspMatrix,
  createPregraspMatrix } from '../visualizations/canonical-grasp/pose';
import {
  createWorldFromCubeContactMatrix,
  createWorldFromCubeMatrix,
  type CubeFace } from '../visualizations/grasp-pose-shared/body-factories';
import { createSimpleGraspMatrix } from '../visualizations/simple-grasp-pose/pose';
import { expectMatrixClose, type FixturePose, loadFixture } from './fixtures';

interface ContactCase {
  pose: FixturePose;
  face: CubeFace;
  worldFromCube: number[];
  worldFromCubeContact: number[];
  simpleGrasp: number[] | null;
}

interface CanonicalCase {
  pose: FixturePose;
  closingAzimuth: number;
  approach: [number, number, number];
  grasp: number[];
  pregrasp: number[];
}

interface GeometryFixture {
  pregraspDistance: number;
  contactCases: ContactCase[];
  canonicalGraspCases: CanonicalCase[];
}

const fixture = loadFixture('geometry.json') as GeometryFixture;

// Frame algebra only — no arm, no IK. When a downstream fixture disagrees, this
// one tells you whether the two languages still share a definition of where the
// cube, the jaw contact and the grasp actually are.
describe('cube and contact transforms', () => {
  it.each(fixture.contactCases)(
    'matches Python for face $face at ($pose.x, $pose.y)',
    testCase => {
      expectMatrixClose(
        createWorldFromCubeMatrix(testCase.pose), testCase.worldFromCube, 'worldFromCube'
      );
      expectMatrixClose(
        createWorldFromCubeContactMatrix(testCase.face, testCase.pose),
        testCase.worldFromCubeContact,
        'worldFromCubeContact'
      );

      const grasp = createSimpleGraspMatrix(testCase.face, testCase.pose);
      if (testCase.simpleGrasp === null) {
        // A non-vertical face has no simple grasp. Both sides must decline it,
        // rather than one of them returning a pose the other never would.
        expect(grasp).toBeUndefined();
        return;
      }
      expect(grasp).toBeDefined();
      if (grasp === undefined) { return; }
      expectMatrixClose(grasp, testCase.simpleGrasp, 'simpleGrasp');
    }
  );
});

describe('canonical grasp transforms', () => {
  it.each(fixture.canonicalGraspCases)(
    'matches Python at ($pose.x, $pose.y) closing $closingAzimuth',
    testCase => {
      const approach = new THREE.Vector3(...testCase.approach);
      const grasp = createGraspMatrix(testCase.pose, testCase.closingAzimuth, approach);
      expectMatrixClose(grasp, testCase.grasp, 'grasp');
      expectMatrixClose(
        createPregraspMatrix(grasp, approach, fixture.pregraspDistance),
        testCase.pregrasp,
        'pregrasp'
      );
    }
  );
});
