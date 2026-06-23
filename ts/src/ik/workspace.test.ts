// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

/// <reference types="node" />

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  CUBE_HALF_SIZE,
  type CubeFace,
  type CubePose
} from '../visualizations/grasp-pose-shared/body-factories';
import { createSimpleGraspMatrix } from '../visualizations/simple-grasp-pose/pose';
import type { WebModel } from '../web-model';
import { deriveSo101Kinematics } from './kinematics';
import { solveSimpleGraspIk } from './simple-ik';
import {
  anyYawCubeCenterBand,
  computeArmWorkspaceAtHeight,
  computeGlobalXyWorkspace,
  computeSimpleWorkspace,
  computeSimpleWorkspaceForCubeZ,
  CUBE_Z_1CM_OVER_GROUND_TOP,
  sectorBoundingBox
} from './workspace';

const model = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../public/so101.json', import.meta.url)),
    'utf8'
  )
) as WebModel;

const VERTICAL_FACES: CubeFace[] = ['+x', '-x', '+y', '-y'];

// At a cube center (x, y) with the given yaw, can at least one vertical face be
// grasped by the simple grasp IK?
function anyFaceSolves(
  k: ReturnType<typeof deriveSo101Kinematics>,
  x: number,
  y: number,
  yaw: number
): boolean {
  const pose: CubePose = { x, y, z: CUBE_HALF_SIZE, roll: 0, pitch: 0, yaw };
  return VERTICAL_FACES.some(face => {
    const matrix = createSimpleGraspMatrix(face, pose);
    if (matrix === undefined) { return false; }
    return solveSimpleGraspIk(k, matrix).type === 'success';
  });
}

describe('computeSimpleWorkspace', () => {
  const k = deriveSo101Kinematics(model);
  const sector = computeSimpleWorkspace(k);

  it('reports the closed-form sector for the loaded model', () => {
    console.log(
      '[simple workspace] radial %s..%s m, azimuth +/-%s deg, ' +
      'targetHeight %s m, faceOffset %s m, anyYawReachable=%s',
      sector.radial.min.toFixed(4),
      sector.radial.max.toFixed(4),
      ((sector.azimuth.max * 180) / Math.PI).toFixed(1),
      sector.targetHeight.toFixed(4),
      sector.faceOffset.toFixed(4),
      sector.anyYawReachable
    );

    // Pin the computed numbers so changes to the kinematics are visible here.
    expect(sector.targetHeight).toBeCloseTo(CUBE_HALF_SIZE, 4);
    expect(sector.radial.min).toBeCloseTo(0.0537, 3);
    expect(sector.radial.max).toBeCloseTo(0.2750, 3);
    expect(sector.azimuth.min).toBeCloseTo(-1.91986, 4);
    expect(sector.azimuth.max).toBeCloseTo(1.91986, 4);
    expect(sector.anyYawReachable).toBe(true);
  });

  it('validates the sector: every interior cube center solves for all yaws', () => {
    // Shrink the radial band by the face offset (target vs. cube center) and the
    // azimuth band slightly, then brute-check that a worst-case yaw sweep always
    // finds a graspable face -- confirming the closed-form claim.
    const rLo = sector.radial.min + sector.faceOffset;
    const rHi = sector.radial.max - sector.faceOffset;
    const azPad = 0.05;

    for (let r = rLo; r <= rHi; r += (rHi - rLo) / 8) {
      for (let az = sector.azimuth.min + azPad; az <= sector.azimuth.max - azPad;
        az += (sector.azimuth.max - sector.azimuth.min - 2 * azPad) / 10) {
        const x = sector.panAxis.x + r * Math.cos(az);
        const y = sector.panAxis.y + r * Math.sin(az);
        for (let yaw = 0; yaw < Math.PI / 2; yaw += Math.PI / 2 / 18) {
          expect(anyFaceSolves(k, x, y, yaw)).toBe(true);
        }
      }
    }
  });

  it('shrinks the target band by the face offset for cube centers', () => {
    const band = anyYawCubeCenterBand(sector);
    expect(band.min).toBeCloseTo(sector.radial.min + sector.faceOffset, 6);
    expect(band.max).toBeCloseTo(sector.radial.max - sector.faceOffset, 6);
  });

  it('bounds the cube-center sector (drives the X/Y slider ranges)', () => {
    const bbox = sectorBoundingBox(sector);
    const band = anyYawCubeCenterBand(sector);
    // Pan range exceeds +/-90 deg, so the outer radius sets both Y extremes and
    // the +X extreme, measured from the pan axis.
    expect(bbox.x.max).toBeCloseTo(sector.panAxis.x + band.max, 4);
    expect(bbox.y.max).toBeCloseTo(sector.panAxis.y + band.max, 4);
    expect(bbox.y.min).toBeCloseTo(sector.panAxis.y - band.max, 4);
    expect(bbox.x.min).toBeLessThan(sector.panAxis.x);
  });

  it('confirms wrist_roll cannot veto every cube yaw', () => {
    const roll = k.jointLimits.wrist_roll;
    const forbiddenArc = 2 * Math.PI - (roll.max - roll.min);
    expect(forbiddenArc).toBeLessThan(Math.PI / 2);
  });
});

describe('computeSimpleWorkspaceForCubeZ (cube 1 cm above ground-cube top)', () => {
  const k = deriveSo101Kinematics(model);
  const clearance = computeSimpleWorkspaceForCubeZ(k, CUBE_Z_1CM_OVER_GROUND_TOP);

  it('reports the clearance sector', () => {
    console.log(
      '[clearance workspace] radial %s..%s m, targetHeight %s m',
      clearance.radial.min.toFixed(4),
      clearance.radial.max.toFixed(4),
      clearance.targetHeight.toFixed(4)
    );
    expect(clearance.targetHeight).toBeCloseTo(CUBE_Z_1CM_OVER_GROUND_TOP, 3);
  });

  it('is smaller than the ground sector (arm reach shrinks at higher z)', () => {
    const ground = computeSimpleWorkspace(k);
    expect(clearance.radial.max).toBeLessThan(ground.radial.max);
  });

  it('shares the same azimuth band as the ground sector', () => {
    const ground = computeSimpleWorkspace(k);
    expect(clearance.azimuth.min).toBeCloseTo(ground.azimuth.min, 6);
    expect(clearance.azimuth.max).toBeCloseTo(ground.azimuth.max, 6);
  });
});

describe('computeArmWorkspaceAtHeight (jaw contact reach at z = 1.5 cm)', () => {
  const k = deriveSo101Kinematics(model);
  const groundHeight = computeSimpleWorkspace(k).targetHeight;
  const armWs = computeArmWorkspaceAtHeight(k, groundHeight);

  it('reports the sector', () => {
    console.log(
      '[arm@ground-height workspace] radial %s..%s m',
      armWs.radial.min.toFixed(4),
      armWs.radial.max.toFixed(4)
    );
    expect(armWs.targetHeight).toBeCloseTo(groundHeight, 4);
    expect(armWs.radial.max).toBeGreaterThan(0);
  });

  it('outer radius strictly exceeds the grasp ground sector', () => {
    const ground = computeSimpleWorkspace(k);
    expect(armWs.radial.max).toBeGreaterThan(ground.radial.max);
  });

  it('outer radius is within the global max reach', () => {
    const global = computeGlobalXyWorkspace(k);
    expect(armWs.radial.max).toBeLessThanOrEqual(global.radial.max + 1e-6);
  });
});

describe('computeGlobalXyWorkspace (arm max reach, any joint config)', () => {
  const k = deriveSo101Kinematics(model);
  const global = computeGlobalXyWorkspace(k);

  it('reports the global sector', () => {
    console.log(
      '[global workspace] radial %s..%s m',
      global.radial.min.toFixed(4),
      global.radial.max.toFixed(4)
    );
    expect(global.radial.min).toBeGreaterThanOrEqual(0);
    expect(global.radial.max).toBeGreaterThan(0);
    expect(Number.isNaN(global.targetHeight)).toBe(true);
    // Must be substantially larger than the grasp-only sector.
    expect(global.radial.max).toBeGreaterThan(0.35);
  });

  it('outer radius exceeds both grasp sectors', () => {
    const ground = computeSimpleWorkspace(k);
    const clearance = computeSimpleWorkspaceForCubeZ(k, CUBE_Z_1CM_OVER_GROUND_TOP);
    expect(global.radial.max).toBeGreaterThan(ground.radial.max);
    expect(global.radial.max).toBeGreaterThan(clearance.radial.max);
  });
});
