// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { type LoadedMesh, type LoadedMeshSet,loadMesh, loadMeshSet } from './mesh-loader';
import { buildWebModel, type WebBody, type WebGeometry, type WebModel } from './web-model';

vi.mock('./mesh-loader', () => ({
  loadMesh: vi.fn(),
  loadMeshSet: vi.fn()
}));

const mockedLoadMesh = vi.mocked(loadMesh);
const mockedLoadMeshSet = vi.mocked(loadMeshSet);

function makeGeometry(overrides: Partial<WebGeometry>): WebGeometry {
  return {
    name: 'geom',
    role: 'visual',
    type: 'mesh',
    position: [0, 0, 0],
    quaternion: [1, 0, 0, 0],
    ...overrides
  };
}

function makeBody(overrides: Partial<WebBody>): WebBody {
  return {
    name: 'body',
    parent: 'world',
    position: [0, 0, 0],
    quaternion: [1, 0, 0, 0],
    joints: [],
    geometries: [],
    ...overrides
  };
}

function makeModel(bodies: WebBody[]): WebModel {
  return { format: 'pick-and-place-web-model', version: 2, materials: {}, bodies };
}

beforeEach(() => {
  mockedLoadMesh.mockReset();
  mockedLoadMeshSet.mockReset();
});

describe('buildWebModel mesh resolution', () => {
  it('resolves a mesh from its packed GLB via loadMeshSet when meshFile is set', async() => {
    const geometry = new THREE.BufferGeometry();
    const set: LoadedMeshSet = { bytes: 100, geometries: new Map([['node_a', geometry]]) };
    mockedLoadMeshSet.mockResolvedValue(set);

    const model = makeModel([
      makeBody({
        name: 'arm',
        geometries: [makeGeometry({ mesh: 'node_a', meshFile: 'arm.glb' })]
      })
    ]);

    const built = buildWebModel(model, '/assets');
    await built.ready;

    expect(mockedLoadMeshSet).toHaveBeenCalledWith('/assets/arm.glb');
    expect(mockedLoadMesh).not.toHaveBeenCalled();
    const mesh = built.bodies.get('arm')?.children[0];
    expect(mesh).toBeInstanceOf(THREE.Mesh);
    expect((mesh as THREE.Mesh).geometry).toBe(geometry);
  });

  it('falls back to loadMesh for legacy per-file geometries without meshFile', async() => {
    const geometry: LoadedMesh = { bytes: 100, geometry: new THREE.BufferGeometry() };
    mockedLoadMesh.mockResolvedValue(geometry);

    const model = makeModel([
      makeBody({ name: 'base', geometries: [makeGeometry({ mesh: 'base_part.glb' })] })
    ]);

    const built = buildWebModel(model, '/so101_assets');
    await built.ready;

    expect(mockedLoadMesh).toHaveBeenCalledWith('/so101_assets/base_part.glb');
    expect(mockedLoadMeshSet).not.toHaveBeenCalled();
  });

  it('only fetches the packed GLB(s) touched by bodies included under subtreeRoot', async() => {
    mockedLoadMeshSet.mockImplementation((url: string) => Promise.resolve({
      bytes: 100,
      geometries: new Map([
        [url.endsWith('gripper.glb') ? 'jaw' : 'link', new THREE.BufferGeometry()]
      ])
    }));

    const model = makeModel([
      makeBody({
        name: 'arm',
        parent: 'world',
        geometries: [makeGeometry({ mesh: 'link', meshFile: 'arm.glb' })]
      }),
      makeBody({
        name: 'gripper',
        parent: 'arm',
        geometries: [makeGeometry({ mesh: 'jaw', meshFile: 'gripper.glb' })]
      })
    ]);

    const built = buildWebModel(model, '/so101_assets', 'gripper');
    await built.ready;

    expect(mockedLoadMeshSet).toHaveBeenCalledTimes(1);
    expect(mockedLoadMeshSet).toHaveBeenCalledWith('/so101_assets/gripper.glb');
    expect(built.bodies.has('arm')).toBe(false);
    expect(built.bodies.has('gripper')).toBe(true);
  });
});
