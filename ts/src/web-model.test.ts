// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { type LoadedMesh, type LoadedMeshSet,loadMesh, loadMeshSet } from './mesh-loader';
import {
  buildWebModel,
  cameraByName,
  createWebCamera,
  type WebBody,
  type WebCamera,
  type WebGeometry,
  type WebModel
} from './web-model';

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

function makeModel(bodies: WebBody[], cameras: WebCamera[] = []): WebModel {
  return { format: 'pick-and-place-web-model', version: 2, materials: {}, bodies, cameras };
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

function makeCamera(overrides: Partial<WebCamera>): WebCamera {
  return {
    name: 'camera',
    body: 'mount',
    position: [0, 0, 0],
    quaternion: [1, 0, 0, 0],
    fovy: 45,
    ...overrides
  };
}

describe('cameraByName', () => {
  it('finds the camera', () => {
    const model = makeModel([], [makeCamera({ name: 'wrist_camera', fovy: 46.9 })]);
    expect(cameraByName(model, 'wrist_camera').fovy).toBeCloseTo(46.9);
  });

  it('throws when the model has no such camera', () => {
    expect(() => cameraByName(makeModel([]), 'wrist_camera')).toThrow(/wrist_camera/);
  });
});

describe('createWebCamera', () => {
  it('takes the vertical field of view straight from the manifest', () => {
    expect(createWebCamera(makeCamera({ fovy: 47.2 })).fov).toBeCloseTo(47.2);
  });

  it('looks down its own -Z, as MuJoCo cameras do', () => {
    const direction = createWebCamera(makeCamera({}))
      .getWorldDirection(new THREE.Vector3());
    expect(direction.x).toBeCloseTo(0);
    expect(direction.y).toBeCloseTo(0);
    expect(direction.z).toBeCloseTo(-1);
  });

  it('reads the quaternion in MuJoCo (w, x, y, z) order', () => {
    // A quarter turn about +X, which swings the view direction from -Z to +Y.
    const quarterTurn = Math.SQRT1_2;
    const camera = createWebCamera(makeCamera({ quaternion: [quarterTurn, quarterTurn, 0, 0] }));

    const direction = camera.getWorldDirection(new THREE.Vector3());
    expect(direction.x).toBeCloseTo(0);
    expect(direction.y).toBeCloseTo(1);
    expect(direction.z).toBeCloseTo(0);
  });

  it('rides the body it is added to', () => {
    const model = makeModel(
      [makeBody({ name: 'mount', position: [0, 0, 0.5] })],
      [makeCamera({ position: [0, 0, -0.02] })]
    );
    const built = buildWebModel(model);
    const camera = createWebCamera(cameraByName(model, 'camera'));
    const mount = built.bodies.get('mount');
    expect(mount).toBeDefined();
    mount?.add(camera);

    const position = camera.getWorldPosition(new THREE.Vector3());
    expect(position.z).toBeCloseTo(0.48);
  });
});
