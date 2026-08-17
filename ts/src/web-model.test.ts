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

// The tag graphic is a material in the manifest, textured for MuJoCo's renderer
// and drawn as geometry here, so it hangs off whichever geom carries that
// material -- a plane laid over the plate, in the manifest the exporter writes.
describe('workspace frame AprilTag plates', () => {
  function buildTagFace(): { face: THREE.Mesh; cells: THREE.Mesh } {
    const model = makeModel([makeBody({
      name: 'workspace_frame_frame',
      geometries: [makeGeometry({
        name: 'workspace_frame_tag_ne_top',
        type: 'plane',
        size: [0.03, 0.03, 0.03],
        position: [0.23, 0.23, 0.00501],
        material: 'workspace_frame_apriltag_12_material'
      })]
    })]);
    const built = buildWebModel(model);
    const face = built.bodies
      .get('workspace_frame_frame')
      ?.getObjectByName('workspace_frame_tag_ne_top') as THREE.Mesh;
    const cells = face.getObjectByName('workspace_frame_tag_ne_top_tag') as THREE.Mesh;
    return { face, cells };
  }

  it('draws the tag pattern on the face that carries the tag material', () => {
    const { cells } = buildTagFace();
    expect(cells).toBeDefined();
    expect(cells.geometry.getAttribute('position').count).toBeGreaterThan(0);
  });

  it('lifts the cells clear of the face they are drawn on', () => {
    const { cells } = buildTagFace();
    expect(cells.position.z).toBeGreaterThan(0);
  });

  // The face is modelled 10 um above its plate, which no depth buffer resolves
  // at the distance the scene is viewed from.
  it('biases the face and its cells out of a depth fight, cells in front', () => {
    const { face, cells } = buildTagFace();
    const faceMaterial = face.material as THREE.MeshStandardMaterial;
    const cellMaterial = cells.material as THREE.MeshStandardMaterial;
    expect(faceMaterial.polygonOffset).toBe(true);
    expect(faceMaterial.polygonOffsetFactor).toBeLessThan(0);
    expect(cellMaterial.polygonOffset).toBe(true);
    expect(cellMaterial.polygonOffsetFactor).toBeLessThan(faceMaterial.polygonOffsetFactor);
  });

  it('lays a tag on each face of the pick cube', () => {
    const halfSize = 0.015;
    const model = makeModel([makeBody({
      name: 'pick_cube',
      geometries: [makeGeometry({
        name: 'pick_cube',
        type: 'box',
        size: [halfSize, halfSize, halfSize],
        material: 'pick_cube_apriltags'
      })]
    })]);
    const cube = buildWebModel(model).bodies
      .get('pick_cube')
      ?.getObjectByName('pick_cube') as THREE.Mesh;
    const tags = cube.getObjectByName('cube_apriltags');

    expect(tags?.children).toHaveLength(6);
    // One per face, each just clear of the surface it is printed on.
    const distances = tags?.children.map(tag => tag.position.length()) ?? [];
    expect(distances.every(distance => distance > halfSize)).toBe(true);
    expect(Math.max(...distances)).toBeLessThan(halfSize * 1.05);
  });

  it('leaves the plates and the rest of the frame to draw normally', () => {
    const model = makeModel([makeBody({
      name: 'workspace_frame_frame',
      geometries: [makeGeometry({
        name: 'workspace_frame_tag_ne',
        type: 'box',
        size: [0.03, 0.03, 0.0025],
        position: [0.23, 0.23, 0.0025]
      })]
    })]);
    const plate = buildWebModel(model).bodies
      .get('workspace_frame_frame')
      ?.getObjectByName('workspace_frame_tag_ne') as THREE.Mesh;

    expect(plate.children).toHaveLength(0);
    expect((plate.material as THREE.MeshStandardMaterial).polygonOffset).toBe(false);
  });
});
