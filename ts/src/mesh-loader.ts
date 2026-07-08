// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { MeshoptDecoder } from 'meshoptimizer';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export interface LoadedMesh {
  bytes: number;
  geometry: THREE.BufferGeometry;
}

export interface LoadedMeshSet {
  bytes: number;
  geometries: Map<string, THREE.BufferGeometry>;
}

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const cache = new Map<string, Promise<LoadedMesh>>();
const setCache = new Map<string, Promise<LoadedMeshSet>>();

export function glbName(mesh: string): string {
  return mesh.replace(/\.stl$/i, '.glb');
}

async function loadNamedGeometries(
  url: string
): Promise<{ bytes: number; geometries: Map<string, THREE.BufferGeometry> }> {
  const response = await fetch(url);
  if (!response.ok) { throw new Error(`Unable to load ${url}: ${response.status}`); }
  const buffer = await response.arrayBuffer();
  const gltf = await loader.parseAsync(buffer, '');
  const geometries = new Map<string, THREE.BufferGeometry>();
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse(object => {
    if (object instanceof THREE.Mesh) {
      const geometry = (object.geometry as THREE.BufferGeometry).clone();
      geometry.applyMatrix4(object.matrixWorld);
      geometries.set(object.name, geometry);
    }
  });
  return { bytes: buffer.byteLength, geometries };
}

export function loadMesh(url: string): Promise<LoadedMesh> {
  const cached = cache.get(url);
  if (cached) { return cached; }

  const promise = loadNamedGeometries(url).then(({ bytes, geometries }) => {
    if (geometries.size !== 1) {
      throw new Error(`Expected one mesh in ${url}, found ${geometries.size}`);
    }
    return { bytes, geometry: [...geometries.values()][0] };
  });
  cache.set(url, promise);
  return promise;
}

/** Load a GLB containing multiple named-node meshes, e.g. a whole robot packed into one file. */
export function loadMeshSet(url: string): Promise<LoadedMeshSet> {
  const cached = setCache.get(url);
  if (cached) { return cached; }

  const promise = loadNamedGeometries(url);
  setCache.set(url, promise);
  return promise;
}
