// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { loadMesh, loadMeshSet } from './mesh-loader';

export interface WebJointMimic {
  joint: string;
  multiplier: number;
  offset: number;
}

export interface WebJoint {
  name: string;
  type: 'free' | 'ball' | 'slide' | 'hinge';
  position: [number, number, number];
  axis: [number, number, number];
  limited: boolean;
  range?: [number, number];
  mimic?: WebJointMimic;
}

export interface WebGeometry {
  name: string;
  role: 'visual' | 'collision';
  type: 'plane' | 'sphere' | 'capsule' | 'ellipsoid' | 'cylinder' | 'box' | 'mesh';
  position: [number, number, number];
  quaternion: [number, number, number, number];
  material?: string;
  rgba?: [number, number, number, number];
  mesh?: string;
  size?: [number, number, number];
}

export interface WebBody {
  name: string;
  parent: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  joints: WebJoint[];
  geometries: WebGeometry[];
}

export interface WebModel {
  format: 'pick-and-place-web-model';
  version: 2;
  materials: Record<string, [number, number, number, number]>;
  bodies: WebBody[];
  /**
   * A single GLB (relative to the model's base path) containing every mesh
   * as a named node; when set, `geometry.mesh` is a node name inside it
   * rather than a standalone file. Absent for models that still ship one
   * GLB per mesh.
   */
  meshFile?: string;
}

export interface BuiltWebModel {
  root: THREE.Group;
  bodies: Map<string, THREE.Group>;
  jointPivots: Map<string, THREE.Group>;
  materialsByName: Map<string, THREE.MeshStandardMaterial[]>;
  ready: Promise<void>;
}

const cache = new Map<string, Promise<WebModel>>();

export function loadWebModel(url = '/so101.json'): Promise<WebModel> {
  const cached = cache.get(url);
  if (cached) { return cached; }
  const promise = fetch(url).then(async response => {
    if (!response.ok) { throw new Error(`Unable to load ${url}: ${response.status}`); }
    return await response.json() as WebModel;
  });
  cache.set(url, promise);
  return promise;
}

function setQuaternion(
  object: THREE.Object3D,
  [w, x, y, z]: [number, number, number, number]
): void {
  // Manifest quaternions come straight from the MJCF spec and may be
  // unnormalized (e.g. "1 0 1 0"); three.js assumes unit quaternions and
  // would otherwise distort the transform.
  object.quaternion.set(x, y, z, w).normalize();
}

export function materialFor(
  geometry: WebGeometry,
  modelMaterials: Record<string, [number, number, number, number]>
): THREE.MeshStandardMaterial {
  const materialKey = geometry.material;
  const isOverlay = geometry.name.startsWith('workspace_');
  const sourceRgba =
    (materialKey !== undefined ? modelMaterials[materialKey] : undefined) ??
    geometry.rgba ??
    [0.5, 0.5, 0.5, 1];
  const [r, g, b, initialAlpha] = sourceRgba;
  const a = isOverlay ? 1.0 : initialAlpha;

  // User request: workspace borders should not be transparent.
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace),
    opacity: a,
    roughness: 0.6,
    transparent: a < 1
  });
}

export function primitiveGeometry(geometry: WebGeometry): THREE.BufferGeometry | undefined {
  const size = geometry.size;
  if (size === undefined) { return undefined; }
  if (geometry.type === 'plane') {
    const width = size[0] > 0 ? size[0] * 2 : 100;
    const height = size[1] > 0 ? size[1] * 2 : 100;
    return new THREE.PlaneGeometry(width, height);
  }
  if (geometry.type === 'box') {
    return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
  }
  if (geometry.type === 'sphere') {
    return new THREE.SphereGeometry(size[0], 24, 16);
  }
  if (geometry.type === 'ellipsoid') {
    const sphere = new THREE.SphereGeometry(1, 24, 16);
    sphere.scale(size[0], size[1], size[2]);
    return sphere;
  }
  if (geometry.type === 'cylinder') {
    const cylinder = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 24);
    cylinder.rotateX(Math.PI / 2);
    return cylinder;
  }
  if (geometry.type === 'capsule') {
    const capsule = new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 16);
    capsule.rotateX(Math.PI / 2);
    return capsule;
  }
  return undefined;
}

function addVisual(
  bodyGroup: THREE.Group,
  geometry: WebGeometry,
  bufferGeometry: THREE.BufferGeometry,
  material: THREE.Material
): void {
  const mesh = new THREE.Mesh(bufferGeometry, material);
  mesh.name = geometry.name;
  mesh.userData.role = geometry.role;
  mesh.position.set(...geometry.position);
  setQuaternion(mesh, geometry.quaternion);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  bodyGroup.add(mesh);
}

export function buildWebModel(
  model: WebModel,
  modelBasePath = '/so101_assets',
  subtreeRoot?: string
): BuiltWebModel {
  const root = new THREE.Group();
  const bodies = new Map<string, THREE.Group>();
  const jointPivots = new Map<string, THREE.Group>();
  const materialsByName = new Map<string, THREE.MeshStandardMaterial[]>();
  const meshLoads: Promise<void>[] = [];
  const basePath = modelBasePath.replace(/\/$/, '');
  const meshSet = model.meshFile !== undefined
    ? loadMeshSet(`${basePath}/${model.meshFile}`)
    : undefined;
  const included = new Set<string>();

  if (subtreeRoot !== undefined) {
    included.add(subtreeRoot);
    let changed = true;
    while (changed) {
      changed = false;
      for (const body of model.bodies) {
        if (!included.has(body.name) && included.has(body.parent)) {
          included.add(body.name);
          changed = true;
        }
      }
    }
  } else {
    for (const body of model.bodies) { included.add(body.name); }
  }

  for (const body of model.bodies) {
    if (!included.has(body.name)) { continue; }
    const bodyGroup = new THREE.Group();
    bodyGroup.name = body.name;
    bodies.set(body.name, bodyGroup);

    const origin = new THREE.Group();
    origin.position.set(...body.position);
    setQuaternion(origin, body.quaternion);
    origin.add(bodyGroup);

    const joint = body.joints.find(
      candidate => candidate.type === 'hinge' || candidate.type === 'slide'
    );
    if (joint) {
      const pivot = new THREE.Group();
      pivot.add(bodyGroup);
      origin.remove(bodyGroup);
      origin.add(pivot);
      jointPivots.set(joint.name, pivot);
    }

    const parent = bodies.get(body.parent);
    if (parent !== undefined && included.has(body.parent) && body.name !== body.parent) {
      parent.add(origin);
    } else {
      root.add(subtreeRoot === body.name ? bodyGroup : origin);
    }

    for (const geometry of body.geometries) {
      if (geometry.role !== 'visual') { continue; }
      const material = materialFor(geometry, model.materials);
      if (geometry.material !== undefined) {
        const slot = materialsByName.get(geometry.material) ?? [];
        slot.push(material);
        materialsByName.set(geometry.material, slot);
      }
      if (geometry.type === 'mesh' && geometry.mesh !== undefined) {
        const meshName = geometry.mesh;
        const geometryLoad: Promise<THREE.BufferGeometry> = meshSet !== undefined
          ? meshSet.then(({ geometries }) => {
            const bufferGeometry = geometries.get(meshName);
            if (bufferGeometry === undefined) {
              throw new Error(`Mesh node "${meshName}" not found in ${model.meshFile ?? ''}`);
            }
            return bufferGeometry;
          })
          : loadMesh(`${basePath}/${meshName}`)
            .then(({ geometry: bufferGeometry }) => bufferGeometry);
        const meshLoad = geometryLoad.then(bufferGeometry => {
          addVisual(bodyGroup, geometry, bufferGeometry, material);
        }).catch((err: unknown) => {
          console.warn(`Failed to load mesh ${meshName}:`, err);
        });
        meshLoads.push(meshLoad);
      } else {
        const bufferGeometry = primitiveGeometry(geometry);
        if (bufferGeometry !== undefined) {
          addVisual(bodyGroup, geometry, bufferGeometry, material);
        }
      }
    }
  }

  return {
    root,
    bodies,
    jointPivots,
    materialsByName,
    ready: Promise.all(meshLoads).then(() => undefined)
  };
}

export function setJointAngle(
  model: WebModel,
  jointPivots: Map<string, THREE.Group>,
  name: string,
  value: number
): void {
  const joint = model.bodies.flatMap(body => body.joints)
    .find(candidate => candidate.name === name);
  const pivot = jointPivots.get(name);
  if (!joint || !pivot) { return; }
  const axis = new THREE.Vector3(...joint.axis).normalize();
  if (joint.type === 'slide') {
    pivot.position.copy(axis.multiplyScalar(value));
  } else {
    pivot.setRotationFromAxisAngle(axis, value);
  }
}
