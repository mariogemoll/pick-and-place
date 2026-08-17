// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// The pick cube's six AprilTags, as geometry rather than a texture.
//
// The simulator textures the cube with one sticker image per face; here the tag
// patterns are real geometry laid just off each face, so they stay crisp at any
// zoom. This lives on its own so both the cubes built from a manifest and the
// ones built by hand place their tags the same way.

import * as THREE from 'three';

import { createAprilTagCellGeometry } from './tag-mesh';

/** One tag per face, in THREE.BoxGeometry material-group order. */
export const CUBE_APRILTAG_IDS = [0, 1, 2, 3, 4, 5] as const;

/** The 30 mm sticker covers the whole cube face; the tag graphic is 20 mm. */
export const CUBE_TAG_SIZE = 0.02;

// The six faces in the same order, as the rotation that carries the tag
// geometry's local +Z normal onto the outward face normal, and the direction of
// the face center from the cube center.
const CUBE_FACE_PLACEMENTS: readonly (readonly [THREE.Euler, THREE.Vector3])[] = [
  [new THREE.Euler(0, Math.PI / 2, 0), new THREE.Vector3(1, 0, 0)],
  [new THREE.Euler(0, -Math.PI / 2, 0), new THREE.Vector3(-1, 0, 0)],
  [new THREE.Euler(-Math.PI / 2, 0, 0), new THREE.Vector3(0, 1, 0)],
  [new THREE.Euler(Math.PI / 2, 0, 0), new THREE.Vector3(0, -1, 0)],
  [new THREE.Euler(0, 0, 0), new THREE.Vector3(0, 0, 1)],
  [new THREE.Euler(Math.PI, 0, 0), new THREE.Vector3(0, 0, -1)]
];

export interface CubeAprilTags {
  group: THREE.Group;
  /** Disposes the tag geometries. The material stays the caller's to dispose. */
  dispose(): void;
}

/**
 * The cube's tags, ready to add to whatever carries the cube's transform.
 *
 * `offset` is the distance from the cube's center to the plane a tag is drawn
 * on, so it is the cube's half size plus however far off the surface the tag
 * should sit. The white margin around each tag is left to the face showing
 * through, so the faces have to be white for the tags to read.
 */
export function createCubeAprilTags(
  offset: number,
  material: THREE.Material
): CubeAprilTags {
  const group = new THREE.Group();
  group.name = 'cube_apriltags';
  const geometries: THREE.BufferGeometry[] = [];

  for (const [index, [euler, direction]] of CUBE_FACE_PLACEMENTS.entries()) {
    const geometry = createAprilTagCellGeometry(CUBE_APRILTAG_IDS[index], CUBE_TAG_SIZE);
    geometries.push(geometry);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `cube_apriltag_${CUBE_APRILTAG_IDS[index]}`;
    mesh.setRotationFromEuler(euler);
    mesh.position.copy(direction).multiplyScalar(offset);
    group.add(mesh);
  }

  return {
    group,
    dispose(): void {
      for (const geometry of geometries) { geometry.dispose(); }
    }
  };
}
