// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

// A large flat plane rather than a true infinite ground: at this size its
// edge is far outside the fog-faded view distance used by the robot viewer,
// so it reads as an unbounded floor without the cost of a custom shader.
const PLANE_SIZE = 500;

export function createInfiniteFloor(color: THREE.ColorRepresentation): THREE.Mesh {
  const geometry = new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE);
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.95 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.receiveShadow = true;
  return mesh;
}
