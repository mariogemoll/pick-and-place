// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

export interface BodyMaterials {
  plastic: THREE.MeshStandardMaterial;
  cube: THREE.MeshStandardMaterial;
  collision: THREE.MeshStandardMaterial;
  marker: THREE.MeshBasicMaterial;
  destroy(): void;
}

export function createBodyMaterials(): BodyMaterials {
  const plastic = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0.6, 0.7, 0.7),
    roughness: 0.6
  });
  const cube = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    roughness: 0.6
  });
  const collision = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    opacity: 0.35,
    roughness: 0.5,
    transparent: true
  });
  const marker = new THREE.MeshBasicMaterial({
    color: 0x111827,
    side: THREE.DoubleSide
  });

  return {
    plastic,
    cube,
    collision,
    marker,
    destroy(): void {
      plastic.dispose();
      cube.dispose();
      collision.dispose();
      marker.dispose();
    }
  };
}
