// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { TAG_41H12_BITS } from './tag-bits';

const TAG_CELLS = 9;

/**
 * Merged geometry of a tag's black cells, centered on the origin in the XY plane
 * with an outward normal of +Z. The pattern spans `tagSize` × `tagSize`; the
 * surrounding white margin is left to the (white) plate or cube surface it sits
 * on. Because it is real geometry rather than a texture, it stays crisp at any
 * zoom level. The caller owns the returned geometry and must dispose it.
 */
export function createAprilTagCellGeometry(
  tagId: number,
  tagSize: number
): THREE.BufferGeometry {
  const grid = TAG_41H12_BITS[tagId];
  if (grid === undefined) {
    throw new Error(`tagStandard41h12 id ${tagId} is not in the local bit table`);
  }

  const cellSize = tagSize / TAG_CELLS;
  const half = tagSize / 2;
  const positions: number[] = [];
  const indices: number[] = [];
  let vertex = 0;
  for (let row = 0; row < TAG_CELLS; row++) {
    for (let col = 0; col < TAG_CELLS; col++) {
      if (grid[row][col] !== 1) { continue; }
      // Row 0 is the top of the printed tag, so it maps to +Y.
      const x0 = -half + col * cellSize;
      const x1 = x0 + cellSize;
      const y1 = half - row * cellSize;
      const y0 = y1 - cellSize;
      positions.push(x0, y0, 0, x1, y0, 0, x1, y1, 0, x0, y1, 0);
      indices.push(vertex, vertex + 1, vertex + 2, vertex, vertex + 2, vertex + 3);
      vertex += 4;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}
