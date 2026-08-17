// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// A row of small square views of one scene, drawn through cameras that live
// inside it. All the views share a single renderer and a single canvas, each
// one scissored to its own square, so the row costs one extra context and one
// extra draw per view rather than a renderer apiece.

import type * as THREE from 'three';
import { WebGLRenderer } from 'three';

export interface CameraStrip {
  render(scene: THREE.Scene): void;
  resize(): void;
  destroy(): void;
}

// Where a view sits inside the strip, in CSS pixels with the origin at the
// canvas's bottom left, as WebGL viewports are measured.
export interface ViewRect {
  x: number;
  y: number;
  size: number;
}

/**
 * Space between one view and the next, in CSS pixels.
 *
 * The frames drawn over the strip have to sit on the same squares, so they
 * divide the host by the same rule; see `flow-policy-viz-camera-view`.
 */
export const VIEW_GAP = 6;

// Equal columns, one per view with the gaps taken out, each holding a
// top-aligned square as large as the column and the strip's height allow.
export function layoutViews(count: number, width: number, height: number): ViewRect[] {
  const column = (width - VIEW_GAP * (count - 1)) / count;
  const size = Math.max(1, Math.min(height, Math.floor(column)));
  return Array.from({ length: count }, (_, index) => ({
    x: Math.round(index * (column + VIEW_GAP)),
    y: height - size,
    size
  }));
}

export function createCameraStrip(
  host: HTMLElement,
  cameras: THREE.PerspectiveCamera[]
): CameraStrip {
  const renderer = new WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  // Each view clears its own square; the strip is cleared as a whole first, so
  // the gaps between the squares stay transparent instead of holding stale
  // pixels from the previous frame.
  renderer.autoClear = false;
  renderer.setClearColor(0x000000, 0);
  renderer.domElement.className = 'flow-policy-viz-cameras-canvas';
  host.appendChild(renderer.domElement);

  let rects: ViewRect[] = [];

  function resize(): void {
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    renderer.setSize(width, height, false);
    rects = layoutViews(cameras.length, width, height);
  }
  resize();

  return {
    render(scene: THREE.Scene): void {
      renderer.setScissorTest(false);
      renderer.clear();
      renderer.setScissorTest(true);
      for (let index = 0; index < cameras.length; index++) {
        const { x, y, size } = rects[index];
        renderer.setViewport(x, y, size, size);
        renderer.setScissor(x, y, size, size);
        renderer.render(scene, cameras[index]);
      }
      renderer.setScissorTest(false);
    },
    resize,
    destroy(): void {
      renderer.domElement.remove();
      renderer.dispose();
    }
  };
}
