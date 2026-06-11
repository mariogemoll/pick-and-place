// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { createGripperScene } from './scene';
import { buildUi } from './ui';

export interface GripperVisualization {
  destroy(): void;
}

export interface GripperVisualizationOptions {
  modelBasePath?: string;
}

export function initializeGripperVisualization(
  parent: HTMLElement,
  options: GripperVisualizationOptions = {}
): Promise<GripperVisualization> {
  const ui = buildUi(parent);
  const vizScene = createGripperScene(ui.viewport, options.modelBasePath);
  const { renderer, camera, scene, orbitControls } = vizScene;

  let animationFrameId = 0;
  let destroyed = false;

  function animate(): void {
    if (destroyed) {
      return;
    }
    animationFrameId = window.requestAnimationFrame(animate);

    orbitControls.update();
    renderer.render(scene, camera);
  }

  animate();

  return Promise.resolve({
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      vizScene.destroy();
      ui.root.remove();
    }
  });
}
