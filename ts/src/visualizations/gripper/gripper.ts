// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadWebModel } from '../../web-model';
import { createGripperScene } from './scene';
import { buildUi } from './ui';

export interface GripperVisualization {
  destroy(): void;
}

export interface GripperVisualizationOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

export async function initializeGripperVisualization(
  parent: HTMLElement,
  options: GripperVisualizationOptions = {}
): Promise<GripperVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const ui = buildUi(parent);
  const vizScene = createGripperScene(ui.viewport, model, options.modelBasePath);
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
