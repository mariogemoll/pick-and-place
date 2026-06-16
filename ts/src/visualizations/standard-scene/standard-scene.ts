// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { createStandardScene } from './scene';
import { buildUi } from './ui';

export interface StandardSceneVisualization {
  destroy(): void;
}

export function initializeStandardSceneVisualization(
  parent: HTMLElement
): Promise<StandardSceneVisualization> {
  const ui = buildUi(parent);
  const vizScene = createStandardScene(ui.viewport);
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

  return vizScene.ready.then(() => ({
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      vizScene.destroy();
      ui.root.remove();
    }
  }));
}
