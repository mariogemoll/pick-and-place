// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { buildUi } from './ui';
import { createDummyScene } from './scene';

export interface DummyVisualization {
  destroy(): void;
}

export async function initializeDummyVisualization(
  parent: HTMLElement
): Promise<DummyVisualization> {
  const ui = buildUi(parent);
  const vizScene = createDummyScene(ui.viewport);
  const { renderer, camera, scene, orbitControls, cube } = vizScene;

  let animationFrameId = 0;
  let destroyed = false;

  function animate(): void {
    if (destroyed) {
      return;
    }
    animationFrameId = window.requestAnimationFrame(animate);

    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;

    orbitControls.update();
    renderer.render(scene, camera);
  }

  animate();

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      vizScene.destroy();
      ui.root.remove();
    }
  };
}
