// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadWebModel } from '../../web-model';
import { createRobotScene } from './scene';
import { buildUi, formatDegrees } from './ui';

export interface RobotVisualization {
  destroy(): void;
}

export interface RobotVisualizationOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

export async function initializeRobotVisualization(
  parent: HTMLElement,
  options: RobotVisualizationOptions = {}
): Promise<RobotVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const joints = model.bodies.flatMap(body => body.joints).flatMap(joint => {
    if (joint.type !== 'hinge' || joint.range === undefined) { return []; }
    return [{
      name: joint.name,
      label: joint.name.replaceAll('_', ' '),
      lower: joint.range[0],
      upper: joint.range[1],
      value: 0
    }];
  });
  const ui = buildUi(parent, joints);
  const vizScene = createRobotScene(ui.viewport, model, options.modelBasePath);
  const { renderer, camera, scene, orbitControls } = vizScene;
  const listeners: (() => void)[] = [];

  for (const joint of joints) {
    const control = ui.controls.get(joint.name);
    if (!control) {continue;}
    const update = (): void => {
      const value = Number(control.input.value);
      control.value.textContent = formatDegrees(value);
      vizScene.setJoint(joint.name, value);
    };
    control.input.addEventListener('input', update);
    listeners.push(() => { control.input.removeEventListener('input', update); });
  }

  const reset = (): void => {
    for (const joint of joints) {
      const control = ui.controls.get(joint.name);
      if (!control) {continue;}
      control.input.value = String(joint.value);
      control.input.dispatchEvent(new Event('input'));
    }
  };
  ui.resetButton.addEventListener('click', reset);
  listeners.push(() => { ui.resetButton.removeEventListener('click', reset); });

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) {return;}
    animationFrameId = window.requestAnimationFrame(animate);
    orbitControls.update();
    renderer.render(scene, camera);
  }
  animate();

  return Promise.resolve({
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      for (const removeListener of listeners) {removeListener();}
      vizScene.destroy();
      ui.root.remove();
    }
  });
}
