// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadWebModel } from '../../web-model';
import {
  type CubeFace,
  type CubePose,
  DEFAULT_CUBE_POSE } from '../pregrasp-pose-shared/body-factories';
import {
  createPregraspPoseScene,
  framePregraspPoseScene
} from '../pregrasp-pose-shared/scene';
import { displayMatrix } from '../pregrasp-pose-shared/ui';
import { createXyDragControls } from '../xy-drag-controls';
import { createSimplePregraspMatrix } from './pose';
import { buildUi } from './ui';

export interface SimplePregraspPoseVisualization {
  destroy(): void;
}

export interface SimplePregraspPoseOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

export async function initializeSimplePregraspPoseVisualization(
  parent: HTMLElement,
  options: SimplePregraspPoseOptions = {}
): Promise<SimplePregraspPoseVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const ui = buildUi(parent);
  let currentFace: CubeFace = '+x';
  let currentPose: CubePose = { ...DEFAULT_CUBE_POSE };
  const vizScene = await createPregraspPoseScene(
    ui.pane.viewport, model, options.modelBasePath, 'combined'
  );
  const gripper = vizScene.bodies.root.getObjectByName('gripper_body');
  const cube = vizScene.bodies.root.getObjectByName('cube_body');

  function updateScene(): void {
    vizScene.bodies.updateCubePose(currentPose);
    const matrix = createSimplePregraspMatrix(currentFace, currentPose);
    if (!gripper) { return; }
    ui.status.textContent = matrix === undefined
      ? 'No solution: the selected face is not vertical.'
      : 'Valid vertical pregrasp pose';
    ui.status.classList.toggle('is-invalid', matrix === undefined);
    if (matrix && ui.pane.matrixOutput) {
      gripper.matrix.copy(matrix);
      gripper.matrix.decompose(gripper.position, gripper.quaternion, gripper.scale);
      displayMatrix(ui.pane.matrixOutput, matrix);
    } else if (ui.pane.matrixOutput) {
      ui.pane.matrixOutput.textContent = 'No solution';
    }
  }

  const faceListeners = ui.faceInputs.map(input => {
    const listener = (): void => {
      if (input.checked) {
        currentFace = input.value as CubeFace;
        updateScene();
      }
    };
    input.addEventListener('change', listener);
    return listener;
  });

  const poseInputs = [
    [ui.xInput, 'x', 1 / 1000],
    [ui.yInput, 'y', 1 / 1000],
    [ui.zInput, 'z', 1 / 1000],
    [ui.yawInput, 'yaw', Math.PI / 180],
    [ui.pitchInput, 'pitch', Math.PI / 180],
    [ui.rollInput, 'roll', Math.PI / 180]
  ] as const;
  const poseListeners = poseInputs.map(([input, property, scale]) => {
    const listener = (): void => {
      currentPose = { ...currentPose, [property]: Number(input.value) * scale };
      updateScene();
    };
    input.addEventListener('input', listener);
    return listener;
  });

  const clampToInput = (input: HTMLInputElement, value: number): number =>
    Math.min(Number(input.max), Math.max(Number(input.min), value));
  const dragControls = cube
    ? createXyDragControls({
      camera: vizScene.camera,
      domElement: vizScene.renderer.domElement,
      object: cube,
      orbitControls: vizScene.orbitControls,
      onDrag(x, y): void {
        ui.xInput.value = String(Math.round(clampToInput(ui.xInput, x * 1000)));
        ui.yInput.value = String(Math.round(clampToInput(ui.yInput, y * 1000)));
        ui.xInput.dispatchEvent(new Event('input'));
        ui.yInput.dispatchEvent(new Event('input'));
      }
    })
    : undefined;

  const resetListener = (): void => {
    currentFace = '+x';
    currentPose = { ...DEFAULT_CUBE_POSE };
    for (const input of ui.faceInputs) {
      input.checked = input.value === currentFace;
    }
    const defaultValues = [
      DEFAULT_CUBE_POSE.x * 1000,
      DEFAULT_CUBE_POSE.y * 1000,
      DEFAULT_CUBE_POSE.z * 1000,
      DEFAULT_CUBE_POSE.yaw * 180 / Math.PI,
      DEFAULT_CUBE_POSE.pitch * 180 / Math.PI,
      DEFAULT_CUBE_POSE.roll * 180 / Math.PI
    ];
    for (const [index, [input]] of poseInputs.entries()) {
      input.value = String(defaultValues[index] ?? 0);
      input.dispatchEvent(new Event('input'));
    }
    framePregraspPoseScene(vizScene);
  };
  ui.resetButton.addEventListener('click', resetListener);

  updateScene();
  framePregraspPoseScene(vizScene);
  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    vizScene.orbitControls.update();
    vizScene.renderer.render(vizScene.scene, vizScene.camera);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      dragControls?.destroy();
      vizScene.destroy();
      for (const [index, input] of ui.faceInputs.entries()) {
        input.removeEventListener('change', faceListeners[index]);
      }
      for (const [index, [input]] of poseInputs.entries()) {
        input.removeEventListener('input', poseListeners[index]);
      }
      ui.resetButton.removeEventListener('click', resetListener);
      ui.root.remove();
    }
  };
}
