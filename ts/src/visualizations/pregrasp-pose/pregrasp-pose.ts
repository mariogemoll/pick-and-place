// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadWebModel } from '../../web-model';
import {
  applyGripperTransform,
  type CubeFace,
  type CubePose
} from '../pregrasp-pose-shared/bodies';
import { DEFAULT_CUBE_POSE } from '../pregrasp-pose-shared/body-factories';
import { createPregraspPoseScene } from '../pregrasp-pose-shared/scene';
import { displayMatrix } from '../pregrasp-pose-shared/ui';
import { buildUi, FLOOR_FACES } from './ui';

export interface PregraspPoseVisualization {
  destroy(): void;
}

export interface PregraspPoseOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

export async function initializePregraspPoseVisualization(
  parent: HTMLElement,
  options: PregraspPoseOptions = {}
): Promise<PregraspPoseVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const ui = buildUi(parent);
  let currentFace: CubeFace = '+x';
  let currentPose: CubePose = { ...DEFAULT_CUBE_POSE };

  const currentHingeAngle = (): number =>
    Number(ui.hingeInput.value) * Math.PI / 180;

  const vizScene = await createPregraspPoseScene(
    ui.pane.viewport,
    model,
    options.modelBasePath,
    'combined',
    'final',
    currentHingeAngle()
  );

  const gripper = vizScene.bodies.root.getObjectByName('gripper_body');

  function updateScene(): void {
    vizScene.bodies.updateCubePose(currentPose);
    if (gripper) {
      applyGripperTransform(gripper, 'final', currentHingeAngle(), currentFace, currentPose);
      if (ui.pane.matrixOutput) {
        displayMatrix(ui.pane.matrixOutput, gripper.matrix);
      }
    }
  }

  updateScene();

  const hingeListener = (): void => { updateScene(); };
  ui.hingeInput.addEventListener('input', hingeListener);

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

  const xListener = (): void => {
    currentPose = { ...currentPose, x: Number(ui.xInput.value) / 1000 };
    updateScene();
  };
  const yListener = (): void => {
    currentPose = { ...currentPose, y: Number(ui.yInput.value) / 1000 };
    updateScene();
  };
  const zListener = (): void => {
    currentPose = { ...currentPose, z: Number(ui.zInput.value) / 1000 };
    updateScene();
  };
  const yawListener = (): void => {
    currentPose = { ...currentPose, yaw: Number(ui.yawInput.value) * Math.PI / 180 };
    updateScene();
  };
  const pitchListener = (): void => {
    currentPose = { ...currentPose, pitch: Number(ui.pitchInput.value) * Math.PI / 180 };
    updateScene();
  };
  const rollListener = (): void => {
    currentPose = { ...currentPose, roll: Number(ui.rollInput.value) * Math.PI / 180 };
    updateScene();
  };
  ui.xInput.addEventListener('input', xListener);
  ui.yInput.addEventListener('input', yListener);
  ui.zInput.addEventListener('input', zListener);
  ui.yawInput.addEventListener('input', yawListener);
  ui.pitchInput.addEventListener('input', pitchListener);
  ui.rollInput.addEventListener('input', rollListener);

  const floorModeListener = (): void => {
    const onFloor = ui.floorModeInput.checked;
    for (const input of ui.faceInputs) {
      input.disabled = onFloor && !FLOOR_FACES.has(input.value);
    }
    ui.zInput.disabled = onFloor;
    ui.pitchInput.disabled = onFloor;
    ui.rollInput.disabled = onFloor;
    if (onFloor) {
      ui.hingeInput.min = '10';
      ui.hingeInput.max = '135';
      if (Number(ui.hingeInput.value) < 10) {ui.hingeInput.value = '10';}
      if (Number(ui.hingeInput.value) > 135) {ui.hingeInput.value = '135';}
    } else {
      ui.hingeInput.min = '0';
      ui.hingeInput.max = '360';
    }
    ui.hingeInput.dispatchEvent(new Event('input'));
    if (onFloor) {
      if (!FLOOR_FACES.has(currentFace)) {
        const fallback = ui.faceInputs.find(i => i.value === '+x');
        if (fallback) {
          fallback.checked = true;
          currentFace = '+x';
        }
      }
      ui.zInput.value = String(DEFAULT_CUBE_POSE.z * 1000);
      ui.pitchInput.value = '0';
      ui.rollInput.value = '0';
      currentPose = { ...currentPose, z: DEFAULT_CUBE_POSE.z, pitch: 0, roll: 0 };
    }
    updateScene();
  };
  ui.floorModeInput.addEventListener('change', floorModeListener);

  let animationFrameId = 0;
  let destroyed = false;

  function animate(): void {
    if (destroyed) {return;}
    animationFrameId = window.requestAnimationFrame(animate);
    vizScene.orbitControls.update();
    vizScene.renderer.render(vizScene.scene, vizScene.camera);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      vizScene.destroy();
      ui.hingeInput.removeEventListener('input', hingeListener);
      for (const [index, input] of ui.faceInputs.entries()) {
        input.removeEventListener('change', faceListeners[index]);
      }
      ui.xInput.removeEventListener('input', xListener);
      ui.yInput.removeEventListener('input', yListener);
      ui.zInput.removeEventListener('input', zListener);
      ui.yawInput.removeEventListener('input', yawListener);
      ui.pitchInput.removeEventListener('input', pitchListener);
      ui.rollInput.removeEventListener('input', rollListener);
      ui.floorModeInput.removeEventListener('change', floorModeListener);
      ui.root.remove();
    }
  };
}
