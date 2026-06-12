// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { loadWebModel } from '../../web-model';
import {
  applyGripperTransform,
  applyGripperTransformProgress,
  type CubeFace,
  type CubePose,
  SAFETY_MARGIN,
  type TransformStage
} from '../pregrasp-pose-shared/bodies';
import {
  createGripperFromContactMatrix,
  createWorldFromCubeContactMatrix,
  DEFAULT_CUBE_POSE
} from '../pregrasp-pose-shared/body-factories';
import { createPregraspPoseScene } from '../pregrasp-pose-shared/scene';
import { displayMatrix, FLOOR_FACES } from '../pregrasp-pose-shared/ui';
import {
  buildUi,
  displayCubeContactMatrix,
  displayHingeRotationMatrix,
  displayJawContactMatrix
} from './ui';

export interface PregraspPoseBreakdownVisualization {
  destroy(): void;
}

export interface PregraspPoseBreakdownOptions {
  animationDurationMs?: number;
  holdDurationMs?: number;
  modelBasePath?: string;
  modelUrl?: string;
}

const DEFAULT_ANIMATION_DURATION_MS = 2000;
const DEFAULT_HOLD_DURATION_MS = 3000;

export async function initializePregraspPoseBreakdownVisualization(
  parent: HTMLElement,
  options: PregraspPoseBreakdownOptions = {}
): Promise<PregraspPoseBreakdownVisualization> {
  const animationDurationMs = Math.max(
    options.animationDurationMs ?? DEFAULT_ANIMATION_DURATION_MS, 0
  );
  const holdDurationMs = Math.max(
    options.holdDurationMs ?? DEFAULT_HOLD_DURATION_MS, 0
  );
  const model = await loadWebModel(options.modelUrl);
  const ui = buildUi(parent);
  const hingePaneIndex = ui.panes.findIndex(pane => pane.hingeInput);
  const hingePane = ui.panes[hingePaneIndex];
  const hingeDegrees = Number(hingePane.hingeInput?.value ?? 0);
  let currentFace: CubeFace = '+x';
  let currentPose: CubePose = { ...DEFAULT_CUBE_POSE };
  const vizScenes = await Promise.all(ui.panes.map(pane =>
    createPregraspPoseScene(
      pane.viewport,
      model,
      options.modelBasePath,
      pane.bodySelection,
      pane.transformStage,
      (pane.transformStage === 'hinge' || pane.transformStage === 'final'
        ? hingeDegrees
        : 0) * Math.PI / 180
    )
  ));
  const combinedScene = vizScenes[2];
  let syncingCameras = false;

  function syncCamerasFrom(source: typeof combinedScene): void {
    if (syncingCameras) {
      return;
    }
    syncingCameras = true;
    for (const destination of vizScenes) {
      if (destination === source) {
        continue;
      }
      destination.camera.position.copy(source.camera.position);
      destination.camera.quaternion.copy(source.camera.quaternion);
      destination.camera.zoom = source.camera.zoom;
      destination.camera.updateProjectionMatrix();
      destination.orbitControls.target.copy(source.orbitControls.target);
      destination.orbitControls.update();
    }
    syncingCameras = false;
  }

  syncCamerasFrom(combinedScene);
  const cameraChangeListeners = vizScenes.map(vizScene => {
    const listener = (): void => {
      syncCamerasFrom(vizScene);
    };
    vizScene.orbitControls.addEventListener('change', listener);
    return listener;
  });
  const finalPaneIndex = ui.panes.findIndex(
    pane => pane.transformStage === 'final'
  );
  const finalPane = ui.panes[finalPaneIndex];
  const alignedPane = ui.panes.find(pane => pane.transformStage === 'aligned');
  const hingeGripper = vizScenes[hingePaneIndex].bodies.root
    .getObjectByName('gripper_body');
  const animatedGrippers = ui.panes.flatMap((pane, index) => {
    const gripper = vizScenes[index].bodies.root.getObjectByName('gripper_body');
    return gripper && pane.matrixOutput &&
      pane.transformStage !== 'unaligned' && pane.transformStage !== 'hinge'
      ? [{
        gripper,
        stage: pane.transformStage,
        targetHingeAngle: 0,
        face: '+x' as const,
        cubePose: { ...DEFAULT_CUBE_POSE }
      }]
      : [];
  });
  const sequenceGrippers = animatedGrippers.filter(
    animatedGripper => animatedGripper.stage !== 'final'
  );
  const finalAnimatedGripper = animatedGrippers.find(
    animatedGripper => animatedGripper.stage === 'final'
  );
  updateTargetHingeAngles(animatedGrippers, hingeDegrees * Math.PI / 180);
  for (const [index, pane] of ui.panes.entries()) {
    const gripper = vizScenes[index].bodies.root.getObjectByName('gripper_body');
    if (gripper && pane.matrixOutput) {
      if (pane.transformStage === 'jaw-contact-origin') {
        displayJawContactMatrix(
          pane.matrixOutput, createGripperFromContactMatrix().invert()
        );
      } else if (pane.transformStage === 'aligned') {
        displayCubeContactMatrix(
          pane.matrixOutput, createWorldFromCubeContactMatrix()
        );
      } else if (pane.transformStage === 'safety-margin') {
        displayMatrix(
          pane.matrixOutput,
          new THREE.Matrix4().makeTranslation(0, 0, -SAFETY_MARGIN)
        );
      } else if (pane.transformStage === 'hinge') {
        displayHingeRotationMatrix(
          pane.matrixOutput, Number(pane.hingeInput?.value ?? 0)
        );
      } else {
        displayMatrix(pane.matrixOutput, gripper.matrix);
      }
    }
  }

  function currentHingeAngle(): number {
    return Number(hingePane.hingeInput?.value ?? 0) * Math.PI / 180;
  }

  function onSceneStateChanged(): void {
    updateFaces(animatedGrippers, currentFace);
    updatePoses(animatedGrippers, currentPose);
    for (const vizScene of vizScenes) {
      vizScene.bodies.updateCubePose(currentPose);
    }
    const hingeAngle = currentHingeAngle();
    if (hingeGripper) {
      applyGripperTransform(hingeGripper, 'hinge', hingeAngle, currentFace, currentPose);
    }
    if (alignedPane?.matrixOutput) {
      displayCubeContactMatrix(
        alignedPane.matrixOutput,
        createWorldFromCubeContactMatrix(currentFace, currentPose),
        currentFace
      );
    }
    if (finalPane.matrixOutput) {
      displayTransformMatrix(
        finalPane.matrixOutput, 'final', hingeAngle, currentFace, currentPose
      );
    }
  }

  const hingeInputListener = (): void => {
    if (hingePane.hingeInput) {
      const hingeAngle = currentHingeAngle();
      updateTargetHingeAngles(animatedGrippers, hingeAngle);
      if (hingeGripper) {
        applyGripperTransform(hingeGripper, 'hinge', hingeAngle, currentFace, currentPose);
      }
      if (hingePane.matrixOutput) {
        displayHingeRotationMatrix(
          hingePane.matrixOutput, Number(hingePane.hingeInput.value)
        );
      }
      if (finalPane.matrixOutput) {
        displayTransformMatrix(
          finalPane.matrixOutput, 'final', hingeAngle, currentFace, currentPose
        );
      }
    }
  };
  hingePane.hingeInput?.addEventListener('input', hingeInputListener);

  const faceListeners = ui.faceInputs.map(input => {
    const listener = (): void => {
      if (input.checked) {
        currentFace = input.value as CubeFace;
        onSceneStateChanged();
      }
    };
    input.addEventListener('change', listener);
    return listener;
  });

  const xListener = (): void => {
    currentPose = { ...currentPose, x: Number(ui.xInput.value) / 1000 };
    onSceneStateChanged();
  };
  const yListener = (): void => {
    currentPose = { ...currentPose, y: Number(ui.yInput.value) / 1000 };
    onSceneStateChanged();
  };
  const zListener = (): void => {
    currentPose = { ...currentPose, z: Number(ui.zInput.value) / 1000 };
    onSceneStateChanged();
  };
  const yawListener = (): void => {
    currentPose = { ...currentPose, yaw: Number(ui.yawInput.value) * Math.PI / 180 };
    onSceneStateChanged();
  };
  const pitchListener = (): void => {
    currentPose = {
      ...currentPose, pitch: Number(ui.pitchInput.value) * Math.PI / 180
    };
    onSceneStateChanged();
  };
  const rollListener = (): void => {
    currentPose = { ...currentPose, roll: Number(ui.rollInput.value) * Math.PI / 180 };
    onSceneStateChanged();
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
    if (hingePane.hingeInput) {
      if (onFloor) {
        hingePane.hingeInput.min = '10';
        hingePane.hingeInput.max = '135';
        if (Number(hingePane.hingeInput.value) < 10) {
          hingePane.hingeInput.value = '10';
        }
        if (Number(hingePane.hingeInput.value) > 135) {
          hingePane.hingeInput.value = '135';
        }
      } else {
        hingePane.hingeInput.min = '0';
        hingePane.hingeInput.max = '360';
      }
      hingePane.hingeInput.dispatchEvent(new Event('input'));
    }
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
    onSceneStateChanged();
  };
  ui.floorModeInput.addEventListener('change', floorModeListener);

  let animationFrameId = 0;
  let destroyed = false;
  let animationStartTime: number | undefined;

  function animate(timestamp: number): void {
    if (destroyed) {
      return;
    }
    animationFrameId = window.requestAnimationFrame(animate);
    animationStartTime ??= timestamp;
    const sequenceDurationMs = animationDurationMs * sequenceGrippers.length;
    const rewindStartMs = sequenceDurationMs + animationDurationMs;
    const rewindEndMs = rewindStartMs + animationDurationMs;
    const cycleDurationMs = rewindEndMs + animationDurationMs;
    const elapsedMs = cycleDurationMs === 0
      ? sequenceDurationMs
      : (timestamp - animationStartTime) % cycleDurationMs;
    for (const [index, animatedGripper] of sequenceGrippers.entries()) {
      let progress;
      if (elapsedMs < rewindStartMs) {
        progress = animationDurationMs === 0
          ? 1
          : THREE.MathUtils.clamp(
            (elapsedMs - index * animationDurationMs) / animationDurationMs,
            0,
            1
          );
      } else {
        progress = animationDurationMs === 0
          ? 0
          : 1 - THREE.MathUtils.clamp(
            (elapsedMs - rewindStartMs) / animationDurationMs,
            0,
            1
          );
      }
      const easedProgress = progress * progress * (3 - 2 * progress);
      applyAnimatedGripperTransform(animatedGripper, easedProgress);
    }
    if (finalAnimatedGripper) {
      const finalElapsedMs = animationDurationMs + holdDurationMs === 0
        ? animationDurationMs
        : (timestamp - animationStartTime) %
          (animationDurationMs + holdDurationMs);
      const finalProgress = animationDurationMs === 0
        ? 1
        : Math.min(finalElapsedMs / animationDurationMs, 1);
      const easedFinalProgress =
        finalProgress * finalProgress * (3 - 2 * finalProgress);
      applyAnimatedGripperTransform(finalAnimatedGripper, easedFinalProgress);
    }

    for (const { renderer, camera, scene, orbitControls } of vizScenes) {
      orbitControls.update();
      renderer.render(scene, camera);
    }
  }

  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      for (const [index, vizScene] of vizScenes.entries()) {
        vizScene.orbitControls.removeEventListener('change', cameraChangeListeners[index]);
        vizScene.destroy();
      }
      hingePane.hingeInput?.removeEventListener('input', hingeInputListener);
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

interface AnimatedGripper {
  gripper: THREE.Object3D;
  stage: TransformStage;
  targetHingeAngle: number;
  face: CubeFace;
  cubePose: CubePose;
}

function applyAnimatedGripperTransform(
  { gripper, stage, targetHingeAngle, face, cubePose }: AnimatedGripper,
  progress: number
): void {
  applyGripperTransformProgress(gripper, stage, targetHingeAngle, progress, face, cubePose);
}

function updateTargetHingeAngles(
  animatedGrippers: AnimatedGripper[],
  hingeAngle: number
): void {
  for (const animatedGripper of animatedGrippers) {
    animatedGripper.targetHingeAngle = hingeAngle;
  }
}

function updateFaces(animatedGrippers: AnimatedGripper[], face: CubeFace): void {
  for (const animatedGripper of animatedGrippers) {
    animatedGripper.face = face;
  }
}

function updatePoses(animatedGrippers: AnimatedGripper[], pose: CubePose): void {
  for (const animatedGripper of animatedGrippers) {
    animatedGripper.cubePose = pose;
  }
}

function displayTransformMatrix(
  output: HTMLOutputElement,
  stage: TransformStage,
  hingeAngle: number,
  face: CubeFace = '+x',
  pose: CubePose = DEFAULT_CUBE_POSE
): void {
  const target = new THREE.Object3D();
  applyGripperTransform(target, stage, hingeAngle, face, pose);
  displayMatrix(output, target.matrix);
}
