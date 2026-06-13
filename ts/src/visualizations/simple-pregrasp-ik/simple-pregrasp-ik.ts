// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { ARM_JOINT_NAMES, deriveSo101Kinematics } from '../../ik/kinematics';
import {
  type SimpleIkBranch,
  type SimpleIkResult,
  solveSimplePregraspIk
} from '../../ik/simple-ik';
import { loadWebModel } from '../../web-model';
import {
  type CubeFace,
  type CubePose,
  DEFAULT_CUBE_POSE
} from '../pregrasp-pose-shared/body-factories';
// The pose math is the shared, DRY core: this viz and the SimplePregraspPose
// viz both derive the gripper pose from the same function.
import { createSimplePregraspMatrix } from '../simple-pregrasp-pose/pose';
import { createSimplePregraspIkScene } from './scene';
import {
  buildUi,
  DEFAULT_IK_CUBE_X,
  DEFAULT_IK_CUBE_Y
} from './ui';

export interface SimplePregraspIkVisualization {
  destroy(): void;
}

export interface SimplePregraspIkOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

type Elbow = SimpleIkBranch['elbow'];

export async function initializeSimplePregraspIkVisualization(
  parent: HTMLElement,
  options: SimplePregraspIkOptions = {}
): Promise<SimplePregraspIkVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const kinematics = deriveSo101Kinematics(model);
  const ui = buildUi(parent);
  const vizScene = createSimplePregraspIkScene(
    ui.viewport, model, options.modelBasePath
  );

  let currentFace: CubeFace = '-x';
  let currentPose: CubePose = {
    ...DEFAULT_CUBE_POSE, x: DEFAULT_IK_CUBE_X, y: DEFAULT_IK_CUBE_Y
  };
  // Persist the operator's elbow choice across pose changes.
  let preferredElbow: Elbow = 'up';
  let result: SimpleIkResult | null = null;

  function applyBranch(branch: SimpleIkBranch): void {
    for (const name of ARM_JOINT_NAMES) {
      vizScene.setJoint(name, branch.joints[name]);
    }
  }

  function restToNeutral(): void {
    for (const name of ARM_JOINT_NAMES) { vizScene.setJoint(name, 0); }
  }

  function renderBranches(branches: SimpleIkBranch[]): void {
    ui.branchContainer.replaceChildren();
    if (branches.length < 2) { return; }
    for (const branch of branches) {
      const label = document.createElement('label');
      label.className = 'simple-pregrasp-ik-viz-branch';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'simple-pregrasp-ik-branch';
      radio.value = branch.elbow;
      radio.checked = branch.elbow === preferredElbow;
      radio.addEventListener('change', () => {
        if (radio.checked) {
          preferredElbow = branch.elbow;
          updateScene();
        }
      });
      const span = document.createElement('span');
      span.textContent = branch.elbow === 'up' ? 'Elbow up' : 'Elbow down';
      label.append(radio, span);
      ui.branchContainer.appendChild(label);
    }
  }

  function updateScene(): void {
    vizScene.updateCubePose(currentPose);

    const matrix = createSimplePregraspMatrix(currentFace, currentPose);
    if (!matrix) {
      result = null;
      ui.status.textContent = 'No solution: the selected face is not vertical.';
      ui.status.classList.add('is-invalid');
      ui.branchContainer.replaceChildren();
      restToNeutral();
      return;
    }

    result = solveSimplePregraspIk(kinematics, matrix);
    if (result.type === 'unreachable') {
      ui.status.textContent = `Unreachable: ${result.reason}.`;
      ui.status.classList.add('is-invalid');
      ui.branchContainer.replaceChildren();
      restToNeutral();
      return;
    }

    const branch = result.branches.find(candidate => candidate.elbow === preferredElbow)
      ?? result.branches[0];
    ui.status.textContent =
      `Reachable (${result.branches.length === 1 ? '1 solution' : '2 solutions'}).`;
    ui.status.classList.remove('is-invalid');
    renderBranches(result.branches);
    applyBranch(branch);
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

  const resetListener = (): void => {
    currentFace = '-x';
    currentPose = {
      ...DEFAULT_CUBE_POSE, x: DEFAULT_IK_CUBE_X, y: DEFAULT_IK_CUBE_Y
    };
    preferredElbow = 'up';
    for (const input of ui.faceInputs) {
      input.checked = input.value === currentFace;
    }
    const defaults = [
      currentPose.x * 1000,
      currentPose.y * 1000,
      currentPose.z * 1000,
      currentPose.yaw * 180 / Math.PI,
      currentPose.pitch * 180 / Math.PI,
      currentPose.roll * 180 / Math.PI
    ];
    for (const [index, [input]] of poseInputs.entries()) {
      input.value = String(defaults[index] ?? 0);
      input.dispatchEvent(new Event('input'));
    }
  };
  ui.resetButton.addEventListener('click', resetListener);

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  updateScene();

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
      resizeObserver.disconnect();
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
