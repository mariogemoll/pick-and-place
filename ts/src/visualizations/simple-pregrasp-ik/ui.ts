// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendDegreeSliderGroup,
  appendFaceInputs,
  appendSliderGroup,
  type CubePoseInputs,
  SIDE_FACES
} from '../pregrasp-pose-shared/ui';

export interface SimplePregraspIkDom extends CubePoseInputs {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  faceInputs: HTMLInputElement[];
  resetButton: HTMLButtonElement;
  status: HTMLOutputElement;
  branchContainer: HTMLDivElement;
}

// The default cube X/Y (metres) used on reset; comfortably reachable on the
// robot-facing −x face.
export const DEFAULT_IK_CUBE_X = 0.2;
export const DEFAULT_IK_CUBE_Y = 0;

export function buildUi(parent: HTMLElement): SimplePregraspIkDom {
  const root = document.createElement('div');
  root.className = 'visualization simple-pregrasp-ik-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'simple-pregrasp-ik-viz-viewport';

  const controls = document.createElement('div');
  controls.className = 'simple-pregrasp-ik-viz-controls';

  const faceInputs = appendFaceInputs(
    controls, 'simple-pregrasp-ik-cube-face', SIDE_FACES
  );
  // Cube is on the robot-facing −x face by default.
  const facePreset = faceInputs.find(input => input.value === '-x');
  if (facePreset) {
    for (const input of faceInputs) { input.checked = input === facePreset; }
  }

  // Wider X/Y ranges than the SimplePregraspPose viz so the cube can be placed
  // within the arm's reach.
  const xInput = appendSliderGroup(controls, 'X', 50, 500, DEFAULT_IK_CUBE_X * 1000, 1);
  const yInput = appendSliderGroup(controls, 'Y', -250, 250, DEFAULT_IK_CUBE_Y * 1000, 1);
  const zInput = appendSliderGroup(controls, 'Z', 0, 300, 15, 1);
  const yawInput = appendDegreeSliderGroup(controls, 'Yaw', -180, 180, 0);
  const pitchInput = appendDegreeSliderGroup(controls, 'Pitch', -180, 180, 0);
  const rollInput = appendDegreeSliderGroup(controls, 'Roll', -180, 180, 0);

  const resetButton = document.createElement('button');
  resetButton.className = 'simple-pregrasp-ik-viz-reset';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  controls.appendChild(resetButton);

  const status = document.createElement('output');
  status.className = 'simple-pregrasp-ik-viz-status';
  controls.appendChild(status);

  const branchContainer = document.createElement('div');
  branchContainer.className = 'simple-pregrasp-ik-viz-branches';
  controls.appendChild(branchContainer);

  const layout = document.createElement('div');
  layout.className = 'simple-pregrasp-ik-viz-layout';
  layout.append(viewport, controls);
  root.appendChild(layout);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root, viewport, faceInputs, resetButton, status, branchContainer,
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  };
}
