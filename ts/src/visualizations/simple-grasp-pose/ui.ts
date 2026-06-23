// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendCubePoseInputs,
  appendFaceInputs,
  createPane,
  type CubePoseInputs,
  type GraspPosePane,
  SIDE_FACES } from '../grasp-pose-shared/ui';

export interface SimpleGraspPoseDom extends CubePoseInputs {
  root: HTMLDivElement;
  pane: GraspPosePane;
  faceInputs: HTMLInputElement[];
  resetButton: HTMLButtonElement;
  status: HTMLOutputElement;
}

export function buildUi(parent: HTMLElement): SimpleGraspPoseDom {
  const root = document.createElement('div');
  root.className = 'visualization grasp-pose-viz-root';

  const controls = document.createElement('div');
  controls.className = 'grasp-pose-breakdown-viz-controls';
  const faceInputs = appendFaceInputs(
    controls, 'simple-grasp-pose-cube-face', SIDE_FACES
  );
  const cubePoseInputs = appendCubePoseInputs(controls);
  const resetButton = document.createElement('button');
  resetButton.className = 'simple-grasp-pose-viz-reset';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  controls.appendChild(resetButton);
  root.appendChild(controls);

  const status = document.createElement('output');
  status.className = 'simple-grasp-pose-viz-status';
  root.appendChild(status);

  const pane = createPane('Simple grasp pose', 'combined', 'final', true);
  root.appendChild(pane.element);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return { root, pane, faceInputs, resetButton, status, ...cubePoseInputs };
}
