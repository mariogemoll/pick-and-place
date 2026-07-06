// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendCubePoseInputs,
  appendFaceInputs,
  appendResetButton,
  appendStatus,
  createPane,
  type CubePoseInputs,
  type GraspPosePane,
  replacePlaceholder,
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
  root.className = 'visualization viz-shell grasp-pose-viz-root';

  const controls = document.createElement('div');
  controls.className = 'viz-top-controls grasp-pose-breakdown-viz-controls';
  const faceInputs = appendFaceInputs(
    controls, 'simple-grasp-pose-cube-face', SIDE_FACES
  );
  const cubePoseInputs = appendCubePoseInputs(controls);
  const resetButton = appendResetButton(controls);
  resetButton.classList.add('simple-grasp-pose-viz-reset');
  root.appendChild(controls);

  const status = appendStatus(root);
  status.classList.add('simple-grasp-pose-viz-status');
  root.appendChild(status);

  const pane = createPane('Simple grasp pose', 'combined', 'final', true);
  root.appendChild(pane.element);

  replacePlaceholder(parent, root);

  return { root, pane, faceInputs, resetButton, status, ...cubePoseInputs };
}
