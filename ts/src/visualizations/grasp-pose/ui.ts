// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendCubePoseInputs,
  appendDegreeSliderGroup,
  appendFaceInputs,
  appendFloorModeInput,
  createPane,
  FLOOR_FACES,
  type GraspPosePane
} from '../grasp-pose-shared/ui';

export { FLOOR_FACES } from '../grasp-pose-shared/ui';

export interface GraspPoseDom {
  root: HTMLDivElement;
  pane: GraspPosePane;
  faceInputs: HTMLInputElement[];
  floorModeInput: HTMLInputElement;
  hingeInput: HTMLInputElement;
  xInput: HTMLInputElement;
  yInput: HTMLInputElement;
  zInput: HTMLInputElement;
  yawInput: HTMLInputElement;
  pitchInput: HTMLInputElement;
  rollInput: HTMLInputElement;
}

export function buildUi(parent: HTMLElement): GraspPoseDom {
  const root = document.createElement('div');
  root.className = 'visualization grasp-pose-viz-root';

  const controls = document.createElement('div');
  controls.className = 'grasp-pose-breakdown-viz-controls';

  const floorModeInput = appendFloorModeInput(controls);
  const faceInputs = appendFaceInputs(
    controls, 'grasp-pose-cube-face', undefined, FLOOR_FACES
  );
  const {
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  } = appendCubePoseInputs(controls, true);
  const hingeInput = appendDegreeSliderGroup(controls, 'Hinge', 0, 360, 0);

  root.appendChild(controls);

  const pane = createPane('Grasp pose', 'combined', 'final', true);
  root.appendChild(pane.element);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root, pane, faceInputs, floorModeInput, hingeInput,
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  };
}
