// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendSliderGroup,
  createPane,
  FLOOR_FACES,
  type PregraspPosePane
} from '../pregrasp-pose-shared/ui';

export { FLOOR_FACES } from '../pregrasp-pose-shared/ui';

export interface PregraspPoseDom {
  root: HTMLDivElement;
  pane: PregraspPosePane;
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

const FACE_OPTIONS = [
  ['+x', '+X'], ['-x', '−X'], ['+y', '+Y'], ['-y', '−Y'], ['+z', '+Z'], ['-z', '−Z']
] as const;

export function buildUi(parent: HTMLElement): PregraspPoseDom {
  const root = document.createElement('div');
  root.className = 'visualization pregrasp-pose-viz-root';

  const controls = document.createElement('div');
  controls.className = 'pregrasp-pose-breakdown-viz-controls';

  const floorGroup = document.createElement('div');
  floorGroup.className = 'pregrasp-pose-breakdown-viz-controls-group';
  const floorLabel = document.createElement('label');
  floorLabel.className = 'pregrasp-pose-breakdown-viz-floor-label';
  const floorModeInput = document.createElement('input');
  floorModeInput.type = 'checkbox';
  floorModeInput.checked = true;
  const floorSpan = document.createElement('span');
  floorSpan.textContent = 'On floor';
  floorLabel.append(floorModeInput, floorSpan);
  floorGroup.appendChild(floorLabel);
  controls.appendChild(floorGroup);

  const faceGroup = document.createElement('div');
  faceGroup.className = 'pregrasp-pose-breakdown-viz-controls-group';
  const faceGroupLabel = document.createElement('span');
  faceGroupLabel.textContent = 'Face';
  const faceOptions = document.createElement('div');
  faceOptions.className = 'pregrasp-pose-breakdown-viz-face-options';
  const faceInputs: HTMLInputElement[] = [];
  for (const [value, label] of FACE_OPTIONS) {
    const wrapper = document.createElement('label');
    wrapper.className = 'pregrasp-pose-breakdown-viz-face-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'pregrasp-pose-cube-face';
    input.value = value;
    if (value === '+x') {input.checked = true;}
    if (!FLOOR_FACES.has(value)) {input.disabled = true;}
    faceInputs.push(input);
    const span = document.createElement('span');
    span.textContent = label;
    wrapper.append(input, span);
    faceOptions.appendChild(wrapper);
  }
  faceGroup.append(faceGroupLabel, faceOptions);
  controls.appendChild(faceGroup);

  const xInput = appendSliderGroup(controls, 'X', -100, 100, 0, 1);
  const yInput = appendSliderGroup(controls, 'Y', -100, 100, 0, 1);
  const zInput = appendSliderGroup(controls, 'Z', 0, 300, 15, 1);
  zInput.disabled = true;
  const yawInput = appendSliderGroup(controls, 'Yaw', -180, 180, 0, 1, '°');
  const pitchInput = appendSliderGroup(controls, 'Pitch', -180, 180, 0, 1, '°');
  pitchInput.disabled = true;
  const rollInput = appendSliderGroup(controls, 'Roll', -180, 180, 0, 1, '°');
  rollInput.disabled = true;
  const hingeInput = appendSliderGroup(controls, 'Hinge', 0, 360, 0, 1, '°');

  root.appendChild(controls);

  const pane = createPane('Pregrasp pose', 'combined', 'final', true);
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
