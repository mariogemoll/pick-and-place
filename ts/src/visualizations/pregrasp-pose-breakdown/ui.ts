// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type * as THREE from 'three';

import type { CubeFace } from '../pregrasp-pose-shared/bodies';
import {
  appendSliderGroup,
  createPane,
  FLOOR_FACES,
  formatTranslation,
  type PregraspPosePane
} from '../pregrasp-pose-shared/ui';

export type { PregraspPosePane as PregraspPoseBreakdownPane } from '../pregrasp-pose-shared/ui';
export {
  appendSliderGroup,
  CANVAS_HEIGHT,
  CANVAS_WIDTH,
  createPane,
  displayMatrix,
  FLOOR_FACES
} from '../pregrasp-pose-shared/ui';

export interface PregraspPoseBreakdownDom {
  root: HTMLDivElement;
  panes: PregraspPosePane[];
  faceInputs: HTMLInputElement[];
  floorModeInput: HTMLInputElement;
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

export function buildUi(parent: HTMLElement): PregraspPoseBreakdownDom {
  const root = document.createElement('div');
  root.className = 'visualization pregrasp-pose-breakdown-viz-root';

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
    input.name = 'cube-face';
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

  root.appendChild(controls);

  const panes: PregraspPosePane[] = [
    createPane('Cube', 'cube'),
    createPane('Gripper', 'gripper'),
    createPane('Combined', 'combined'),
    createPane('0. Combined', 'combined', 'unaligned', true),
    createPane('1. Jaw contact to origin', 'combined', 'jaw-contact-origin', true),
    createPane('2. Origin to cube contact', 'combined', 'aligned', true),
    createPane('3. Back off 1 cm', 'combined', 'safety-margin', true),
    createPane('4. Rotate around contact normal', 'combined', 'hinge', true, true),
    createPane('0. Combined', 'combined', 'unaligned', true),
    createPane('Final combined transform', 'combined', 'final', true)
  ];
  root.append(...panes.slice(0, 3).map(pane => pane.element));

  const transformRow = document.createElement('div');
  transformRow.className = 'pregrasp-pose-breakdown-viz-transform-row';
  transformRow.append(...panes.slice(3, 8).map(pane => pane.element));
  root.appendChild(transformRow);

  const finalRow = document.createElement('div');
  finalRow.className = 'pregrasp-pose-breakdown-viz-final-row';
  finalRow.append(...panes.slice(8).map(pane => pane.element));
  root.appendChild(finalRow);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root, panes, faceInputs, floorModeInput,
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  };
}

export function displayHingeRotationMatrix(
  output: HTMLOutputElement,
  degrees: number
): void {
  const angle = `-90° - ${degrees}°`;
  output.textContent = [
    ` cos(${angle})  -sin(${angle})   0   0`,
    ` sin(${angle})   cos(${angle})   0   0`,
    '        0          0   1   0',
    '        0          0   0   1'
  ].join('\n');
}

export function displayJawContactMatrix(
  output: HTMLOutputElement,
  matrix: THREE.Matrix4
): void {
  const translation = formatTranslation(matrix);
  output.textContent = [
    ` cos(90°)                  0   sin(90°)  ${translation[0]}`,
    ` sin(180°)sin(90°)  cos(180°)  -sin(180°)cos(90°)  ${translation[1]}`,
    `-cos(180°)sin(90°)  sin(180°)   cos(180°)cos(90°)  ${translation[2]}`,
    '        0                  0          0        1'
  ].join('\n');
}

export function displayCubeContactMatrix(
  output: HTMLOutputElement,
  matrix: THREE.Matrix4,
  face: CubeFace = '+x'
): void {
  const t = formatTranslation(matrix);
  const rows = cubeFaceMatrixRows(face, t);
  output.textContent = rows.join('\n');
}

function cubeFaceMatrixRows(face: CubeFace, t: string[]): string[] {
  switch (face) {
  case '+x': return [
    ` cos(-90°)   0   sin(-90°)  ${t[0]}`,
    `         0   1           0  ${t[1]}`,
    `-sin(-90°)   0   cos(-90°)  ${t[2]}`,
    '         0   0           0        1'
  ];
  case '-x': return [
    ` cos(90°)   0   sin(90°)  ${t[0]}`,
    `        0   1          0  ${t[1]}`,
    `-sin(90°)   0   cos(90°)  ${t[2]}`,
    '        0   0          0        1'
  ];
  case '+y': return [
    `        1           0            0  ${t[0]}`,
    `        0   cos(90°)   -sin(90°)  ${t[1]}`,
    `        0   sin(90°)    cos(90°)  ${t[2]}`,
    '        0           0            0        1'
  ];
  case '-y': return [
    `        1            0             0  ${t[0]}`,
    `        0   cos(-90°)   -sin(-90°)  ${t[1]}`,
    `        0   sin(-90°)    cos(-90°)  ${t[2]}`,
    '        0            0             0        1'
  ];
  case '+z': return [
    ` cos(90°)   0   sin(90°)  ${t[0]}`,
    `        0   1          0  ${t[1]}`,
    `-sin(90°)   0   cos(90°)  ${t[2]}`,
    '        0   0          0        1'
  ];
  case '-z': return [
    ` cos(-90°)   0   sin(-90°)  ${t[0]}`,
    `          0  1           0  ${t[1]}`,
    `-sin(-90°)   0   cos(-90°)  ${t[2]}`,
    '          0  0           0        1'
  ];
  }
}
