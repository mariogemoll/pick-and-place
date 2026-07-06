// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type * as THREE from 'three';

import type { CubeFace } from '../grasp-pose-shared/bodies';
import {
  appendCubePoseInputs,
  appendFaceInputs,
  appendFloorModeInput,
  createPane,
  FLOOR_FACES,
  formatTranslation,
  type GraspPosePane,
  replacePlaceholder
} from '../grasp-pose-shared/ui';

export type { GraspPosePane as GraspPoseBreakdownPane } from '../grasp-pose-shared/ui';
export {
  appendSliderGroup,
  CANVAS_HEIGHT,
  CANVAS_WIDTH,
  createPane,
  displayMatrix,
  FLOOR_FACES
} from '../grasp-pose-shared/ui';

export interface GraspPoseBreakdownDom {
  root: HTMLDivElement;
  panes: GraspPosePane[];
  faceInputs: HTMLInputElement[];
  floorModeInput: HTMLInputElement;
  xInput: HTMLInputElement;
  yInput: HTMLInputElement;
  zInput: HTMLInputElement;
  yawInput: HTMLInputElement;
  pitchInput: HTMLInputElement;
  rollInput: HTMLInputElement;
}

export function buildUi(parent: HTMLElement): GraspPoseBreakdownDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell grasp-pose-breakdown-viz-root';

  const controls = document.createElement('div');
  controls.className = 'viz-top-controls grasp-pose-breakdown-viz-controls';

  const floorModeInput = appendFloorModeInput(controls);
  const faceInputs = appendFaceInputs(
    controls, 'grasp-pose-breakdown-cube-face', undefined, FLOOR_FACES
  );
  const {
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  } = appendCubePoseInputs(controls, true);

  root.appendChild(controls);

  const panes: GraspPosePane[] = [
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
  transformRow.className = 'grasp-pose-breakdown-viz-transform-row';
  transformRow.append(...panes.slice(3, 8).map(pane => pane.element));
  root.appendChild(transformRow);

  const finalRow = document.createElement('div');
  finalRow.className = 'grasp-pose-breakdown-viz-final-row';
  finalRow.append(...panes.slice(8).map(pane => pane.element));
  root.appendChild(finalRow);

  replacePlaceholder(parent, root);

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
