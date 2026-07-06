// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type * as THREE from 'three';

import type { TransformStage } from './bodies';
import type { CubeFace } from './body-factories';

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 300;
export const DEGREE_SLIDER_STEPS = 360;

export const FLOOR_FACES: ReadonlySet<string> = new Set(['+x', '-x', '+y', '-y']);
export const SIDE_FACES: readonly CubeFace[] = ['+x', '-x', '+y', '-y'];

const FACE_LABELS: Record<CubeFace, string> = {
  '+x': '+X',
  '-x': '−X',
  '+y': '+Y',
  '-y': '−Y',
  '+z': '+Z',
  '-z': '−Z'
};

export interface CubePoseInputs {
  xInput: HTMLInputElement;
  yInput: HTMLInputElement;
  zInput: HTMLInputElement;
  yawInput: HTMLInputElement;
  pitchInput: HTMLInputElement;
  rollInput: HTMLInputElement;
}

export interface GraspPosePane {
  bodySelection: 'combined' | 'cube' | 'gripper';
  element: HTMLElement;
  hingeInput?: HTMLInputElement;
  matrixOutput?: HTMLOutputElement;
  transformStage: TransformStage;
  viewport: HTMLDivElement;
}

export function replacePlaceholder(parent: HTMLElement, root: HTMLElement): void {
  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }
}

export function appendResetButton(parent: HTMLElement, label = 'Reset'): HTMLButtonElement {
  const button = document.createElement('button');
  button.className = 'viz-button viz-reset-button';
  button.type = 'button';
  button.textContent = label;
  parent.appendChild(button);
  return button;
}

export function appendStatus(parent: HTMLElement): HTMLOutputElement {
  const status = document.createElement('output');
  status.className = 'viz-status';
  parent.appendChild(status);
  return status;
}

export function appendCheckbox(
  parent: HTMLElement,
  labelText: string,
  className = ''
): HTMLInputElement {
  const label = document.createElement('label');
  label.className = `viz-checkbox ${className}`.trim();
  const input = document.createElement('input');
  input.type = 'checkbox';
  const text = document.createElement('span');
  text.textContent = labelText;
  label.append(input, text);
  parent.appendChild(label);
  return input;
}

export function appendRadioGroup(
  parent: HTMLElement,
  name: string,
  groupLabel: string,
  modes: { value: string; label: string; disabled?: boolean }[]
): HTMLInputElement[] {
  const group = document.createElement('div');
  group.className = 'viz-control-group grasp-pose-breakdown-viz-controls-group';
  const label = document.createElement('span');
  label.className = 'viz-control-label';
  label.textContent = groupLabel;
  const options = document.createElement('div');
  options.className = 'viz-segmented grasp-pose-breakdown-viz-face-options';
  const inputs = modes.map((mode, index) => {
    const wrapper = document.createElement('label');
    wrapper.className = 'viz-segmented-option grasp-pose-breakdown-viz-face-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = name;
    input.value = mode.value;
    input.checked = index === 0;
    input.disabled = mode.disabled === true;
    const optionLabel = document.createElement('span');
    optionLabel.textContent = mode.label;
    wrapper.append(input, optionLabel);
    options.appendChild(wrapper);
    return input;
  });
  group.append(label, options);
  parent.appendChild(group);
  return inputs;
}

export function appendSliderGroup(
  parent: HTMLElement,
  label: string,
  min: number,
  max: number,
  value: number,
  step: number,
  unit = ' mm'
): HTMLInputElement {
  const group = document.createElement('div');
  group.className = 'viz-slider grasp-pose-breakdown-viz-controls-group';
  const labelEl = document.createElement('span');
  labelEl.className = 'viz-slider-label';
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'range';
  input.className = 'grasp-pose-breakdown-viz-control-slider';
  input.min = String(min);
  input.max = String(max);
  input.value = String(value);
  input.step = String(step);
  const outputEl = document.createElement('output');
  outputEl.className = 'viz-slider-value grasp-pose-breakdown-viz-control-output';
  outputEl.textContent = `${value}${unit}`;
  input.addEventListener('input', () => {
    outputEl.textContent = `${input.value}${unit}`;
  });
  group.append(labelEl, input, outputEl);
  parent.appendChild(group);
  return input;
}

export function appendDegreeSliderGroup(
  parent: HTMLElement,
  label: string,
  min: number,
  max: number,
  value: number
): HTMLInputElement {
  const input = appendSliderGroup(parent, label, min, max, value, 1, '°');
  setDegreeSliderRange(input, min, max);
  return input;
}

export function setDegreeSliderRange(
  input: HTMLInputElement,
  min: number,
  max: number
): void {
  input.min = String(min);
  input.max = String(max);
  input.step = String((max - min) / DEGREE_SLIDER_STEPS);
}

export function appendFloorModeInput(parent: HTMLElement): HTMLInputElement {
  const group = document.createElement('div');
  group.className = 'viz-control-group grasp-pose-breakdown-viz-controls-group';
  const input = appendCheckbox(group, 'On floor', 'grasp-pose-breakdown-viz-floor-label');
  input.checked = true;
  parent.appendChild(group);
  return input;
}

export function appendFaceInputs(
  parent: HTMLElement,
  name: string,
  faces: readonly CubeFace[] = [
    '+x', '-x', '+y', '-y', '+z', '-z'
  ],
  enabledFaces?: ReadonlySet<string>
): HTMLInputElement[] {
  return appendRadioGroup(
    parent,
    name,
    'Face',
    faces.map(face => ({
      value: face,
      label: FACE_LABELS[face],
      disabled: enabledFaces !== undefined && !enabledFaces.has(face)
    }))
  );
}

export function appendCubePoseInputs(
  parent: HTMLElement,
  onFloor = false
): CubePoseInputs {
  const xInput = appendSliderGroup(parent, 'X', -100, 100, 0, 1);
  const yInput = appendSliderGroup(parent, 'Y', -100, 100, 0, 1);
  const zInput = appendSliderGroup(parent, 'Z', 0, 300, 15, 1);
  const yawInput = appendDegreeSliderGroup(parent, 'Yaw', -180, 180, 0);
  const pitchInput = appendDegreeSliderGroup(parent, 'Pitch', -180, 180, 0);
  const rollInput = appendDegreeSliderGroup(parent, 'Roll', -180, 180, 0);
  zInput.disabled = onFloor;
  pitchInput.disabled = onFloor;
  rollInput.disabled = onFloor;
  return { xInput, yInput, zInput, yawInput, pitchInput, rollInput };
}

export function createPane(
  title: string,
  bodySelection: GraspPosePane['bodySelection'],
  transformStage: TransformStage = 'unaligned',
  withMatrix = false,
  withHingeInput = false
): GraspPosePane {
  const pane = document.createElement('section');
  pane.className = 'viz-pane grasp-pose-breakdown-viz-pane';

  const header = document.createElement('header');
  const heading = document.createElement('h3');
  heading.textContent = title;
  header.appendChild(heading);

  let hingeInput: HTMLInputElement | undefined;
  if (withHingeInput) {
    const output = document.createElement('output');
    output.className = 'grasp-pose-breakdown-viz-hinge-output';
    output.textContent = '0°';
    hingeInput = document.createElement('input');
    hingeInput.className = 'grasp-pose-breakdown-viz-hinge-input';
    hingeInput.type = 'range';
    setDegreeSliderRange(hingeInput, 0, 360);
    hingeInput.value = '0';
    hingeInput.addEventListener('input', () => {
      output.textContent = `${hingeInput?.value ?? 0}°`;
    });
    header.append(hingeInput, output);
  }
  pane.appendChild(header);

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport grasp-pose-breakdown-viz-viewport';
  viewport.style.height = `${CANVAS_HEIGHT}px`;
  pane.appendChild(viewport);

  let matrixOutput: HTMLOutputElement | undefined;
  if (withMatrix) {
    const matrix = document.createElement('div');
    matrix.className = 'grasp-pose-breakdown-viz-matrix';
    const label = document.createElement('span');
    label.textContent = matrixLabel(transformStage);
    matrixOutput = document.createElement('output');
    matrixOutput.className = 'grasp-pose-breakdown-viz-matrix-output';
    matrix.append(label, matrixOutput);
    pane.appendChild(matrix);
  }

  return {
    bodySelection,
    element: pane,
    hingeInput,
    matrixOutput,
    transformStage,
    viewport
  };
}

export function displayMatrix(
  output: HTMLOutputElement,
  matrix: THREE.Matrix4
): void {
  const values = matrix.elements;
  output.textContent = Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) =>
      formatMatrixValue(values[column * 4 + row] ?? 0)
    ).join('  ')
  ).join('\n');
}

export function formatMatrixValue(value: number): string {
  const rounded = Math.abs(value) < 5e-4 ? 0 : value;
  return rounded.toFixed(3).padStart(7);
}

export function formatTranslation(matrix: THREE.Matrix4): string[] {
  return [matrix.elements[12], matrix.elements[13], matrix.elements[14]]
    .map(value => formatMatrixValue(value));
}

function matrixLabel(stage: TransformStage): string {
  if (stage === 'jaw-contact-origin') {
    return 'contactFromGripper (translation in meters)';
  }
  if (stage === 'aligned') {
    return 'worldFromCubeContact (translation in meters)';
  }
  if (stage === 'safety-margin') {
    return 'cubeContactFromSafeContact (translation in meters)';
  }
  if (stage === 'hinge') {
    return 'safeContactFromJawContact';
  }
  if (stage === 'final') {
    return 'worldFromGripper (combined matrix)';
  }
  return 'worldFromGripper';
}
