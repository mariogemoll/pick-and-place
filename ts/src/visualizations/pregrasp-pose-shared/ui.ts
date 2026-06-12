// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type * as THREE from 'three';

import type { TransformStage } from './bodies';

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 300;

export const FLOOR_FACES: ReadonlySet<string> = new Set(['+x', '-x', '+y', '-y']);

export interface PregraspPosePane {
  bodySelection: 'combined' | 'cube' | 'gripper';
  element: HTMLElement;
  hingeInput?: HTMLInputElement;
  matrixOutput?: HTMLOutputElement;
  transformStage: TransformStage;
  viewport: HTMLDivElement;
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
  group.className = 'pregrasp-pose-breakdown-viz-controls-group';
  const labelEl = document.createElement('span');
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'range';
  input.className = 'pregrasp-pose-breakdown-viz-control-slider';
  input.min = String(min);
  input.max = String(max);
  input.value = String(value);
  input.step = String(step);
  const outputEl = document.createElement('output');
  outputEl.className = 'pregrasp-pose-breakdown-viz-control-output';
  outputEl.textContent = `${value}${unit}`;
  input.addEventListener('input', () => {
    outputEl.textContent = `${input.value}${unit}`;
  });
  group.append(labelEl, input, outputEl);
  parent.appendChild(group);
  return input;
}

export function createPane(
  title: string,
  bodySelection: PregraspPosePane['bodySelection'],
  transformStage: TransformStage = 'unaligned',
  withMatrix = false,
  withHingeInput = false
): PregraspPosePane {
  const pane = document.createElement('section');
  pane.className = 'pregrasp-pose-breakdown-viz-pane';

  const header = document.createElement('header');
  const heading = document.createElement('h3');
  heading.textContent = title;
  header.appendChild(heading);

  let hingeInput: HTMLInputElement | undefined;
  if (withHingeInput) {
    const output = document.createElement('output');
    output.className = 'pregrasp-pose-breakdown-viz-hinge-output';
    output.textContent = '0°';
    hingeInput = document.createElement('input');
    hingeInput.className = 'pregrasp-pose-breakdown-viz-hinge-input';
    hingeInput.type = 'range';
    hingeInput.min = '0';
    hingeInput.max = '360';
    hingeInput.value = '0';
    hingeInput.addEventListener('input', () => {
      output.textContent = `${hingeInput?.value ?? 0}°`;
    });
    header.append(hingeInput, output);
  }
  pane.appendChild(header);

  const viewport = document.createElement('div');
  viewport.className = 'pregrasp-pose-breakdown-viz-viewport';
  viewport.style.height = `${CANVAS_HEIGHT}px`;
  pane.appendChild(viewport);

  let matrixOutput: HTMLOutputElement | undefined;
  if (withMatrix) {
    const matrix = document.createElement('div');
    matrix.className = 'pregrasp-pose-breakdown-viz-matrix';
    const label = document.createElement('span');
    label.textContent = matrixLabel(transformStage);
    matrixOutput = document.createElement('output');
    matrixOutput.className = 'pregrasp-pose-breakdown-viz-matrix-output';
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
