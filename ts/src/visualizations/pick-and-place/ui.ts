// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendDegreeSliderGroup,
  appendSliderGroup
} from '../pregrasp-pose-shared/ui';

export interface PickAndPlaceCubeInputs {
  xInput: HTMLInputElement;
  yInput: HTMLInputElement;
  yawInput: HTMLInputElement;
}

export interface PickAndPlaceDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  sourceInputs: PickAndPlaceCubeInputs;
  targetInputs: PickAndPlaceCubeInputs;
  resetButton: HTMLButtonElement;
}

export interface PickAndPlaceUiOptions {
  xRange: { min: number; max: number };
  yRange: { min: number; max: number };
  source: { x: number; y: number; yaw: number };
  target: { x: number; y: number; yaw: number };
}

function appendCubeInputs(
  parent: HTMLElement,
  className: string,
  labelText: string,
  options: PickAndPlaceUiOptions,
  pose: PickAndPlaceUiOptions['source']
): PickAndPlaceCubeInputs {
  const heading = document.createElement('div');
  heading.className = 'pick-and-place-viz-cube-heading';

  const swatch = document.createElement('span');
  swatch.className = `pick-and-place-viz-swatch ${className}`;

  const label = document.createElement('strong');
  label.textContent = labelText;
  heading.append(swatch, label);
  parent.appendChild(heading);

  const xInput = appendSliderGroup(
    parent, 'X', options.xRange.min, options.xRange.max, pose.x, 1
  );
  const yInput = appendSliderGroup(
    parent, 'Y', options.yRange.min, options.yRange.max, pose.y, 1
  );
  const yawInput = appendDegreeSliderGroup(parent, 'Yaw', -180, 180, pose.yaw);
  return { xInput, yInput, yawInput };
}

export function buildUi(
  parent: HTMLElement,
  options: PickAndPlaceUiOptions
): PickAndPlaceDom {
  const root = document.createElement('div');
  root.className = 'visualization simple-pregrasp-ik-viz-root pick-and-place-viz-root';

  const viewport = document.createElement('div');
  viewport.className =
    'simple-pregrasp-ik-viz-viewport pick-and-place-viz-viewport';

  const controls = document.createElement('div');
  controls.className = 'simple-pregrasp-ik-viz-controls';
  const sourceInputs = appendCubeInputs(
    controls, 'source', 'Source cube', options, options.source
  );
  const targetInputs = appendCubeInputs(
    controls, 'target', 'Target cube', options, options.target
  );

  const resetButton = document.createElement('button');
  resetButton.className = 'simple-pregrasp-ik-viz-reset';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  controls.appendChild(resetButton);

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

  return { root, viewport, sourceInputs, targetInputs, resetButton };
}
