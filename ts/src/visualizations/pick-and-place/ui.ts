// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendDegreeSliderGroup,
  appendSliderGroup
} from '../grasp-pose-shared/ui';

export interface PickAndPlaceCubeInputs {
  xInput: HTMLInputElement;
  yInput: HTMLInputElement;
  yawInput: HTMLInputElement;
}

export interface PickAndPlaceDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  controls: HTMLElement;
  setupControls: HTMLDivElement;
  setupActions: HTMLDivElement;
  runControls: HTMLDivElement;
  sourceInputs: PickAndPlaceCubeInputs;
  targetInputs: PickAndPlaceCubeInputs;
  resetButton: HTMLButtonElement;
  runButton: HTMLButtonElement;
  playPauseButton: HTMLButtonElement;
  cancelButton: HTMLButtonElement;
  seekInput: HTMLInputElement;
  playbackTime: HTMLOutputElement;
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
  root.className = 'visualization simple-grasp-ik-viz-root pick-and-place-viz-root';

  const viewport = document.createElement('div');
  viewport.className =
    'simple-grasp-ik-viz-viewport pick-and-place-viz-viewport';

  const controls = document.createElement('aside');
  controls.className = 'pick-and-place-viz-controls';
  const setupControls = document.createElement('div');
  setupControls.className = 'pick-and-place-viz-setup-controls';
  const sourceControls = document.createElement('section');
  sourceControls.className = 'pick-and-place-viz-setup-panel';
  const targetControls = document.createElement('section');
  targetControls.className = 'pick-and-place-viz-setup-panel';
  const sourceInputs = appendCubeInputs(
    sourceControls, 'source', 'Source cube', options, options.source
  );
  const targetInputs = appendCubeInputs(
    targetControls, 'target', 'Target cube', options, options.target
  );

  const resetButton = document.createElement('button');
  resetButton.className = 'simple-grasp-ik-viz-reset';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  const runButton = document.createElement('button');
  runButton.className = 'pick-and-place-viz-run-button';
  runButton.type = 'button';
  runButton.textContent = 'Run';
  const setupActions = document.createElement('div');
  setupActions.className = 'pick-and-place-viz-setup-actions';
  setupActions.appendChild(runButton);
  setupControls.append(sourceControls, targetControls, resetButton);

  const runControls = document.createElement('div');
  runControls.className = 'pick-and-place-viz-run-controls';
  runControls.hidden = true;
  const cancelButton = document.createElement('button');
  cancelButton.className = 'pick-and-place-viz-cancel-button';
  cancelButton.type = 'button';
  cancelButton.textContent = 'X';
  cancelButton.setAttribute('aria-label', 'Cancel trajectory playback');
  const playbackRow = document.createElement('div');
  playbackRow.className = 'pick-and-place-viz-playback-row';
  const playPauseButton = document.createElement('button');
  playPauseButton.className = 'pick-and-place-viz-play-button';
  playPauseButton.type = 'button';
  playPauseButton.textContent = 'Play';
  playPauseButton.setAttribute('aria-label', 'Play trajectory');
  const playbackTime = document.createElement('output');
  playbackTime.className = 'pick-and-place-viz-playback-time';
  playbackTime.textContent = '0:00.0 / 0:03.0';
  const seekInput = document.createElement('input');
  seekInput.className = 'pick-and-place-viz-seek';
  seekInput.type = 'range';
  seekInput.min = '0';
  seekInput.max = '3';
  seekInput.step = '0.01';
  seekInput.value = '0';
  seekInput.setAttribute('aria-label', 'Trajectory playback position');
  playbackRow.append(playPauseButton, playbackTime, cancelButton);
  runControls.append(seekInput, playbackRow);
  controls.appendChild(setupControls);
  viewport.append(controls, setupActions, runControls);

  const layout = document.createElement('div');
  layout.className = 'simple-grasp-ik-viz-layout';
  layout.appendChild(viewport);
  root.appendChild(layout);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root,
    viewport,
    controls,
    setupControls,
    setupActions,
    runControls,
    sourceInputs,
    targetInputs,
    resetButton,
    runButton,
    playPauseButton,
    cancelButton,
    seekInput,
    playbackTime
  };
}
