// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendDegreeSliderGroup,
  appendFaceInputs,
  appendSliderGroup,
  type CubePoseInputs,
  SIDE_FACES
} from '../grasp-pose-shared/ui';

export type CoordinateMode = 'cartesian' | 'radial';

export interface SimpleGraspIkDom extends CubePoseInputs {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  faceInputs: HTMLInputElement[];
  // Cube placement can be driven in Cartesian (X/Y) or radial (radius/azimuth
  // about the pan axis) coordinates; the toggle swaps which pair is shown.
  coordModeInputs: HTMLInputElement[];
  cartesianGroup: HTMLDivElement;
  radialGroup: HTMLDivElement;
  radiusInput: HTMLInputElement;
  azimuthInput: HTMLInputElement;
  resetButton: HTMLButtonElement;
  status: HTMLOutputElement;
  branchContainer: HTMLDivElement;
}

// The default cube X/Y (metres) used on reset; comfortably reachable on the
// robot-facing −x face.
export const DEFAULT_IK_CUBE_X = 0.2;
export const DEFAULT_IK_CUBE_Y = 0;

export interface SliderRange { min: number; max: number; }

export interface SimpleGraspIkUiOptions {
  // X/Y slider ranges in millimetres. Default to the previous wide ranges.
  xRange?: SliderRange;
  yRange?: SliderRange;
  // Radius (mm) and azimuth (degrees) ranges + initial values for the radial
  // controls. Azimuth is measured from the pan axis.
  radiusRange?: SliderRange;
  azimuthRange?: SliderRange;
  radiusDefault?: number;
  azimuthDefault?: number;
}

function appendCoordinateModeInputs(parent: HTMLElement): HTMLInputElement[] {
  const group = document.createElement('div');
  group.className = 'grasp-pose-breakdown-viz-controls-group';
  const groupLabel = document.createElement('span');
  groupLabel.textContent = 'Coordinates';
  const options = document.createElement('div');
  options.className = 'grasp-pose-breakdown-viz-face-options';
  const modes: { value: CoordinateMode; label: string }[] = [
    { value: 'cartesian', label: 'X / Y' },
    { value: 'radial', label: 'Radial' }
  ];
  const inputs = modes.map((mode, index) => {
    const wrapper = document.createElement('label');
    wrapper.className = 'grasp-pose-breakdown-viz-face-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'simple-grasp-ik-coord-mode';
    input.value = mode.value;
    input.checked = index === 0;
    const label = document.createElement('span');
    label.textContent = mode.label;
    wrapper.append(input, label);
    options.appendChild(wrapper);
    return input;
  });
  group.append(groupLabel, options);
  parent.appendChild(group);
  return inputs;
}

export function buildUi(
  parent: HTMLElement,
  options: SimpleGraspIkUiOptions = {}
): SimpleGraspIkDom {
  const xRange = options.xRange ?? { min: 50, max: 500 };
  const yRange = options.yRange ?? { min: -250, max: 250 };
  const radiusRange = options.radiusRange ?? { min: 50, max: 300 };
  const azimuthRange = options.azimuthRange ?? { min: -110, max: 110 };
  const radiusDefault = options.radiusDefault ?? DEFAULT_IK_CUBE_X * 1000;
  const azimuthDefault = options.azimuthDefault ?? 0;
  const root = document.createElement('div');
  root.className = 'visualization simple-grasp-ik-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'simple-grasp-ik-viz-viewport';

  const controls = document.createElement('div');
  controls.className = 'simple-grasp-ik-viz-controls';

  const faceInputs = appendFaceInputs(
    controls, 'simple-grasp-ik-cube-face', SIDE_FACES
  );
  // Cube is on the robot-facing −x face by default.
  const facePreset = faceInputs.find(input => input.value === '-x');
  if (facePreset) {
    for (const input of faceInputs) { input.checked = input === facePreset; }
  }

  const coordModeInputs = appendCoordinateModeInputs(controls);

  // X/Y and radial ranges both bound the cube to the usable workspace (the
  // any-yaw graspable sector); see src/ik/workspace.ts. Only one group is shown
  // at a time; the controller keeps them in sync.
  const cartesianGroup = document.createElement('div');
  const xInput = appendSliderGroup(
    cartesianGroup, 'X', xRange.min, xRange.max, DEFAULT_IK_CUBE_X * 1000, 1
  );
  const yInput = appendSliderGroup(
    cartesianGroup, 'Y', yRange.min, yRange.max, DEFAULT_IK_CUBE_Y * 1000, 1
  );
  controls.appendChild(cartesianGroup);

  const radialGroup = document.createElement('div');
  const radiusInput = appendSliderGroup(
    radialGroup, 'Radius', radiusRange.min, radiusRange.max, radiusDefault, 1
  );
  const azimuthInput = appendDegreeSliderGroup(
    radialGroup, 'Azimuth', azimuthRange.min, azimuthRange.max, azimuthDefault
  );
  radialGroup.style.display = 'none';
  controls.appendChild(radialGroup);

  const zInput = appendSliderGroup(controls, 'Z', 0, 300, 15, 1);
  const yawInput = appendDegreeSliderGroup(controls, 'Yaw', -180, 180, 0);
  const pitchInput = appendDegreeSliderGroup(controls, 'Pitch', -180, 180, 0);
  const rollInput = appendDegreeSliderGroup(controls, 'Roll', -180, 180, 0);

  const resetButton = document.createElement('button');
  resetButton.className = 'simple-grasp-ik-viz-reset';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  controls.appendChild(resetButton);

  const status = document.createElement('output');
  status.className = 'simple-grasp-ik-viz-status';
  controls.appendChild(status);

  const branchContainer = document.createElement('div');
  branchContainer.className = 'simple-grasp-ik-viz-branches';
  controls.appendChild(branchContainer);

  const layout = document.createElement('div');
  layout.className = 'simple-grasp-ik-viz-layout';
  layout.append(viewport, controls);
  root.appendChild(layout);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root, viewport, faceInputs, resetButton, status, branchContainer,
    coordModeInputs, cartesianGroup, radialGroup, radiusInput, azimuthInput,
    xInput, yInput, zInput, yawInput, pitchInput, rollInput
  };
}
