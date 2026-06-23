// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendDegreeSliderGroup,
  appendSliderGroup
} from '../grasp-pose-shared/ui';

export type CoordinateMode = 'cartesian' | 'radial';

export interface CanonicalGraspDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  // Cube placement can be driven in Cartesian (X/Y) or radial (radius/azimuth
  // about the pan axis) coordinates; the toggle swaps which pair is shown.
  coordModeInputs: HTMLInputElement[];
  cartesianGroup: HTMLDivElement;
  radialGroup: HTMLDivElement;
  xInput: HTMLInputElement;
  yInput: HTMLInputElement;
  radiusInput: HTMLInputElement;
  azimuthInput: HTMLInputElement;
  yawInput: HTMLInputElement;
  showPregraspInput: HTMLInputElement;
  resetButton: HTMLButtonElement;
  status: HTMLOutputElement;
  branchContainer: HTMLDivElement;
}

// The default cube X/Y (metres) used on reset; comfortably reachable in front
// of the robot.
export const DEFAULT_CUBE_X = 0.2;
export const DEFAULT_CUBE_Y = 0;

// Yaw is measured from the radial direction (not the world frame), so the grasp
// geometry is the same at every azimuth. A cube has 4-fold rotational symmetry
// about its vertical axis, so two yaws differing by a multiple of 90° present an
// identical cube; the slider therefore spans one representative quarter-turn.
// The lower boundary is excluded because -45° aliases to +45° and makes the
// canonical grasp appear to jump at the slider edge.
export const YAW_MIN_DEG = -44;
export const YAW_MAX_DEG = 45;

export interface SliderRange { min: number; max: number; }

export interface CanonicalGraspUiOptions {
  // X/Y slider ranges in millimetres.
  xRange?: SliderRange;
  yRange?: SliderRange;
  // Radius (mm) and azimuth (degrees) ranges + initial values for the radial
  // controls. Azimuth is measured from the pan axis.
  radiusRange?: SliderRange;
  azimuthRange?: SliderRange;
  radiusDefault?: number;
  azimuthDefault?: number;
}

function appendRadioGroup(
  parent: HTMLElement,
  name: string,
  groupLabel: string,
  modes: { value: string; label: string }[]
): HTMLInputElement[] {
  const group = document.createElement('div');
  group.className = 'grasp-pose-breakdown-viz-controls-group';
  const label = document.createElement('span');
  label.textContent = groupLabel;
  const options = document.createElement('div');
  options.className = 'grasp-pose-breakdown-viz-face-options';
  const inputs = modes.map((mode, index) => {
    const wrapper = document.createElement('label');
    wrapper.className = 'grasp-pose-breakdown-viz-face-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = name;
    input.value = mode.value;
    input.checked = index === 0;
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

export function buildUi(
  parent: HTMLElement,
  options: CanonicalGraspUiOptions = {}
): CanonicalGraspDom {
  const xRange = options.xRange ?? { min: 50, max: 500 };
  const yRange = options.yRange ?? { min: -250, max: 250 };
  const radiusRange = options.radiusRange ?? { min: 50, max: 300 };
  const azimuthRange = options.azimuthRange ?? { min: -110, max: 110 };
  const radiusDefault = options.radiusDefault ?? DEFAULT_CUBE_X * 1000;
  const azimuthDefault = options.azimuthDefault ?? 0;

  const root = document.createElement('div');
  root.className = 'visualization canonical-grasp-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'canonical-grasp-viz-viewport';

  const controls = document.createElement('div');
  controls.className = 'canonical-grasp-viz-controls';

  const coordModeInputs = appendRadioGroup(
    controls, 'canonical-grasp-coord-mode', 'Coordinates',
    [{ value: 'cartesian', label: 'X / Y' }, { value: 'radial', label: 'Radial' }]
  );

  // X/Y and radial groups both drive the cube center; only one is shown at a
  // time and the controller keeps them in sync.
  const cartesianGroup = document.createElement('div');
  const xInput = appendSliderGroup(
    cartesianGroup, 'X', xRange.min, xRange.max, DEFAULT_CUBE_X * 1000, 1
  );
  const yInput = appendSliderGroup(
    cartesianGroup, 'Y', yRange.min, yRange.max, DEFAULT_CUBE_Y * 1000, 1
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

  // Yaw relative to the radial direction; (-45°, 45°] covers all distinct
  // orientations under the cube's 4-fold symmetry.
  const yawInput = appendDegreeSliderGroup(
    controls, 'Yaw (from radius)', YAW_MIN_DEG, YAW_MAX_DEG, 0
  );

  const pregraspLabel = document.createElement('label');
  pregraspLabel.className = 'canonical-grasp-viz-checkbox';
  const showPregraspInput = document.createElement('input');
  showPregraspInput.type = 'checkbox';
  const pregraspText = document.createElement('span');
  pregraspText.textContent = 'Show pregrasp pose';
  pregraspLabel.append(showPregraspInput, pregraspText);
  controls.appendChild(pregraspLabel);

  const resetButton = document.createElement('button');
  resetButton.className = 'canonical-grasp-viz-reset';
  resetButton.type = 'button';
  resetButton.textContent = 'Reset';
  controls.appendChild(resetButton);

  const status = document.createElement('output');
  status.className = 'canonical-grasp-viz-status';
  controls.appendChild(status);

  const branchContainer = document.createElement('div');
  branchContainer.className = 'canonical-grasp-viz-branches';
  controls.appendChild(branchContainer);

  const layout = document.createElement('div');
  layout.className = 'canonical-grasp-viz-layout';
  layout.append(viewport, controls);
  root.appendChild(layout);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root, viewport, coordModeInputs, cartesianGroup, radialGroup, xInput,
    yInput, radiusInput, azimuthInput, yawInput, showPregraspInput,
    resetButton, status, branchContainer
  };
}
