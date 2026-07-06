// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  appendCheckbox,
  appendDegreeSliderGroup,
  appendRadioGroup,
  appendResetButton,
  appendSliderGroup,
  appendStatus,
  replacePlaceholder
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
  dropModeInput: HTMLInputElement;
  resetButton: HTMLButtonElement;
  status: HTMLOutputElement;
  branchContainer: HTMLDivElement;
}

// The default cube X/Y (metres) used on reset; comfortably reachable in front
// of the robot.
export const DEFAULT_CUBE_X = 0.2;
export const DEFAULT_CUBE_Y = 0;

// Cube-center height used for the "drop mode" sweep, matching Python's
// DROP_CUBE_CENTER_Z (py/src/pick_and_place/trajectory.py).
export const DROP_POSE_Z_MM = 45;

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
  root.className = 'visualization viz-shell canonical-grasp-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport canonical-grasp-viz-viewport';

  const controls = document.createElement('div');
  controls.className = 'viz-side-controls canonical-grasp-viz-controls';

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

  const showPregraspInput = appendCheckbox(
    controls, 'Show pregrasp pose', 'canonical-grasp-viz-checkbox'
  );
  const dropModeInput = appendCheckbox(
    controls,
    `Drop mode (ignore orientation, z = ${DROP_POSE_Z_MM} mm)`,
    'canonical-grasp-viz-checkbox'
  );

  const resetButton = appendResetButton(controls);
  resetButton.classList.add('canonical-grasp-viz-reset');

  const status = appendStatus(controls);
  status.classList.add('canonical-grasp-viz-status');

  const branchContainer = document.createElement('div');
  branchContainer.className = 'canonical-grasp-viz-branches';
  controls.appendChild(branchContainer);

  const layout = document.createElement('div');
  layout.className = 'canonical-grasp-viz-layout';
  layout.append(viewport, controls);
  root.appendChild(layout);

  replacePlaceholder(parent, root);

  return {
    root, viewport, coordModeInputs, cartesianGroup, radialGroup, xInput,
    yInput, radiusInput, azimuthInput, yawInput, showPregraspInput,
    dropModeInput, resetButton, status, branchContainer
  };
}
