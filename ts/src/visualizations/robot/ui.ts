// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 520;

export interface JointControl {
  input: HTMLInputElement;
  value: HTMLOutputElement;
}

export interface RobotVizDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  controls: Map<string, JointControl>;
  poseButtons: Map<string, HTMLButtonElement>;
  geometryModeInputs: Map<RobotGeometryMode, HTMLInputElement>;
  colorInputs: Map<string, HTMLInputElement>;
  extentColorInput: HTMLInputElement;
  extentVisibleInput: HTMLInputElement;
  backgroundColorInput: HTMLInputElement;
}

export interface JointControlDefinition {
  name: string;
  label: string;
  lower: number;
  upper: number;
  value: number;
}

export interface MaterialColorDefinition {
  name: string;
  label: string;
  hexColor: string;
}

const PRESET_PLASTIC_COLORS = [
  '#bac6a4', '#e5d8ca', '#b1ab96', '#7cafa3', '#a7bed1',
  '#e97175', '#a1bcde', '#dcc4a9', '#e7ddd3', '#98bac4'
];

const DEFAULT_PLASTIC_COLOR = '#dbc4a8';
const DEFAULT_EXTENT_COLOR = '#cccccc';
const DEFAULT_BACKGROUND_COLOR = '#f4f8ff';

function randomPlasticColor(): string {
  const h = Math.random();
  const s = 0.2 + Math.random() * 0.8;
  const l = 0.25 + Math.random() * 0.55;
  const a = s * Math.min(l, 1 - l);
  const channel = (n: number): number => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  const toHex = (x: number): string => Math.round(x * 255).toString(16).padStart(2, '0');
  return `#${toHex(channel(0))}${toHex(channel(8))}${toHex(channel(4))}`;
}

export interface RobotPoseButtonDefinition {
  name: string;
  label: string;
}

export type RobotGeometryMode = 'visual' | 'collision' | 'both';

export function buildUi(
  parent: HTMLElement,
  joints: JointControlDefinition[],
  materialColors: MaterialColorDefinition[],
  poseButtons: RobotPoseButtonDefinition[]
): RobotVizDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell robot-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport robot-viz-viewport';
  root.appendChild(viewport);

  const panel = document.createElement('div');
  panel.className = 'viz-side-controls robot-viz-controls';

  const poseButtonGroup = document.createElement('div');
  poseButtonGroup.className = 'robot-viz-pose-buttons';
  const poseButtonElements = new Map<string, HTMLButtonElement>();
  for (const pose of poseButtons) {
    const button = document.createElement('button');
    button.className = 'viz-button robot-viz-pose-button';
    button.type = 'button';
    button.textContent = pose.label;
    poseButtonGroup.appendChild(button);
    poseButtonElements.set(pose.name, button);
  }

  const controls = new Map<string, JointControl>();
  for (const joint of joints) {
    const row = document.createElement('label');
    row.className = 'viz-slider robot-viz-joint';

    const label = document.createElement('span');
    label.className = 'viz-slider-label';
    label.textContent = joint.label;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(joint.lower);
    input.max = String(joint.upper);
    input.step = '0.01';
    input.value = String(joint.value);

    const value = document.createElement('output');
    value.className = 'viz-slider-value';
    value.textContent = formatDegrees(joint.value);

    row.append(label, input, value);
    panel.appendChild(row);
    controls.set(joint.name, { input, value });
  }
  panel.appendChild(poseButtonGroup);

  const geometryModeGroup = document.createElement('div');
  geometryModeGroup.className = 'viz-segmented robot-viz-geometry-mode';
  const geometryModeInputs = new Map<RobotGeometryMode, HTMLInputElement>();
  const geometryModes: { mode: RobotGeometryMode; label: string }[] = [
    { mode: 'visual', label: 'Visual' },
    { mode: 'collision', label: 'Collision' },
    { mode: 'both', label: 'Both' }
  ];
  for (const { mode, label } of geometryModes) {
    const option = document.createElement('label');
    option.className = 'viz-segmented-option';

    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'robot-geometry-mode';
    input.value = mode;
    input.checked = mode === 'visual';

    option.append(input, label);
    geometryModeGroup.appendChild(option);
    geometryModeInputs.set(mode, input);
  }
  panel.appendChild(geometryModeGroup);

  const extentVisibleRow = document.createElement('label');
  extentVisibleRow.className = 'robot-viz-extent-visible-row';

  const extentVisibleInput = document.createElement('input');
  extentVisibleInput.type = 'checkbox';
  extentVisibleInput.checked = true;

  const extentVisibleLabel = document.createElement('span');
  extentVisibleLabel.textContent = 'Show extent area';

  extentVisibleRow.append(extentVisibleInput, extentVisibleLabel);
  panel.appendChild(extentVisibleRow);

  const colorsToggleButton = document.createElement('button');
  colorsToggleButton.type = 'button';
  colorsToggleButton.className = 'viz-button robot-viz-colors-toggle';
  colorsToggleButton.textContent = 'Colors';
  colorsToggleButton.setAttribute('aria-expanded', 'false');
  panel.appendChild(colorsToggleButton);

  const colorInputs = new Map<string, HTMLInputElement>();
  const colorSection = document.createElement('div');
  colorSection.className = 'robot-viz-color-section';
  colorSection.hidden = true;

  colorsToggleButton.addEventListener('click', () => {
    colorSection.hidden = false;
    colorsToggleButton.setAttribute('aria-expanded', 'true');
    colorsToggleButton.hidden = true;
  });

  if (materialColors.length > 0) {
    for (const mat of materialColors) {
      const wrapper = document.createElement('div');
      wrapper.className = 'robot-viz-color-entry';

      const row = document.createElement('label');
      row.className = 'robot-viz-color-row';

      const label = document.createElement('span');
      label.textContent = mat.label;

      const initialHex = mat.name === 'plastic' ? DEFAULT_PLASTIC_COLOR : mat.hexColor;

      const readout = document.createElement('output');
      readout.className = 'robot-viz-color-readout';
      readout.textContent = hexToRgbText(initialHex);

      const input = document.createElement('input');
      input.type = 'color';
      input.value = initialHex;
      input.addEventListener('input', () => {
        readout.textContent = hexToRgbText(input.value);
      });

      row.append(label, readout, input);

      wrapper.append(row);

      if (mat.name === 'plastic') {
        const presetRow = document.createElement('div');
        presetRow.className = 'robot-viz-color-presets';
        for (const preset of PRESET_PLASTIC_COLORS) {
          const swatch = document.createElement('button');
          swatch.type = 'button';
          swatch.className = 'robot-viz-color-preset';
          swatch.style.backgroundColor = preset;
          swatch.title = preset;
          swatch.addEventListener('click', () => {
            input.value = preset;
            input.dispatchEvent(new Event('input'));
          });
          presetRow.appendChild(swatch);
        }
        wrapper.appendChild(presetRow);

        const randomizeButton = document.createElement('button');
        randomizeButton.type = 'button';
        randomizeButton.className = 'viz-button robot-viz-color-randomize';
        randomizeButton.textContent = 'Random';
        randomizeButton.addEventListener('click', () => {
          input.value = randomPlasticColor();
          input.dispatchEvent(new Event('input'));
        });
        wrapper.appendChild(randomizeButton);
      }

      colorSection.appendChild(wrapper);
      colorInputs.set(mat.name, input);
    }
  }

  const extentWrapper = document.createElement('div');
  extentWrapper.className = 'robot-viz-color-entry';

  const extentRow = document.createElement('label');
  extentRow.className = 'robot-viz-color-row';

  const extentLabel = document.createElement('span');
  extentLabel.textContent = 'Extent area';

  const extentReadout = document.createElement('output');
  extentReadout.className = 'robot-viz-color-readout';
  extentReadout.textContent = hexToRgbText(DEFAULT_EXTENT_COLOR);

  const extentColorInput = document.createElement('input');
  extentColorInput.type = 'color';
  extentColorInput.value = DEFAULT_EXTENT_COLOR;
  extentColorInput.addEventListener('input', () => {
    extentReadout.textContent = hexToRgbText(extentColorInput.value);
  });

  extentRow.append(extentLabel, extentReadout, extentColorInput);

  extentWrapper.append(extentRow);
  colorSection.appendChild(extentWrapper);

  const backgroundWrapper = document.createElement('div');
  backgroundWrapper.className = 'robot-viz-color-entry';

  const backgroundRow = document.createElement('label');
  backgroundRow.className = 'robot-viz-color-row';

  const backgroundLabel = document.createElement('span');
  backgroundLabel.textContent = 'Background';

  const backgroundReadout = document.createElement('output');
  backgroundReadout.className = 'robot-viz-color-readout';
  backgroundReadout.textContent = hexToRgbText(DEFAULT_BACKGROUND_COLOR);

  const backgroundColorInput = document.createElement('input');
  backgroundColorInput.type = 'color';
  backgroundColorInput.value = DEFAULT_BACKGROUND_COLOR;
  backgroundColorInput.addEventListener('input', () => {
    backgroundReadout.textContent = hexToRgbText(backgroundColorInput.value);
  });

  backgroundRow.append(backgroundLabel, backgroundReadout, backgroundColorInput);

  backgroundWrapper.append(backgroundRow);
  colorSection.appendChild(backgroundWrapper);

  panel.appendChild(colorSection);

  root.appendChild(panel);

  replacePlaceholder(parent, root);

  return {
    root, viewport, controls, poseButtons: poseButtonElements, geometryModeInputs, colorInputs,
    extentColorInput, extentVisibleInput, backgroundColorInput
  };
}

export function formatDegrees(radians: number): string {
  return `${Math.round(radians * 180 / Math.PI)}°`;
}

function hexToRgbText(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  const r = ((n >> 16) & 0xff) / 255;
  const g = ((n >> 8) & 0xff) / 255;
  const b = (n & 0xff) / 255;
  return `${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)}`;
}
