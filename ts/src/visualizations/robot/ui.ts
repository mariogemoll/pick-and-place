// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { appendResetButton, replacePlaceholder } from '../grasp-pose-shared/ui';

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
  resetButton: HTMLButtonElement;
  colorInputs: Map<string, HTMLInputElement>;
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

export function buildUi(
  parent: HTMLElement,
  joints: JointControlDefinition[],
  materialColors: MaterialColorDefinition[]
): RobotVizDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell robot-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport robot-viz-viewport';
  root.appendChild(viewport);

  const panel = document.createElement('div');
  panel.className = 'viz-side-controls robot-viz-controls';

  const header = document.createElement('div');
  header.className = 'robot-viz-controls-header';
  const title = document.createElement('strong');
  title.textContent = 'Joint angles';
  const resetButton = appendResetButton(header, 'Reset pose');
  header.append(title, resetButton);
  panel.appendChild(header);

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

  const colorInputs = new Map<string, HTMLInputElement>();
  if (materialColors.length > 0) {
    const colorSection = document.createElement('div');
    colorSection.className = 'robot-viz-color-section';

    const colorTitle = document.createElement('strong');
    colorTitle.textContent = 'Colors';
    colorSection.appendChild(colorTitle);

    for (const mat of materialColors) {
      const wrapper = document.createElement('div');
      wrapper.className = 'robot-viz-color-entry';

      const row = document.createElement('label');
      row.className = 'robot-viz-color-row';

      const label = document.createElement('span');
      label.textContent = mat.label;

      const input = document.createElement('input');
      input.type = 'color';
      input.value = mat.hexColor;

      row.append(label, input);

      const readout = document.createElement('output');
      readout.className = 'robot-viz-color-readout';
      readout.textContent = hexToRgbText(mat.hexColor);
      input.addEventListener('input', () => {
        readout.textContent = hexToRgbText(input.value);
      });

      wrapper.append(row, readout);
      colorSection.appendChild(wrapper);
      colorInputs.set(mat.name, input);
    }

    panel.appendChild(colorSection);
  }

  root.appendChild(panel);

  replacePlaceholder(parent, root);

  return { root, viewport, controls, resetButton, colorInputs };
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
