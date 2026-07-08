// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_WIDTH = 420;
export const CANVAS_HEIGHT = 360;

export interface JointControl {
  input: HTMLInputElement;
  value: HTMLOutputElement;
}

export interface JointControlDefinition {
  name: string;
  label: string;
  type: 'hinge' | 'slide';
  lower: number;
  upper: number;
  value: number;
}

export interface PanelDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  controlsHost: HTMLDivElement;
}

// Builds and attaches the panel synchronously so that multiple viewers
// created in the same container keep their call order regardless of how
// fast each robot's model loads.
export function buildPanel(parent: HTMLElement, label: string): PanelDom {
  const root = document.createElement('div');
  root.className = 'visualization robot-viewer-panel';

  const title = document.createElement('h3');
  title.className = 'robot-viewer-title';
  title.textContent = label;
  root.appendChild(title);

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport robot-viewer-viewport';
  root.appendChild(viewport);

  const controlsHost = document.createElement('div');
  controlsHost.className = 'robot-viewer-controls';
  root.appendChild(controlsHost);

  replacePlaceholder(parent, root);

  return { root, viewport, controlsHost };
}

export function addJointControls(
  host: HTMLDivElement,
  joints: JointControlDefinition[]
): Map<string, JointControl> {
  const controls = new Map<string, JointControl>();
  for (const joint of joints) {
    const row = document.createElement('label');
    row.className = 'viz-slider robot-viewer-joint';

    const jointLabel = document.createElement('span');
    jointLabel.className = 'viz-slider-label';
    jointLabel.textContent = joint.label;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(joint.lower);
    input.max = String(joint.upper);
    input.step = joint.type === 'slide' ? '0.001' : '0.01';
    input.value = String(joint.value);

    const value = document.createElement('output');
    value.className = 'viz-slider-value';
    value.textContent = formatJointValue(joint.type, joint.value);

    row.append(jointLabel, input, value);
    host.appendChild(row);
    controls.set(joint.name, { input, value });
  }
  return controls;
}

export function formatDegrees(radians: number): string {
  return `${Math.round(radians * 180 / Math.PI)}°`;
}

export function formatMillimeters(meters: number): string {
  return `${Math.round(meters * 1000)} mm`;
}

export function formatJointValue(type: 'hinge' | 'slide', value: number): string {
  return type === 'slide' ? formatMillimeters(value) : formatDegrees(value);
}
