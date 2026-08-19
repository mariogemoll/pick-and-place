// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_WIDTH = 720;
export const CANVAS_HEIGHT = 480;

export interface LivePolicyDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  run: HTMLButtonElement;
  reset: HTMLButtonElement;
  status: HTMLDivElement;
  hint: HTMLDivElement;
}

function button(label: string): HTMLButtonElement {
  const element = document.createElement('button');
  element.type = 'button';
  element.textContent = label;
  return element;
}

export function buildUi(parent: HTMLElement): LivePolicyDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell live-policy-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport live-policy-viz-viewport';
  root.appendChild(viewport);

  const hint = document.createElement('div');
  hint.className = 'live-policy-viz-hint';
  hint.textContent = 'Drag the cube and the target plate, then run.';
  viewport.appendChild(hint);

  const controls = document.createElement('div');
  controls.className = 'live-policy-viz-controls';

  const run = button('Run');
  const reset = button('Reset');
  controls.append(run, reset);
  root.appendChild(controls);

  const status = document.createElement('div');
  status.className = 'live-policy-viz-status';
  root.appendChild(status);

  replacePlaceholder(parent, root);
  return { root, viewport, run, reset, status, hint };
}
