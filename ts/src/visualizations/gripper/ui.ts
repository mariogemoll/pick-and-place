// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 300;

export interface GripperVizDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
}

export function buildUi(parent: HTMLElement): GripperVizDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell gripper-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport gripper-viz-viewport';
  viewport.style.width = `${CANVAS_WIDTH}px`;
  viewport.style.height = `${CANVAS_HEIGHT}px`;
  viewport.style.margin = '0 auto';
  root.appendChild(viewport);

  replacePlaceholder(parent, root);

  return {
    root,
    viewport
  };
}
