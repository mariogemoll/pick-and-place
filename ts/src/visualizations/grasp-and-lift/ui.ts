// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_HEIGHT = 260;

export interface GraspAndLiftVizDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
}

export function buildUi(parent: HTMLElement): GraspAndLiftVizDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell grasp-and-lift-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport grasp-and-lift-viz-viewport';
  root.appendChild(viewport);

  replacePlaceholder(parent, root);

  return {
    root,
    viewport
  };
}
