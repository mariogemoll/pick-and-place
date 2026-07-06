// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_WIDTH = 640;
export const CANVAS_HEIGHT = 480;

export interface StandardSceneUi {
  root: HTMLElement;
  viewport: HTMLElement;
}

export function buildUi(parent: HTMLElement): StandardSceneUi {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell standard-scene-ui';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport viewport';
  viewport.style.width = `${CANVAS_WIDTH}px`;
  viewport.style.height = `${CANVAS_HEIGHT}px`;
  root.appendChild(viewport);

  replacePlaceholder(parent, root);

  return { root, viewport };
}
