// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export const CANVAS_WIDTH = 640;
export const CANVAS_HEIGHT = 480;

export interface StandardSceneUi {
  root: HTMLElement;
  viewport: HTMLElement;
}

export function buildUi(parent: HTMLElement): StandardSceneUi {
  const root = document.createElement('div');
  root.className = 'standard-scene-ui';

  const viewport = document.createElement('div');
  viewport.className = 'viewport';
  viewport.style.width = `${CANVAS_WIDTH}px`;
  viewport.style.height = `${CANVAS_HEIGHT}px`;
  viewport.style.backgroundColor = '#111';
  root.appendChild(viewport);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return { root, viewport };
}
