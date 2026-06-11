// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 300;

export interface DummyVizDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
}

export function buildUi(parent: HTMLElement): DummyVizDom {
  const root = document.createElement('div');
  root.className = 'visualization dummy-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'dummy-viz-viewport';
  viewport.style.width = `${CANVAS_WIDTH}px`;
  viewport.style.height = `${CANVAS_HEIGHT}px`;
  viewport.style.margin = '0 auto';
  root.appendChild(viewport);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.appendChild(root);
  }

  return {
    root,
    viewport
  };
}
