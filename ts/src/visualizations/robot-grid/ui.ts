// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

export interface RobotGridTile {
  viewport: HTMLDivElement;
  swatch: HTMLSpanElement;
  hex: HTMLSpanElement;
  rgb: HTMLSpanElement;
}

export interface RobotGridDom {
  root: HTMLDivElement;
  resampleButton: HTMLButtonElement;
  tiles: RobotGridTile[];
}

export function buildUi(parent: HTMLElement, robotCount: number): RobotGridDom {
  const root = document.createElement('div');
  root.className = 'visualization robot-grid-viz-root';

  const toolbar = document.createElement('div');
  toolbar.className = 'robot-grid-toolbar';

  const resampleButton = document.createElement('button');
  resampleButton.type = 'button';
  resampleButton.className = 'viz-button-primary';
  resampleButton.textContent = 'Resample';
  toolbar.appendChild(resampleButton);
  root.appendChild(toolbar);

  const grid = document.createElement('div');
  grid.className = 'robot-grid';
  root.appendChild(grid);

  const tiles: RobotGridTile[] = [];
  for (let index = 0; index < robotCount; index += 1) {
    const tile = document.createElement('div');
    tile.className = 'robot-grid-tile';

    const viewport = document.createElement('div');
    viewport.className = 'viz-viewport robot-grid-viewport';

    const readout = document.createElement('div');
    readout.className = 'robot-grid-readout';

    const swatch = document.createElement('span');
    swatch.className = 'robot-grid-swatch';

    const hex = document.createElement('span');
    hex.className = 'robot-grid-hex';

    const rgb = document.createElement('span');
    rgb.className = 'robot-grid-rgb';

    readout.append(swatch, hex, rgb);
    viewport.appendChild(readout);
    tile.appendChild(viewport);
    grid.appendChild(tile);
    tiles.push({ viewport, swatch, hex, rgb });
  }

  replacePlaceholder(parent, root);

  return { root, resampleButton, tiles };
}
