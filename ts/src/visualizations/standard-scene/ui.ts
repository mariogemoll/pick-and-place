// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  buildPlaybackControls,
  type PlaybackControlsDom
} from '../grasp-pose-shared/playback-controls';
import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_WIDTH = 640;
export const CANVAS_HEIGHT = 480;
const DEFAULT_FLOOR_COLOR = '#70f2f7';
const DEFAULT_PEDESTAL_COLOR = '#3d5cad';
const DEFAULT_SKY_COLOR = '#f27782';
const DEFAULT_ROBOT_PLASTIC_COLOR = '#f5f6fa';
const DEFAULT_ENVIRONMENT_MATERIAL_COLOR = '#fff7e8';

export interface StandardSceneUi {
  root: HTMLElement;
  viewport: HTMLElement;
  floorColorInput: HTMLInputElement;
  pedestalColorInput: HTMLInputElement;
  skyColorInput: HTMLInputElement;
  robotPlasticColorInput: HTMLInputElement;
  environmentMaterialColorInput: HTMLInputElement;
  episodeLabel: HTMLDivElement;
  episodeOverlay: HTMLDivElement;
  playback: PlaybackControlsDom;
}

function hexToRgbText(hex: string): string {
  const n = parseInt(hex.slice(1), 16);
  const r = ((n >> 16) & 0xff) / 255;
  const g = ((n >> 8) & 0xff) / 255;
  const b = (n & 0xff) / 255;
  return `${r.toFixed(2)}, ${g.toFixed(2)}, ${b.toFixed(2)}`;
}

interface ColorControl {
  wrapper: HTMLDivElement;
  input: HTMLInputElement;
}

function createColorControl(labelText: string, initialValue: string): ColorControl {
  const wrapper = document.createElement('div');
  wrapper.className = 'robot-viz-color-entry';

  const row = document.createElement('label');
  row.className = 'robot-viz-color-row';

  const label = document.createElement('span');
  label.textContent = labelText;

  const readout = document.createElement('output');
  readout.className = 'robot-viz-color-readout';
  readout.textContent = hexToRgbText(initialValue);

  const input = document.createElement('input');
  input.type = 'color';
  input.value = initialValue;
  input.addEventListener('input', () => {
    readout.textContent = hexToRgbText(input.value);
  });

  row.append(label, readout, input);
  wrapper.append(row);

  return { wrapper, input };
}

export function buildUi(
  parent: HTMLElement,
  showColorControls: boolean
): StandardSceneUi {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell standard-scene-ui standard-scene-viz-root';
  if (!showColorControls) {
    root.classList.add('standard-scene-viz-root-no-controls');
  }

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport viewport standard-scene-viz-viewport';
  viewport.style.width = `${CANVAS_WIDTH}px`;
  viewport.style.height = `${CANVAS_HEIGHT}px`;
  root.appendChild(viewport);

  const episodeLabel = document.createElement('div');
  episodeLabel.className = 'standard-scene-viz-episode-label';
  episodeLabel.hidden = true;
  viewport.appendChild(episodeLabel);

  const episodeOverlay = document.createElement('div');
  episodeOverlay.className = 'viz-playback-overlay standard-scene-viz-playback';
  episodeOverlay.hidden = true;
  const playback = buildPlaybackControls(episodeOverlay, 'episode');
  viewport.appendChild(episodeOverlay);

  const floorColor = createColorControl('Floor', DEFAULT_FLOOR_COLOR);
  const pedestalColor = createColorControl('Pedestal', DEFAULT_PEDESTAL_COLOR);
  const skyColor = createColorControl('Sky', DEFAULT_SKY_COLOR);
  const robotPlasticColor =
    createColorControl('Robot', DEFAULT_ROBOT_PLASTIC_COLOR);
  const environmentMaterialColor =
    createColorControl('Environment', DEFAULT_ENVIRONMENT_MATERIAL_COLOR);

  if (showColorControls) {
    const panel = document.createElement('div');
    panel.className = 'viz-side-controls standard-scene-viz-controls';

    const colorsToggleButton = document.createElement('button');
    colorsToggleButton.type = 'button';
    colorsToggleButton.className = 'viz-button robot-viz-colors-toggle';
    colorsToggleButton.textContent = 'Colors';
    colorsToggleButton.setAttribute('aria-expanded', 'false');
    panel.appendChild(colorsToggleButton);

    const colorSection = document.createElement('div');
    colorSection.className = 'robot-viz-color-section';
    colorSection.hidden = true;

    colorsToggleButton.addEventListener('click', () => {
      colorSection.hidden = false;
      colorsToggleButton.setAttribute('aria-expanded', 'true');
      colorsToggleButton.hidden = true;
    });

    colorSection.append(
      floorColor.wrapper,
      pedestalColor.wrapper,
      skyColor.wrapper,
      robotPlasticColor.wrapper,
      environmentMaterialColor.wrapper
    );
    panel.appendChild(colorSection);
    root.appendChild(panel);
  }

  replacePlaceholder(parent, root);

  return {
    root,
    viewport,
    floorColorInput: floorColor.input,
    pedestalColorInput: pedestalColor.input,
    skyColorInput: skyColor.input,
    robotPlasticColorInput: robotPlasticColor.input,
    environmentMaterialColorInput: environmentMaterialColor.input,
    episodeLabel,
    episodeOverlay,
    playback
  };
}
