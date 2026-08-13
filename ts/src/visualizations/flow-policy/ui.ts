// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';

// Captions sit in the same order as the cameras the strip renders.
export const CAMERA_CAPTIONS = ['overhead', 'wrist'] as const;

export interface FlowPolicyDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  cameras: HTMLDivElement;
  panelHost: HTMLDivElement;
  label: HTMLDivElement;
  status: HTMLDivElement;
  phaseRow: HTMLDivElement;
  phases: Record<string, HTMLSpanElement>;
  playPauseButton: HTMLButtonElement;
  modeButton: HTMLButtonElement;
}

export function buildUi(parent: HTMLElement): FlowPolicyDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell flow-policy-viz-root';

  const scene = document.createElement('div');
  scene.className = 'flow-policy-viz-scene';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport flow-policy-viz-viewport';
  scene.appendChild(viewport);

  const label = document.createElement('div');
  label.className = 'flow-policy-viz-label';
  viewport.appendChild(label);

  const cameras = document.createElement('div');
  cameras.className = 'flow-policy-viz-cameras';
  for (let index = 0; index < CAMERA_CAPTIONS.length; index++) {
    const caption = document.createElement('span');
    caption.className = 'flow-policy-viz-camera-caption';
    caption.style.left = `${(index * 100) / CAMERA_CAPTIONS.length}%`;
    caption.textContent = CAMERA_CAPTIONS[index];
    cameras.appendChild(caption);
  }
  scene.appendChild(cameras);

  const side = document.createElement('div');
  side.className = 'flow-policy-viz-side';

  // The three beats of one horizon, lit in turn so the cycle is readable
  // without watching the whole loop.
  const phaseRow = document.createElement('div');
  phaseRow.className = 'flow-policy-viz-phases';
  const phases: Record<string, HTMLSpanElement> = {};
  for (const [key, text] of [
    ['sample', '1 · sample noise'],
    ['flow', '2 · run the flow'],
    ['execute', '3 · execute chunk']
  ]) {
    const chip = document.createElement('span');
    chip.className = 'flow-policy-viz-phase';
    chip.textContent = text;
    phases[key] = chip;
    phaseRow.appendChild(chip);
  }
  side.appendChild(phaseRow);

  const status = document.createElement('div');
  status.className = 'flow-policy-viz-status';
  side.appendChild(status);

  const panelHost = document.createElement('div');
  panelHost.className = 'flow-policy-viz-panel-host';
  side.appendChild(panelHost);

  root.append(scene, side);

  const overlay = document.createElement('div');
  overlay.className = 'viz-playback-overlay';
  const controls = document.createElement('div');
  controls.className = 'flow-policy-viz-controls';
  const playPauseButton = document.createElement('button');
  playPauseButton.className = 'viz-button viz-button-primary viz-play-button';
  playPauseButton.type = 'button';
  playPauseButton.textContent = 'Pause';
  playPauseButton.setAttribute('aria-label', 'Pause rollout');
  const modeButton = document.createElement('button');
  modeButton.className = 'viz-button viz-play-button flow-policy-viz-mode-button';
  modeButton.type = 'button';
  controls.append(playPauseButton, modeButton);
  overlay.appendChild(controls);
  viewport.appendChild(overlay);

  replacePlaceholder(parent, root);

  return {
    root, viewport, cameras, panelHost, label, status,
    phaseRow, phases, playPauseButton, modeButton
  };
}
