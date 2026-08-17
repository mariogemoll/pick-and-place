// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { replacePlaceholder } from '../grasp-pose-shared/ui';
import { VIEW_GAP } from './camera-strip';

// Captions sit in the same order as the cameras the strip renders.
export const CAMERA_CAPTIONS = ['overhead', 'wrist'] as const;

export interface FlowPolicyDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  cameras: HTMLDivElement;
  panelHost: HTMLDivElement;
  label: HTMLDivElement;
  status: HTMLDivElement;
  playPauseButton: HTMLButtonElement;
  modeButton: HTMLButtonElement;
}

// Everything but the scene is inlaid over it: the panel and the camera views
// float on the render rather than taking a column of their own, which keeps the
// whole visualization to one compact stage.
export function buildUi(parent: HTMLElement): FlowPolicyDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell flow-policy-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport flow-policy-viz-viewport';

  const label = document.createElement('div');
  label.className = 'flow-policy-viz-label';

  // The views the policy is actually looking through, inlaid over the scene
  // they are looking into.
  // The strip renders every view into one canvas, so each view is framed and
  // captioned by an element laid over its own column of it. The columns are the
  // ones `layoutViews` scissors, gaps and all.
  const cameras = document.createElement('div');
  cameras.className = 'flow-policy-viz-cameras';
  const gaps = (CAMERA_CAPTIONS.length - 1) * VIEW_GAP;
  const column = `((100% - ${gaps}px) / ${CAMERA_CAPTIONS.length})`;
  for (let index = 0; index < CAMERA_CAPTIONS.length; index++) {
    const view = document.createElement('span');
    view.className = 'flow-policy-viz-camera-view';
    view.style.left = `calc((${column} + ${VIEW_GAP}px) * ${index})`;
    view.style.width = `calc(${column})`;
    const caption = document.createElement('span');
    caption.className = 'flow-policy-viz-camera-caption';
    caption.textContent = CAMERA_CAPTIONS[index];
    view.appendChild(caption);
    cameras.appendChild(view);
  }

  const panelHost = document.createElement('div');
  panelHost.className = 'flow-policy-viz-panel-host';

  const overlay = document.createElement('div');
  overlay.className = 'viz-playback-overlay flow-policy-viz-overlay';
  const status = document.createElement('div');
  status.className = 'flow-policy-viz-status';
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
  overlay.append(status, controls);

  viewport.append(label, cameras, panelHost, overlay);
  root.appendChild(viewport);

  replacePlaceholder(parent, root);

  return {
    root, viewport, cameras, panelHost, label, status, playPauseButton, modeButton
  };
}
