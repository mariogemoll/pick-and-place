// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import {
  buildPlaybackControls,
  type PlaybackControlsDom
} from '../grasp-pose-shared/playback-controls';
import { replacePlaceholder } from '../grasp-pose-shared/ui';

export const CANVAS_WIDTH = 720;
export const CANVAS_HEIGHT = 480;

export interface EpisodeReplayDom {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  label: HTMLDivElement;
  playback: PlaybackControlsDom;
}

export function buildUi(parent: HTMLElement): EpisodeReplayDom {
  const root = document.createElement('div');
  root.className = 'visualization viz-shell episode-replay-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'viz-viewport episode-replay-viz-viewport';
  root.appendChild(viewport);

  const label = document.createElement('div');
  label.className = 'episode-replay-viz-label';
  viewport.appendChild(label);

  const overlay = document.createElement('div');
  overlay.className = 'viz-playback-overlay';
  const playback = buildPlaybackControls(overlay, 'episode');
  viewport.appendChild(overlay);

  replacePlaceholder(parent, root);

  return { root, viewport, label, playback };
}
