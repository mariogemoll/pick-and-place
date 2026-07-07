// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// A play/pause button + seek slider + elapsed/total time readout, shared by
// any visualization that scrubs through a timed animation (pick-and-place's
// trajectory run, episode-replay's recorded rollouts).

export interface PlaybackControlsDom {
  row: HTMLDivElement;
  playPauseButton: HTMLButtonElement;
  seekInput: HTMLInputElement;
  playbackTime: HTMLOutputElement;
}

export function formatPlaybackTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = (seconds % 60).toFixed(1).padStart(4, '0');
  return `${minutes}:${remainingSeconds}`;
}

export function buildPlaybackControls(
  parent: HTMLElement,
  label = 'playback'
): PlaybackControlsDom {
  const seekInput = document.createElement('input');
  seekInput.className = 'viz-seek';
  seekInput.type = 'range';
  seekInput.min = '0';
  seekInput.max = '1';
  seekInput.step = '0.01';
  seekInput.value = '0';
  seekInput.setAttribute('aria-label', `${label} position`);

  const row = document.createElement('div');
  row.className = 'viz-playback-row';
  const playPauseButton = document.createElement('button');
  playPauseButton.className = 'viz-button viz-button-primary viz-play-button';
  playPauseButton.type = 'button';
  playPauseButton.textContent = 'Play';
  playPauseButton.setAttribute('aria-label', `Play ${label}`);
  const playbackTime = document.createElement('output');
  playbackTime.className = 'viz-playback-time';
  playbackTime.textContent = '0:00.0 / 0:00.0';
  row.append(playPauseButton, playbackTime);

  parent.append(seekInput, row);

  return { row, playPauseButton, seekInput, playbackTime };
}

export function renderPlaybackControls(
  dom: Pick<PlaybackControlsDom, 'playPauseButton' | 'seekInput' | 'playbackTime'>,
  seconds: number,
  duration: number,
  playing: boolean,
  label = 'playback'
): void {
  dom.seekInput.max = String(duration);
  dom.seekInput.value = String(seconds);
  dom.playbackTime.textContent =
    `${formatPlaybackTime(seconds)} / ${formatPlaybackTime(duration)}`;
  dom.playPauseButton.textContent = playing ? 'Pause' : 'Play';
  dom.playPauseButton.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${label}`);
}
