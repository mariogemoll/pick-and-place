// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { ARM_JOINT_NAMES } from '../../ik/kinematics';
import { loadWebModel } from '../../web-model';
import { renderPlaybackControls } from '../grasp-pose-shared/playback-controls';
import { parseRollout, type Rollout } from './rollout';
import { createEpisodeReplayScene } from './scene';
import { buildUi } from './ui';

const JOINT_NAMES = [...ARM_JOINT_NAMES, 'gripper'] as const;
const NUM_JOINTS = JOINT_NAMES.length;

const DEFAULT_EPISODE_URLS = Array.from(
  { length: 5 }, (_, index) => `/episodes/episode_${String(index).padStart(2, '0')}.bin`
);

export interface EpisodeReplayVisualization {
  destroy(): void;
}

export interface EpisodeReplayOptions {
  modelBasePath?: string;
  modelUrl?: string;
  episodeUrls?: string[];
}

async function loadEpisodes(urls: string[]): Promise<Rollout[]> {
  return Promise.all(urls.map(async url => {
    const response = await fetch(url);
    if (!response.ok) { throw new Error(`Unable to load ${url}: ${response.status}`); }
    return parseRollout(await response.arrayBuffer());
  }));
}

// Which recorded frame pair (and blend fraction between them) ``seconds`` into
// an episode falls on, so playback rate is decoupled from the file's own fps.
function frameAt(episode: Rollout, seconds: number): { i0: number; i1: number; t: number } {
  const framePosition = Math.min(seconds * episode.fps, episode.nframes - 1);
  const i0 = Math.floor(framePosition);
  const i1 = Math.min(i0 + 1, episode.nframes - 1);
  return { i0, i1, t: framePosition - i0 };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export async function EpisodeReplay(
  parent: HTMLElement,
  options: EpisodeReplayOptions = {}
): Promise<EpisodeReplayVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const episodeUrls = options.episodeUrls ?? DEFAULT_EPISODE_URLS;
  const episodes = await loadEpisodes(episodeUrls);
  const durations = episodes.map(episode => (episode.nframes - 1) / episode.fps);

  const ui = buildUi(parent);
  const vizScene = createEpisodeReplayScene(ui.viewport, model, options.modelBasePath);

  let episodeIndex = 0;
  let episodeSeconds = 0;
  let playing = true;
  let previousFrameTime: number | null = null;

  const cubeQuat0 = new THREE.Quaternion();
  const cubeQuat1 = new THREE.Quaternion();
  const cubeQuat = new THREE.Quaternion();

  const applyFrame = (seconds: number): void => {
    const episode = episodes[episodeIndex];
    const { i0, i1, t } = frameAt(episode, seconds);
    const frame0 = episode.qpos.subarray(i0 * episode.nq, (i0 + 1) * episode.nq);
    const frame1 = episode.qpos.subarray(i1 * episode.nq, (i1 + 1) * episode.nq);

    for (let jointIndex = 0; jointIndex < NUM_JOINTS; jointIndex++) {
      vizScene.setJoint(JOINT_NAMES[jointIndex], lerp(frame0[jointIndex], frame1[jointIndex], t));
    }

    const cubeX = lerp(frame0[NUM_JOINTS], frame1[NUM_JOINTS], t);
    const cubeY = lerp(frame0[NUM_JOINTS + 1], frame1[NUM_JOINTS + 1], t);
    const cubeZ = lerp(frame0[NUM_JOINTS + 2], frame1[NUM_JOINTS + 2], t);
    // Cube pose is [pos.x, pos.y, pos.z, quat.w, quat.x, quat.y, quat.z] (MuJoCo order).
    cubeQuat0.set(
      frame0[NUM_JOINTS + 4], frame0[NUM_JOINTS + 5],
      frame0[NUM_JOINTS + 6], frame0[NUM_JOINTS + 3]
    );
    cubeQuat1.set(
      frame1[NUM_JOINTS + 4], frame1[NUM_JOINTS + 5],
      frame1[NUM_JOINTS + 6], frame1[NUM_JOINTS + 3]
    );
    cubeQuat.copy(cubeQuat0).slerp(cubeQuat1, t);
    vizScene.setCubeTransform(
      cubeX, cubeY, cubeZ, [cubeQuat.w, cubeQuat.x, cubeQuat.y, cubeQuat.z]
    );

    vizScene.setTarget(episode.targetX, episode.targetY);
  };

  const renderPlayback = (): void => {
    ui.label.textContent = `Episode ${episodeIndex + 1} / ${episodes.length}`;
    renderPlaybackControls(
      ui.playback, episodeSeconds, durations[episodeIndex], playing, 'episode'
    );
  };
  const setPlaying = (nextPlaying: boolean): void => {
    playing = nextPlaying;
    previousFrameTime = null;
    renderPlayback();
  };

  applyFrame(episodeSeconds);
  renderPlayback();

  const playPauseListener = (): void => { setPlaying(!playing); };
  const seekListener = (): void => {
    episodeSeconds = Number(ui.playback.seekInput.value);
    previousFrameTime = null;
    applyFrame(episodeSeconds);
    renderPlayback();
  };
  ui.playback.playPauseButton.addEventListener('click', playPauseListener);
  ui.playback.seekInput.addEventListener('input', seekListener);

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  let animationFrameId = 0;
  let destroyed = false;
  function animate(time: number): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    if (playing) {
      if (previousFrameTime !== null) {
        episodeSeconds += (time - previousFrameTime) / 1000;
        if (episodeSeconds >= durations[episodeIndex]) {
          episodeIndex = (episodeIndex + 1) % episodes.length;
          episodeSeconds = 0;
        }
        applyFrame(episodeSeconds);
        renderPlayback();
      }
      previousFrameTime = time;
    }
    vizScene.orbitControls.update();
    vizScene.renderer.render(vizScene.scene, vizScene.camera);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      ui.playback.playPauseButton.removeEventListener('click', playPauseListener);
      ui.playback.seekInput.removeEventListener('input', seekListener);
      vizScene.destroy();
      ui.root.remove();
    }
  };
}
