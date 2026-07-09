// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { ARM_JOINT_NAMES } from '../../ik/kinematics';
import { parseRollout, type Rollout } from '../episode-replay/rollout';
import { renderPlaybackControls } from '../grasp-pose-shared/playback-controls';
import { createStandardScene, type StandardSceneOptions } from './scene';
import { buildUi } from './ui';

const JOINT_NAMES = [...ARM_JOINT_NAMES, 'gripper'] as const;
const NUM_JOINTS = JOINT_NAMES.length;

const DEFAULT_EPISODE_URLS = Array.from(
  { length: 5 }, (_, index) => `/episodes/episode_${String(index).padStart(2, '0')}.bin`
);
const TARGET_PLATE_HALF_SIZE = 0.05;
const TARGET_PLATE_CLEARANCE = 0.002;
const WORKSPACE_FRAME_POS = { x: 0.279579, y: 0.0000305 };
const WORKSPACE_FRAME_QUAT = { w: -0.707107, x: 0, y: 0, z: -0.707107 };
const WORKSPACE_FRAME_INNER_HALF_EXTENT = 0.2813 - 0.0187;
const WORKSPACE_FRAME_APRILTAG_HALF_SIZE = 0.03;
const WORKSPACE_FRAME_APRILTAG_PLATES = [
  { x: 0.230, y: 0.230 },
  { x: -0.230, y: 0.230 },
  { x: -0.230, y: -0.230 },
  { x: 0.230, y: -0.230 }
] as const;

export interface StandardSceneVisualization {
  destroy(): void;
}

export interface StandardSceneVisualizationOptions extends StandardSceneOptions {
  episodeUrls?: string[];
  showColorControls?: boolean;
}

async function loadEpisodes(urls: string[]): Promise<Rollout[]> {
  return Promise.all(urls.map(async url => {
    const response = await fetch(url);
    if (!response.ok) { throw new Error(`Unable to load ${url}: ${response.status}`); }
    return parseRollout(await response.arrayBuffer());
  }));
}

function frameAt(episode: Rollout, seconds: number): { i0: number; i1: number; t: number } {
  const framePosition = Math.min(seconds * episode.fps, episode.nframes - 1);
  const i0 = Math.floor(framePosition);
  const i1 = Math.min(i0 + 1, episode.nframes - 1);
  return { i0, i1, t: framePosition - i0 };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function randomUnit(seed: number): number {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function worldToFrameXy(x: number, y: number): THREE.Vector2 {
  const { w, x: qx, y: qy, z: qz } = WORKSPACE_FRAME_QUAT;
  const r00 = 1 - 2 * (qy * qy + qz * qz);
  const r01 = 2 * (qx * qy - qz * w);
  const r10 = 2 * (qx * qy + qz * w);
  const r11 = 1 - 2 * (qx * qx + qz * qz);
  const worldDx = x - WORKSPACE_FRAME_POS.x;
  const worldDy = y - WORKSPACE_FRAME_POS.y;
  return new THREE.Vector2(
    r00 * worldDx + r10 * worldDy,
    r01 * worldDx + r11 * worldDy
  );
}

function targetPlateCornersInFrame(x: number, y: number, yaw: number): THREE.Vector2[] {
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
  ].map(([sx, sy]) => worldToFrameXy(
    x + (sx * c - sy * s) * TARGET_PLATE_HALF_SIZE,
    y + (sx * s + sy * c) * TARGET_PLATE_HALF_SIZE
  ));
}

function project(points: readonly THREE.Vector2[], axis: THREE.Vector2): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const point of points) {
    const value = point.dot(axis);
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return [min, max];
}

function polygonsOverlap(
  a: readonly THREE.Vector2[],
  b: readonly THREE.Vector2[]
): boolean {
  for (const polygon of [a, b]) {
    for (let index = 0; index < polygon.length; index++) {
      const p0 = polygon[index];
      const p1 = polygon[(index + 1) % polygon.length];
      const edge = new THREE.Vector2().subVectors(p1, p0);
      const axis = new THREE.Vector2(-edge.y, edge.x).normalize();
      const [aMin, aMax] = project(a, axis);
      const [bMin, bMax] = project(b, axis);
      if (aMax <= bMin || bMax <= aMin) { return false; }
    }
  }
  return true;
}

function targetPlateIsClear(x: number, y: number, yaw: number): boolean {
  const corners = targetPlateCornersInFrame(x, y, yaw);
  const frameLimit = WORKSPACE_FRAME_INNER_HALF_EXTENT - TARGET_PLATE_CLEARANCE;
  if (corners.some(corner => Math.abs(corner.x) > frameLimit || Math.abs(corner.y) > frameLimit)) {
    return false;
  }

  const tagHalfSize = WORKSPACE_FRAME_APRILTAG_HALF_SIZE + TARGET_PLATE_CLEARANCE;
  return !WORKSPACE_FRAME_APRILTAG_PLATES.some(tag => {
    const tagCorners = [
      new THREE.Vector2(tag.x - tagHalfSize, tag.y - tagHalfSize),
      new THREE.Vector2(tag.x + tagHalfSize, tag.y - tagHalfSize),
      new THREE.Vector2(tag.x + tagHalfSize, tag.y + tagHalfSize),
      new THREE.Vector2(tag.x - tagHalfSize, tag.y + tagHalfSize)
    ];
    return polygonsOverlap(corners, tagCorners);
  });
}

function targetPlateYawForEpisode(
  index: number,
  targetX: number,
  targetY: number
): number | null {
  for (let attempt = 0; attempt < 96; attempt++) {
    const yaw = randomUnit((index + 1) * 101 + attempt) * Math.PI * 2;
    if (targetPlateIsClear(targetX, targetY, yaw)) {
      return yaw;
    }
  }
  return null;
}

export function initializeStandardSceneVisualization(
  parent: HTMLElement,
  options: StandardSceneVisualizationOptions = {}
): Promise<StandardSceneVisualization> {
  const ui = buildUi(parent, options.showColorControls ?? false);
  const vizScene = createStandardScene(ui.viewport, options);
  const { renderer, camera, scene, orbitControls } = vizScene;
  const listeners: (() => void)[] = [];

  let animationFrameId = 0;
  let destroyed = false;
  let episodeIndex = 0;
  let episodeSeconds = 0;
  let playing = true;
  let previousFrameTime: number | null = null;
  let episodes: Rollout[] = [];
  let durations: number[] = [];
  let targetPlateYaws: (number | null)[] = [];

  const cubeQuat0 = new THREE.Quaternion();
  const cubeQuat1 = new THREE.Quaternion();
  const cubeQuat = new THREE.Quaternion();

  const updateFloorColor = (): void => {
    vizScene.setFloorColor(new THREE.Color(ui.floorColorInput.value));
  };
  ui.floorColorInput.addEventListener('input', updateFloorColor);
  listeners.push(() => { ui.floorColorInput.removeEventListener('input', updateFloorColor); });
  updateFloorColor();

  const updatePedestalColor = (): void => {
    vizScene.setPedestalColor(new THREE.Color(ui.pedestalColorInput.value));
  };
  ui.pedestalColorInput.addEventListener('input', updatePedestalColor);
  listeners.push(() => {
    ui.pedestalColorInput.removeEventListener('input', updatePedestalColor);
  });
  updatePedestalColor();

  const updateSkyColor = (): void => {
    vizScene.setBackgroundColor(new THREE.Color(ui.skyColorInput.value));
  };
  ui.skyColorInput.addEventListener('input', updateSkyColor);
  listeners.push(() => { ui.skyColorInput.removeEventListener('input', updateSkyColor); });
  updateSkyColor();

  const updateRobotPlasticColor = (): void => {
    vizScene.setRobotPlasticColor(new THREE.Color(ui.robotPlasticColorInput.value));
  };
  ui.robotPlasticColorInput.addEventListener('input', updateRobotPlasticColor);
  listeners.push(() => {
    ui.robotPlasticColorInput.removeEventListener('input', updateRobotPlasticColor);
  });
  updateRobotPlasticColor();

  const updateEnvironmentMaterialColor = (): void => {
    vizScene.setEnvironmentMaterialColor(
      new THREE.Color(ui.environmentMaterialColorInput.value)
    );
  };
  ui.environmentMaterialColorInput.addEventListener('input', updateEnvironmentMaterialColor);
  listeners.push(() => {
    ui.environmentMaterialColorInput.removeEventListener(
      'input',
      updateEnvironmentMaterialColor
    );
  });
  updateEnvironmentMaterialColor();

  function applyFrame(seconds: number): void {
    const episode = episodes.at(episodeIndex);
    if (episode === undefined) { return; }
    const { i0, i1, t } = frameAt(episode, seconds);
    const frame0 = episode.qpos.subarray(i0 * episode.nq, (i0 + 1) * episode.nq);
    const frame1 = episode.qpos.subarray(i1 * episode.nq, (i1 + 1) * episode.nq);

    for (let jointIndex = 0; jointIndex < NUM_JOINTS; jointIndex++) {
      vizScene.setJoint(JOINT_NAMES[jointIndex], lerp(frame0[jointIndex], frame1[jointIndex], t));
    }

    const cubeX = lerp(frame0[NUM_JOINTS], frame1[NUM_JOINTS], t);
    const cubeY = lerp(frame0[NUM_JOINTS + 1], frame1[NUM_JOINTS + 1], t);
    const cubeZ = lerp(frame0[NUM_JOINTS + 2], frame1[NUM_JOINTS + 2], t);
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

    const targetPlateYaw = targetPlateYaws.at(episodeIndex);
    if (targetPlateYaw === null || targetPlateYaw === undefined) {
      vizScene.setTargetVisible(false);
    } else {
      vizScene.setTarget(episode.targetX, episode.targetY, targetPlateYaw);
    }
  }

  function renderPlayback(): void {
    if (episodes.length === 0) { return; }
    ui.episodeLabel.textContent = `Episode ${episodeIndex + 1} / ${episodes.length}`;
    renderPlaybackControls(
      ui.playback, episodeSeconds, durations[episodeIndex], playing, 'episode'
    );
  }

  function setPlaying(nextPlaying: boolean): void {
    playing = nextPlaying;
    previousFrameTime = null;
    renderPlayback();
  }

  const playPauseListener = (): void => { setPlaying(!playing); };
  const seekListener = (): void => {
    episodeSeconds = Number(ui.playback.seekInput.value);
    previousFrameTime = null;
    applyFrame(episodeSeconds);
    renderPlayback();
  };
  ui.playback.playPauseButton.addEventListener('click', playPauseListener);
  ui.playback.seekInput.addEventListener('input', seekListener);
  listeners.push(() => {
    ui.playback.playPauseButton.removeEventListener('click', playPauseListener);
    ui.playback.seekInput.removeEventListener('input', seekListener);
  });

  const episodePlaybackReady = Promise.all([
    vizScene.ready,
    loadEpisodes(options.episodeUrls ?? DEFAULT_EPISODE_URLS)
  ]).then(([, loadedEpisodes]) => {
    episodes = loadedEpisodes;
    durations = episodes.map(episode => (episode.nframes - 1) / episode.fps);
    targetPlateYaws = episodes.map((episode, index) =>
      targetPlateYawForEpisode(index, episode.targetX, episode.targetY)
    );
    ui.episodeLabel.hidden = false;
    ui.episodeOverlay.hidden = false;
    applyFrame(episodeSeconds);
    renderPlayback();
  }).catch((error: unknown) => {
    console.warn('Episode playback unavailable in standard scene:', error);
    ui.episodeLabel.hidden = true;
    ui.episodeOverlay.hidden = true;
  });

  function animate(time: number): void {
    if (destroyed) {
      return;
    }
    animationFrameId = window.requestAnimationFrame(animate);

    if (playing && episodes.length > 0) {
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

    orbitControls.update();
    renderer.render(scene, camera);
  }

  animationFrameId = window.requestAnimationFrame(animate);

  return Promise.all([vizScene.ready, episodePlaybackReady]).then(() => {
    updateRobotPlasticColor();
    updateEnvironmentMaterialColor();

    return {
      destroy(): void {
        destroyed = true;
        window.cancelAnimationFrame(animationFrameId);
        for (const removeListener of listeners) { removeListener(); }
        vizScene.destroy();
        ui.root.remove();
      }
    };
  });
}
