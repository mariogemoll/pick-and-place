// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// A recorded flow-policy rollout, replayed one horizon at a time.
//
// The policy works in chunks: draw noise, integrate it into a 16-step horizon,
// execute the first 8 of those steps, repeat. Stepped, the visualization walks
// that cycle beat by beat, holding the arm still while a horizon is being
// generated and moving it only while that horizon is being executed, so each
// phase is watchable on its own. Continuous drops the beats and plays the
// rollout at its recorded rate, showing only each finished horizon.

import * as THREE from 'three';

import { ARM_JOINT_NAMES } from '../../ik/kinematics';
import { cameraByName, createWebCamera, loadWebModel, type WebModel } from '../../web-model';
import { createEpisodeReplayScene } from '../episode-replay/scene';
import { createCameraStrip } from './camera-strip';
import { createFlowPanel, type FlowPanel } from './flow-panel';
import {
  buildSchedule,
  momentAt,
  scheduleDuration,
  type ScheduleMode,
  type Segment
} from './schedule';
import { type FlowTrace, frame, parseFlowTrace, pathState } from './trace';
import { buildUi } from './ui';

const JOINT_NAMES = [...ARM_JOINT_NAMES, 'gripper'] as const;
const NUM_JOINTS = JOINT_NAMES.length;

const DEFAULT_TRACE_URLS = [
  '/flow-traces/dppo-train-000000.bin',
  '/flow-traces/dppo-train-000001.bin',
  '/flow-traces/dppo-train-000002.bin'
];

export interface FlowPolicyVisualization {
  destroy(): void;
}

export interface FlowPolicyOptions {
  modelBasePath?: string;
  modelUrl?: string;
  environmentModelUrl?: string;
  traceUrls?: string[];
}

async function loadTraces(urls: string[]): Promise<FlowTrace[]> {
  return Promise.all(urls.map(async url => {
    const response = await fetch(url);
    if (!response.ok) { throw new Error(`Unable to load ${url}: ${response.status}`); }
    return parseFlowTrace(await response.arrayBuffer());
  }));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// Hangs a camera off the body it is bolted to in the manifest, so it tracks
// that body as the arm moves instead of being re-posed every frame.
function mountCamera(
  bodies: Map<string, THREE.Group>,
  model: WebModel,
  name: string
): THREE.PerspectiveCamera {
  const definition = cameraByName(model, name);
  const body = bodies.get(definition.body);
  if (body === undefined) {
    throw new Error(`Scene has no body "${definition.body}" to mount ${name} on`);
  }
  const camera = createWebCamera(definition);
  body.add(camera);
  return camera;
}

export async function FlowPolicy(
  parent: HTMLElement,
  options: FlowPolicyOptions = {}
): Promise<FlowPolicyVisualization> {
  const [model, environmentModel] = await Promise.all([
    loadWebModel(options.modelUrl),
    loadWebModel(options.environmentModelUrl ?? '/environment.json')
  ]);
  const traces = await loadTraces(options.traceUrls ?? DEFAULT_TRACE_URLS);
  // Both timings are laid out up front; switching modes is then just a matter
  // of reading from the other one.
  const schedules: Record<ScheduleMode, Segment[][]> = {
    stepped: traces.map(trace => buildSchedule(trace, { mode: 'stepped' })),
    continuous: traces.map(trace => buildSchedule(trace, { mode: 'continuous' }))
  };
  const durations: Record<ScheduleMode, number[]> = {
    stepped: schedules.stepped.map(scheduleDuration),
    continuous: schedules.continuous.map(scheduleDuration)
  };

  const ui = buildUi(parent);
  // The overhead camera hangs off the environment's mast, and the workspace
  // frame it looks down on is what the real camera sees, so the environment
  // comes into the scene rather than the camera being placed in mid-air.
  const vizScene = createEpisodeReplayScene(ui.viewport, model, {
    modelBasePath: options.modelBasePath,
    environmentModel
  });
  const panel: FlowPanel = createFlowPanel(ui.panelHost);

  // Order matches CAMERA_CAPTIONS in the UI.
  const cameraStrip = createCameraStrip(ui.cameras, [
    mountCamera(vizScene.environmentBodies, environmentModel, 'overhead_camera'),
    mountCamera(vizScene.bodies, model, 'wrist_camera')
  ]);

  let traceIndex = 0;
  let seconds = 0;
  let playing = true;
  let mode: ScheduleMode = 'stepped';
  let previousFrameTime: number | null = null;

  // Reused so the animation does not allocate a horizon per rendered frame,
  // sized for the largest horizon in case the rollouts came from different runs.
  const blended = new Float32Array(
    Math.max(...traces.map(trace => trace.steps * trace.joints))
  );

  const cubeQuat0 = new THREE.Quaternion();
  const cubeQuat1 = new THREE.Quaternion();
  const cubeQuat = new THREE.Quaternion();

  const applyScene = (trace: FlowTrace, tickFloat: number): void => {
    const i0 = Math.floor(tickFloat);
    const t = tickFloat - i0;
    const frame0 = frame(trace, i0);
    const frame1 = frame(trace, i0 + 1);

    for (let joint = 0; joint < NUM_JOINTS; joint++) {
      vizScene.setJoint(JOINT_NAMES[joint], lerp(frame0[joint], frame1[joint], t));
    }
    // Cube pose is [x, y, z, qw, qx, qy, qz] in MuJoCo order.
    cubeQuat0.set(frame0[NUM_JOINTS + 4], frame0[NUM_JOINTS + 5],
      frame0[NUM_JOINTS + 6], frame0[NUM_JOINTS + 3]);
    cubeQuat1.set(frame1[NUM_JOINTS + 4], frame1[NUM_JOINTS + 5],
      frame1[NUM_JOINTS + 6], frame1[NUM_JOINTS + 3]);
    cubeQuat.copy(cubeQuat0).slerp(cubeQuat1, t);
    vizScene.setCubeTransform(
      lerp(frame0[NUM_JOINTS], frame1[NUM_JOINTS], t),
      lerp(frame0[NUM_JOINTS + 1], frame1[NUM_JOINTS + 1], t),
      lerp(frame0[NUM_JOINTS + 2], frame1[NUM_JOINTS + 2], t),
      [cubeQuat.w, cubeQuat.x, cubeQuat.y, cubeQuat.z]
    );
    vizScene.setTarget(trace.targetX, trace.targetY);
  };

  const applyTime = (value: number): void => {
    const trace = traces[traceIndex];
    const { beat, chunk, opacity, phase, progress, tickFloat } =
      momentAt(schedules[mode][traceIndex], value);
    if (chunk < 0) { return; }

    applyScene(trace, tickFloat);

    // The integration only runs during the flow beat: the sample beat holds the
    // noise it starts from, and the execute beat holds the finished horizon.
    const integration = phase === 'sample' ? 0 : phase === 'flow' ? progress : 1;
    // Walk it continuously rather than snapping between the recorded Euler
    // states, so the dots slide instead of jumping.
    const position = integration * trace.eulerSteps;
    const state = Math.min(Math.floor(position), trace.eulerSteps - 1);
    const within = position - state;
    const from = pathState(trace, chunk, state);
    const to = pathState(trace, chunk, state + 1);
    for (let index = 0; index < from.length; index++) {
      blended[index] = lerp(from[index], to[index], within);
    }

    panel.draw({
      values: blended.subarray(0, from.length),
      // The trail is where the integration started; without an integration to
      // watch there is nothing for it to say.
      trail: mode === 'stepped' ? pathState(trace, chunk, 0) : null,
      steps: trace.steps,
      joints: trace.joints,
      actSteps: trace.actSteps,
      executingStep: phase === 'execute' && beat === 'run'
        ? Math.min(Math.floor(tickFloat - trace.chunkTicks[chunk]), trace.actSteps - 1)
        : -1,
      progress: integration,
      opacity,
      phase
    });

    for (const [key, chip] of Object.entries(ui.phases)) {
      chip.classList.toggle('is-active', key === phase);
    }
    ui.status.textContent =
      `horizon ${chunk + 1} / ${trace.chunks} · generated at tick ` +
      `${trace.chunkTicks[chunk]} · tick ${Math.floor(tickFloat)} of ${trace.frames - 1}`;
  };

  const renderPlayback = (): void => {
    ui.label.textContent = `Rollout ${traceIndex + 1} / ${traces.length}`;
    ui.playPauseButton.textContent = playing ? 'Pause' : 'Play';
    ui.playPauseButton.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} rollout`);
    ui.modeButton.textContent = mode === 'stepped' ? 'Run continuously' : 'Step through';
    ui.phaseRow.hidden = mode === 'continuous';
  };
  const setPlaying = (next: boolean): void => {
    playing = next;
    previousFrameTime = null;
    renderPlayback();
  };

  // Switching modes lands on the same horizon rather than restarting, so the
  // rollout carries on from where it was being watched.
  const setMode = (next: ScheduleMode): void => {
    const { chunk } = momentAt(schedules[mode][traceIndex], seconds);
    mode = next;
    seconds = schedules[mode][traceIndex]
      .find(segment => segment.chunk === chunk)?.start ?? 0;
    previousFrameTime = null;
    applyTime(seconds);
    renderPlayback();
  };

  applyTime(seconds);
  renderPlayback();

  const playPauseListener = (): void => { setPlaying(!playing); };
  const modeListener = (): void => {
    setMode(mode === 'stepped' ? 'continuous' : 'stepped');
  };
  ui.playPauseButton.addEventListener('click', playPauseListener);
  ui.modeButton.addEventListener('click', modeListener);

  const resizeObserver = new ResizeObserver(() => {
    vizScene.resize();
    panel.resize();
    cameraStrip.resize();
  });
  resizeObserver.observe(ui.viewport);
  resizeObserver.observe(ui.panelHost);
  resizeObserver.observe(ui.cameras);

  let animationFrameId = 0;
  let destroyed = false;
  function animate(time: number): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    if (playing) {
      if (previousFrameTime !== null) {
        seconds += (time - previousFrameTime) / 1000;
        if (seconds >= durations[mode][traceIndex]) {
          traceIndex = (traceIndex + 1) % traces.length;
          seconds = 0;
        }
        applyTime(seconds);
        renderPlayback();
      }
      previousFrameTime = time;
    }
    vizScene.orbitControls.update();
    vizScene.renderer.render(vizScene.scene, vizScene.camera);
    cameraStrip.render(vizScene.scene);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      ui.playPauseButton.removeEventListener('click', playPauseListener);
      ui.modeButton.removeEventListener('click', modeListener);
      panel.destroy();
      cameraStrip.destroy();
      vizScene.destroy();
      ui.root.remove();
    }
  };
}
