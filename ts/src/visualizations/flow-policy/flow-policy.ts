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
//
// Executing a horizon scrolls it off the side of the panel, which leaves the
// last two commands the arm carried out sitting where the next horizon will be
// predicted from -- so the cycles run into each other instead of the panel
// being cleared between them.

import * as THREE from 'three';

import { ARM_JOINT_NAMES } from '../../ik/kinematics';
import { cameraByName, createWebCamera, loadWebModel, type WebModel } from '../../web-model';
import { createEpisodeReplayScene } from '../episode-replay/scene';
import { createCameraStrip } from './camera-strip';
import { createFlowPanel, type FlowPanel, HISTORY_COLUMNS } from './flow-panel';
import {
  buildSchedule,
  momentAt,
  scheduleDuration,
  type ScheduleMode,
  type Segment
} from './schedule';
import {
  executedSteps,
  executedTail,
  type FlowTrace,
  frame,
  parseFlowTrace,
  pathState
} from './trace';
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
//
// Squared up to its mount rather than posed as the manifest has it. What the
// manifest carries is what calibration measured on the real rig, a degree or two
// of tilt and yaw included, and the dataset is rendered through exactly that.
// These views are for watching, and off-axis by a degree reads as a mistake, so
// they look straight down their mount's axis. The mount's own angle stays: that
// is where the camera is actually bolted, not a calibration residual.
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
  camera.quaternion.identity();
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

  // Reused so the animation does not allocate a horizon per rendered frame.
  // Only the steps that get executed are ever drawn, and the largest of those
  // in case the rollouts came from different runs.
  const blended = new Float32Array(
    Math.max(...traces.map(trace => trace.actSteps * trace.joints))
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
    const { chunk, phase, progress, tickFloat } =
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
    // Only the steps that get executed are shown, so only those are blended.
    const shown = trace.actSteps * trace.joints;
    const from = pathState(trace, chunk, state);
    const to = pathState(trace, chunk, state + 1);
    for (let index = 0; index < shown; index++) {
      blended[index] = lerp(from[index], to[index], within);
    }

    panel.draw({
      history: executedTail(trace, chunk - 1, HISTORY_COLUMNS),
      values: blended.subarray(0, shown),
      // The trail is where the integration started; once the horizon is being
      // executed that is behind it, and a trail on a scrolling column would
      // only smear.
      trail: mode === 'stepped' && phase !== 'execute'
        ? pathState(trace, chunk, 0).subarray(0, shown)
        : null,
      joints: trace.joints,
      actSteps: trace.actSteps,
      // Executing the horizon is what scrolls it: one column per step carried
      // out, which leaves the last two executed sitting left of the seam.
      slide: phase === 'execute' ? executedSteps(trace, chunk) * progress : 0,
      progress: integration
    });

    ui.status.textContent =
      `horizon ${chunk + 1}/${trace.chunks} · tick ` +
      `${Math.floor(tickFloat)}/${trace.frames - 1}`;
  };

  const renderPlayback = (): void => {
    ui.label.textContent = `Rollout ${traceIndex + 1} / ${traces.length}`;
    ui.playPauseButton.textContent = playing ? 'Pause' : 'Play';
    ui.playPauseButton.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} rollout`);
    ui.modeButton.textContent = mode === 'stepped' ? 'Run continuously' : 'Step through';
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
