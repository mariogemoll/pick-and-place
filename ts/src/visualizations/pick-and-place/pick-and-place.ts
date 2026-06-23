// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { ARM_JOINT_NAMES, deriveSo101Kinematics } from '../../ik/kinematics';
import {
  anyYawCubeCenterBand,
  computeSimpleWorkspaceForCubeZ,
  CUBE_Z_1CM_OVER_GROUND_TOP,
  sectorBoundingBox,
  type SimpleWorkspaceSector
} from '../../ik/workspace';
import { loadWebModel } from '../../web-model';
import {
  CUBE_HALF_SIZE,
  type CubePose
} from '../grasp-pose-shared/body-factories';
import { createXyMultiDragControls } from '../xy-drag-controls';
import { createCarryProfilePlot } from './carry-profile-plot';
import { createPickAndPlaceScene } from './scene';
import {
  computeTrajectory,
  NEUTRAL_FRAME,
  REST_FRAME,
  type Trajectory
} from './trajectory';
import {
  buildUi,
  type PickAndPlaceCubeInputs
} from './ui';

export interface PickAndPlaceVisualization {
  destroy(): void;
}

export interface PickAndPlaceOptions {
  modelBasePath?: string;
  modelUrl?: string;
  initialJointPositions?: Readonly<Record<string, number>>;
  sourcePosition?: Readonly<{ x: number; y: number; yaw?: number }>;
  targetPosition?: Readonly<{ x: number; y: number; yaw?: number }>;
  // Build the carry height profile charts and sample the carry profile data.
  includeCarryProfilePlots?: boolean;
  // Wrap the existing neutral-to-neutral motion with rest-to-neutral and
  // neutral-to-rest phases.
  startFromAndReturnToRestPose?: boolean;
}

const DEFAULT_SOURCE = { x: 0.2, y: -0.08, yaw: 0 };
const DEFAULT_TARGET = { x: 0.2, y: 0.08, yaw: 0 };

type PickAndPlaceStage = 'setup' | 'run';

function formatPlaybackTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = (seconds % 60).toFixed(1).padStart(4, '0');
  return `${minutes}:${remainingSeconds}`;
}

function fixedCubePose(position: Readonly<{
  x: number;
  y: number;
  yaw?: number;
}>): CubePose {
  return {
    x: position.x,
    y: position.y,
    z: CUBE_HALF_SIZE,
    roll: 0,
    pitch: 0,
    yaw: position.yaw ?? 0
  };
}

function clampToWorkspace(
  x: number,
  y: number,
  workspace: SimpleWorkspaceSector
): { x: number; y: number } {
  const band = anyYawCubeCenterBand(workspace);
  const dx = x - workspace.panAxis.x;
  const dy = y - workspace.panAxis.y;
  const radius = Math.min(band.max, Math.max(band.min, Math.hypot(dx, dy)));
  const azimuth = Math.min(
    workspace.azimuth.max,
    Math.max(workspace.azimuth.min, Math.atan2(dy, dx))
  );
  return {
    x: workspace.panAxis.x + radius * Math.cos(azimuth),
    y: workspace.panAxis.y + radius * Math.sin(azimuth)
  };
}

export async function PickAndPlace(
  parent: HTMLElement,
  options: PickAndPlaceOptions = {}
): Promise<PickAndPlaceVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const kinematics = deriveSo101Kinematics(model);
  const workspace = computeSimpleWorkspaceForCubeZ(
    kinematics, CUBE_Z_1CM_OVER_GROUND_TOP
  );
  const band = anyYawCubeCenterBand(workspace);
  const bbox = sectorBoundingBox(workspace);
  const sourceStart = options.sourcePosition ?? DEFAULT_SOURCE;
  const targetStart = options.targetPosition ?? DEFAULT_TARGET;
  const initialSource = fixedCubePose({
    ...sourceStart,
    ...clampToWorkspace(sourceStart.x, sourceStart.y, workspace)
  });
  const initialTarget = fixedCubePose({
    ...targetStart,
    ...clampToWorkspace(targetStart.x, targetStart.y, workspace)
  });
  const ui = buildUi(parent, {
    xRange: {
      min: Math.floor(bbox.x.min * 1000),
      max: Math.ceil(bbox.x.max * 1000)
    },
    yRange: {
      min: Math.floor(bbox.y.min * 1000),
      max: Math.ceil(bbox.y.max * 1000)
    },
    source: {
      x: Math.round(initialSource.x * 1000),
      y: Math.round(initialSource.y * 1000),
      yaw: initialSource.yaw * 180 / Math.PI
    },
    target: {
      x: Math.round(initialTarget.x * 1000),
      y: Math.round(initialTarget.y * 1000),
      yaw: initialTarget.yaw * 180 / Math.PI
    }
  });
  const vizScene = createPickAndPlaceScene(
    ui.viewport,
    model,
    options.modelBasePath,
    {
      center: workspace.panAxis,
      innerRadius: band.min,
      outerRadius: band.max,
      thetaStart: workspace.azimuth.min,
      thetaLength: workspace.azimuth.max - workspace.azimuth.min
    }
  );
  const includeCarryProfilePlots = options.includeCarryProfilePlots ?? false;
  const profilePlots = includeCarryProfilePlots
    ? [
      createCarryProfilePlot('time'),
      createCarryProfilePlot('distance')
    ]
    : [];
  if (includeCarryProfilePlots) {
    for (const profilePlot of profilePlots) {
      ui.viewport.appendChild(profilePlot.element);
    }
  }

  let sourcePose = { ...initialSource };
  let targetPose = { ...initialTarget };

  for (const [name, radians] of Object.entries(options.initialJointPositions ?? {})) {
    vizScene.setJoint(name, radians);
  }

  const renderCubePoses = (): void => {
    vizScene.updateSourceCube(sourcePose);
    vizScene.updateTargetCube(targetPose);
  };

  let syncingInputs = false;
  const syncInputs = (inputs: PickAndPlaceCubeInputs, pose: CubePose): void => {
    syncingInputs = true;
    const values = [
      [inputs.xInput, Math.round(pose.x * 1000)],
      [inputs.yInput, Math.round(pose.y * 1000)],
      [inputs.yawInput, pose.yaw * 180 / Math.PI]
    ] as const;
    for (const [input, value] of values) {
      input.value = String(value);
      input.dispatchEvent(new Event('input'));
    }
    syncingInputs = false;
  };
  const applyInputs = (
    inputs: PickAndPlaceCubeInputs,
    updatePose: (pose: CubePose) => void,
    currentPose: () => CubePose
  ): void => {
    if (syncingInputs) { return; }
    const xy = clampToWorkspace(
      Number(inputs.xInput.value) / 1000,
      Number(inputs.yInput.value) / 1000,
      workspace
    );
    const pose = {
      ...currentPose(),
      ...xy,
      yaw: Number(inputs.yawInput.value) * Math.PI / 180
    };
    updatePose(pose);
    syncInputs(inputs, pose);
    renderCubePoses();
  };
  const sourceInputListener = (): void => {
    applyInputs(ui.sourceInputs, pose => { sourcePose = pose; }, () => sourcePose);
  };
  const targetInputListener = (): void => {
    applyInputs(ui.targetInputs, pose => { targetPose = pose; }, () => targetPose);
  };
  const cubeInputs = (inputs: PickAndPlaceCubeInputs): HTMLInputElement[] => [
    inputs.xInput, inputs.yInput, inputs.yawInput
  ];
  for (const input of cubeInputs(ui.sourceInputs)) {
    input.addEventListener('input', sourceInputListener);
  }
  for (const input of cubeInputs(ui.targetInputs)) {
    input.addEventListener('input', targetInputListener);
  }

  const dragControls = createXyMultiDragControls({
    camera: vizScene.camera,
    domElement: vizScene.renderer.domElement,
    orbitControls: vizScene.orbitControls,
    targets: [
      {
        object: vizScene.sourceCube,
        onDrag(x, y): void {
          sourcePose = { ...sourcePose, ...clampToWorkspace(x, y, workspace) };
          syncInputs(ui.sourceInputs, sourcePose);
          renderCubePoses();
        }
      },
      {
        object: vizScene.targetCube,
        onDrag(x, y): void {
          targetPose = { ...targetPose, ...clampToWorkspace(x, y, workspace) };
          syncInputs(ui.targetInputs, targetPose);
          renderCubePoses();
        }
      }
    ]
  });

  let stage: PickAndPlaceStage = 'setup';
  let trajectory: Trajectory | null = null;
  let playbackSeconds = 0;
  let playing = false;
  let previousFrameTime: number | null = null;

  const applyFrame = (t: number): void => {
    if (trajectory === null) { return; }
    const frame = trajectory.evaluate(t);
    for (const name of ARM_JOINT_NAMES) {
      vizScene.setJoint(name, frame.joints[name]);
    }
    vizScene.setJoint('gripper', frame.gripper);
    vizScene.updateSourceCube(frame.sourceCube);
    if (includeCarryProfilePlots) {
      for (const profilePlot of profilePlots) {
        profilePlot.setMarker(trajectory.carryFraction(t));
      }
    }
  };
  const resetFrame = (): void => {
    const setupFrame =
      options.startFromAndReturnToRestPose === true ? REST_FRAME : NEUTRAL_FRAME;
    for (const name of ARM_JOINT_NAMES) {
      vizScene.setJoint(name, setupFrame.joints[name]);
    }
    vizScene.setJoint('gripper', setupFrame.gripper);
    vizScene.updateSourceCube(sourcePose);
  };

  const renderPlayback = (): void => {
    ui.seekInput.value = String(playbackSeconds);
    ui.playbackTime.textContent =
      `${formatPlaybackTime(playbackSeconds)} / ` +
      formatPlaybackTime(trajectory?.duration ?? 0);
    ui.playPauseButton.textContent = playing ? 'Pause' : 'Play';
    ui.playPauseButton.setAttribute(
      'aria-label', playing ? 'Pause trajectory' : 'Play trajectory'
    );
  };
  const setPlaying = (nextPlaying: boolean): void => {
    playing = nextPlaying;
    previousFrameTime = null;
    renderPlayback();
  };
  const setStage = (nextStage: PickAndPlaceStage): void => {
    stage = nextStage;
    const isSetup = stage === 'setup';
    ui.root.classList.toggle('is-running', !isSetup);
    ui.controls.hidden = !isSetup;
    ui.setupActions.hidden = !isSetup;
    ui.setupControls.hidden = !isSetup;
    ui.runControls.hidden = isSetup;
    for (const input of [
      ...cubeInputs(ui.sourceInputs), ...cubeInputs(ui.targetInputs)
    ]) {
      input.disabled = !isSetup;
    }
    dragControls.setEnabled(isSetup);
    if (isSetup) {
      trajectory = null;
      playbackSeconds = 0;
      setPlaying(false);
      resetFrame();
      for (const profilePlot of profilePlots) {
        profilePlot.element.hidden = true;
      }
    } else {
      trajectory = computeTrajectory(kinematics, sourcePose, targetPose, {
        startFromAndReturnToRestPose: options.startFromAndReturnToRestPose
      });
      if (trajectory === null) { setStage('setup'); return; }
      ui.seekInput.max = String(trajectory.duration);
      playbackSeconds = 0;
      applyFrame(playbackSeconds);
      if (includeCarryProfilePlots) {
        const profile = trajectory.carryProfile();
        for (const profilePlot of profilePlots) {
          profilePlot.setProfile(profile);
          profilePlot.setMarker(trajectory.carryFraction(0));
          profilePlot.element.hidden = false;
        }
      }
      setPlaying(true);
    }
  };
  const runListener = (): void => { setStage('run'); };
  const cancelListener = (): void => { setStage('setup'); };
  const playPauseListener = (): void => {
    if (trajectory !== null && playbackSeconds >= trajectory.duration) {
      playbackSeconds = 0;
    }
    setPlaying(!playing);
  };
  const seekListener = (): void => {
    playbackSeconds = Number(ui.seekInput.value);
    previousFrameTime = null;
    applyFrame(playbackSeconds);
    renderPlayback();
  };
  ui.runButton.addEventListener('click', runListener);
  ui.cancelButton.addEventListener('click', cancelListener);
  ui.playPauseButton.addEventListener('click', playPauseListener);
  ui.seekInput.addEventListener('input', seekListener);
  renderPlayback();

  const resetListener = (): void => {
    sourcePose = { ...initialSource };
    targetPose = { ...initialTarget };
    syncInputs(ui.sourceInputs, sourcePose);
    syncInputs(ui.targetInputs, targetPose);
    renderCubePoses();
  };
  ui.resetButton.addEventListener('click', resetListener);

  const resizeObserver = new ResizeObserver(() => {
    vizScene.resize();
    for (const profilePlot of profilePlots) { profilePlot.resize(); }
  });
  resizeObserver.observe(ui.viewport);
  renderCubePoses();
  if (options.startFromAndReturnToRestPose === true) { resetFrame(); }

  let animationFrameId = 0;
  let destroyed = false;
  function animate(time: number): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    if (stage === 'run' && playing && trajectory !== null) {
      if (previousFrameTime !== null) {
        playbackSeconds = Math.min(
          trajectory.duration,
          playbackSeconds + (time - previousFrameTime) / 1000
        );
        applyFrame(playbackSeconds);
        renderPlayback();
        if (playbackSeconds >= trajectory.duration) {
          setPlaying(false);
        }
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
      dragControls.destroy();
      for (const input of cubeInputs(ui.sourceInputs)) {
        input.removeEventListener('input', sourceInputListener);
      }
      for (const input of cubeInputs(ui.targetInputs)) {
        input.removeEventListener('input', targetInputListener);
      }
      ui.resetButton.removeEventListener('click', resetListener);
      ui.runButton.removeEventListener('click', runListener);
      ui.cancelButton.removeEventListener('click', cancelListener);
      ui.playPauseButton.removeEventListener('click', playPauseListener);
      ui.seekInput.removeEventListener('input', seekListener);
      for (const profilePlot of profilePlots) { profilePlot.destroy(); }
      vizScene.destroy();
      ui.root.remove();
    }
  };
}
