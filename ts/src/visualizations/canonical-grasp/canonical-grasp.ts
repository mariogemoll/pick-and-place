// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  type CanonicalGraspChoice,
  MAX_CANONICAL_AZIMUTH,
  MAX_CANONICAL_GRASP_RADIUS,
  MIN_CANONICAL_AZIMUTH,
  selectCanonicalGrasp
} from '../../ik/canonical-grasp';
import {
  ARM_JOINT_NAMES,
  deriveSo101Kinematics,
  NEUTRAL_ARM_JOINTS
} from '../../ik/kinematics';
import type { SimpleIkBranch } from '../../ik/simple-ik';
import {
  computeGlobalXyWorkspace,
  sectorBoundingBox
} from '../../ik/workspace';
import { loadWebModel } from '../../web-model';
import {
  type CubePose,
  DEFAULT_CUBE_POSE
} from '../grasp-pose-shared/body-factories';
import { robotModelWithBaseOnFloor } from '../robot-model';
import { buildWorkspaceOverlaySpecs } from '../workspace-overlay';
import { createXyDragControls } from '../xy-drag-controls';
import { createCanonicalGraspScene } from './scene';
import {
  buildUi,
  DEFAULT_CUBE_X,
  DEFAULT_CUBE_Y,
  DROP_POSE_Z_MM
} from './ui';

// Conservative floor-pick band for the canonical pick-lift motion.
const MIN_GRASP_RADIUS = 0.110;

// Cube-center height swept in "drop mode" — the held cube's orientation is a
// don't-care once grasped, so this only overrides z/yaw, never x/y.
const DROP_POSE_Z = DROP_POSE_Z_MM / 1000;

export interface CanonicalGraspVisualization {
  destroy(): void;
}

export interface CanonicalGraspOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

export async function initializeCanonicalGraspVisualization(
  parent: HTMLElement,
  options: CanonicalGraspOptions = {}
): Promise<CanonicalGraspVisualization> {
  const model = robotModelWithBaseOnFloor(await loadWebModel(options.modelUrl));
  const kinematics = deriveSo101Kinematics(model);

  // Full floor reach (max arm reach projected onto the floor) drives the
  // placement slider ranges. faceOffset is zeroed so the radial band is the raw
  // reach rather than the any-yaw-graspable centre band, then capped to the
  // radius/yaw-valid canonical grasp boundary.
  const workspace = { ...computeGlobalXyWorkspace(kinematics), faceOffset: 0 };
  const maxGraspRadius = Math.min(
    workspace.radial.max,
    MAX_CANONICAL_GRASP_RADIUS
  );
  const minGraspRadius = MIN_GRASP_RADIUS;
  const minAzimuth = Math.max(workspace.azimuth.min, MIN_CANONICAL_AZIMUTH);
  const maxAzimuth = Math.min(workspace.azimuth.max, MAX_CANONICAL_AZIMUTH);
  const canonicalWorkspace = {
    ...workspace,
    radial: { ...workspace.radial, min: minGraspRadius, max: maxGraspRadius },
    azimuth: { min: minAzimuth, max: maxAzimuth }
  };
  const bbox = sectorBoundingBox(canonicalWorkspace);
  const panX = workspace.panAxis.x;
  const panY = workspace.panAxis.y;
  // Radial coordinates are measured from the pan axis (the sector's center).
  const radialFromCartesian = (x: number, y: number): {
    radiusMm: number; azimuthDeg: number;
  } => ({
    radiusMm: Math.hypot(x - panX, y - panY) * 1000,
    azimuthDeg: (Math.atan2(y - panY, x - panX) * 180) / Math.PI
  });
  const cartesianFromRadial = (radiusMm: number, azimuthDeg: number): {
    x: number; y: number;
  } => {
    const radius = radiusMm / 1000;
    const azimuth = (azimuthDeg * Math.PI) / 180;
    return {
      x: panX + radius * Math.cos(azimuth),
      y: panY + radius * Math.sin(azimuth)
    };
  };
  const clampCartesianToReach = (x: number, y: number): { x: number; y: number } => {
    const radial = radialFromCartesian(x, y);
    const radius = radial.radiusMm / 1000;
    const clampedRadius = THREE.MathUtils.clamp(
      radius,
      minGraspRadius,
      maxGraspRadius
    );
    const clampedAzimuthDeg = THREE.MathUtils.clamp(
      radial.azimuthDeg,
      (minAzimuth * 180) / Math.PI,
      (maxAzimuth * 180) / Math.PI
    );
    if (
      Math.abs(clampedRadius - radius) < 1e-9 &&
      Math.abs(clampedAzimuthDeg - radial.azimuthDeg) < 1e-9
    ) {
      return { x, y };
    }
    return cartesianFromRadial(clampedRadius * 1000, clampedAzimuthDeg);
  };
  const defaultRadial = radialFromCartesian(DEFAULT_CUBE_X, DEFAULT_CUBE_Y);
  const ui = buildUi(parent, {
    xRange: {
      min: Math.floor(bbox.x.min * 1000),
      max: Math.ceil(bbox.x.max * 1000)
    },
    yRange: {
      min: Math.floor(bbox.y.min * 1000),
      max: Math.ceil(bbox.y.max * 1000)
    },
    radiusRange: {
      min: Math.max(
        Math.round(minGraspRadius * 1000),
        Math.floor(canonicalWorkspace.radial.min * 1000)
      ),
      max: Math.floor(canonicalWorkspace.radial.max * 1000)
    },
    azimuthRange: {
      min: Math.ceil((canonicalWorkspace.azimuth.min * 180) / Math.PI),
      max: Math.floor((canonicalWorkspace.azimuth.max * 180) / Math.PI)
    },
    radiusDefault: Math.round(defaultRadial.radiusMm),
    azimuthDefault: Math.round(defaultRadial.azimuthDeg)
  });
  const vizScene = await createCanonicalGraspScene(
    ui.viewport, model, options.modelBasePath,
    buildWorkspaceOverlaySpecs(kinematics)
  );

  // The cube always rests flat on the ground; only its X/Y and yaw vary.
  let currentPose: CubePose = {
    ...DEFAULT_CUBE_POSE, x: DEFAULT_CUBE_X, y: DEFAULT_CUBE_Y
  };
  // Cube yaw measured from the radial direction (the slider value). The cube's
  // world yaw is this plus the azimuth, so the grasp geometry — and thus the
  // camera-down bands — stays the same at every azimuth.
  let yawFromRadius = 0;
  let showPregrasp = false;
  let dropMode = false;

  function applyBranch(branch: SimpleIkBranch): void {
    for (const name of ARM_JOINT_NAMES) {
      vizScene.setJoint(name, branch.joints[name]);
    }
  }

  function restToNeutral(): void {
    for (const name of ARM_JOINT_NAMES) {
      vizScene.setJoint(name, NEUTRAL_ARM_JOINTS[name]);
    }
  }

  function selectedBranch(choice: CanonicalGraspChoice): SimpleIkBranch {
    return {
      elbow: choice.elbow,
      joints: showPregrasp ? choice.hoverJoints : choice.graspJoints
    };
  }

  function updateScene(): void {
    // World yaw keeps the cube at a constant offset from the radius as it moves.
    const azimuth = Math.atan2(currentPose.y - panY, currentPose.x - panX);
    currentPose = { ...currentPose, yaw: yawFromRadius + azimuth };
    vizScene.updateCubePose(currentPose);

    // In drop mode, orientation is a don't-care (the cube is held rigidly once
    // grasped), so only x/y are swept; z/yaw are pinned to the drop height and 0.
    const searchPose: CubePose = dropMode
      ? { ...currentPose, z: DROP_POSE_Z, yaw: 0 }
      : currentPose;
    const solution = selectCanonicalGrasp(kinematics, searchPose);
    if (solution === null) {
      vizScene.updateGhostGraspPose(null);
      restToNeutral();
      return;
    }

    if (showPregrasp && !dropMode) {
      vizScene.updateGhostGraspPose(solution.graspMatrix);
    } else {
      vizScene.updateGhostGraspPose(null);
    }

    applyBranch(selectedBranch(solution));
  }

  const yawListener = (): void => {
    yawFromRadius = (Number(ui.yawInput.value) * Math.PI) / 180;
    updateScene();
  };
  ui.yawInput.addEventListener('input', yawListener);

  const pregraspListener = (): void => {
    showPregrasp = ui.showPregraspInput.checked;
    updateScene();
  };
  ui.showPregraspInput.addEventListener('change', pregraspListener);

  const dropModeListener = (): void => {
    dropMode = ui.dropModeInput.checked;
    // Yaw and pregrasp are meaningless once orientation is a don't-care.
    ui.yawInput.disabled = dropMode;
    ui.showPregraspInput.disabled = dropMode;
    updateScene();
  };
  ui.dropModeInput.addEventListener('change', dropModeListener);

  // X/Y and radius/azimuth drive the same cube center; keep both in sync so a
  // mode switch is seamless. The guard stops the programmatic value updates
  // (which fire 'input' to refresh the slider labels) from recursing.
  let syncing = false;
  const setSlider = (input: HTMLInputElement, value: number): void => {
    input.value = String(value);
    input.dispatchEvent(new Event('input'));
  };
  const applyCartesian = (): void => {
    if (syncing) { return; }
    const rawX = Number(ui.xInput.value) / 1000;
    const rawY = Number(ui.yInput.value) / 1000;
    const { x, y } = clampCartesianToReach(rawX, rawY);
    currentPose = { ...currentPose, x, y };
    const radial = radialFromCartesian(x, y);
    syncing = true;
    setSlider(ui.xInput, Math.round(x * 1000));
    setSlider(ui.yInput, Math.round(y * 1000));
    setSlider(ui.radiusInput, Math.round(radial.radiusMm));
    setSlider(ui.azimuthInput, Math.round(radial.azimuthDeg * 10) / 10);
    syncing = false;
    updateScene();
  };
  const applyRadial = (): void => {
    if (syncing) { return; }
    const { x, y } = cartesianFromRadial(
      Number(ui.radiusInput.value), Number(ui.azimuthInput.value)
    );
    currentPose = { ...currentPose, x, y };
    syncing = true;
    setSlider(ui.xInput, Math.round(x * 1000));
    setSlider(ui.yInput, Math.round(y * 1000));
    syncing = false;
    updateScene();
  };
  ui.xInput.addEventListener('input', applyCartesian);
  ui.yInput.addEventListener('input', applyCartesian);
  ui.radiusInput.addEventListener('input', applyRadial);
  ui.azimuthInput.addEventListener('input', applyRadial);

  const clampToInput = (input: HTMLInputElement, value: number): number =>
    Math.min(Number(input.max), Math.max(Number(input.min), value));
  const dragControls = createXyDragControls({
    camera: vizScene.camera,
    domElement: vizScene.renderer.domElement,
    object: vizScene.cube,
    orbitControls: vizScene.orbitControls,
    onDrag(x, y): void {
      ui.xInput.value = String(Math.round(clampToInput(ui.xInput, x * 1000)));
      ui.yInput.value = String(Math.round(clampToInput(ui.yInput, y * 1000)));
      ui.xInput.dispatchEvent(new Event('input'));
      ui.yInput.dispatchEvent(new Event('input'));
    }
  });

  const coordModeListeners = ui.coordModeInputs.map(input => {
    const listener = (): void => {
      if (!input.checked) { return; }
      const radial = input.value === 'radial';
      ui.cartesianGroup.style.display = radial ? 'none' : '';
      ui.radialGroup.style.display = radial ? '' : 'none';
    };
    input.addEventListener('change', listener);
    return listener;
  });

  const resetListener = (): void => {
    currentPose = {
      ...DEFAULT_CUBE_POSE, x: DEFAULT_CUBE_X, y: DEFAULT_CUBE_Y
    };
    showPregrasp = false;
    ui.showPregraspInput.checked = false;
    ui.showPregraspInput.disabled = false;
    dropMode = false;
    ui.dropModeInput.checked = false;
    ui.yawInput.disabled = false;
    for (const input of ui.coordModeInputs) {
      input.checked = input.value === 'radial';
      input.dispatchEvent(new Event('change'));
    }
    // Setting X/Y drives the radius/azimuth sliders via applyCartesian.
    ui.xInput.value = String(Math.round(DEFAULT_CUBE_X * 1000));
    ui.yInput.value = String(Math.round(DEFAULT_CUBE_Y * 1000));
    ui.xInput.dispatchEvent(new Event('input'));
    ui.yInput.dispatchEvent(new Event('input'));
    ui.yawInput.value = '0';
    ui.yawInput.dispatchEvent(new Event('input'));
  };
  ui.resetButton.addEventListener('click', resetListener);

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  updateScene();

  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
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
      vizScene.destroy();
      ui.yawInput.removeEventListener('input', yawListener);
      ui.showPregraspInput.removeEventListener('change', pregraspListener);
      ui.dropModeInput.removeEventListener('change', dropModeListener);
      ui.xInput.removeEventListener('input', applyCartesian);
      ui.yInput.removeEventListener('input', applyCartesian);
      ui.radiusInput.removeEventListener('input', applyRadial);
      ui.azimuthInput.removeEventListener('input', applyRadial);
      for (const [index, input] of ui.coordModeInputs.entries()) {
        input.removeEventListener('change', coordModeListeners[index]);
      }
      ui.resetButton.removeEventListener('click', resetListener);
      ui.root.remove();
    }
  };
}
