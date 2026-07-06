// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { deriveSo101Kinematics, NEUTRAL_ARM_JOINTS } from '../../ik/kinematics';
import { loadWebModel } from '../../web-model';
import { robotModelWithBaseOnFloor } from '../robot-model';
import { buildWorkspaceOverlaySpecs } from '../workspace-overlay';
import { createRobotScene } from './scene';
import { buildUi, formatDegrees } from './ui';

type RobotPoseName = 'grasp' | 'rest' | 'neutral' | 'extended';

const ROBOT_POSES: Record<RobotPoseName, {
  label: string;
  joints: Partial<Record<string, number>>;
}> = {
  grasp: {
    label: 'Grasp',
    joints: {
      shoulder_pan: 0,
      shoulder_lift: -0.1829202616552561,
      elbow_flex: 0.6162301587559451,
      wrist_flex: 1.1374864296942075,
      wrist_roll: -1.6279467415829163,
      gripper: 40 * (Math.PI / 180)
    }
  },
  rest: {
    label: 'Rest',
    joints: {
      shoulder_pan: 0,
      shoulder_lift: -100 * (Math.PI / 180),
      elbow_flex: Math.PI / 2,
      wrist_flex: 1.2865569914701056,
      wrist_roll: -1.509038522493559,
      gripper: 0
    }
  },
  neutral: {
    label: 'Neutral',
    joints: {
      ...NEUTRAL_ARM_JOINTS,
      gripper: 0
    }
  },
  extended: {
    label: 'Extended',
    joints: {
      ...NEUTRAL_ARM_JOINTS,
      shoulder_lift: Math.PI / 2,
      elbow_flex: -Math.PI / 2,
      gripper: 0
    }
  }
};

const POSE_ANIMATION_DURATION_MS = 1600;

interface PoseTransition {
  durationMs: number;
  startedAt: number;
  starts: Map<string, number>;
  targets: Map<string, number>;
}

export interface RobotVisualization {
  destroy(): void;
}

export interface RobotVisualizationOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

function rgbToHex(r: number, g: number, b: number): string {
  const hex = [r, g, b].map(component =>
    Math.round(Math.min(1, Math.max(0, component)) * 255)
      .toString(16)
      .padStart(2, '0')
  ).join('');
  return `#${hex}`;
}

function easeInOutCosine(t: number): number {
  return (1 - Math.cos(Math.PI * t)) / 2;
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function isRobotPoseName(poseName: string): poseName is RobotPoseName {
  return poseName === 'grasp' || poseName === 'rest' || poseName === 'neutral' ||
    poseName === 'extended';
}

export async function initializeRobotVisualization(
  parent: HTMLElement,
  options: RobotVisualizationOptions = {}
): Promise<RobotVisualization> {
  const model = robotModelWithBaseOnFloor(await loadWebModel(options.modelUrl));
  const initialJointValues = ROBOT_POSES.grasp.joints;
  const joints = model.bodies.flatMap(body => body.joints).flatMap(joint => {
    if (joint.type !== 'hinge' || joint.range === undefined) { return []; }
    return [{
      name: joint.name,
      label: capitalize(joint.name.replaceAll('_', ' ')),
      lower: joint.range[0],
      upper: joint.range[1],
      value: initialJointValues[joint.name] ?? 0
    }];
  });

  const visualMaterialNames = new Set(
    model.bodies.flatMap(b => b.geometries)
      .filter(g => g.role === 'visual')
      .map(g => g.material)
  );
  const materialColors = Object.entries(model.materials)
    .filter(([name]) => visualMaterialNames.has(name))
    .map(([name, [r, g, b]]) => ({
      name,
      label: capitalize(name),
      hexColor: rgbToHex(r, g, b)
    }));

  const kinematics = deriveSo101Kinematics(model);
  const ui = buildUi(parent, joints, materialColors, Object.entries(ROBOT_POSES).map(
    ([name, pose]) => ({ name, label: pose.label })
  ));
  const vizScene = createRobotScene(
    ui.viewport, model, options.modelBasePath,
    buildWorkspaceOverlaySpecs(kinematics).slice(0, 1)
  );
  const { renderer, camera, scene, orbitControls } = vizScene;
  const listeners: (() => void)[] = [];
  const currentJointValues = new Map(joints.map(joint => [joint.name, joint.value]));
  let poseTransition: PoseTransition | undefined;

  const applyJointValue = (jointName: string, value: number): void => {
    const control = ui.controls.get(jointName);
    if (!control) {return;}
    currentJointValues.set(jointName, value);
    control.input.value = String(value);
    control.value.textContent = formatDegrees(value);
    vizScene.setJoint(jointName, value);
  };

  for (const joint of joints) {
    const control = ui.controls.get(joint.name);
    if (!control) {continue;}
    const update = (): void => {
      poseTransition = undefined;
      applyJointValue(joint.name, Number(control.input.value));
    };
    control.input.addEventListener('input', update);
    listeners.push(() => { control.input.removeEventListener('input', update); });
  }

  for (const [matName, colorInput] of ui.colorInputs) {
    const update = (): void => {
      vizScene.setMaterialColor(matName, new THREE.Color(colorInput.value));
    };
    colorInput.addEventListener('input', update);
    listeners.push(() => { colorInput.removeEventListener('input', update); });
    update();
  }

  const updateExtentColor = (): void => {
    vizScene.setOverlayColor(0, new THREE.Color(ui.extentColorInput.value));
  };
  ui.extentColorInput.addEventListener('input', updateExtentColor);
  listeners.push(() => { ui.extentColorInput.removeEventListener('input', updateExtentColor); });
  updateExtentColor();

  const updateExtentVisible = (): void => {
    vizScene.setOverlayVisible(0, ui.extentVisibleInput.checked);
  };
  ui.extentVisibleInput.addEventListener('change', updateExtentVisible);
  listeners.push(() => {
    ui.extentVisibleInput.removeEventListener('change', updateExtentVisible);
  });
  updateExtentVisible();

  const updateBackgroundColor = (): void => {
    vizScene.setBackgroundColor(new THREE.Color(ui.backgroundColorInput.value));
  };
  ui.backgroundColorInput.addEventListener('input', updateBackgroundColor);
  listeners.push(() => {
    ui.backgroundColorInput.removeEventListener('input', updateBackgroundColor);
  });
  updateBackgroundColor();

  const applyPoseImmediately = (poseName: RobotPoseName): void => {
    const pose = ROBOT_POSES[poseName];
    for (const joint of joints) {
      applyJointValue(joint.name, pose.joints[joint.name] ?? 0);
    }
  };
  const animateToPose = (poseName: string): void => {
    if (!isRobotPoseName(poseName)) { return; }
    const pose = ROBOT_POSES[poseName];
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      poseTransition = undefined;
      applyPoseImmediately(poseName);
      return;
    }

    poseTransition = {
      durationMs: POSE_ANIMATION_DURATION_MS,
      startedAt: performance.now(),
      starts: new Map(joints.map(joint => [
        joint.name, currentJointValues.get(joint.name) ?? 0
      ])),
      targets: new Map(joints.map(joint => [
        joint.name, pose.joints[joint.name] ?? 0
      ]))
    };
  };
  for (const [poseName, button] of ui.poseButtons) {
    const moveToPose = (): void => { animateToPose(poseName); };
    button.addEventListener('click', moveToPose);
    listeners.push(() => { button.removeEventListener('click', moveToPose); });
  }
  applyPoseImmediately('grasp');

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  let animationFrameId = 0;
  let destroyed = false;
  function updatePoseTransition(now: number): void {
    if (poseTransition === undefined) {return;}
    const progress = Math.min(
      1,
      (now - poseTransition.startedAt) / poseTransition.durationMs
    );
    const easedProgress = easeInOutCosine(progress);

    for (const joint of joints) {
      const start = poseTransition.starts.get(joint.name) ?? 0;
      const target = poseTransition.targets.get(joint.name) ?? 0;
      applyJointValue(joint.name, start + (target - start) * easedProgress);
    }
    if (progress >= 1) {
      poseTransition = undefined;
    }
  }

  function animate(now: number): void {
    if (destroyed) {return;}
    animationFrameId = window.requestAnimationFrame(animate);
    updatePoseTransition(now);
    orbitControls.update();
    renderer.render(scene, camera);
    vizScene.renderInsets(currentJointValues.get('shoulder_pan') ?? 0);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return Promise.resolve({
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      for (const removeListener of listeners) {removeListener();}
      vizScene.destroy();
      ui.root.remove();
    }
  });
}
