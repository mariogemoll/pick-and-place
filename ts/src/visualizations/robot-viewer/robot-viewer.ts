// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { loadWebModel, type WebModel } from '../../web-model';
import { robotModelWithBaseOnFloor } from '../robot-model';
import { createRobotViewerScene } from './scene';
import {
  addJointControls,
  buildPanel,
  CANVAS_HEIGHT,
  CANVAS_WIDTH,
  formatJointValue,
  type JointControlDefinition
} from './ui';

export interface RobotViewerConfig {
  label: string;
  modelUrl: string;
  modelBasePath: string;
  defaultJointDegrees?: Record<string, number>;
  defaultJointMillimeters?: Record<string, number>;
  // Mirrors the default camera's starting side across the robot's Y axis;
  // defaults to false (the original UR5e-derived viewing angle).
  mirrorCameraY?: boolean;
}

export interface RobotViewerVisualization {
  destroy(): void;
}

function jointsFromModel(
  model: WebModel,
  defaultJointDegrees: Record<string, number> | undefined,
  defaultJointMillimeters: Record<string, number> | undefined
): JointControlDefinition[] {
  return model.bodies.flatMap(body => body.joints).flatMap(joint => {
    if (
      (joint.type !== 'hinge' && joint.type !== 'slide') ||
      joint.range === undefined ||
      joint.mimic !== undefined
    ) {
      return [];
    }
    const [lower, upper] = joint.range;
    const defaultValue = joint.type === 'hinge'
      ? defaultJointDegrees?.[joint.name] !== undefined
        ? defaultJointDegrees[joint.name] * Math.PI / 180
        : Math.min(Math.max(0, lower), upper)
      : defaultJointMillimeters?.[joint.name] !== undefined
        ? defaultJointMillimeters[joint.name] / 1000
        : Math.min(Math.max(0, lower), upper);
    return [{
      name: joint.name,
      label: jointLabel(joint.name),
      type: joint.type,
      lower,
      upper,
      value: Math.min(Math.max(defaultValue, lower), upper)
    }];
  });
}

interface DependentJoint {
  name: string;
  multiplier: number;
  offset: number;
}

// Joints an underactuated gripper's linkage follows but doesn't drive
// directly (see WebJointMimic); these move in lockstep with their primary
// joint instead of getting their own slider.
function mimicsByPrimaryJoint(model: WebModel): Map<string, DependentJoint[]> {
  const mimics = new Map<string, DependentJoint[]>();
  for (const body of model.bodies) {
    for (const joint of body.joints) {
      if (joint.mimic === undefined) { continue; }
      const forPrimary = mimics.get(joint.mimic.joint) ?? [];
      forPrimary.push({
        name: joint.name,
        multiplier: joint.mimic.multiplier,
        offset: joint.mimic.offset
      });
      mimics.set(joint.mimic.joint, forPrimary);
    }
  }
  return mimics;
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// Turns a MuJoCo joint name into a display label, dropping the "joint" noise
// word (e.g. "shoulder_pan_joint" -> "Shoulder pan") and special-casing each
// robot's primary gripper joint, whose slider is the only gripper control
// shown (the rest of the gripper linkage is mimic joints filtered out of the
// panel).
const GRIPPER_JOINT_NAMES = new Set(['finger_joint1', 'gripper_right_driver_joint']);

function jointLabel(name: string): string {
  if (GRIPPER_JOINT_NAMES.has(name)) { return 'Gripper'; }
  const words = name
    .split('_')
    .filter(word => word !== 'joint')
    .map(word => word.replace(/(\D)(\d+)$/, '$1 $2'));
  return capitalize(words.join(' '));
}

export async function initRobotViewerVisualization(
  parent: HTMLElement,
  config: RobotViewerConfig
): Promise<RobotViewerVisualization> {
  // The panel is attached before the model loads: the scene sizes its canvas
  // from the viewport's live clientWidth/clientHeight, which reads as 0 (and
  // falls back to a mismatched aspect ratio) on a detached element.
  const panel = buildPanel(parent, config.label);

  const model = robotModelWithBaseOnFloor(await loadWebModel(config.modelUrl));
  const joints = jointsFromModel(model, config.defaultJointDegrees, config.defaultJointMillimeters);
  const controls = addJointControls(panel.controlsHost, joints);
  const mimicsByPrimary = mimicsByPrimaryJoint(model);

  const scene = createRobotViewerScene(
    panel.viewport, model, config.modelBasePath, CANVAS_WIDTH, CANVAS_HEIGHT,
    config.mirrorCameraY ?? false
  );

  const setPrimaryAndMimics = (name: string, value: number): void => {
    scene.setJoint(name, value);
    for (const dependent of mimicsByPrimary.get(name) ?? []) {
      scene.setJoint(dependent.name, dependent.multiplier * value + dependent.offset);
    }
  };

  const listeners: (() => void)[] = [];
  for (const joint of joints) {
    const control = controls.get(joint.name);
    if (!control) { continue; }
    setPrimaryAndMimics(joint.name, joint.value);
    const update = (): void => {
      const value = Number(control.input.value);
      control.value.textContent = formatJointValue(joint.type, value);
      setPrimaryAndMimics(joint.name, value);
    };
    control.input.addEventListener('input', update);
    listeners.push(() => { control.input.removeEventListener('input', update); });
  }

  const resizeObserver = new ResizeObserver(() => { scene.resize(); });
  resizeObserver.observe(panel.viewport);

  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    scene.orbitControls.update();
    scene.renderer.render(scene.scene, scene.camera);
  }
  animate();

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      for (const removeListener of listeners) { removeListener(); }
      scene.destroy();
      panel.root.remove();
    }
  };
}
