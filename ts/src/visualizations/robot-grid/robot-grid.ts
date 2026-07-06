// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { type ArmJointName } from '../../ik/kinematics';
import { loadWebModel, type WebGeometry } from '../../web-model';
import { createRobotScene, type RobotScene } from '../robot/scene';
import { robotModelWithBaseOnFloor } from '../robot-model';
import { buildUi, type RobotGridTile } from './ui';

const ROBOT_COUNT = 4;
const ROBOT_VIEW_TARGET = new THREE.Vector3(0, 0, 0.1);
const ROBOT_VIEW_CAMERA = new THREE.Vector3(0.4, 0.4, 0.22);
// Static display pose generated from the canonical grasp for cube pose
// x=0.2, y=0, yaw=0. The color grid does not need runtime IK.
const ROBOT_GRID_JOINT_POSE: Record<ArmJointName, number> = {
  shoulder_pan: -0.10582991602672699,
  shoulder_lift: -0.1829202616552561,
  elbow_flex: 0.6162301587559451,
  wrist_flex: 1.1374864296942075,
  wrist_roll: -1.6279467415829163
};
const robotBounds = new THREE.Box3();
const robotCenter = new THREE.Vector3();

export interface RobotGridVisualization {
  destroy(): void;
}

export interface RobotGridVisualizationOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

interface RobotSample {
  color: THREE.Color;
  hex: string;
  rgbText: string;
}

function hasVisualMaterial(geometry: WebGeometry): geometry is WebGeometry & {
  material: string;
} {
  return geometry.role === 'visual' && geometry.material !== undefined;
}

function isLightMaterial([r, g, b]: [number, number, number, number]): boolean {
  return (r + g + b) / 3 > 0.45;
}

function randomRobotSample(): RobotSample {
  const color = new THREE.Color().setHSL(
    Math.random(),
    0.2 + Math.random() * 0.8,
    0.25 + Math.random() * 0.55
  );
  const hex = `#${color.getHexString()}`;
  return {
    color,
    hex,
    rgbText: `${color.r.toFixed(2)}, ${color.g.toFixed(2)}, ${color.b.toFixed(2)}`
  };
}

function centerViewOnRobotX(vizScene: RobotScene): void {
  robotBounds.setFromObject(vizScene.robotRoot);
  if (robotBounds.isEmpty()) { return; }
  robotBounds.getCenter(robotCenter);
  const xDelta = robotCenter.x - vizScene.orbitControls.target.x;
  vizScene.orbitControls.target.x = robotCenter.x;
  vizScene.camera.position.x += xDelta;
  vizScene.orbitControls.update();
}

export async function initializeRobotGridVisualization(
  parent: HTMLElement,
  options: RobotGridVisualizationOptions = {}
): Promise<RobotGridVisualization> {
  const model = robotModelWithBaseOnFloor(await loadWebModel(options.modelUrl));
  const sampledMaterialNames = Array.from(new Set(
    model.bodies.flatMap(body => body.geometries)
      .filter(hasVisualMaterial)
      .map(geometry => geometry.material)
      .filter(materialName => isLightMaterial(model.materials[materialName]))
  ));
  const ui = buildUi(parent, ROBOT_COUNT);
  const scenes: RobotScene[] = [];
  const listeners: (() => void)[] = [];

  for (const tile of ui.tiles) {
    const vizScene = createRobotScene(tile.viewport, model, options.modelBasePath);
    vizScene.scene.background = new THREE.Color(0xffffff);
    vizScene.camera.fov = 42;
    vizScene.camera.position.copy(ROBOT_VIEW_CAMERA);
    vizScene.camera.updateProjectionMatrix();
    vizScene.orbitControls.target.copy(ROBOT_VIEW_TARGET);
    vizScene.orbitControls.update();
    vizScene.resize();
    for (const [jointName, radians] of Object.entries(ROBOT_GRID_JOINT_POSE)) {
      vizScene.setJoint(jointName, radians);
    }
    void vizScene.ready.then(() => {
      centerViewOnRobotX(vizScene);
    });
    scenes.push(vizScene);
  }

  function applySample(tile: RobotGridTile, scene: RobotScene, sample: RobotSample): void {
    for (const materialName of sampledMaterialNames) {
      scene.setMaterialColor(materialName, sample.color);
    }
    tile.swatch.style.backgroundColor = sample.hex;
    tile.hex.textContent = sample.hex.toUpperCase();
    tile.rgb.textContent = sample.rgbText;
  }

  const resample = (): void => {
    ui.tiles.forEach((tile, index) => {
      applySample(tile, scenes[index], randomRobotSample());
    });
  };
  ui.resampleButton.addEventListener('click', resample);
  listeners.push(() => { ui.resampleButton.removeEventListener('click', resample); });
  resample();

  const resizeObserver = new ResizeObserver(() => {
    for (const scene of scenes) { scene.resize(); }
  });
  for (const tile of ui.tiles) { resizeObserver.observe(tile.viewport); }

  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    for (const scene of scenes) {
      scene.orbitControls.update();
      scene.renderer.render(scene.scene, scene.camera);
    }
  }
  animate();

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      for (const removeListener of listeners) { removeListener(); }
      for (const scene of scenes) { scene.destroy(); }
      ui.root.remove();
    }
  };
}
