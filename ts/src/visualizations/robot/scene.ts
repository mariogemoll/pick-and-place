// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import {
  buildWebModel,
  setJointAngle,
  type WebModel
} from '../../web-model';
import {
  addWorkspaceOverlaysToScene,
  type WorkspaceOverlaySpec
} from '../workspace-overlay';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

const INSET_WIDTH = 160;
const INSET_HEIGHT = 120;
const INSET_TARGET_HEIGHT = 0.15;
const INSET_DISTANCE = 0.55;
// Screen-down for the top camera (up = (-1, 0, 0)) is world +X; shifting the
// look-at point that way moves the robot base toward the top of the frame,
// making room to show the whole extent-ring annulus.
const TOP_CENTER_SHIFT = 0.13;
// Aim the side camera slightly off-axis (in its co-rotating frame) so the
// robot isn't dead-center in the inset.
const SIDE_TARGET_SHIFT = new THREE.Vector3(0.2, 0, 0);

export interface RobotScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  robotRoot: THREE.Group;
  ready: Promise<void>;
  setJoint(name: string, radians: number): void;
  setMaterialColor(materialName: string, color: THREE.Color): void;
  setOverlayColor(index: number, color: THREE.Color): void;
  setOverlayVisible(index: number, visible: boolean): void;
  setBackgroundColor(color: THREE.Color): void;
  resize(): void;
  renderInsets(shoulderPanRadians: number): void;
  destroy(): void;
}

function createInsetRenderer(container: HTMLElement): {
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
} {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(INSET_WIDTH, INSET_HEIGHT, false);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  container.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(42, INSET_WIDTH / INSET_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);

  return { renderer, camera };
}

export function createRobotScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets',
  workspaces: WorkspaceOverlaySpec[] = []
): RobotScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT, false);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(42, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(0.58, -0.48, 0.38);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0.2, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(1, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const overlays = addWorkspaceOverlaysToScene(scene, workspaces);

  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);
  builtModel.root.updateMatrixWorld(true);

  const panAxisWorld = new THREE.Vector3();
  builtModel.jointPivots.get('shoulder_pan')?.getWorldPosition(panAxisWorld);
  const insetTarget = new THREE.Vector3(panAxisWorld.x, panAxisWorld.y, INSET_TARGET_HEIGHT);

  const insetContainer = document.createElement('div');
  insetContainer.className = 'robot-viz-insets';
  viewport.appendChild(insetContainer);

  const sideInsetContainer = document.createElement('div');
  sideInsetContainer.className = 'robot-viz-inset robot-viz-inset-side';
  const sideInsetLabel = document.createElement('span');
  sideInsetLabel.className = 'robot-viz-inset-label';
  sideInsetLabel.textContent = 'Side (following robot)';
  const { renderer: sideRenderer, camera: sideCamera } = createInsetRenderer(sideInsetContainer);
  sideInsetContainer.appendChild(sideInsetLabel);

  const topInsetContainer = document.createElement('div');
  topInsetContainer.className = 'robot-viz-inset robot-viz-inset-top';
  const topInsetLabel = document.createElement('span');
  topInsetLabel.className = 'robot-viz-inset-label';
  topInsetLabel.textContent = 'Top';
  const { renderer: topRenderer, camera: topCamera } = createInsetRenderer(topInsetContainer);
  topInsetContainer.appendChild(topInsetLabel);

  insetContainer.append(topInsetContainer, sideInsetContainer);

  const topTarget = insetTarget.clone().add(new THREE.Vector3(TOP_CENTER_SHIFT, 0, 0));
  topCamera.up.set(-1, 0, 0);
  topCamera.position.set(topTarget.x, topTarget.y, INSET_TARGET_HEIGHT + 0.75);
  topCamera.lookAt(topTarget);

  const sideOffset = new THREE.Vector3(0, -INSET_DISTANCE, 0);

  function resize(): void {
    const width = viewport.clientWidth || CANVAS_WIDTH;
    const height = viewport.clientHeight || CANVAS_HEIGHT;
    renderer.setSize(width, height, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  resize();

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    robotRoot: builtModel.root,
    ready: builtModel.ready,
    setJoint(name: string, radians: number): void {
      setJointAngle(model, builtModel.jointPivots, name, radians);
    },
    setMaterialColor(materialName: string, color: THREE.Color): void {
      for (const mat of builtModel.materialsByName.get(materialName) ?? []) {
        mat.color.copy(color);
      }
    },
    setOverlayColor(index: number, color: THREE.Color): void {
      overlays.setColor(index, color);
    },
    setOverlayVisible(index: number, visible: boolean): void {
      overlays.setVisible(index, visible);
    },
    setBackgroundColor(color: THREE.Color): void {
      scene.background = color;
    },
    resize,
    renderInsets(shoulderPanRadians: number): void {
      const axis = new THREE.Vector3(0, 0, 1);
      const rotation = -shoulderPanRadians;
      sideCamera.position
        .copy(sideOffset)
        .add(SIDE_TARGET_SHIFT)
        .applyAxisAngle(axis, rotation)
        .add(insetTarget);
      const sideTarget = SIDE_TARGET_SHIFT.clone()
        .applyAxisAngle(axis, rotation)
        .add(insetTarget);
      sideCamera.lookAt(sideTarget);
      sideRenderer.render(scene, sideCamera);
      topRenderer.render(scene, topCamera);
    },
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      sideRenderer.dispose();
      topRenderer.dispose();
      overlays.dispose();
      for (const mats of builtModel.materialsByName.values()) {
        for (const mat of mats) { mat.dispose(); }
      }
    }
  };
}
