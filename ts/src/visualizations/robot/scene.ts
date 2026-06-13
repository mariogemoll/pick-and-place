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

export interface RobotScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  setJoint(name: string, radians: number): void;
  setMaterialColor(materialName: string, color: THREE.Color): void;
  resize(): void;
  destroy(): void;
}

export function createRobotScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets',
  workspaces: WorkspaceOverlaySpec[] = []
): RobotScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(42, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(0.48, 0.48, 0.38);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(1, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const disposeOverlays = addWorkspaceOverlaysToScene(scene, workspaces);

  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);

  function resize(): void {
    const width = viewport.clientWidth || CANVAS_WIDTH;
    const height = viewport.clientHeight || CANVAS_HEIGHT;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  resize();

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    setJoint(name: string, radians: number): void {
      setJointAngle(model, builtModel.jointPivots, name, radians);
    },
    setMaterialColor(materialName: string, color: THREE.Color): void {
      for (const mat of builtModel.materialsByName.get(materialName) ?? []) {
        mat.color.copy(color);
      }
    },
    resize,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      disposeOverlays();
      for (const mats of builtModel.materialsByName.values()) {
        for (const mat of mats) { mat.dispose(); }
      }
    }
  };
}
