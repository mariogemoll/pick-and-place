// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import type { WebModel } from '../../web-model';
import {
  type BodySelection,
  createBodies,
  type PregraspPoseBodies,
  type TransformStage
} from './bodies';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

export interface PregraspPoseScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  bodies: PregraspPoseBodies;
  destroy(): void;
}

export async function createPregraspPoseScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets',
  bodySelection: BodySelection = 'combined',
  transformStage: TransformStage = 'unaligned',
  hingeAngle = 0
): Promise<PregraspPoseScene> {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  const width = viewport.clientWidth || CANVAS_WIDTH;
  const height = viewport.clientHeight || CANVAS_HEIGHT;
  renderer.setSize(width, height);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(42, width / height, 0.001, 100);
  camera.up.set(0, 0, 1);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(1, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.05));

  const bodies = await createBodies(
    model, modelBasePath, bodySelection, transformStage, hingeAngle
  );
  scene.add(bodies.root);

  const bounds = new THREE.Box3().setFromObject(bodies.root);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = Math.max(bounds.getSize(new THREE.Vector3()).length(), 0.05);
  orbitControls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size, size, size * 0.75));
  orbitControls.update();

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    bodies,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      bodies.destroy();
    }
  };
}
