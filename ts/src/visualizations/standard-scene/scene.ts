// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { deriveSo101Kinematics } from '../../ik/kinematics';
import { buildWebModel, loadWebModel } from '../../web-model';
import { buildEnvironmentModel } from '../environment-model';
import {
  addWorkspaceOverlaysToScene,
  buildWorkspaceOverlaySpecs
} from '../workspace-overlay';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

export interface StandardScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  ready: Promise<void>;
  destroy(): void;
}

export function createStandardScene(viewport: HTMLElement): StandardScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(42, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.0, 1.0, 0.8);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.target.set(0, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(2, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  let disposeOverlays: (() => void) | undefined;

  const ready = Promise.all([
    loadWebModel('/so101.json'),
    loadWebModel('/environment.json')
  ]).then(([robotModel, environmentModel]) => {
    // The robot is defined once (so101) and the environment is overlaid on top;
    // both trees are rooted at the world origin, so no stitching is needed.
    const robot = buildWebModel(robotModel, '/so101_assets');
    const environment = buildEnvironmentModel(environmentModel, '/so101_assets');
    scene.add(robot.root);
    scene.add(environment.root);

    // Derive kinematics from the robot model to add workspace overlays
    const kinematics = deriveSo101Kinematics(robotModel);
    const overlaySpecs = buildWorkspaceOverlaySpecs(kinematics);
    disposeOverlays = addWorkspaceOverlaysToScene(scene, overlaySpecs);

    return Promise.all([robot.ready, environment.ready]).then(() => undefined);
  });

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    ready,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      disposeOverlays?.();
    }
  };
}
