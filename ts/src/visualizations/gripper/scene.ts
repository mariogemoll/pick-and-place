// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

export interface GripperScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  gripper: THREE.Group;
  destroy(): void;
}

export function createGripperScene(
  viewport: HTMLElement,
  modelBasePath = '/so101_assets'
): GripperScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(45, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(0.3, 0.3, 0.3);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0, 0, 0.04);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  scene.add(directionalLight);

  const gridHelper = new THREE.GridHelper(
    1,
    20,
    0x9aa9bc,
    0xd5dde8
  );
  gridHelper.rotation.x = Math.PI / 2;
  scene.add(gridHelper);

  const gripper = new THREE.Group();
  gripper.position.set(0, 0, 0.11);
  scene.add(gripper);

  const loader = new STLLoader();
  const material = new THREE.MeshStandardMaterial({ color: 0xffa500 });
  const motorMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 });
  const basePath = modelBasePath.endsWith('/') ? modelBasePath.slice(0, -1) : modelBasePath;

  loader.load(`${basePath}/wrist_roll_follower_so101_v1.stl`, (geometry) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.set(-Math.PI, 0, 0);
    mesh.position.set(0, -0.0002, 0.0009);
    gripper.add(mesh);
  });

  loader.load(`${basePath}/sts3215_03a_v1.stl`, (geometry) => {
    const mesh = new THREE.Mesh(geometry, motorMaterial);
    mesh.rotation.set(-Math.PI / 2, 0, 0);
    mesh.position.set(0.0077, 0.0001, -0.0234);
    gripper.add(mesh);
  });

  const jawGroup = new THREE.Group();
  jawGroup.position.set(0.0202, 0.0188, -0.0234);
  jawGroup.rotation.set(Math.PI / 2, 0, 0);
  gripper.add(jawGroup);

  loader.load(`${basePath}/moving_jaw_so101_v1.stl`, (geometry) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 0, 0.0189);
    jawGroup.add(mesh);
  });

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    gripper,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      material.dispose();
      motorMaterial.dispose();
    }
  };
}
