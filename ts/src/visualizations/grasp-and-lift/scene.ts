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
  createCubeBody,
  type CubePose
} from '../grasp-pose-shared/body-factories';
import { createBodyMaterials } from '../grasp-pose-shared/materials';
import { CANVAS_HEIGHT } from './ui';

// One colour per cube face, in THREE.BoxGeometry group order
// (+x, -x, +y, -y, +z, -z) – matches the pick-and-place cube so the two
// visualizations read as one system.
const FACE_COLORS = [0xef4444, 0xf97316, 0x22c55e, 0x06b6d4, 0x3b82f6, 0xeab308];

export interface GraspAndLiftScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  setGripperPose(position: THREE.Vector3, quaternion: THREE.Quaternion): void;
  setGripperAngle(radians: number): void;
  setCubePose(pose: CubePose): void;
  resize(): void;
  destroy(): void;
}

export function createGraspAndLiftScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets'
): GraspAndLiftScene {
  const initialWidth = viewport.clientWidth || 600;
  const initialHeight = viewport.clientHeight || CANVAS_HEIGHT;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(initialWidth, initialHeight, false);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  // Side profile: camera looks along +y so the vertical lift (z) and the
  // jaw pinch (x) both read clearly in the x–z plane.
  const camera = new THREE.PerspectiveCamera(33, initialWidth / initialHeight, 0.001, 10);
  camera.up.set(0, 0, 1);
  camera.position.set(0, 0.37, 0.075);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0, 0, 0.075);
  orbitControls.minDistance = 0.08;
  orbitControls.maxDistance = 1.0;
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(0.25, 0.25, 0.6);
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(0.3, 12, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const builtModel = buildWebModel(model, modelBasePath, 'gripper');
  void builtModel.ready.catch(console.error);
  const gripper = builtModel.root;
  scene.add(gripper);

  const materials = createBodyMaterials();
  const faceMaterials = FACE_COLORS.map(color => new THREE.MeshStandardMaterial({
    color,
    roughness: 0.6
  }));
  const cubePart = createCubeBody(materials, faceMaterials);
  scene.add(cubePart.body);

  function resize(): void {
    const width = viewport.clientWidth || 600;
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
    setGripperPose(position, quaternion): void {
      gripper.position.copy(position);
      gripper.quaternion.copy(quaternion);
    },
    setGripperAngle(radians: number): void {
      setJointAngle(model, builtModel.jointPivots, 'gripper', radians);
    },
    setCubePose(pose: CubePose): void {
      cubePart.body.position.set(pose.x, pose.y, pose.z);
      cubePart.body.quaternion.setFromEuler(
        new THREE.Euler(pose.roll, pose.pitch, pose.yaw, 'ZYX')
      );
    },
    resize,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      cubePart.destroy();
      materials.destroy();
      for (const material of faceMaterials) { material.dispose(); }
      for (const mats of builtModel.materialsByName.values()) {
        for (const mat of mats) { mat.dispose(); }
      }
    }
  };
}
