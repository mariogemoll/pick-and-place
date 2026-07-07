// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import {
  buildWebModel,
  setJointAngle,
  type WebModel
} from '../../web-model';
import { createCubeBody } from '../grasp-pose-shared/body-factories';
import { createBodyMaterials } from '../grasp-pose-shared/materials';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from '../grasp-pose-shared/ui';

const TARGET_MARKER_RADIUS = 0.006;
const TARGET_MARKER_HEIGHT = 0.0005;

export interface EpisodeReplayScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  setJoint(name: string, radians: number): void;
  setCubeTransform(x: number, y: number, z: number, quat: [number, number, number, number]): void;
  setTarget(x: number, y: number): void;
  resize(): void;
  destroy(): void;
}

export function createEpisodeReplayScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets'
): EpisodeReplayScene {
  const initialWidth = viewport.clientWidth || CANVAS_WIDTH;
  const initialHeight = viewport.clientHeight || CANVAS_HEIGHT;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(initialWidth, initialHeight, false);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(
    42, initialWidth / initialHeight, 0.001, 100
  );
  camera.up.set(0, 0, 1);
  camera.position.set(0.55, 0.45, 0.42);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0.2, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(0.8, 16, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(0.2, 0, 0);
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.05));

  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);

  const materials = createBodyMaterials();
  const cubePart = createCubeBody(materials);
  scene.add(cubePart.body);

  // A tiny flat disc on the floor marking the episode's drop target. MuJoCo's
  // floor is the world XY plane (up = +Z), but THREE.CylinderGeometry's axis
  // defaults to Y, so it needs a quarter-turn about X to lie flat.
  const targetGeometry = new THREE.CylinderGeometry(
    TARGET_MARKER_RADIUS, TARGET_MARKER_RADIUS, TARGET_MARKER_HEIGHT, 24
  );
  targetGeometry.rotateX(Math.PI / 2);
  const targetMaterial = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    roughness: 0.5
  });
  const targetMarker = new THREE.Mesh(targetGeometry, targetMaterial);
  targetMarker.position.z = TARGET_MARKER_HEIGHT / 2;
  scene.add(targetMarker);

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
    setCubeTransform(x, y, z, [qw, qx, qy, qz]): void {
      cubePart.body.position.set(x, y, z);
      cubePart.body.quaternion.set(qx, qy, qz, qw);
    },
    setTarget(x, y): void {
      targetMarker.position.x = x;
      targetMarker.position.y = y;
    },
    resize,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      for (const mats of builtModel.materialsByName.values()) {
        for (const mat of mats) { mat.dispose(); }
      }
      cubePart.destroy();
      materials.destroy();
      targetGeometry.dispose();
      targetMaterial.dispose();
    }
  };
}
