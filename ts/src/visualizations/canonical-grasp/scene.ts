// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { ARM_JOINT_NAMES, NEUTRAL_ARM_JOINTS } from '../../ik/kinematics';
import {
  buildWebModel,
  setJointAngle,
  type WebModel
} from '../../web-model';
import {
  createCubeBody,
  createWorldFromCubeMatrix,
  type CubePose
} from '../pregrasp-pose-shared/body-factories';
import { createBodyMaterials } from '../pregrasp-pose-shared/materials';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from '../pregrasp-pose-shared/ui';
import {
  addWorkspaceOverlaysToScene,
  type WorkspaceOverlaySpec
} from '../workspace-overlay';

export type { WorkspaceOverlaySpec };

const GRIPPER_OPEN_ANGLE = Math.PI / 3;

export interface CanonicalGraspScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  cube: THREE.Object3D;
  setJoint(name: string, radians: number): void;
  updateCubePose(pose: CubePose): void;
  resize(): void;
  destroy(): void;
}

export function createCanonicalGraspScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets',
  workspaces?: WorkspaceOverlaySpec[]
): CanonicalGraspScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(
    42, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100
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

  const disposeOverlays = addWorkspaceOverlaysToScene(scene, workspaces ?? []);

  // The arm stays in its neutral rest pose: this visualization is about the
  // cube's placement, not the grasp solution.
  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);
  for (const name of ARM_JOINT_NAMES) {
    setJointAngle(model, builtModel.jointPivots, name, NEUTRAL_ARM_JOINTS[name]);
  }
  setJointAngle(model, builtModel.jointPivots, 'gripper', GRIPPER_OPEN_ANGLE);

  const materials = createBodyMaterials();
  const cubePart = createCubeBody(materials);
  scene.add(cubePart.body);

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
    cube: cubePart.body,
    setJoint(name: string, radians: number): void {
      setJointAngle(model, builtModel.jointPivots, name, radians);
    },
    updateCubePose(pose: CubePose): void {
      createWorldFromCubeMatrix(pose).decompose(
        cubePart.body.position, cubePart.body.quaternion, cubePart.body.scale
      );
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
      disposeOverlays();
    }
  };
}
