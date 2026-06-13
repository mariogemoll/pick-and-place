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
  createWorldFromCubeMatrix,
  type CubePose
} from '../pregrasp-pose-shared/body-factories';
import { createBodyMaterials } from '../pregrasp-pose-shared/materials';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from '../pregrasp-pose-shared/ui';
import {
  addWorkspaceOverlaysToScene,
  type WorkspaceOverlaySpec
} from '../workspace-overlay';

export interface PickAndPlaceScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  sourceCube: THREE.Object3D;
  targetCube: THREE.Object3D;
  setJoint(name: string, radians: number): void;
  updateSourceCube(pose: CubePose): void;
  updateTargetCube(pose: CubePose): void;
  resize(): void;
  destroy(): void;
}

export function createPickAndPlaceScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets',
  workspace?: WorkspaceOverlaySpec
): PickAndPlaceScene {
  const initialWidth = viewport.clientWidth || CANVAS_WIDTH;
  const initialHeight = viewport.clientHeight || CANVAS_HEIGHT;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(initialWidth, initialHeight, false);
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(
    42, initialWidth / initialHeight, 0.001, 100
  );
  camera.up.set(0, 0, 1);
  camera.position.set(0.55, 0.48, 0.42);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0.18, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(0.8, 16, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  grid.position.x = 0.2;
  scene.add(grid);
  scene.add(new THREE.AxesHelper(0.05));
  const disposeOverlays = addWorkspaceOverlaysToScene(
    scene,
    workspace ? [{ ...workspace, color: 0x22c55e, opacity: 0.18 }] : []
  );

  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);

  const sourceMaterials = createBodyMaterials();
  const sourcePart = createCubeBody(sourceMaterials);
  sourcePart.body.name = 'source_cube';
  scene.add(sourcePart.body);

  const targetMaterials = createBodyMaterials();
  targetMaterials.cube.color.set(0x22c55e);
  targetMaterials.cube.opacity = 0.28;
  targetMaterials.cube.transparent = true;
  targetMaterials.cube.depthWrite = false;
  targetMaterials.marker.opacity = 0.35;
  targetMaterials.marker.transparent = true;
  targetMaterials.marker.depthWrite = false;
  const targetPart = createCubeBody(targetMaterials);
  targetPart.body.name = 'target_cube';
  scene.add(targetPart.body);

  const updateCube = (object: THREE.Object3D, pose: CubePose): void => {
    createWorldFromCubeMatrix(pose).decompose(
      object.position, object.quaternion, object.scale
    );
  };

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
    sourceCube: sourcePart.body,
    targetCube: targetPart.body,
    setJoint(name: string, radians: number): void {
      setJointAngle(model, builtModel.jointPivots, name, radians);
    },
    updateSourceCube(pose: CubePose): void { updateCube(sourcePart.body, pose); },
    updateTargetCube(pose: CubePose): void { updateCube(targetPart.body, pose); },
    resize,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      for (const mats of builtModel.materialsByName.values()) {
        for (const mat of mats) { mat.dispose(); }
      }
      sourcePart.destroy();
      sourceMaterials.destroy();
      targetPart.destroy();
      targetMaterials.destroy();
      disposeOverlays();
    }
  };
}
