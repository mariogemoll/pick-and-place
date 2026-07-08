// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { buildWebModel, setJointAngle, type WebModel } from '../../web-model';
import { createInfiniteFloor } from './infinite-floor';

const CAMERA_DISTANCE_FACTOR = 1.6;
const GRID_SIZE_FACTOR = 3;
const GRID_DIVISIONS = 20;
const BACKGROUND_COLOR = 0xc7d3e8;
const FLOOR_COLOR = 0x8b93a0;

export interface RobotViewerScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  robotRoot: THREE.Group;
  bodies: Map<string, THREE.Group>;
  baseBodyName: string | undefined;
  grid: THREE.GridHelper;
  floor: THREE.Mesh;
  ready: Promise<void>;
  setJoint(name: string, value: number): void;
  resize(): void;
  destroy(): void;
}

// The root of the kinematic chain: the manifest always has a synthetic
// "world" body parenting itself, and the true robot base is whichever body
// is attached directly to it (not necessarily named "base" - Panda's is
// "link0").
function findBaseBody(model: WebModel): string | undefined {
  return model.bodies.find(body => body.parent === 'world' && body.name !== 'world')?.name;
}

function fitCameraToRobot(scene: RobotViewerScene, mirrorCameraY: boolean): void {
  const bounds = new THREE.Box3().setFromObject(scene.robotRoot);
  if (bounds.isEmpty()) { return; }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  bounds.getCenter(center);
  bounds.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 0.05) * CAMERA_DISTANCE_FACTOR;

  scene.orbitControls.target.copy(center);
  scene.camera.position.set(
    center.x + radius,
    center.y + (mirrorCameraY ? radius : -radius),
    center.z + radius * 0.4
  );
  scene.camera.near = radius / 100;
  scene.camera.far = radius * 100;
  scene.camera.updateProjectionMatrix();
  scene.orbitControls.update();

  const baseGroup = scene.baseBodyName !== undefined
    ? scene.bodies.get(scene.baseBodyName)
    : undefined;
  const baseOrigin = baseGroup?.getWorldPosition(new THREE.Vector3()) ?? new THREE.Vector3();
  scene.floor.position.set(baseOrigin.x, baseOrigin.y, 0);

  const gridSize = Math.max(size.x, size.y, 0.05) * GRID_SIZE_FACTOR;
  scene.grid.scale.set(gridSize, 1, gridSize);
  scene.grid.position.set(baseOrigin.x, baseOrigin.y, 0.001);
}

export function createRobotViewerScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath: string,
  canvasWidth: number,
  canvasHeight: number,
  mirrorCameraY: boolean
): RobotViewerScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(canvasWidth, canvasHeight, false);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BACKGROUND_COLOR);
  scene.fog = new THREE.Fog(BACKGROUND_COLOR, 2, 40);

  const camera = new THREE.PerspectiveCamera(42, canvasWidth / canvasHeight, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(0.58, -0.48, 0.38);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  const floor = createInfiniteFloor(FLOOR_COLOR);
  scene.add(floor);

  const grid = new THREE.GridHelper(1, GRID_DIVISIONS, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);
  builtModel.root.updateMatrixWorld(true);

  function resize(): void {
    const width = viewport.clientWidth || canvasWidth;
    const height = viewport.clientHeight || canvasHeight;
    renderer.setSize(width, height, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  resize();

  const viewerScene: RobotViewerScene = {
    scene,
    renderer,
    camera,
    orbitControls,
    robotRoot: builtModel.root,
    bodies: builtModel.bodies,
    baseBodyName: findBaseBody(model),
    grid,
    floor,
    ready: builtModel.ready,
    setJoint(name: string, value: number): void {
      setJointAngle(model, builtModel.jointPivots, name, value);
    },
    resize,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
    }
  };

  void builtModel.ready.then(() => {
    builtModel.root.updateMatrixWorld(true);
    fitCameraToRobot(viewerScene, mirrorCameraY);
  });

  return viewerScene;
}
