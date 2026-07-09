// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import {
  buildWebModel,
  type BuiltWebModel,
  loadWebModel,
  setJointAngle,
  type WebModel
} from '../../web-model';
import { buildEnvironmentModel } from '../environment-model';
import { createCubeAprilTagBody } from '../grasp-pose-shared/body-factories';
import { createBodyMaterials } from '../grasp-pose-shared/materials';
import { createInfiniteFloor } from '../robot-viewer/infinite-floor';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

export const DEFAULT_FLOOR_COLOR = '#70f2f7';
export const DEFAULT_PEDESTAL_COLOR = '#3d5cad';
export const DEFAULT_SKY_COLOR = '#f27782';

const FLOOR_Z = -0.101;
const PEDESTAL_SIZE = new THREE.Vector3(0.8, 0.8, 0.1);
const PEDESTAL_CENTER = new THREE.Vector3(0.3, 0, -0.05);
const WORKSPACE_INSERT_SIZE = 0.525;
const WORKSPACE_INSERT_Z = 0.0004;
const TARGET_PLATE_HALF_SIZE = 0.05;
const TARGET_PLATE_Z = WORKSPACE_INSERT_Z + 0.0004;
const BACKGROUND_TOP_COLOR = '#f27782';
const BACKGROUND_BOTTOM_COLOR = '#ffd0d3';
const BACKGROUND_BOTTOM_BLEND = new THREE.Color(BACKGROUND_BOTTOM_COLOR);
const DEFAULT_WORKSPACE_INSERT_COLOR = '#a9dfff';

function createSoftBackgroundTexture(
  topColor: THREE.ColorRepresentation,
  bottomColor: THREE.ColorRepresentation
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 256;
  const context = canvas.getContext('2d');
  if (context === null) {
    throw new Error('Unable to create standard scene background texture');
  }
  const gradient = context.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, new THREE.Color(topColor).getStyle());
  gradient.addColorStop(1, new THREE.Color(bottomColor).getStyle());
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  return texture;
}

function formatVector(vector: THREE.Vector3): string {
  return [vector.x, vector.y, vector.z]
    .map(value => Number(value.toFixed(3)))
    .join(', ');
}

export interface StandardScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  ready: Promise<void>;
  setJoint(name: string, radians: number): void;
  setCubeTransform(x: number, y: number, z: number, quat: [number, number, number, number]): void;
  setTarget(x: number, y: number, yaw: number): void;
  setTargetVisible(visible: boolean): void;
  setFloorColor(color: THREE.Color): void;
  setPedestalColor(color: THREE.Color): void;
  setBackgroundColor(color: THREE.Color): void;
  setRobotPlasticColor(color: THREE.Color): void;
  setEnvironmentMaterialColor(color: THREE.Color): void;
  destroy(): void;
}

export interface StandardSceneOptions {
  environmentBasePath?: string;
  environmentUrl?: string;
  modelBasePath?: string;
  modelUrl?: string;
}

function setMaterialsColor(
  builtModel: BuiltWebModel | undefined,
  materialNames: string[],
  color: THREE.Color
): void {
  if (builtModel === undefined) { return; }
  for (const materialName of materialNames) {
    for (const material of builtModel.materialsByName.get(materialName) ?? []) {
      material.color.copy(color);
    }
  }
}

function configureShadows(root: THREE.Object3D): void {
  root.traverse(object => {
    if (!(object instanceof THREE.Mesh)) { return; }
    object.castShadow = true;
    object.receiveShadow = true;
  });
}

export function createStandardScene(
  viewport: HTMLElement,
  options: StandardSceneOptions = {}
): StandardScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  let backgroundTexture = createSoftBackgroundTexture(
    BACKGROUND_TOP_COLOR,
    BACKGROUND_BOTTOM_COLOR
  );
  scene.background = backgroundTexture;
  scene.fog = null;

  const camera = new THREE.PerspectiveCamera(38, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(-0.794, 1.849, 0.697);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.target.set(0.36, -0.015, 0.15);
  orbitControls.update();
  orbitControls.addEventListener('end', () => {
    console.info(
      `[standard-scene camera] camera.position.set(${formatVector(camera.position)}); ` +
      `orbitControls.target.set(${formatVector(orbitControls.target)});`
    );
  });

  scene.add(new THREE.HemisphereLight(0xeaf1ff, 0xd7e3f2, 0.35));

  const keyLight = new THREE.DirectionalLight(0xfff4dd, 1.6);
  keyLight.position.set(3, 5, 4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.1;
  keyLight.shadow.camera.far = 12;
  keyLight.shadow.camera.left = -2;
  keyLight.shadow.camera.right = 2;
  keyLight.shadow.camera.top = 2;
  keyLight.shadow.camera.bottom = -2;
  keyLight.shadow.bias = -0.0002;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xcfe6ff, 0.55);
  fillLight.position.set(-4, 2, 3);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xa4c8ff, 0.9);
  rimLight.position.set(-2, 5, -3);
  scene.add(rimLight);

  const floor = createInfiniteFloor(DEFAULT_FLOOR_COLOR);
  floor.position.set(0.3, 0, FLOOR_Z);
  floor.receiveShadow = true;
  if (floor.material instanceof THREE.MeshStandardMaterial) {
    floor.material.roughness = 0.55;
  }
  scene.add(floor);

  const floorGrid = new THREE.GridHelper(4, 32, 0xb5eff0, 0xb5eff0);
  floorGrid.position.set(0.3, 0, FLOOR_Z + 0.001);
  floorGrid.rotation.x = Math.PI / 2;
  floorGrid.material.opacity = 0.45;
  floorGrid.material.transparent = true;
  scene.add(floorGrid);

  const pedestalGeometry = new THREE.BoxGeometry(
    PEDESTAL_SIZE.x,
    PEDESTAL_SIZE.y,
    PEDESTAL_SIZE.z
  );
  const pedestalMaterial = new THREE.MeshStandardMaterial({
    color: DEFAULT_PEDESTAL_COLOR,
    roughness: 0.55
  });
  const pedestal = new THREE.Mesh(pedestalGeometry, pedestalMaterial);
  pedestal.position.copy(PEDESTAL_CENTER);
  pedestal.castShadow = true;
  pedestal.receiveShadow = true;
  scene.add(pedestal);

  const workspaceInsertGeometry = new THREE.PlaneGeometry(
    WORKSPACE_INSERT_SIZE,
    WORKSPACE_INSERT_SIZE
  );
  const workspaceInsertMaterial = new THREE.MeshStandardMaterial({
    color: DEFAULT_WORKSPACE_INSERT_COLOR,
    roughness: 0.38
  });
  const workspaceInsert = new THREE.Mesh(workspaceInsertGeometry, workspaceInsertMaterial);
  workspaceInsert.position.set(0.279579, 0.0000305, WORKSPACE_INSERT_Z);
  workspaceInsert.receiveShadow = true;
  scene.add(workspaceInsert);

  let builtRobot: BuiltWebModel | undefined;
  let robotModel: WebModel | undefined;
  let builtEnvironment: BuiltWebModel | undefined;

  const bodyMaterials = createBodyMaterials();
  const cubePart = createCubeAprilTagBody(bodyMaterials);
  cubePart.body.visible = false;
  configureShadows(cubePart.body);
  scene.add(cubePart.body);

  const targetGeometry = new THREE.PlaneGeometry(
    TARGET_PLATE_HALF_SIZE * 2,
    TARGET_PLATE_HALF_SIZE * 2
  );
  const targetMaterial = new THREE.MeshStandardMaterial({
    color: 0x4a4f57,
    roughness: 0.78
  });
  const targetMarker = new THREE.Mesh(targetGeometry, targetMaterial);
  targetMarker.position.z = TARGET_PLATE_Z;
  targetMarker.visible = false;
  targetMarker.receiveShadow = true;
  scene.add(targetMarker);

  const ready = Promise.all([
    loadWebModel(options.modelUrl ?? '/so101.json'),
    loadWebModel(options.environmentUrl ?? '/environment.json')
  ]).then(([loadedRobotModel, environmentModel]) => {
    // The robot is defined once (so101) and the environment is overlaid on top;
    // both trees are rooted at the world origin, so no stitching is needed.
    robotModel = loadedRobotModel;
    const robot = buildWebModel(robotModel, options.modelBasePath ?? '/so101_assets');
    const environment = buildEnvironmentModel(
      environmentModel,
      options.environmentBasePath ?? options.modelBasePath ?? '/so101_assets'
    );
    builtRobot = robot;
    builtEnvironment = environment;
    configureShadows(robot.root);
    configureShadows(environment.root);
    scene.add(robot.root);
    scene.add(environment.root);

    return Promise.all([robot.ready, environment.ready]).then(() => {
      configureShadows(robot.root);
      configureShadows(environment.root);
    });
  });

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    ready,
    setJoint(name: string, radians: number): void {
      if (robotModel === undefined || builtRobot === undefined) { return; }
      setJointAngle(robotModel, builtRobot.jointPivots, name, radians);
    },
    setCubeTransform(x, y, z, [qw, qx, qy, qz]): void {
      cubePart.body.visible = true;
      cubePart.body.position.set(x, y, z);
      cubePart.body.quaternion.set(qx, qy, qz, qw);
    },
    setTarget(x, y, yaw): void {
      targetMarker.visible = true;
      targetMarker.position.x = x;
      targetMarker.position.y = y;
      targetMarker.rotation.z = yaw;
    },
    setTargetVisible(visible): void {
      targetMarker.visible = visible;
    },
    setFloorColor(color: THREE.Color): void {
      if (!(floor.material instanceof THREE.MeshStandardMaterial)) { return; }
      floor.material.color.copy(color);
    },
    setPedestalColor(color: THREE.Color): void {
      pedestalMaterial.color.copy(color);
    },
    setBackgroundColor(color: THREE.Color): void {
      backgroundTexture.dispose();
      backgroundTexture = createSoftBackgroundTexture(
        color,
        color.clone().lerp(BACKGROUND_BOTTOM_BLEND, 0.65)
      );
      scene.background = backgroundTexture;
    },
    setRobotPlasticColor(color: THREE.Color): void {
      setMaterialsColor(builtRobot, ['plastic'], color);
    },
    setEnvironmentMaterialColor(color: THREE.Color): void {
      setMaterialsColor(builtEnvironment, ['environment_plastic', 'mdf'], color);
    },
    destroy(): void {
      orbitControls.dispose();
      floor.geometry.dispose();
      if (Array.isArray(floor.material)) {
        for (const material of floor.material) { material.dispose(); }
      } else {
        floor.material.dispose();
      }
      floorGrid.geometry.dispose();
      (floorGrid.material as THREE.Material).dispose();
      pedestal.geometry.dispose();
      pedestalMaterial.dispose();
      workspaceInsert.geometry.dispose();
      workspaceInsertMaterial.dispose();
      cubePart.destroy();
      bodyMaterials.destroy();
      targetGeometry.dispose();
      targetMaterial.dispose();
      backgroundTexture.dispose();
      renderer.dispose();
    }
  };
}
