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
import { type CollisionBoxDefinition, SO101_COLLISION_BOXES } from './collision-boxes';
import { CANVAS_HEIGHT, CANVAS_WIDTH } from './ui';

const INSET_WIDTH = 160;
const INSET_HEIGHT = 120;
const INSET_TARGET_HEIGHT = 0.15;
const INSET_DISTANCE = 0.55;
// Screen-down for the top camera (up = (-1, 0, 0)) is world +X; shifting the
// look-at point that way moves the robot base toward the top of the frame,
// making room to show the whole extent-ring annulus.
const TOP_CENTER_SHIFT = 0.13;
// Aim the side camera slightly off-axis (in its co-rotating frame) so the
// robot isn't dead-center in the inset.
const SIDE_TARGET_SHIFT = new THREE.Vector3(0.2, 0, 0);

export type RobotGeometryMode = 'visual' | 'collision' | 'both';

export interface RobotScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  robotRoot: THREE.Group;
  ready: Promise<void>;
  setJoint(name: string, radians: number): void;
  setMaterialColor(materialName: string, color: THREE.Color): void;
  setOverlayColor(index: number, color: THREE.Color): void;
  setOverlayVisible(index: number, visible: boolean): void;
  setGeometryMode(mode: RobotGeometryMode): void;
  setBackgroundColor(color: THREE.Color): void;
  resize(): void;
  renderInsets(shoulderPanRadians: number): void;
  destroy(): void;
}

function setQuaternion(
  object: THREE.Object3D,
  [w, x, y, z]: [number, number, number, number]
): void {
  object.quaternion.set(x, y, z, w);
}

function createCollisionBox(
  box: CollisionBoxDefinition,
  material: THREE.Material
): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(
    box.size[0] * 2,
    box.size[1] * 2,
    box.size[2] * 2
  );
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = box.name;
  mesh.position.set(...box.position);
  setQuaternion(mesh, box.quaternion);
  mesh.userData.role = 'collision';
  return mesh;
}

function createCollisionOutline(
  box: CollisionBoxDefinition,
  material: THREE.Material
): THREE.LineSegments {
  const boxGeometry = new THREE.BoxGeometry(
    box.size[0] * 2,
    box.size[1] * 2,
    box.size[2] * 2
  );
  const geometry = new THREE.EdgesGeometry(boxGeometry);
  boxGeometry.dispose();
  const line = new THREE.LineSegments(geometry, material);
  line.name = `${box.name}_outline`;
  line.position.set(...box.position);
  setQuaternion(line, box.quaternion);
  line.userData.role = 'collision';
  return line;
}

function addCollisionBoxes(
  builtBodies: Map<string, THREE.Group>,
  material: THREE.Material
): {
  meshes: THREE.Mesh[];
  geometries: THREE.BufferGeometry[];
} {
  const meshes: THREE.Mesh[] = [];
  const geometries: THREE.BufferGeometry[] = [];

  for (const [bodyName, boxes] of Object.entries(SO101_COLLISION_BOXES)) {
    const body = builtBodies.get(bodyName);
    if (body === undefined) { continue; }
    const group = new THREE.Group();
    group.name = `${bodyName}_collision_boxes`;
    for (const box of boxes) {
      const mesh = createCollisionBox(box, material);
      meshes.push(mesh);
      geometries.push(mesh.geometry);
      group.add(mesh);
    }
    body.add(group);
  }

  return { meshes, geometries };
}

function addCollisionOutlines(
  builtBodies: Map<string, THREE.Group>,
  material: THREE.Material
): {
  lines: THREE.LineSegments[];
  geometries: THREE.BufferGeometry[];
} {
  const lines: THREE.LineSegments[] = [];
  const geometries: THREE.BufferGeometry[] = [];

  for (const [bodyName, boxes] of Object.entries(SO101_COLLISION_BOXES)) {
    const body = builtBodies.get(bodyName);
    if (body === undefined) { continue; }
    const group = new THREE.Group();
    group.name = `${bodyName}_collision_outlines`;
    for (const box of boxes) {
      const line = createCollisionOutline(box, material);
      lines.push(line);
      geometries.push(line.geometry);
      group.add(line);
    }
    body.add(group);
  }

  return { lines, geometries };
}

function createInsetRenderer(container: HTMLElement): {
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
} {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(INSET_WIDTH, INSET_HEIGHT, false);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  container.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(42, INSET_WIDTH / INSET_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);

  return { renderer, camera };
}

export function createRobotScene(
  viewport: HTMLElement,
  model: WebModel,
  modelBasePath = '/so101_assets',
  workspaces: WorkspaceOverlaySpec[] = []
): RobotScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT, false);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.shadowMap.enabled = true;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f8ff);

  const camera = new THREE.PerspectiveCamera(42, CANVAS_WIDTH / CANVAS_HEIGHT, 0.001, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(0.58, -0.48, 0.38);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.target.set(0.2, 0, 0.1);
  orbitControls.update();

  scene.add(new THREE.HemisphereLight(0xddeeff, 0xffffff, 2.2));
  const directionalLight = new THREE.DirectionalLight(0xfff2d6, 3);
  directionalLight.position.set(2, 2, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(1, 20, 0x9aa9bc, 0xd5dde8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const overlays = addWorkspaceOverlaysToScene(scene, workspaces);

  const builtModel = buildWebModel(model, modelBasePath);
  scene.add(builtModel.root);
  const collisionMaterial = new THREE.MeshStandardMaterial({
    color: 0x0f7f8c,
    opacity: 0.34,
    roughness: 0.45,
    transparent: true,
    depthWrite: false
  });
  const collisionOutlineMaterial = new THREE.LineBasicMaterial({
    color: 0x064f59,
    transparent: true,
    opacity: 0.74
  });
  const collisionBoxes = addCollisionBoxes(builtModel.bodies, collisionMaterial);
  const collisionOutlines = addCollisionOutlines(builtModel.bodies, collisionOutlineMaterial);
  builtModel.root.updateMatrixWorld(true);

  const panAxisWorld = new THREE.Vector3();
  builtModel.jointPivots.get('shoulder_pan')?.getWorldPosition(panAxisWorld);
  const insetTarget = new THREE.Vector3(panAxisWorld.x, panAxisWorld.y, INSET_TARGET_HEIGHT);

  const insetContainer = document.createElement('div');
  insetContainer.className = 'robot-viz-insets';
  viewport.appendChild(insetContainer);

  const sideInsetContainer = document.createElement('div');
  sideInsetContainer.className = 'robot-viz-inset robot-viz-inset-side';
  const sideInsetLabel = document.createElement('span');
  sideInsetLabel.className = 'robot-viz-inset-label';
  sideInsetLabel.textContent = 'Side (following robot)';
  const { renderer: sideRenderer, camera: sideCamera } = createInsetRenderer(sideInsetContainer);
  sideInsetContainer.appendChild(sideInsetLabel);

  const topInsetContainer = document.createElement('div');
  topInsetContainer.className = 'robot-viz-inset robot-viz-inset-top';
  const topInsetLabel = document.createElement('span');
  topInsetLabel.className = 'robot-viz-inset-label';
  topInsetLabel.textContent = 'Top';
  const { renderer: topRenderer, camera: topCamera } = createInsetRenderer(topInsetContainer);
  topInsetContainer.appendChild(topInsetLabel);

  insetContainer.append(topInsetContainer, sideInsetContainer);

  const topTarget = insetTarget.clone().add(new THREE.Vector3(TOP_CENTER_SHIFT, 0, 0));
  topCamera.up.set(-1, 0, 0);
  topCamera.position.set(topTarget.x, topTarget.y, INSET_TARGET_HEIGHT + 0.75);
  topCamera.lookAt(topTarget);

  const sideOffset = new THREE.Vector3(0, -INSET_DISTANCE, 0);

  function resize(): void {
    const width = viewport.clientWidth || CANVAS_WIDTH;
    const height = viewport.clientHeight || CANVAS_HEIGHT;
    renderer.setSize(width, height, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  resize();

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    robotRoot: builtModel.root,
    ready: builtModel.ready,
    setJoint(name: string, radians: number): void {
      setJointAngle(model, builtModel.jointPivots, name, radians);
    },
    setMaterialColor(materialName: string, color: THREE.Color): void {
      for (const mat of builtModel.materialsByName.get(materialName) ?? []) {
        mat.color.copy(color);
      }
    },
    setOverlayColor(index: number, color: THREE.Color): void {
      overlays.setColor(index, color);
    },
    setOverlayVisible(index: number, visible: boolean): void {
      overlays.setVisible(index, visible);
    },
    setGeometryMode(mode: RobotGeometryMode): void {
      const visualVisible = mode === 'visual' || mode === 'both';
      const collisionVisible = mode === 'collision' || mode === 'both';
      builtModel.root.traverse(object => {
        if (object.userData.role === 'visual') {
          object.visible = visualVisible;
        }
      });
      for (const mesh of collisionBoxes.meshes) {
        mesh.visible = collisionVisible;
      }
      for (const line of collisionOutlines.lines) {
        line.visible = collisionVisible;
      }
    },
    setBackgroundColor(color: THREE.Color): void {
      scene.background = color;
    },
    resize,
    renderInsets(shoulderPanRadians: number): void {
      const axis = new THREE.Vector3(0, 0, 1);
      const rotation = -shoulderPanRadians;
      sideCamera.position
        .copy(sideOffset)
        .add(SIDE_TARGET_SHIFT)
        .applyAxisAngle(axis, rotation)
        .add(insetTarget);
      const sideTarget = SIDE_TARGET_SHIFT.clone()
        .applyAxisAngle(axis, rotation)
        .add(insetTarget);
      sideCamera.lookAt(sideTarget);
      sideRenderer.render(scene, sideCamera);
      topRenderer.render(scene, topCamera);
    },
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
      sideRenderer.dispose();
      topRenderer.dispose();
      overlays.dispose();
      collisionMaterial.dispose();
      collisionOutlineMaterial.dispose();
      for (const geometry of collisionBoxes.geometries) { geometry.dispose(); }
      for (const geometry of collisionOutlines.geometries) { geometry.dispose(); }
      for (const mats of builtModel.materialsByName.values()) {
        for (const mat of mats) { mat.dispose(); }
      }
    }
  };
}
