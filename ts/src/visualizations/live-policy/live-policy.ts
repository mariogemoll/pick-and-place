// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// A policy flying the arm, live, in the reader's browser.
//
// Everything here runs on the page: MuJoCo's WebAssembly build steps the same
// compiled scene the Python evaluation steps, and the state flow policy runs
// through onnxruntime-web from the checkpoint's own weights. Drag the cube and
// the target plate anywhere in the workspace and press run; nothing about the
// resulting episode was decided in advance.
//
// What it is not: a benchmark. The engine is a different build, the arithmetic
// rounds differently, and one episode is one episode. The scored numbers come
// from the Python harness over frozen scenario manifests, and this is a way to
// watch the policy work, not a way to measure it.

import * as THREE from 'three';

import { ARM_JOINT_NAMES } from '../../ik/kinematics';
import { simFrameToReal } from '../../joint-frames';
import { loadWebModel } from '../../web-model';
import { createEpisodeReplayScene, type EpisodeReplayScene } from '../episode-replay/scene';
import { createXyMultiDragControls, type XyDragControls } from '../xy-drag-controls';
import { type CubePose, loadPolicyEnvironment } from './environment';
import { type FlowPolicy, loadFlowPolicy } from './flow-policy-runner';
import { buildUi, CANVAS_HEIGHT, CANVAS_WIDTH, type LivePolicyDom } from './ui';

const CUBE_HALF_SIZE = 0.015;
const PLATE_THICKNESS = 0.001;
/** Episodes are scored over 15 s at 10 Hz; the page holds to the same ceiling. */
const MAX_TICKS = 150;
/** How close the cube has to settle to count as placed, matching the oracle. */
const SUCCESS_TOLERANCE_M = 0.03;

export interface LivePolicyOptions {
  modelBasePath?: string;
  modelUrl?: string;
  environmentModelUrl?: string;
  sceneBaseUrl?: string;
  policyBaseUrl?: string;
}

export interface LivePolicyVisualization {
  destroy(): void;
}

interface Placement {
  cube: { x: number; y: number; yaw: number };
  target: { x: number; y: number };
}

function cubePoseFrom(placement: Placement): CubePose {
  const half = placement.cube.yaw / 2;
  return {
    position: [placement.cube.x, placement.cube.y, CUBE_HALF_SIZE],
    quaternion: [Math.cos(half), 0, 0, Math.sin(half)]
  };
}

/** A flat square standing in for the sheet of paper the rig drops onto. */
function createTargetPlate(halfSize: number): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(halfSize * 2, halfSize * 2, PLATE_THICKNESS);
  const material = new THREE.MeshStandardMaterial({ color: 0x1f2933, roughness: 0.9 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.z = PLATE_THICKNESS / 2;
  return mesh;
}

/**
 * An invisible box on the cube, so the pointer has something to pick up.
 *
 * The cube itself is drawn deep inside the replay scene and is not exposed as
 * an object to drag; this rides on top of it and is what the raycaster hits.
 */
function createCubeHandle(): THREE.Mesh {
  const edge = CUBE_HALF_SIZE * 2.4;
  const geometry = new THREE.BoxGeometry(edge, edge, edge);
  const material = new THREE.MeshBasicMaterial({ visible: false });
  return new THREE.Mesh(geometry, material);
}

export async function initLivePolicyVisualization(
  parent: HTMLElement,
  options: LivePolicyOptions = {}
): Promise<LivePolicyVisualization> {
  const dom: LivePolicyDom = buildUi(parent);
  dom.status.textContent = 'Loading the simulator and the policy...';

  const [model, environmentModel] = await Promise.all([
    loadWebModel(options.modelUrl ?? '/so101.json'),
    loadWebModel(options.environmentModelUrl ?? '/environment.json')
  ]);
  const environment = await loadPolicyEnvironment(options.sceneBaseUrl ?? '/policy-scene');
  const flowPolicy: FlowPolicy = await loadFlowPolicy(options.policyBaseUrl ?? '/flow-policy');

  const scene: EpisodeReplayScene = createEpisodeReplayScene(dom.viewport, model, {
    modelBasePath: options.modelBasePath,
    environmentModel
  });

  const plate = createTargetPlate(environment.manifest.dropZoneHalfSize);
  scene.scene.add(plate);
  const cubeHandle = createCubeHandle();
  scene.scene.add(cubeHandle);

  const placement: Placement = {
    cube: { x: 0.187, y: 0.018, yaw: -0.573 },
    target: { x: 0.248, y: -0.143 }
  };

  let running = false;
  let tick = 0;
  let disposed = false;
  let outcome = '';

  function draw(): void {
    const joints = environment.jointAnglesRad();
    for (const [index, name] of [...ARM_JOINT_NAMES, 'gripper'].entries()) {
      scene.setJoint(name, joints[index]);
    }
    const pose = environment.cubePose();
    scene.setCubeTransform(pose.position[0], pose.position[1], pose.position[2], pose.quaternion);
    cubeHandle.position.set(pose.position[0], pose.position[1], pose.position[2]);
    const [targetX, targetY] = environment.targetXy();
    scene.setTarget(targetX, targetY);
    plate.position.x = targetX;
    plate.position.y = targetY;
  }

  function distanceToTarget(): number {
    const pose = environment.cubePose();
    const [targetX, targetY] = environment.targetXy();
    return Math.hypot(pose.position[0] - targetX, pose.position[1] - targetY);
  }

  function updateStatus(): void {
    const seconds = (tick / environment.manifest.policyHz).toFixed(1);
    const error = (distanceToTarget() * 1000).toFixed(0);
    dom.status.textContent = outcome !== ''
      ? `${outcome} — ${seconds}s, cube ${error} mm from the target.`
      : `${seconds}s — cube ${error} mm from the target.`;
  }

  function resetEpisode(): void {
    running = false;
    tick = 0;
    outcome = '';
    flowPolicy.reset();
    environment.reset({
      cube: cubePoseFrom(placement),
      targetXy: [placement.target.x, placement.target.y],
      initialJointsReal: [...simFrameToReal(environment.manifest.neutralJointsRad)]
    });
    dom.run.textContent = 'Run';
    dom.hint.textContent = 'Drag the cube and the target plate, then run.';
    draw();
    updateStatus();
  }

  function finish(message: string): void {
    running = false;
    outcome = message;
    dom.run.textContent = 'Run';
    updateStatus();
  }

  async function stepOnce(): Promise<void> {
    environment.step(
      await flowPolicy.act(
        environment.observe(),
        environment.cubePose(),
        environment.targetXy()
      )
    );
    tick += 1;
    draw();
    updateStatus();
    if (distanceToTarget() < SUCCESS_TOLERANCE_M && environment.cubePose().position[2] < 0.02) {
      finish('Placed');
    } else if (tick >= MAX_TICKS) {
      finish('Out of time');
    }
  }

  // Stepping is asynchronous because inference is, so the loop is a chain
  // rather than a timer: each tick is scheduled only once the previous one has
  // finished, which keeps the arm from running ahead of the policy on a slow
  // machine instead of dropping frames.
  async function loop(): Promise<void> {
    while (running && !disposed) {
      const started = performance.now();
      await stepOnce();
      const remaining = environment.stepSeconds * 1000 - (performance.now() - started);
      if (remaining > 0) {
        await new Promise(resolve => setTimeout(resolve, remaining));
      }
    }
  }

  const drag: XyDragControls = createXyMultiDragControls({
    camera: scene.camera,
    domElement: scene.renderer.domElement,
    orbitControls: scene.orbitControls,
    targets: [
      {
        object: cubeHandle,
        onDrag: (x: number, y: number): void => {
          placement.cube.x = x;
          placement.cube.y = y;
          resetEpisode();
        }
      },
      {
        object: plate,
        onDrag: (x: number, y: number): void => {
          placement.target.x = x;
          placement.target.y = y;
          resetEpisode();
        }
      }
    ]
  });

  dom.run.addEventListener('click', () => {
    if (running) {
      finish('Stopped');
      return;
    }
    if (tick >= MAX_TICKS || outcome !== '') {
      resetEpisode();
    }
    running = true;
    outcome = '';
    dom.run.textContent = 'Stop';
    drag.setEnabled(false);
    void loop().finally(() => { drag.setEnabled(true); });
  });

  dom.reset.addEventListener('click', () => { resetEpisode(); });

  const onResize = (): void => { scene.resize(); };
  window.addEventListener('resize', onResize);

  function render(): void {
    if (disposed) { return; }
    requestAnimationFrame(render);
    scene.orbitControls.update();
    scene.renderer.render(scene.scene, scene.camera);
  }

  resetEpisode();
  render();

  return {
    destroy(): void {
      disposed = true;
      running = false;
      window.removeEventListener('resize', onResize);
      drag.destroy();
      scene.destroy();
      environment.destroy();
      void flowPolicy.destroy();
      plate.geometry.dispose();
      (plate.material as THREE.Material).dispose();
      cubeHandle.geometry.dispose();
      (cubeHandle.material as THREE.Material).dispose();
    }
  };
}

export { CANVAS_HEIGHT, CANVAS_WIDTH };
