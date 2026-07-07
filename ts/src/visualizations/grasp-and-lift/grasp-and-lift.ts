// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { loadWebModel } from '../../web-model';
import { SAFETY_MARGIN } from '../grasp-pose-shared/bodies';
import {
  createWorldFromCubeContactMatrix,
  CUBE_HALF_SIZE,
  type CubePose,
  DEFAULT_CUBE_POSE } from '../grasp-pose-shared/body-factories';
import {
  GRIPPER_CLOSED,
  GRIPPER_CONTACT,
  GRIPPER_OPEN
} from '../pick-and-place/trajectory';
import { createSimpleGraspMatrix } from '../simple-grasp-pose/pose';
import { createGraspAndLiftScene } from './scene';
import { buildUi } from './ui';

export interface GraspAndLiftVisualization {
  destroy(): void;
}

export interface GraspAndLiftOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

// The cube face the jaws close onto. +x keeps the jaw-closing direction
// perpendicular to the side camera (which looks along +y), so the pinch is
// read edge-on as a horizontal squeeze and the lift reads as pure vertical.
const GRASP_FACE = '+x' as const;

// Sub-phase boundaries inside one 0 → 1 forward pass of the loop. The
// gripper is pre-positioned at the cube, so the visible action is just
// pause → close → lift → hold. The wrap itself is the reset – an instant
// teleport back to the open, cube-on-the-floor start state.
const START_PAUSE_END = 0.10;
const CLOSE_END = 0.42;
const LIFT_END = 0.85;
// [LIFT_END, 1] = hold at the top, then teleport on the wrap.

// How far the grasped cube is lifted – large enough that the lift reads
// clearly in the side view.
const LIFT_HEIGHT = 0.05;
// Length of one pause → close → lift → hold loop.
const LOOP_SECONDS = 3.2;

function smoothstep(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

interface GraspAndLiftState {
  gripperAngle: number;
  zOffset: number;
  push: number;
  lift: number;
}

// The open, cube-on-the-floor start state – held during the start pause and
// snapped to on every loop wrap (the teleport reset).
const START_STATE: GraspAndLiftState = {
  gripperAngle: GRIPPER_OPEN,
  zOffset: 0,
  push: 0,
  lift: 0
};

// One forward pass (p ∈ [0, 1]): start pause → close on the cube → lift →
// hold at the top. The wrap back to p = 0 is the teleport reset.
function evaluateLoop(p: number): GraspAndLiftState {
  if (p < START_PAUSE_END) {
    // Short open hold at the start before closing.
    return START_STATE;
  }
  if (p < CLOSE_END) {
    // Close: the moving jaw shoves the cube flush against the fixed jaw –
    // the same contact schedule as pick-and-place stage 3.
    const ease = smoothstep((p - START_PAUSE_END) / (CLOSE_END - START_PAUSE_END));
    const gripperAngle = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * ease;
    const push = gripperAngle < GRIPPER_CONTACT
      ? SAFETY_MARGIN * (GRIPPER_CONTACT - gripperAngle) /
        (GRIPPER_CONTACT - GRIPPER_CLOSED)
      : 0;
    return { gripperAngle, zOffset: 0, push, lift: 0 };
  }
  if (p < LIFT_END) {
    // Lift off with the grasped cube.
    const ease = smoothstep((p - CLOSE_END) / (LIFT_END - CLOSE_END));
    return {
      gripperAngle: GRIPPER_CLOSED,
      zOffset: LIFT_HEIGHT * ease,
      push: SAFETY_MARGIN,
      lift: LIFT_HEIGHT * ease
    };
  }
  // Hold the lifted pose until the loop wraps and teleports back to start.
  return {
    gripperAngle: GRIPPER_CLOSED,
    zOffset: LIFT_HEIGHT,
    push: SAFETY_MARGIN,
    lift: LIFT_HEIGHT
  };
}

export async function GraspAndLift(
  parent: HTMLElement,
  options: GraspAndLiftOptions = {}
): Promise<GraspAndLiftVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const ui = buildUi(parent);
  const vizScene = createGraspAndLiftScene(ui.viewport, model, options.modelBasePath);

  // The grasp matrix is the geometric truth shared with pick-and-place: it
  // places the gripper pointing straight down at the cube with its jaws
  // straddling the grasped face. We hold its orientation fixed and only
  // translate vertically for the hover/approach and the lift.
  const graspMatrix = createSimpleGraspMatrix(GRASP_FACE, DEFAULT_CUBE_POSE);
  if (!graspMatrix) {
    throw new Error(`Unable to compute grasp matrix for face ${GRASP_FACE}`);
  }
  const graspPosition = new THREE.Vector3();
  const graspQuaternion = new THREE.Quaternion();
  graspMatrix.decompose(
    graspPosition, graspQuaternion, new THREE.Vector3()
  );
  // Direction from the origin toward the cube's open-start offset. Closing the
  // gripper removes this offset, so the cube slides into the origin before the
  // straight vertical lift.
  const startOffsetDirection = new THREE.Vector3(0, 0, 1).transformDirection(
    createWorldFromCubeContactMatrix(GRASP_FACE, DEFAULT_CUBE_POSE)
  );

  const gripperPosition = new THREE.Vector3();
  const renderFrame = (seconds: number): void => {
    const phase = (seconds % LOOP_SECONDS) / LOOP_SECONDS;
    const state = evaluateLoop(phase);
    const horizontalOffset = SAFETY_MARGIN - state.push;
    gripperPosition.set(
      graspPosition.x + startOffsetDirection.x * SAFETY_MARGIN,
      graspPosition.y + startOffsetDirection.y * SAFETY_MARGIN,
      graspPosition.z + state.zOffset
    );
    vizScene.setGripperPose(gripperPosition, graspQuaternion);
    vizScene.setGripperAngle(state.gripperAngle);
    const cubePose: CubePose = {
      x: DEFAULT_CUBE_POSE.x + startOffsetDirection.x * horizontalOffset,
      y: DEFAULT_CUBE_POSE.y + startOffsetDirection.y * horizontalOffset,
      z: CUBE_HALF_SIZE + state.lift,
      roll: 0,
      pitch: 0,
      yaw: 0
    };
    vizScene.setCubePose(cubePose);
  };

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  let animationFrameId = 0;
  let destroyed = false;
  function animate(time: number): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    renderFrame(time / 1000);
    vizScene.orbitControls.update();
    vizScene.renderer.render(vizScene.scene, vizScene.camera);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      vizScene.destroy();
      ui.root.remove();
    }
  };
}
