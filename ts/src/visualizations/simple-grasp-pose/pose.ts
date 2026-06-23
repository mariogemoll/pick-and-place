// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { SAFETY_MARGIN } from '../grasp-pose-shared/bodies';
import {
  createGripperFromContactMatrix,
  createWorldFromCubeContactMatrix,
  type CubeFace,
  type CubePose
} from '../grasp-pose-shared/body-factories';

const VERTICAL_TOLERANCE = 1e-9;
const WORLD_UP = new THREE.Vector3(0, 0, 1);

export function createSimpleGraspMatrix(
  face: CubeFace,
  pose: CubePose
): THREE.Matrix4 | undefined {
  const worldFromCubeContact = createWorldFromCubeContactMatrix(face, pose);
  const inwardNormal = new THREE.Vector3(0, 0, 1)
    .transformDirection(worldFromCubeContact);
  if (Math.abs(inwardNormal.dot(WORLD_UP)) > VERTICAL_TOLERANCE) {
    return undefined;
  }

  const gripperY = new THREE.Vector3().crossVectors(WORLD_UP, inwardNormal);
  const worldFromGripper = new THREE.Matrix4().makeBasis(
    inwardNormal,
    gripperY,
    WORLD_UP
  );
  const cubeContactPosition = new THREE.Vector3()
    .setFromMatrixPosition(worldFromCubeContact);
  const jawContactPosition = cubeContactPosition
    .addScaledVector(inwardNormal, -SAFETY_MARGIN);
  const gripperFromContact = createGripperFromContactMatrix();
  const jawOffset = new THREE.Vector3()
    .setFromMatrixPosition(gripperFromContact)
    .applyMatrix3(new THREE.Matrix3().setFromMatrix4(worldFromGripper));
  worldFromGripper.setPosition(jawContactPosition.sub(jawOffset));

  return worldFromGripper;
}
