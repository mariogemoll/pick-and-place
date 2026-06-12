// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import type { WebModel } from '../../web-model';
import {
  type BodyPart,
  createCubeBody,
  createGripperBody,
  createGripperFromContactMatrix,
  createWorldFromCubeContactMatrix,
  createWorldFromCubeMatrix,
  type CubeFace,
  type CubePose,
  DEFAULT_CUBE_POSE
} from './body-factories';
import { createBodyMaterials } from './materials';

export interface PregraspPoseBodies {
  root: THREE.Group;
  updateCubePose(pose: CubePose): void;
  destroy(): void;
}

export type { CubeFace, CubePose } from './body-factories';
export type BodySelection = 'combined' | 'cube' | 'gripper';
export type TransformStage =
  'unaligned' | 'jaw-contact-origin' | 'aligned' | 'safety-margin' | 'hinge' |
  'final';

export const SAFETY_MARGIN = 0.01;
export const HORIZONTAL_CAMERA_ON_TOP_ANGLE = -Math.PI / 2;

export async function createBodies(
  model: WebModel,
  modelBasePath: string,
  selection: BodySelection = 'combined',
  transformStage: TransformStage = 'unaligned',
  hingeAngle = 0
): Promise<PregraspPoseBodies> {
  const root = new THREE.Group();
  root.name = 'bodies';
  const materials = createBodyMaterials();
  const parts: BodyPart[] = [];

  let gripper: BodyPart | undefined;
  if (selection !== 'cube') {
    gripper = await createGripperBody(model, modelBasePath, materials);
    parts.push(gripper);
  }
  if (selection !== 'gripper') {
    parts.push(createCubeBody(materials));
  }
  root.add(...parts.map(part => part.body));
  if (gripper) {
    applyGripperTransform(gripper.body, transformStage, hingeAngle);
  }

  return {
    root,
    updateCubePose(pose: CubePose): void {
      const cubeBody = root.getObjectByName('cube_body');
      if (cubeBody) {
        createWorldFromCubeMatrix(pose).decompose(
          cubeBody.position, cubeBody.quaternion, cubeBody.scale
        );
      }
    },
    destroy(): void {
      for (const part of parts) { part.destroy(); }
      materials.destroy();
    }
  };
}

export function applyGripperTransform(
  gripper: THREE.Object3D,
  stage: TransformStage,
  hingeAngle: number,
  face: CubeFace = '+x',
  pose: CubePose = DEFAULT_CUBE_POSE
): void {
  applyGripperTransformProgress(gripper, stage, hingeAngle, 1, face, pose);
}

export function applyGripperTransformProgress(
  gripper: THREE.Object3D,
  stage: TransformStage,
  hingeAngle: number,
  progress: number,
  face: CubeFace = '+x',
  pose: CubePose = DEFAULT_CUBE_POSE
): void {
  const clampedProgress = THREE.MathUtils.clamp(progress, 0, 1);
  const contactFromGripper = interpolateFromIdentity(
    createGripperFromContactMatrix().invert(),
    stage === 'jaw-contact-origin' || stage === 'final' ? clampedProgress : 1
  );

  if (stage === 'unaligned') {
    gripper.matrix.identity();
  } else if (stage === 'jaw-contact-origin') {
    gripper.matrix.copy(contactFromGripper);
  } else {
    const worldFromCubeContact = interpolateFromIdentity(
      createWorldFromCubeContactMatrix(face, pose),
      stage === 'aligned' || stage === 'final' ? clampedProgress : 1
    );
    const cubeContactFromSafeContact = new THREE.Matrix4().makeTranslation(
      0,
      0,
      stage === 'safety-margin' || stage === 'final'
        ? -SAFETY_MARGIN * clampedProgress :
        stage === 'aligned' ? 0 : -SAFETY_MARGIN
    );
    const contactFromJawContact = new THREE.Matrix4().makeRotationZ(
      stage === 'hinge' || stage === 'final'
        ? (HORIZONTAL_CAMERA_ON_TOP_ANGLE - hingeAngle) * clampedProgress
        : 0
    );
    gripper.matrix.copy(worldFromCubeContact)
      .multiply(cubeContactFromSafeContact)
      .multiply(contactFromJawContact)
      .multiply(contactFromGripper);
  }

  gripper.matrix.decompose(gripper.position, gripper.quaternion, gripper.scale);
}

function interpolateFromIdentity(
  target: THREE.Matrix4,
  progress: number
): THREE.Matrix4 {
  const targetPosition = new THREE.Vector3();
  const targetQuaternion = new THREE.Quaternion();
  const targetScale = new THREE.Vector3();
  target.decompose(targetPosition, targetQuaternion, targetScale);

  return new THREE.Matrix4().compose(
    targetPosition.multiplyScalar(progress),
    new THREE.Quaternion().slerp(targetQuaternion, progress),
    new THREE.Vector3(1, 1, 1).lerp(targetScale, progress)
  );
}
