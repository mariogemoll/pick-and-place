// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  applyGripperTransform,
  applyGripperTransformProgress,
  HORIZONTAL_CAMERA_ON_TOP_ANGLE,
  SAFETY_MARGIN,
  type TransformStage
} from './bodies';
import {
  createCubeFromContactMatrix,
  createGripperFromContactMatrix,
  createWorldFromCubeContactMatrix,
  createWorldFromCubeMatrix,
  CUBE_HALF_SIZE } from './body-factories';

describe('pregrasp-pose-breakdown contact transforms', () => {
  const cameraModulePosition = new THREE.Vector3(0.0025, 0.073357, 0.007515);
  const animatedStagePairs: [TransformStage, TransformStage][] = [
    ['jaw-contact-origin', 'unaligned'],
    ['aligned', 'jaw-contact-origin'],
    ['safety-margin', 'aligned'],
    ['hinge', 'safety-margin']
  ];

  it('places the cube body on the floor', () => {
    const worldFromCube = createWorldFromCubeMatrix();
    const cubeCenter = new THREE.Vector3().setFromMatrixPosition(worldFromCube);

    expect(cubeCenter.z).toBe(CUBE_HALF_SIZE);
    expect(cubeCenter.z - CUBE_HALF_SIZE).toBe(0);
  });

  it('composes the cube contact frame through the grounded cube body', () => {
    const expected = createWorldFromCubeMatrix()
      .multiply(createCubeFromContactMatrix());

    expect(createWorldFromCubeContactMatrix().equals(expected)).toBe(true);
  });

  it('expresses the contact point directly in the gripper body frame', () => {
    const gripperFromContact = createGripperFromContactMatrix();
    const contactPosition = new THREE.Vector3()
      .setFromMatrixPosition(gripperFromContact);

    expect(contactPosition.x).toBeCloseTo(-0.00788);
    // y is deliberately zeroed (see JAW_CONTACT_POSITION), not the vendor
    // box's -0.00015.
    expect(contactPosition.y).toBe(0);
    expect(contactPosition.z).toBeCloseTo(-0.099363);
  });

  it.each([-Math.PI, -0.4, 0, 0.7, Math.PI])(
    'keeps the jaw at the safety margin and flush at angle %s',
    angle => {
      const gripper = new THREE.Object3D();
      applyGripperTransform(gripper, 'hinge', angle);

      const worldFromJawContact = gripper.matrix.clone()
        .multiply(createGripperFromContactMatrix());
      const actualPosition = new THREE.Vector3()
        .setFromMatrixPosition(worldFromJawContact);
      const expectedPosition = new THREE.Vector3()
        .set(0, 0, -SAFETY_MARGIN)
        .applyMatrix4(createWorldFromCubeContactMatrix());
      const actualNormal = new THREE.Vector3(0, 0, 1)
        .transformDirection(worldFromJawContact);
      const expectedNormal = new THREE.Vector3(0, 0, 1)
        .transformDirection(createWorldFromCubeContactMatrix());

      expect(actualPosition.distanceTo(expectedPosition)).toBeLessThan(1e-12);
      expect(actualNormal.distanceTo(expectedNormal)).toBeLessThan(1e-12);
    }
  );

  it('defines zero degrees as horizontal with the camera on top', () => {
    const gripper = new THREE.Object3D();
    applyGripperTransform(gripper, 'hinge', 0);

    const cameraPosition = cameraModulePosition.clone()
      .applyMatrix4(gripper.matrix);
    const jawContactPosition = new THREE.Vector3()
      .setFromMatrixPosition(
        gripper.matrix.clone().multiply(createGripperFromContactMatrix())
      );

    expect(cameraPosition.z).toBeGreaterThan(jawContactPosition.z);
  });

  it.each(animatedStagePairs)(
    'animates %s from the completed %s transform',
    (stage, previousStage) => {
      const start = new THREE.Object3D();
      const previous = new THREE.Object3D();
      const end = new THREE.Object3D();
      const completed = new THREE.Object3D();

      applyGripperTransformProgress(start, stage, 0.7, 0);
      applyGripperTransform(previous, previousStage, 0.7);
      applyGripperTransformProgress(end, stage, 0.7, 1);
      applyGripperTransform(completed, stage, 0.7);

      expectMatrixToBeCloseTo(start.matrix, previous.matrix);
      expectMatrixToBeCloseTo(end.matrix, completed.matrix);
    }
  );

  it.each([0, 0.25, 0.5, 0.75, 1])(
    'backs off only along the contact normal at progress %s',
    progress => {
      const gripper = new THREE.Object3D();
      applyGripperTransformProgress(gripper, 'safety-margin', 0, progress);

      const jawContactPosition = new THREE.Vector3().setFromMatrixPosition(
        gripper.matrix.clone().multiply(createGripperFromContactMatrix())
      );
      const expectedPosition = new THREE.Vector3()
        .set(0, 0, -SAFETY_MARGIN * progress)
        .applyMatrix4(createWorldFromCubeContactMatrix());

      expect(jawContactPosition.distanceTo(expectedPosition)).toBeLessThan(1e-12);
    }
  );

  it.each([0, 0.25, 0.5, 0.75, 1])(
    'keeps the jaw contact fixed while hinging at progress %s',
    progress => {
      const gripper = new THREE.Object3D();
      applyGripperTransformProgress(gripper, 'hinge', 0.7, progress);

      const jawContactPosition = new THREE.Vector3().setFromMatrixPosition(
        gripper.matrix.clone().multiply(createGripperFromContactMatrix())
      );
      const expectedPosition = new THREE.Vector3()
        .set(0, 0, -SAFETY_MARGIN)
        .applyMatrix4(createWorldFromCubeContactMatrix());

      expect(jawContactPosition.distanceTo(expectedPosition)).toBeLessThan(1e-12);
    }
  );

  it('backs the jaw contact point 1 cm away from the cube', () => {
    const gripper = new THREE.Object3D();
    applyGripperTransform(gripper, 'safety-margin', 0);

    const actualPosition = new THREE.Vector3().setFromMatrixPosition(
      gripper.matrix.clone().multiply(createGripperFromContactMatrix())
    );
    const cubeContactPosition = new THREE.Vector3().setFromMatrixPosition(
      createWorldFromCubeContactMatrix()
    );

    expect(actualPosition.distanceTo(cubeContactPosition))
      .toBeCloseTo(SAFETY_MARGIN);
  });

  it('combines the final transform into the same pose as the hinge stage', () => {
    const hingeGripper = new THREE.Object3D();
    const finalGripper = new THREE.Object3D();

    applyGripperTransform(hingeGripper, 'hinge', 0.7);
    applyGripperTransform(finalGripper, 'final', 0.7);

    expect(finalGripper.matrix.equals(hingeGripper.matrix)).toBe(true);
  });

  it.each([0, 0.25, 0.5, 0.75, 1])(
    'faithfully composes the combined transform at progress %s',
    progress => {
      const angle = 0.7;
      const gripper = new THREE.Object3D();
      applyGripperTransformProgress(gripper, 'final', angle, progress);

      const expected = interpolateFromIdentity(
        createWorldFromCubeContactMatrix(), progress
      )
        .multiply(new THREE.Matrix4().makeTranslation(
          0, 0, -SAFETY_MARGIN * progress
        ))
        .multiply(new THREE.Matrix4().makeRotationZ(
          (HORIZONTAL_CAMERA_ON_TOP_ANGLE - angle) * progress
        ))
        .multiply(interpolateFromIdentity(
          createGripperFromContactMatrix().invert(), progress
        ));

      expectMatrixToBeCloseTo(gripper.matrix, expected);
    }
  );
});

function expectMatrixToBeCloseTo(
  actual: THREE.Matrix4,
  expected: THREE.Matrix4
): void {
  for (const [index, value] of actual.elements.entries()) {
    expect(value).toBeCloseTo(expected.elements[index] ?? 0);
  }
}

function interpolateFromIdentity(
  target: THREE.Matrix4,
  progress: number
): THREE.Matrix4 {
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  target.decompose(position, quaternion, scale);

  return new THREE.Matrix4().compose(
    position.multiplyScalar(progress),
    new THREE.Quaternion().slerp(quaternion, progress),
    new THREE.Vector3(1, 1, 1).lerp(scale, progress)
  );
}
