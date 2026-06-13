// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import type { WebModel } from '../web-model';
import type { ArmJointName } from './kinematics';

function quaternionFromWeb(
  [w, x, y, z]: [number, number, number, number]
): THREE.Quaternion {
  return new THREE.Quaternion(x, y, z, w);
}

// World transform of the named body for a given joint pose. Bodies are listed
// parent before child, so a single forward pass suffices. Each hinge joint
// rotates its body about the body origin around the joint's local axis (every
// SO-101 joint sits at its body origin with a local +z axis), matching how
// `buildWebModel` inserts the rotation pivots.
export function bodyWorldTransform(
  model: WebModel,
  jointAngles: Partial<Record<ArmJointName, number>>,
  bodyName: string
): THREE.Matrix4 {
  const worlds = new Map<string, THREE.Matrix4>();
  for (const body of model.bodies) {
    const local = new THREE.Matrix4().compose(
      new THREE.Vector3(...body.position),
      quaternionFromWeb(body.quaternion),
      new THREE.Vector3(1, 1, 1)
    );
    for (const joint of body.joints) {
      const angle = jointAngles[joint.name as ArmJointName];
      if (joint.type === 'hinge' && angle !== undefined && angle !== 0) {
        local.multiply(
          new THREE.Matrix4().makeRotationFromQuaternion(
            new THREE.Quaternion().setFromAxisAngle(
              new THREE.Vector3(...joint.axis).normalize(),
              angle
            )
          )
        );
      }
    }
    const parentWorld = worlds.get(body.parent);
    const world = parentWorld ? parentWorld.clone().multiply(local) : local;
    worlds.set(body.name, world);
    if (body.name === bodyName) {
      return world;
    }
  }
  throw new Error(`Body ${bodyName} not found in model`);
}
