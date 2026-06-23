// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { SAFETY_MARGIN } from '../pregrasp-pose-shared/bodies';
import {
  CUBE_HALF_SIZE,
  type CubePose,
  GRIPPER_TARGET_POSITION,
  JAW_CONTACT_POSITION
} from '../pregrasp-pose-shared/body-factories';

// Distance, along the jaw-closing axis, from the cube centre to the IK target
// (the jaw contact projected onto the roll axis). As in the simple pregrasp, the
// fixed jaw is parked SAFETY_MARGIN clear of the cube's near face rather than
// straddling the centre, so closing the jaws pushes the cube onto the fixed jaw.
// JAW_CONTACT_POSITION.x is the (negative) inset of the fixed-jaw contact from
// the roll axis, so adding it backs the offset off by that much.
const FACE_OFFSET = CUBE_HALF_SIZE + SAFETY_MARGIN + JAW_CONTACT_POSITION.x;

// World-from-gripper matrix for a grasp whose tool reaches the cube along
// `approach` (a unit world direction from wrist to target) with the jaws closing
// along the horizontal `closingAzimuth`. The fixed jaw sits one face offset to
// the −closing side, SAFETY_MARGIN clear of the cube's near face. A straight-down
// approach gives the square top-down grasp; tilting it trades squareness for
// reach when the cube is outside the top-down region.
export function createGraspMatrix(
  pose: CubePose,
  closingAzimuth: number,
  approach: THREE.Vector3
): THREE.Matrix4 {
  // gripper z is the tool/roll axis, pointing opposite the approach.
  const z = approach.clone().negate().normalize();
  const x = new THREE.Vector3(Math.cos(closingAzimuth), Math.sin(closingAzimuth), 0);
  const y = new THREE.Vector3().crossVectors(z, x).normalize();
  // Re-derive x from y and z so the basis stays orthonormal for a tilted z.
  x.crossVectors(y, z).normalize();

  const matrix = new THREE.Matrix4().makeBasis(x, y, z);
  // IK target: cube centre backed off along −closing so the fixed jaw stands off
  // the near face. Then place the gripper so R * GRIPPER_TARGET_POSITION lands there.
  const target = new THREE.Vector3(pose.x, pose.y, pose.z).addScaledVector(x, -FACE_OFFSET);
  const offset = GRIPPER_TARGET_POSITION.clone()
    .applyMatrix3(new THREE.Matrix3().setFromMatrix4(matrix));
  matrix.setPosition(target.sub(offset));
  return matrix;
}
