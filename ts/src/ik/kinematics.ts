// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  GRIPPER_TARGET_POSITION
} from '../visualizations/grasp-pose-shared/body-factories';
import type { WebJoint, WebModel } from '../web-model';

// The five actuated joints of the SO-101 arm, base to tool, in chain order.
export const ARM_JOINT_NAMES = [
  'shoulder_pan',
  'shoulder_lift',
  'elbow_flex',
  'wrist_flex',
  'wrist_roll'
] as const;

export type ArmJointName = typeof ARM_JOINT_NAMES[number];

export const NEUTRAL_ARM_JOINTS: Record<ArmJointName, number> = {
  shoulder_pan: 0,
  shoulder_lift: 0,
  elbow_flex: 0,
  wrist_flex: 0,
  wrist_roll: -Math.PI / 2
};

// A link of the planar arm, expressed in the radial–height plane that the
// closed-form IK solves in. `radial` runs outward from the pan axis, `height`
// is world z; the link's constant out-of-plane (lateral) offset is dropped, so
// `length` is the in-plane projection that the 2R solver treats as a link.
export interface PlanarSegment {
  radial: number;
  height: number;
  length: number;
}

export interface JointLimit {
  min: number;
  max: number;
}

// Kinematic constants the closed-form IK needs, all derived from the loaded
// model so they stay in sync with `so101.json`.
// Lengths are metres, limits radians.
export interface So101Kinematics {
  // Horizontal location of the vertical shoulder_pan axis (world x, y).
  panAxis: THREE.Vector2;
  // shoulder_lift pivot in the radial–height plane: radial offset from the pan
  // axis, height above the floor.
  shoulderLift: { radial: number; height: number };
  upperArm: PlanarSegment; // shoulder_lift -> elbow_flex
  lowerArm: PlanarSegment; // elbow_flex -> wrist_flex
  // wrist_flex -> gripper target along the approach direction; the tool that
  // the wrist pitch carries.
  toolLength: number;
  // The constant roll angle offset (in radians) of the gripper x-axis relative
  // to the ideal zero-roll pitch axis (due to the 2.8° arm twist).
  wristRollZeroTwist: number;
  // Joint limits for the five arm joints.
  jointLimits: Record<ArmJointName, JointLimit>;
}

function quaternionFromWeb(
  [w, x, y, z]: [number, number, number, number]
): THREE.Quaternion {
  return new THREE.Quaternion(x, y, z, w);
}

// World transform of every body at the zero pose. Bodies are listed parent
// before child, so a single forward pass suffices.
function worldMatrices(model: WebModel): Map<string, THREE.Matrix4> {
  const worlds = new Map<string, THREE.Matrix4>();
  for (const body of model.bodies) {
    const local = new THREE.Matrix4().compose(
      new THREE.Vector3(...body.position),
      quaternionFromWeb(body.quaternion),
      new THREE.Vector3(1, 1, 1)
    );
    const parentWorld = worlds.get(body.parent);
    worlds.set(
      body.name,
      parentWorld ? parentWorld.clone().multiply(local) : local
    );
  }
  return worlds;
}

interface JointFrame {
  position: THREE.Vector3;
  axis: THREE.Vector3;
  joint: WebJoint;
}

function jointFrame(
  model: WebModel,
  worlds: Map<string, THREE.Matrix4>,
  name: ArmJointName
): JointFrame {
  for (const body of model.bodies) {
    const joint = body.joints.find(candidate => candidate.name === name);
    if (joint === undefined) { continue; }
    const world = worlds.get(body.name);
    if (world === undefined) { break; }
    return {
      position: new THREE.Vector3(...joint.position).applyMatrix4(world),
      axis: new THREE.Vector3(...joint.axis).transformDirection(world).normalize(),
      joint
    };
  }
  throw new Error(`Joint ${name} not found in model`);
}

function jointLimit(joint: WebJoint): JointLimit {
  if (joint.limited && joint.range) {
    return { min: joint.range[0], max: joint.range[1] };
  }
  return { min: -Infinity, max: Infinity };
}

export function deriveSo101Kinematics(model: WebModel): So101Kinematics {
  const worlds = worldMatrices(model);
  const pan = jointFrame(model, worlds, 'shoulder_pan');
  const lift = jointFrame(model, worlds, 'shoulder_lift');
  const elbow = jointFrame(model, worlds, 'elbow_flex');
  const wristFlex = jointFrame(model, worlds, 'wrist_flex');
  const wristRoll = jointFrame(model, worlds, 'wrist_roll');

  const panAxis = new THREE.Vector2(pan.position.x, pan.position.y);

  // The radial axis is horizontal and perpendicular to the (lateral) pitch
  // axis: pitch × up, projected to the floor and oriented outward toward the
  // arm. This drops the arm's constant lateral offset automatically.
  const radialDir = new THREE.Vector3()
    .crossVectors(lift.axis, new THREE.Vector3(0, 0, 1));
  const radial = new THREE.Vector2(radialDir.x, radialDir.y).normalize();
  const toWrist = new THREE.Vector2(
    wristFlex.position.x - panAxis.x,
    wristFlex.position.y - panAxis.y
  );
  if (radial.dot(toWrist) < 0) { radial.negate(); }

  const radialOf = (position: THREE.Vector3): number =>
    (position.x - panAxis.x) * radial.x + (position.y - panAxis.y) * radial.y;
  const segment = (
    from: THREE.Vector3,
    to: THREE.Vector3
  ): PlanarSegment => {
    const dr = radialOf(to) - radialOf(from);
    const dh = to.z - from.z;
    return { radial: dr, height: dh, length: Math.hypot(dr, dh) };
  };

  const target = GRIPPER_TARGET_POSITION.clone()
    .applyMatrix4(worlds.get('gripper') ?? new THREE.Matrix4());

  const gripperWorld = worlds.get('gripper') ?? new THREE.Matrix4();
  const gripperX = new THREE.Vector3(1, 0, 0).transformDirection(gripperWorld).normalize();
  const a = new THREE.Vector3().subVectors(target, wristFlex.position).normalize();
  // Pitch axis n_V = (0, 1, 0) at pan=0 (normal to the radial plane).
  const pitchAxis = new THREE.Vector3(0, 1, 0);
  const idealX = new THREE.Vector3().crossVectors(a, pitchAxis).normalize();
  const idealY = pitchAxis;
  const wristRollZeroTwist = Math.atan2(gripperX.dot(idealY), gripperX.dot(idealX));

  return {
    panAxis,
    shoulderLift: { radial: radialOf(lift.position), height: lift.position.z },
    upperArm: segment(lift.position, elbow.position),
    lowerArm: segment(elbow.position, wristFlex.position),
    toolLength: segment(wristFlex.position, target).length,
    wristRollZeroTwist,
    jointLimits: {
      shoulder_pan: jointLimit(pan.joint),
      shoulder_lift: jointLimit(lift.joint),
      elbow_flex: jointLimit(elbow.joint),
      wrist_flex: jointLimit(wristFlex.joint),
      wrist_roll: jointLimit(wristRoll.joint)
    }
  };
}
