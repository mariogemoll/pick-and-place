// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { createCubeAprilTags } from '../../apriltag/cube-tags';
import {
  buildWebModel,
  setJointAngle,
  type WebModel
} from '../../web-model';
import type { BodyMaterials } from './materials';

export interface BodyPart {
  body: THREE.Group;
  destroy(): void;
}

const FIXED_JAW_COLLISION_BOXES = [
  {
    position: [-0.00260000024, -0.00199999986, -0.0180251466],
    size: [0.0325999996, 0.0260000005, 0.0190251462]
  },
  { position: [-0.0244, 0, -0.041025], size: [0.00941, 0.016, 0.00412] },
  { position: [-0.02291, 0, -0.05485], size: [0.00962, 0.01134, 0.00981] },
  { position: [-0.0176, 0, -0.0745], size: [0.00601, 0.00805, 0.0098] },
  { position: [-0.01492, -0.00018, -0.0893], size: [0.00503, 0.00703, 0.005] },
  { position: [-0.01189, -0.00015, -0.099363], size: [0.004, 0.00545, 0.005063] }
] as const;

const TIP_BOX = FIXED_JAW_COLLISION_BOXES[5];
export const CUBE_HALF_SIZE = 0.015;
const MARKER_SURFACE_OFFSET = 0.00001;
const TAG_SURFACE_OFFSET = 0.0001;
// The sticker is white, but pure white is not paintable under these lights: at
// an albedo of 1 every channel saturates and all six faces come out the same
// flat white, so the cube reads as a silhouette. A little grey keeps enough
// headroom for the faces to shade apart, which is why the blue cube always
// looked solid -- two of its three channels sit nowhere near clipping.
const CUBE_FACE_COLOR = 0xc7ccd3;

export type CubeFace = '+x' | '-x' | '+y' | '-y' | '+z' | '-z';

export interface CubePose {
  x: number;
  y: number;
  z: number;
  roll: number;
  pitch: number;
  yaw: number;
}

export const DEFAULT_CUBE_POSE: CubePose = {
  x: 0, y: 0, z: CUBE_HALF_SIZE, roll: 0, pitch: 0, yaw: 0
};

export const JAW_CONTACT_POSITION = new THREE.Vector3(
  TIP_BOX.position[0] + TIP_BOX.size[0] + MARKER_SURFACE_OFFSET,
  // Deliberately 0, not TIP_BOX.position[1] (-0.15 mm): that tiny y-offset is
  // an incidental artifact of the hand-tuned collision box, not a real
  // feature of the jaw. Zeroing it keeps the contact point in the pan-axis
  // center plane, which the closed-form IK relies on. The collision box
  // itself keeps its tuned value and must not be "cleaned up".
  0,
  TIP_BOX.position[2]
);

// How far below the cube's vertical center the jaws take hold. Gripping a
// little low leaves the jaw tips clear of the cube's top edge as they close.
export const GRIP_Z_OFFSET = -0.005;

export const CUBE_CONTACT_POSITION = new THREE.Vector3(
  CUBE_HALF_SIZE + MARKER_SURFACE_OFFSET,
  0,
  GRIP_Z_OFFSET
);

// IK position target: the jaw contact point projected onto the wrist-roll axis
// (the gripper frame's z-axis). The displacement from the contact point to this
// point runs along gripper x (the jaw-closing direction and face normal), so
// wrist roll does not change the target's world position. That invariance makes
// the closed-form IK decomposition exact.
export const GRIPPER_TARGET_POSITION = new THREE.Vector3(
  0,
  0,
  TIP_BOX.position[2]
);

export function createWorldFromCubeMatrix(pose: CubePose = DEFAULT_CUBE_POSE): THREE.Matrix4 {
  return new THREE.Matrix4()
    .makeTranslation(pose.x, pose.y, pose.z)
    .multiply(
      new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(pose.roll, pose.pitch, pose.yaw, 'ZYX')
      )
    );
}

export function createCubeFromContactMatrix(): THREE.Matrix4 {
  return new THREE.Matrix4().compose(
    CUBE_CONTACT_POSITION,
    // Point the target frame into the cube so the two surfaces are flush.
    new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, 0)),
    new THREE.Vector3(1, 1, 1)
  );
}

export function createGripperFromContactMatrix(): THREE.Matrix4 {
  return new THREE.Matrix4().compose(
    JAW_CONTACT_POSITION,
    new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 2, 0)),
    new THREE.Vector3(1, 1, 1)
  );
}

// A cube whose six faces carry crisp, geometry-based AprilTags on a white
// surface (rather than textures), so they stay sharp at any zoom level.
export function createCubeAprilTagBody(materials: BodyMaterials): BodyPart {
  const faceMaterial = new THREE.MeshStandardMaterial({
    color: CUBE_FACE_COLOR,
    roughness: 0.72
  });
  const tagMaterial = new THREE.MeshStandardMaterial({ color: 0x000000, roughness: 0.72 });
  const cubePart = createCubeBody(
    materials, Array.from({ length: 6 }, () => faceMaterial), false
  );

  const tags = createCubeAprilTags(CUBE_HALF_SIZE + TAG_SURFACE_OFFSET, tagMaterial);
  cubePart.body.add(tags.group);

  return {
    body: cubePart.body,
    destroy(): void {
      cubePart.destroy();
      tags.dispose();
      faceMaterial.dispose();
      tagMaterial.dispose();
    }
  };
}

export function createWorldFromCubeContactMatrix(
  face: CubeFace = '+x',
  pose: CubePose = DEFAULT_CUBE_POSE
): THREE.Matrix4 {
  return createWorldFromCubeMatrix(pose)
    .multiply(cubeFaceRotation(face))
    .multiply(createCubeFromContactMatrix());
}

function cubeFaceRotation(face: CubeFace): THREE.Matrix4 {
  switch (face) {
  case '+x': return new THREE.Matrix4();
  case '+y': return new THREE.Matrix4().makeRotationZ(-Math.PI / 2);
  case '-x': return new THREE.Matrix4().makeRotationZ(Math.PI);
  case '-y': return new THREE.Matrix4().makeRotationZ(Math.PI / 2);
  case '+z': return new THREE.Matrix4().makeRotationY(Math.PI / 2);
  case '-z': return new THREE.Matrix4().makeRotationY(-Math.PI / 2);
  }
}

export async function createGripperBody(
  model: WebModel,
  modelBasePath: string,
  materials: BodyMaterials
): Promise<BodyPart> {
  const builtModel = buildWebModel(model, modelBasePath, 'gripper');
  await builtModel.ready;

  const body = new THREE.Group();
  body.name = 'gripper_body';
  body.add(builtModel.root);
  setJointAngle(model, builtModel.jointPivots, 'gripper', Math.PI / 3);

  const bodyFrameOverlays = new THREE.Group();
  bodyFrameOverlays.name = 'gripper_body_frame_overlays';
  body.add(bodyFrameOverlays);

  const collisionBoxes = new THREE.Group();
  collisionBoxes.name = 'fixed_jaw_collision_boxes';
  collisionBoxes.visible = false;
  const collisionGeometries: THREE.BoxGeometry[] = [];
  for (const [index, box] of FIXED_JAW_COLLISION_BOXES.entries()) {
    const geometry = new THREE.BoxGeometry(
      box.size[0] * 2,
      box.size[1] * 2,
      box.size[2] * 2
    );
    collisionGeometries.push(geometry);
    const collisionBox = new THREE.Mesh(geometry, materials.collision);
    collisionBox.name = `fixed_jaw_col${index}`;
    collisionBox.position.set(box.position[0], box.position[1], box.position[2]);
    collisionBoxes.add(collisionBox);
  }
  bodyFrameOverlays.add(collisionBoxes);

  const markerGeometry = new THREE.CircleGeometry(0.001, 24);
  const marker = new THREE.Mesh(markerGeometry, materials.marker);
  marker.name = 'fixed_jaw_tip_inner_face_center';
  marker.position.copy(JAW_CONTACT_POSITION);
  marker.rotation.y = Math.PI / 2;
  bodyFrameOverlays.add(marker);

  return {
    body,
    destroy(): void {
      markerGeometry.dispose();
      for (const geometry of collisionGeometries) { geometry.dispose(); }
      for (const modelMaterials of builtModel.materialsByName.values()) {
        for (const material of modelMaterials) { material.dispose(); }
      }
    }
  };
}

// `faceMaterials`, when given, paints the six cube faces individually (in
// THREE.BoxGeometry group order: +x, -x, +y, -y, +z, -z) so the cube's
// orientation is legible as it is carried and reoriented. The caller owns and
// disposes those materials; when omitted the cube uses the shared `cube`
// material as before.
export function createCubeBody(
  materials: BodyMaterials,
  faceMaterials?: readonly THREE.Material[],
  showFaceCenterMarkers = true
): BodyPart {
  const body = new THREE.Group();
  body.name = 'cube_body';
  createWorldFromCubeMatrix().decompose(
    body.position,
    body.quaternion,
    body.scale
  );

  const cubeGeometry = new THREE.BoxGeometry(
    CUBE_HALF_SIZE * 2,
    CUBE_HALF_SIZE * 2,
    CUBE_HALF_SIZE * 2
  );
  const cubeVisual = new THREE.Mesh(
    cubeGeometry,
    faceMaterials ? [...faceMaterials] : materials.cube
  );
  cubeVisual.name = 'cube_visual';
  body.add(cubeVisual);

  const markerGeometry = showFaceCenterMarkers
    ? new THREE.CircleGeometry(0.002, 24)
    : undefined;
  const markerPoses = [
    [CUBE_HALF_SIZE + MARKER_SURFACE_OFFSET, 0, 0, 0, Math.PI / 2, 0],
    [-CUBE_HALF_SIZE - MARKER_SURFACE_OFFSET, 0, 0, 0, Math.PI / 2, 0],
    [0, CUBE_HALF_SIZE + MARKER_SURFACE_OFFSET, 0, Math.PI / 2, 0, 0],
    [0, -CUBE_HALF_SIZE - MARKER_SURFACE_OFFSET, 0, Math.PI / 2, 0, 0],
    [0, 0, CUBE_HALF_SIZE + MARKER_SURFACE_OFFSET, 0, 0, 0],
    [0, 0, -CUBE_HALF_SIZE - MARKER_SURFACE_OFFSET, Math.PI, 0, 0]
  ] as const;
  if (markerGeometry !== undefined) {
    for (const [index, [x, y, z, rotationX, rotationY, rotationZ]]
      of markerPoses.entries()) {
      const marker = new THREE.Mesh(markerGeometry, materials.marker);
      marker.name = `cube_horizontal_face_center_${index}`;
      marker.position.set(x, y, z);
      marker.rotation.set(rotationX, rotationY, rotationZ);
      body.add(marker);
    }
  }

  return {
    body,
    destroy(): void {
      cubeGeometry.dispose();
      markerGeometry?.dispose();
    }
  };
}
