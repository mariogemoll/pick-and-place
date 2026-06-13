// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import type { So101Kinematics } from '../ik/kinematics';
import {
  anyYawCubeCenterBand,
  computeArmWorkspaceAtHeight,
  computeGlobalXyWorkspace,
  computeSimpleWorkspace,
  computeSimpleWorkspaceForCubeZ,
  CUBE_Z_1CM_OVER_GROUND_TOP
} from '../ik/workspace';

export interface WorkspaceOverlaySpec {
  center: THREE.Vector2;
  innerRadius: number;
  outerRadius: number;
  thetaStart: number;
  thetaLength: number;
  color?: number;
  opacity?: number;
}

// Build the four standard workspace overlay specs from kinematics.
// Returned in back-to-front (largest → smallest) order for correct layering.
export function buildWorkspaceOverlaySpecs(k: So101Kinematics): WorkspaceOverlaySpec[] {
  const ground = computeSimpleWorkspace(k);
  const clearance = computeSimpleWorkspaceForCubeZ(k, CUBE_Z_1CM_OVER_GROUND_TOP);
  const groundHeightArm = computeArmWorkspaceAtHeight(k, ground.targetHeight);
  const global = computeGlobalXyWorkspace(k);

  const groundBand = anyYawCubeCenterBand(ground);
  const clearanceBand = anyYawCubeCenterBand(clearance);

  return [
    // Maximum arm reach at any joint config, projected onto the floor.
    {
      center: global.panAxis,
      innerRadius: global.radial.min,
      outerRadius: global.radial.max,
      thetaStart: global.azimuth.min,
      thetaLength: global.azimuth.max - global.azimuth.min
    },
    // Max jaw-contact reach at z = ground-cube-center height, any joint config.
    {
      center: groundHeightArm.panAxis,
      innerRadius: groundHeightArm.radial.min,
      outerRadius: groundHeightArm.radial.max,
      thetaStart: groundHeightArm.azimuth.min,
      thetaLength: groundHeightArm.azimuth.max - groundHeightArm.azimuth.min
    },
    // Ground-cube simple-pregrasp workspace.
    {
      center: ground.panAxis,
      innerRadius: groundBand.min,
      outerRadius: groundBand.max,
      thetaStart: ground.azimuth.min,
      thetaLength: ground.azimuth.max - ground.azimuth.min
    },
    // Simple-pregrasp workspace with 1 cm clearance above a ground cube.
    {
      center: clearance.panAxis,
      innerRadius: clearanceBand.min,
      outerRadius: clearanceBand.max,
      thetaStart: clearance.azimuth.min,
      thetaLength: clearance.azimuth.max - clearance.azimuth.min
    }
  ];
}

// Add overlay meshes to `scene` and return a disposer that removes and frees them.
export function addWorkspaceOverlaysToScene(
  scene: THREE.Scene,
  specs: WorkspaceOverlaySpec[]
): () => void {
  const geometries: THREE.RingGeometry[] = [];
  const materials: THREE.MeshBasicMaterial[] = [];

  for (const [i, ws] of specs.entries()) {
    const geo = new THREE.RingGeometry(
      ws.innerRadius, ws.outerRadius, 96, 1, ws.thetaStart, ws.thetaLength
    );
    const mat = new THREE.MeshBasicMaterial({
      color: ws.color ?? 0xff7700,
      transparent: true,
      opacity: ws.opacity ?? 0.22,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const mesh = new THREE.Mesh(geo, mat);
    // RingGeometry lies in the XY plane; stagger z slightly to avoid z-fighting.
    mesh.position.set(ws.center.x, ws.center.y, 0.0002 + i * 0.0002);
    mesh.renderOrder = i - specs.length;
    scene.add(mesh);
    geometries.push(geo);
    materials.push(mat);
  }

  return () => {
    for (const geo of geometries) { geo.dispose(); }
    for (const mat of materials) { mat.dispose(); }
  };
}
