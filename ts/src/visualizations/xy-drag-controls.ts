// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface XyDragControlsOptions {
  camera: THREE.Camera;
  domElement: HTMLElement;
  object: THREE.Object3D;
  orbitControls: OrbitControls;
  onDrag: (x: number, y: number) => void;
}

export interface XyDragControls {
  destroy(): void;
}

export interface XyDragTarget {
  object: THREE.Object3D;
  onDrag: (x: number, y: number) => void;
}

export interface XyMultiDragControlsOptions {
  camera: THREE.Camera;
  domElement: HTMLElement;
  targets: XyDragTarget[];
  orbitControls: OrbitControls;
}

export function createXyMultiDragControls({
  camera,
  domElement,
  targets,
  orbitControls
}: XyMultiDragControlsOptions): XyDragControls {
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1));
  const intersection = new THREE.Vector3();
  const offset = new THREE.Vector3();
  let pointerId: number | null = null;
  let activeTarget: XyDragTarget | null = null;

  function updateRaycaster(event: PointerEvent): void {
    const bounds = domElement.getBoundingClientRect();
    pointer.set(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1
    );
    raycaster.setFromCamera(pointer, camera);
  }

  function targetUnderPointer(event: PointerEvent): XyDragTarget | null {
    updateRaycaster(event);
    const intersections = raycaster.intersectObjects(
      targets.map(target => target.object),
      true
    );
    for (const hit of intersections) {
      const target = targets.find(candidate => {
        let object: THREE.Object3D | null = hit.object;
        while (object !== null) {
          if (object === candidate.object) { return true; }
          object = object.parent;
        }
        return false;
      });
      if (target) { return target; }
    }
    return null;
  }

  function finishDrag(): void {
    if (pointerId === null) { return; }
    if (domElement.hasPointerCapture(pointerId)) {
      domElement.releasePointerCapture(pointerId);
    }
    pointerId = null;
    activeTarget = null;
    orbitControls.enabled = true;
    domElement.style.cursor = '';
  }

  const pointerDownListener = (event: PointerEvent): void => {
    if (pointerId !== null || event.button !== 0) { return; }
    const target = targetUnderPointer(event);
    if (!target) { return; }
    dragPlane.constant = -target.object.position.z;
    if (!raycaster.ray.intersectPlane(dragPlane, intersection)) { return; }
    offset.copy(target.object.position).sub(intersection);
    pointerId = event.pointerId;
    activeTarget = target;
    domElement.setPointerCapture(event.pointerId);
    orbitControls.enabled = false;
    domElement.style.cursor = 'grabbing';
    event.preventDefault();
  };
  const pointerMoveListener = (event: PointerEvent): void => {
    if (pointerId === null) {
      domElement.style.cursor = targetUnderPointer(event) ? 'grab' : '';
      return;
    }
    if (event.pointerId !== pointerId || !activeTarget) { return; }
    updateRaycaster(event);
    if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
      activeTarget.onDrag(intersection.x + offset.x, intersection.y + offset.y);
    }
    event.preventDefault();
  };
  const pointerUpListener = (event: PointerEvent): void => {
    if (event.pointerId === pointerId) { finishDrag(); }
  };
  const pointerLeaveListener = (): void => {
    if (pointerId === null) { domElement.style.cursor = ''; }
  };

  domElement.addEventListener('pointerdown', pointerDownListener);
  domElement.addEventListener('pointermove', pointerMoveListener);
  domElement.addEventListener('pointerup', pointerUpListener);
  domElement.addEventListener('pointercancel', pointerUpListener);
  domElement.addEventListener('pointerleave', pointerLeaveListener);

  return {
    destroy(): void {
      finishDrag();
      domElement.removeEventListener('pointerdown', pointerDownListener);
      domElement.removeEventListener('pointermove', pointerMoveListener);
      domElement.removeEventListener('pointerup', pointerUpListener);
      domElement.removeEventListener('pointercancel', pointerUpListener);
      domElement.removeEventListener('pointerleave', pointerLeaveListener);
    }
  };
}

export function createXyDragControls({
  camera,
  domElement,
  object,
  orbitControls,
  onDrag
}: XyDragControlsOptions): XyDragControls {
  return createXyMultiDragControls({
    camera,
    domElement,
    orbitControls,
    targets: [{ object, onDrag }]
  });
}
