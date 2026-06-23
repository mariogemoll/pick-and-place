// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  ARM_JOINT_NAMES,
  deriveSo101Kinematics,
  NEUTRAL_ARM_JOINTS
} from '../../ik/kinematics';
import {
  type SimpleIkBranch,
  solveSimpleGraspIk
} from '../../ik/simple-ik';
import {
  computeGlobalXyWorkspace,
  sectorBoundingBox
} from '../../ik/workspace';
import { loadWebModel } from '../../web-model';
import {
  type CubePose,
  DEFAULT_CUBE_POSE
} from '../grasp-pose-shared/body-factories';
import { buildWorkspaceOverlaySpecs } from '../workspace-overlay';
import { createXyDragControls } from '../xy-drag-controls';
import {
  createGraspMatrix,
  createPregraspMatrix,
  PREGRASP_DISTANCE
} from './pose';
import { createCanonicalGraspScene } from './scene';
import {
  type ApproachMode,
  buildUi,
  DEFAULT_CUBE_X,
  DEFAULT_CUBE_Y
} from './ui';

type Elbow = SimpleIkBranch['elbow'];

interface GraspSolution {
  graspBranches: SimpleIkBranch[];
  pregraspBranches: SimpleIkBranch[] | null;
  // Approach pitch (radians); π/2 is straight down.
  pitch: number;
  // Whether the wrist camera faces outward (true) or is forced inward (false).
  cameraOut: boolean;
  // Whether the most-top-down reachable grasp would point the camera inward, so
  // facing it outward costs extra tilt (or, if cameraOut is false, isn't possible).
  extraTiltForCamera: boolean;
  // World-from-gripper matrix for the actual contact grasp.
  matrix: THREE.Matrix4;
  // Unit world direction from wrist to target.
  approach: THREE.Vector3;
  pregraspMatrix: THREE.Matrix4 | null;
}

interface GraspCandidate {
  graspBranches: SimpleIkBranch[];
  pregraspBranches: SimpleIkBranch[] | null;
  cameraOutward: number;
  matrix: THREE.Matrix4;
  pregraspMatrix: THREE.Matrix4 | null;
}

// Cubes closer than this to the pan axis are out of bounds: the gripper (and
// camera mount) collide with the robot body regardless of orientation.
const MIN_GRASP_RADIUS = 0.08;
// The canonical grasp/pregrasp solver is radius/yaw invariant across azimuth.
// A dense sweep found the first invalid all-yaw radius at 428.75 mm; clamp just
// below it so both the actual grasp and the 3 cm pregrasp stay reachable.
const MAX_CANONICAL_GRASP_RADIUS = 0.428;
// Keep away from the shoulder-pan edge where the camera-facing convention flips
// inward. The loaded model exposes ±110°; clamp to ±100° to leave a guard band
// that also covers the smaller-radius cases.
const MAX_CANONICAL_AZIMUTH = (100 * Math.PI) / 180;

// Snap `nominal` to the nearest of the cube's four face normals (at
// `cubeYaw + k·90°`). The result is within ±45° of `nominal`, so the wrist twist
// needed to square the jaws onto the cube never exceeds the symmetry window.
function squareToCubeFace(nominal: number, cubeYaw: number): number {
  const quarter = Math.PI / 2;
  return cubeYaw + Math.round((nominal - cubeYaw) / quarter) * quarter;
}

// Unit approach direction (wrist → target) in the radial–vertical plane at
// `radialAzimuth`. `pitch` is measured from horizontal: π/2 points straight down
// (the square top-down grasp); tilting away from π/2 lets the tool reach in
// diagonally for cubes outside the top-down region.
function approachVector(radialAzimuth: number, pitch: number): THREE.Vector3 {
  const horizontal = Math.cos(pitch);
  return new THREE.Vector3(
    Math.cos(radialAzimuth) * horizontal,
    Math.sin(radialAzimuth) * horizontal,
    -Math.sin(pitch)
  );
}

// Approach pitches to try (radians). The square top-down grasp (π/2) is always
// tried first, so inside the top-down region the grasp is square regardless of
// mode. The remaining pitches are ordered by preference per mode — 'tilt' keeps
// the approach as vertical as possible, 'side' as horizontal as possible — and
// the first that reaches wins.
const SQUARE_PITCH = Math.PI / 2;
const toRad = (deg: number): number => (deg * Math.PI) / 180;
const DEGRADE_PITCHES_DEG = ((): number[] => {
  const degrees: number[] = [];
  for (let deg = 10; deg <= 170; deg += 2) {
    if (deg !== 90) { degrees.push(deg); }
  }
  return degrees;
})();
const TILT_PITCHES = [
  SQUARE_PITCH,
  ...[...DEGRADE_PITCHES_DEG].sort((a, b) => Math.abs(a - 90) - Math.abs(b - 90)).map(toRad)
];
const SIDE_PITCHES = [
  SQUARE_PITCH,
  ...[...DEGRADE_PITCHES_DEG]
    .sort((a, b) => Math.abs(Math.sin(toRad(a))) - Math.abs(Math.sin(toRad(b))))
    .map(toRad)
];

export interface CanonicalGraspVisualization {
  destroy(): void;
}

export interface CanonicalGraspOptions {
  modelBasePath?: string;
  modelUrl?: string;
}

export async function initializeCanonicalGraspVisualization(
  parent: HTMLElement,
  options: CanonicalGraspOptions = {}
): Promise<CanonicalGraspVisualization> {
  const model = await loadWebModel(options.modelUrl);
  const kinematics = deriveSo101Kinematics(model);

  // Full floor reach (max arm reach projected onto the floor) drives the
  // placement slider ranges. faceOffset is zeroed so the radial band is the raw
  // reach rather than the any-yaw-graspable centre band, then capped to the
  // radius/yaw-valid canonical grasp boundary.
  const workspace = { ...computeGlobalXyWorkspace(kinematics), faceOffset: 0 };
  const maxGraspRadius = Math.min(
    workspace.radial.max,
    MAX_CANONICAL_GRASP_RADIUS
  );
  const minGraspRadius = MIN_GRASP_RADIUS;
  const minAzimuth = Math.max(workspace.azimuth.min, -MAX_CANONICAL_AZIMUTH);
  const maxAzimuth = Math.min(workspace.azimuth.max, MAX_CANONICAL_AZIMUTH);
  const canonicalWorkspace = {
    ...workspace,
    radial: { ...workspace.radial, min: minGraspRadius, max: maxGraspRadius },
    azimuth: { min: minAzimuth, max: maxAzimuth }
  };
  const bbox = sectorBoundingBox(canonicalWorkspace);
  const panX = workspace.panAxis.x;
  const panY = workspace.panAxis.y;
  // Radial coordinates are measured from the pan axis (the sector's center).
  const radialFromCartesian = (x: number, y: number): {
    radiusMm: number; azimuthDeg: number;
  } => ({
    radiusMm: Math.hypot(x - panX, y - panY) * 1000,
    azimuthDeg: (Math.atan2(y - panY, x - panX) * 180) / Math.PI
  });
  const cartesianFromRadial = (radiusMm: number, azimuthDeg: number): {
    x: number; y: number;
  } => {
    const radius = radiusMm / 1000;
    const azimuth = (azimuthDeg * Math.PI) / 180;
    return {
      x: panX + radius * Math.cos(azimuth),
      y: panY + radius * Math.sin(azimuth)
    };
  };
  const clampCartesianToReach = (x: number, y: number): { x: number; y: number } => {
    const radial = radialFromCartesian(x, y);
    const radius = radial.radiusMm / 1000;
    const clampedRadius = THREE.MathUtils.clamp(
      radius,
      minGraspRadius,
      maxGraspRadius
    );
    const clampedAzimuthDeg = THREE.MathUtils.clamp(
      radial.azimuthDeg,
      (minAzimuth * 180) / Math.PI,
      (maxAzimuth * 180) / Math.PI
    );
    if (
      Math.abs(clampedRadius - radius) < 1e-9 &&
      Math.abs(clampedAzimuthDeg - radial.azimuthDeg) < 1e-9
    ) {
      return { x, y };
    }
    return cartesianFromRadial(clampedRadius * 1000, clampedAzimuthDeg);
  };
  const defaultRadial = radialFromCartesian(DEFAULT_CUBE_X, DEFAULT_CUBE_Y);
  const ui = buildUi(parent, {
    xRange: {
      min: Math.floor(bbox.x.min * 1000),
      max: Math.ceil(bbox.x.max * 1000)
    },
    yRange: {
      min: Math.floor(bbox.y.min * 1000),
      max: Math.ceil(bbox.y.max * 1000)
    },
    radiusRange: {
      min: Math.max(
        Math.round(minGraspRadius * 1000),
        Math.floor(canonicalWorkspace.radial.min * 1000)
      ),
      max: Math.floor(canonicalWorkspace.radial.max * 1000)
    },
    azimuthRange: {
      min: Math.ceil((canonicalWorkspace.azimuth.min * 180) / Math.PI),
      max: Math.floor((canonicalWorkspace.azimuth.max * 180) / Math.PI)
    },
    radiusDefault: Math.round(defaultRadial.radiusMm),
    azimuthDefault: Math.round(defaultRadial.azimuthDeg)
  });
  const vizScene = await createCanonicalGraspScene(
    ui.viewport, model, options.modelBasePath,
    buildWorkspaceOverlaySpecs(kinematics)
  );

  // The cube always rests flat on the ground; only its X/Y and yaw vary.
  let currentPose: CubePose = {
    ...DEFAULT_CUBE_POSE, x: DEFAULT_CUBE_X, y: DEFAULT_CUBE_Y
  };
  // Cube yaw measured from the radial direction (the slider value). The cube's
  // world yaw is this plus the azimuth, so the grasp geometry — and thus the
  // camera-down bands — stays the same at every azimuth.
  let yawFromRadius = 0;
  // Persist the operator's elbow choice across pose changes.
  let preferredElbow: Elbow = 'up';
  // How to degrade the grasp outside the top-down-reachable region.
  let approachMode: ApproachMode = 'tilt';
  let showPregrasp = false;

  function applyBranch(branch: SimpleIkBranch): void {
    for (const name of ARM_JOINT_NAMES) {
      vizScene.setJoint(name, branch.joints[name]);
    }
  }

  function restToNeutral(): void {
    for (const name of ARM_JOINT_NAMES) {
      vizScene.setJoint(name, NEUTRAL_ARM_JOINTS[name]);
    }
  }

  function renderBranches(branches: SimpleIkBranch[]): void {
    ui.branchContainer.replaceChildren();
    if (branches.length < 2) { return; }
    for (const branch of branches) {
      const label = document.createElement('label');
      label.className = 'canonical-grasp-viz-branch';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'canonical-grasp-branch';
      radio.value = branch.elbow;
      radio.checked = branch.elbow === preferredElbow;
      radio.addEventListener('change', () => {
        if (radio.checked) {
          preferredElbow = branch.elbow;
          updateScene();
        }
      });
      const span = document.createElement('span');
      span.textContent = branch.elbow === 'up' ? 'Elbow up' : 'Elbow down';
      label.append(radio, span);
      ui.branchContainer.appendChild(label);
    }
  }

  // Solve the grasp for a cube pose. The nominal jaw axis is perpendicular to the
  // radius from the pan axis, twisted to square onto the nearest cube face (≤±45°
  // away by symmetry). The approach starts straight down and tilts only as far as
  // needed to reach: inside the top-down region it stays vertical (a square
  // grasp); outside, it tilts in — no longer a square grasp, but the best the
  // fixed-approach arm can do.
  //
  // Of the two squared orientations (180° apart) one points the wrist camera
  // outward (away from the base), the other inward — and tilting an outward
  // camera lifts it up, tilting an inward one drops it toward the floor. We
  // require the camera to face outward and accept a few extra degrees of tilt to
  // get it, because the outward orientation sometimes needs tilt to clear the
  // wrist-roll limit. Inward is taken only when outward is unreachable at any
  // pitch. Choosing by outward-facing (rather than the camera's height, which is
  // ~0 for both at a pure top-down grasp) keeps the pick continuous across the
  // top-down boundary instead of flipping the camera around.
  function solveGrasp(
    pose: CubePose,
    mode: ApproachMode,
    requirePregrasp: boolean
  ): GraspSolution | null {
    const poseRadius = Math.hypot(pose.x - panX, pose.y - panY);
    if (poseRadius < minGraspRadius || poseRadius > maxGraspRadius) { return null; }
    const azimuth = Math.atan2(pose.y - panY, pose.x - panX);
    const closings = [azimuth + Math.PI / 2, azimuth - Math.PI / 2]
      .map(nominal => squareToCubeFace(nominal, pose.yaw));

    const radial = new THREE.Vector3(Math.cos(azimuth), Math.sin(azimuth), 0);
    const pitches = mode === 'side' ? SIDE_PITCHES : TILT_PITCHES;
    let fallback: {
      pitch: number;
      candidate: GraspCandidate;
      matrix: THREE.Matrix4;
      approach: THREE.Vector3;
    } | null = null;
    // Whether the most-top-down reachable orientation faces the camera inward —
    // i.e. facing it outward costs extra tilt. This is the region of interest.
    let extraTiltForCamera = false;
    for (const pitch of pitches) {
      const approach = approachVector(azimuth, pitch);
      const graspSolutions = closings.flatMap<GraspCandidate>(closingAzimuth => {
        const matrix = createGraspMatrix(pose, closingAzimuth, approach);
        const result = solveSimpleGraspIk(kinematics, matrix);
        if (result.type !== 'success') { return []; }
        const pregraspMatrix = createPregraspMatrix(matrix, approach);
        const pregraspResult = solveSimpleGraspIk(kinematics, pregraspMatrix);
        // Camera sits on the gripper's +Y axis; its outward component is that
        // axis dotted with the radial. > 0 faces away from the base, < 0 inward.
        const cameraOutward = new THREE.Vector3().setFromMatrixColumn(matrix, 1).dot(radial);
        return [{
          graspBranches: result.branches,
          pregraspBranches: pregraspResult.type === 'success' ? pregraspResult.branches : null,
          cameraOutward,
          matrix,
          pregraspMatrix: pregraspResult.type === 'success' ? pregraspMatrix : null
        }];
      });
      if (graspSolutions.length === 0) { continue; }
      const solutions = requirePregrasp
        ? graspSolutions.filter(solution => solution.pregraspBranches !== null)
        : graspSolutions;

      const bestGrasp = graspSolutions.reduce((b, c) =>
        c.cameraOutward > b.cameraOutward ? c : b
      );
      const best = solutions.length > 0
        ? solutions.reduce((b, c) => c.cameraOutward > b.cameraOutward ? c : b)
        : bestGrasp;
      if (
        fallback === null ||
        (
          requirePregrasp &&
          fallback.candidate.pregraspBranches === null &&
          solutions.length > 0
        )
      ) {
        // First (most-top-down) reachable pitch.
        fallback = {
          pitch,
          candidate: best,
          matrix: best.matrix,
          approach: approach.clone()
        };
        extraTiltForCamera = best.cameraOutward < 0;
      }
      if (solutions.length > 0 && best.cameraOutward > 0) {
        return {
          graspBranches: best.graspBranches,
          pregraspBranches: best.pregraspBranches,
          pitch,
          cameraOut: true,
          extraTiltForCamera,
          matrix: best.matrix,
          approach: approach.clone(),
          pregraspMatrix: best.pregraspMatrix
        };
      }
    }
    if (fallback === null) { return null; }
    return {
      graspBranches: fallback.candidate.graspBranches,
      pregraspBranches: fallback.candidate.pregraspBranches,
      pitch: fallback.pitch,
      cameraOut: false, extraTiltForCamera,
      matrix: fallback.matrix,
      approach: fallback.approach,
      pregraspMatrix: fallback.candidate.pregraspMatrix
    };
  }

  function updateScene(): void {
    // World yaw keeps the cube at a constant offset from the radius as it moves.
    const azimuth = Math.atan2(currentPose.y - panY, currentPose.x - panX);
    currentPose = { ...currentPose, yaw: yawFromRadius + azimuth };
    vizScene.updateCubePose(currentPose);

    const radius = Math.hypot(currentPose.x - panX, currentPose.y - panY);
    const solution = solveGrasp(currentPose, approachMode, showPregrasp);
    if (solution === null) {
      vizScene.updateGhostGraspPose(null);
      ui.status.textContent = radius < minGraspRadius
        ? `Too close to the base: keep the cube ≥ ${Math.round(minGraspRadius * 1000)} mm out.`
        : 'Unreachable: the cube is outside the arm’s reach.';
      ui.status.classList.add('is-invalid');
      ui.branchContainer.replaceChildren();
      restToNeutral();
      return;
    }

    let branches = solution.graspBranches;
    let pregraspNote = '';
    if (showPregrasp) {
      if (solution.pregraspBranches !== null && solution.pregraspMatrix !== null) {
        branches = solution.pregraspBranches;
        vizScene.updateGhostGraspPose(solution.matrix);
        pregraspNote = ` Showing pregrasp ${Math.round(PREGRASP_DISTANCE * 1000)} mm back.`;
      } else {
        vizScene.updateGhostGraspPose(null);
        pregraspNote = ' Pregrasp pose is unreachable here.';
      }
    } else {
      vizScene.updateGhostGraspPose(null);
    }

    const branch = branches.find(candidate => candidate.elbow === preferredElbow)
      ?? branches[0];
    const tiltDeg = Math.round(Math.abs(90 - (solution.pitch * 180) / Math.PI));
    const base = tiltDeg === 0
      ? 'Reachable: square top-down grasp.'
      : `Reachable: approach tilted ${tiltDeg}° (not a square grasp).`;
    let note = '';
    if (!solution.cameraOut) {
      note = ' Camera forced inward — no outward grasp reachable here.';
    } else if (solution.extraTiltForCamera) {
      note = ' Tilted extra to face the camera outward.';
    }
    ui.status.textContent = `${base}${note}${pregraspNote}`;
    ui.status.classList.toggle(
      'is-invalid',
      showPregrasp && solution.pregraspBranches === null
    );
    renderBranches(branches);
    applyBranch(branch);
  }

  const yawListener = (): void => {
    yawFromRadius = (Number(ui.yawInput.value) * Math.PI) / 180;
    updateScene();
  };
  ui.yawInput.addEventListener('input', yawListener);

  const pregraspListener = (): void => {
    showPregrasp = ui.showPregraspInput.checked;
    updateScene();
  };
  ui.showPregraspInput.addEventListener('change', pregraspListener);

  // X/Y and radius/azimuth drive the same cube center; keep both in sync so a
  // mode switch is seamless. The guard stops the programmatic value updates
  // (which fire 'input' to refresh the slider labels) from recursing.
  let syncing = false;
  const setSlider = (input: HTMLInputElement, value: number): void => {
    input.value = String(value);
    input.dispatchEvent(new Event('input'));
  };
  const applyCartesian = (): void => {
    if (syncing) { return; }
    const rawX = Number(ui.xInput.value) / 1000;
    const rawY = Number(ui.yInput.value) / 1000;
    const { x, y } = clampCartesianToReach(rawX, rawY);
    currentPose = { ...currentPose, x, y };
    const radial = radialFromCartesian(x, y);
    syncing = true;
    setSlider(ui.xInput, Math.round(x * 1000));
    setSlider(ui.yInput, Math.round(y * 1000));
    setSlider(ui.radiusInput, Math.round(radial.radiusMm));
    setSlider(ui.azimuthInput, Math.round(radial.azimuthDeg * 10) / 10);
    syncing = false;
    updateScene();
  };
  const applyRadial = (): void => {
    if (syncing) { return; }
    const { x, y } = cartesianFromRadial(
      Number(ui.radiusInput.value), Number(ui.azimuthInput.value)
    );
    currentPose = { ...currentPose, x, y };
    syncing = true;
    setSlider(ui.xInput, Math.round(x * 1000));
    setSlider(ui.yInput, Math.round(y * 1000));
    syncing = false;
    updateScene();
  };
  ui.xInput.addEventListener('input', applyCartesian);
  ui.yInput.addEventListener('input', applyCartesian);
  ui.radiusInput.addEventListener('input', applyRadial);
  ui.azimuthInput.addEventListener('input', applyRadial);

  const clampToInput = (input: HTMLInputElement, value: number): number =>
    Math.min(Number(input.max), Math.max(Number(input.min), value));
  const dragControls = createXyDragControls({
    camera: vizScene.camera,
    domElement: vizScene.renderer.domElement,
    object: vizScene.cube,
    orbitControls: vizScene.orbitControls,
    onDrag(x, y): void {
      ui.xInput.value = String(Math.round(clampToInput(ui.xInput, x * 1000)));
      ui.yInput.value = String(Math.round(clampToInput(ui.yInput, y * 1000)));
      ui.xInput.dispatchEvent(new Event('input'));
      ui.yInput.dispatchEvent(new Event('input'));
    }
  });

  const coordModeListeners = ui.coordModeInputs.map(input => {
    const listener = (): void => {
      if (!input.checked) { return; }
      const radial = input.value === 'radial';
      ui.cartesianGroup.style.display = radial ? 'none' : '';
      ui.radialGroup.style.display = radial ? '' : 'none';
    };
    input.addEventListener('change', listener);
    return listener;
  });

  const approachModeListeners = ui.approachModeInputs.map(input => {
    const listener = (): void => {
      if (!input.checked) { return; }
      approachMode = input.value === 'side' ? 'side' : 'tilt';
      updateScene();
    };
    input.addEventListener('change', listener);
    return listener;
  });

  const resetListener = (): void => {
    currentPose = {
      ...DEFAULT_CUBE_POSE, x: DEFAULT_CUBE_X, y: DEFAULT_CUBE_Y
    };
    preferredElbow = 'up';
    approachMode = 'tilt';
    showPregrasp = false;
    ui.showPregraspInput.checked = false;
    for (const input of ui.approachModeInputs) {
      input.checked = input.value === 'tilt';
    }
    // Back to Cartesian mode on reset.
    for (const input of ui.coordModeInputs) {
      input.checked = input.value === 'cartesian';
      input.dispatchEvent(new Event('change'));
    }
    // Setting X/Y drives the radius/azimuth sliders via applyCartesian.
    ui.xInput.value = String(Math.round(DEFAULT_CUBE_X * 1000));
    ui.yInput.value = String(Math.round(DEFAULT_CUBE_Y * 1000));
    ui.xInput.dispatchEvent(new Event('input'));
    ui.yInput.dispatchEvent(new Event('input'));
    ui.yawInput.value = '0';
    ui.yawInput.dispatchEvent(new Event('input'));
  };
  ui.resetButton.addEventListener('click', resetListener);

  const resizeObserver = new ResizeObserver(() => { vizScene.resize(); });
  resizeObserver.observe(ui.viewport);

  updateScene();

  let animationFrameId = 0;
  let destroyed = false;
  function animate(): void {
    if (destroyed) { return; }
    animationFrameId = window.requestAnimationFrame(animate);
    vizScene.orbitControls.update();
    vizScene.renderer.render(vizScene.scene, vizScene.camera);
  }
  animationFrameId = window.requestAnimationFrame(animate);

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      dragControls.destroy();
      vizScene.destroy();
      ui.yawInput.removeEventListener('input', yawListener);
      ui.showPregraspInput.removeEventListener('change', pregraspListener);
      ui.xInput.removeEventListener('input', applyCartesian);
      ui.yInput.removeEventListener('input', applyCartesian);
      ui.radiusInput.removeEventListener('input', applyRadial);
      ui.azimuthInput.removeEventListener('input', applyRadial);
      for (const [index, input] of ui.coordModeInputs.entries()) {
        input.removeEventListener('change', coordModeListeners[index]);
      }
      for (const [index, input] of ui.approachModeInputs.entries()) {
        input.removeEventListener('change', approachModeListeners[index]);
      }
      ui.resetButton.removeEventListener('click', resetListener);
      ui.root.remove();
    }
  };
}
