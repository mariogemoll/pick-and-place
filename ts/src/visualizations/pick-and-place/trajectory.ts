// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import {
  ARM_JOINT_NAMES,
  type ArmJointName,
  NEUTRAL_ARM_JOINTS,
  type So101Kinematics
} from '../../ik/kinematics';
import { solveSimpleGraspIk } from '../../ik/simple-ik';
import { SAFETY_MARGIN } from '../grasp-pose-shared/bodies';
import {
  createWorldFromCubeContactMatrix,
  createWorldFromCubeMatrix,
  type CubeFace,
  type CubePose
} from '../grasp-pose-shared/body-factories';
import { createSimpleGraspMatrix } from '../simple-grasp-pose/pose';

// Hover keyframes are specified by the height of the tip contact point (center
// of the tip collision-box face) above the floor. The source hover clears the
// cube top (3 cm) by 1 cm so the approach swing doesn't clip the cube; the drop
// hover sits lower for a gentle release. At the grasp pose the tip contact is at
// the cube-center height (`pose.z`), so the world-Z offset applied to a hover is
// `tipZ - pose.z`.
const SOURCE_HOVER_TIP_Z = 0.04;
const PREDROP_HOVER_TIP_Z = 0.02;
const POSTDROP_HOVER_TIP_Z = 0.04;

// Gripper joint angle at hover grasp: 40° open.
export const GRIPPER_OPEN = 40 * (Math.PI / 180);
// Gripper joint angle once the jaws have pinched the cube. Geometrically
// estimated: at this angle the moving jaw's fingertips reach the cube's far
// face after the cube has slid flush against the fixed jaw. (Faked, not solved
// from a physics engine – fine-tune visually if the pinch looks off.)
export const GRIPPER_CLOSED = 10.5 * (Math.PI / 180);
// Gripper angle at which the cube starts being shoved toward the fixed jaw.
// The true geometric contact is a bit later (~14°), but starting the slide
// sooner reads better: the cube eases over rather than snapping flush at the
// very end of the close.
export const GRIPPER_CONTACT = 20 * (Math.PI / 180);

export type JointAngles = Record<ArmJointName, number>;

export interface RobotPose {
  joints: JointAngles;
  gripper: number;
}

export interface TrajectoryFrame extends RobotPose {
  sourceCube: CubePose;
}

export const NEUTRAL_FRAME: RobotPose = {
  joints: NEUTRAL_ARM_JOINTS,
  gripper: 0
};

// The physical follower's safe sleeping pose, converted from calibrated
// encoder degrees into the MuJoCo joint frame.
export const REST_FRAME: RobotPose = {
  joints: {
    shoulder_pan: 4.967032967032967 * (Math.PI / 180),
    shoulder_lift: -100 * (Math.PI / 180),
    elbow_flex: 90 * (Math.PI / 180),
    wrist_flex: 73.71428571428571 * (Math.PI / 180),
    wrist_roll: -86.46153846153847 * (Math.PI / 180)
  },
  gripper: ((10.5 - 2.3) / 96.2 * 130 - 10) * (Math.PI / 180)
};

const REST_PHASE_DURATION = 2.0;

// Duration of stage 1 – neutral → hover grasp above source cube.
const STAGE1_DURATION = 2.0;
// Duration of stage 2 – hover grasp → grasp at source cube center.
const STAGE2_DURATION = 1.0;
// Duration of stage 3 – close the gripper, pushing the cube against the fixed
// jaw.
const STAGE3_DURATION = 1.0;
// Duration of stage 4 – carry the grasped cube to the drop hover above the
// target (tip 2 cm above the floor).
const STAGE4_DURATION = 2.0;
// Stage 5 is one continuous retreat spline through the postdrop hover and back
// to neutral. The hover is a clearance waypoint, not a stopping point.
const POSTDROP_HOVER_DURATION = 1.5;
const RETURN_TO_NEUTRAL_DURATION = 2.0;
const STAGE5_DURATION = POSTDROP_HOVER_DURATION + RETURN_TO_NEUTRAL_DURATION;
const RELEASE_OPENING_ANGLE = 5 * (Math.PI / 180);
const MOVEMENT_OPENING_ANGLE = 10 * (Math.PI / 180);
// Real gravity covers the 5 mm release gap in about 32 ms. Stretch the
// ballistic fall enough to remain visible at ordinary playback speed.
const DROP_DURATION = 0.06;

// Cube-center height of the level cruise. Above the drop hover (2.5 cm) so the
// cube genuinely rises then descends; clears the cube top with room to spare
// mid-traverse.
const CARRY_CRUISE_Z = 0.03;
// Quintic Hermite segments share position, velocity, and acceleration at the
// cruise waypoints, making the joins C2 without stopping. Endpoint derivatives
// constrain the carry to leave the pick vertically and arrive vertically.
const CARRY_WAYPOINTS = [
  { phase: 0, label: 'Pick' },
  { phase: 0.4, label: 'Cruise start' },
  { phase: 0.6, label: 'Cruise end' },
  { phase: 1, label: 'Place' }
] as const;
const CARRY_CORNER_TRAVEL = 0.25;
// How many points along the carry to check for reachability when planning it.
const CARRY_SAMPLES = 24;
const CARRY_ARC_LENGTH_SAMPLES = 2048;
const CARRY_EASE_FRACTION = 0.2;

const UNIT_SCALE = new THREE.Vector3(1, 1, 1);

// The four vertical faces tried in order of how naturally the robot approaches
// a cube that is roughly in the +x direction from the pan axis.
const VERTICAL_FACES: CubeFace[] = ['+x', '-x', '+y', '-y'];

// Simple-grasp matrix for `face` shifted up along world Z, or null if the
// face is not vertical for this pose.
function graspMatrix(
  face: CubeFace,
  sourcePose: CubePose,
  zOffset = 0
): THREE.Matrix4 | null {
  const grasp = createSimpleGraspMatrix(face, sourcePose);
  if (!grasp) { return null; }
  const pos = new THREE.Vector3().setFromMatrixPosition(grasp);
  grasp.setPosition(pos.x, pos.y, pos.z + zOffset);
  return grasp;
}

function smoothstep(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
}

function lerpJoints(a: JointAngles, b: JointAngles, alpha: number): JointAngles {
  const out = {} as JointAngles;
  for (const name of ARM_JOINT_NAMES) {
    out[name] = a[name] + (b[name] - a[name]) * alpha;
  }
  return out;
}

function quinticBezier(
  p0: number,
  p1: number,
  p2: number,
  p3: number,
  p4: number,
  p5: number,
  phase: number
): number {
  const t = Math.min(1, Math.max(0, phase));
  const u = 1 - t;
  return u ** 5 * p0 +
    5 * u ** 4 * t * p1 +
    10 * u ** 3 * t ** 2 * p2 +
    10 * u ** 2 * t ** 3 * p3 +
    5 * u * t ** 4 * p4 +
    t ** 5 * p5;
}

// One smooth rest-to-hover movement. Neutral steers the curve but is not
// visited; duplicated endpoints make the arm start and finish at rest.
function restToHoverJoints(
  hover: JointAngles,
  phase: number
): JointAngles {
  const out = {} as JointAngles;
  for (const name of ARM_JOINT_NAMES) {
    out[name] = quinticBezier(
      REST_FRAME.joints[name],
      REST_FRAME.joints[name],
      NEUTRAL_FRAME.joints[name],
      NEUTRAL_FRAME.joints[name],
      hover[name],
      hover[name],
      phase
    );
  }
  return out;
}

// Mirror the opening movement after the drop: leave hover, bend toward neutral
// without visiting it, and settle at rest.
function hoverToRestJoints(
  hover: JointAngles,
  phase: number
): JointAngles {
  const out = {} as JointAngles;
  for (const name of ARM_JOINT_NAMES) {
    out[name] = quinticBezier(
      hover[name],
      hover[name],
      NEUTRAL_FRAME.joints[name],
      NEUTRAL_FRAME.joints[name],
      REST_FRAME.joints[name],
      REST_FRAME.joints[name],
      phase
    );
  }
  return out;
}

// C2 spline from start through a non-stopping waypoint to end. Both segments
// share the waypoint velocity and have zero acceleration at the join.
function splineJointsThroughWaypoint(
  start: JointAngles,
  waypoint: JointAngles,
  end: JointAngles,
  waypointPhase: number,
  phase: number
): JointAngles {
  const p = Math.min(1, Math.max(0, phase));
  if (p === 0) { return start; }
  if (p === 1) { return end; }
  const out = {} as JointAngles;
  for (const name of ARM_JOINT_NAMES) {
    const waypointVelocity =
      0.5 * (end[name] - waypoint[name]) / (1 - waypointPhase);
    if (p <= waypointPhase) {
      out[name] = quinticHermite(
        start[name], waypoint[name], 0, waypointVelocity,
        waypointPhase, p / waypointPhase
      );
    } else {
      out[name] = quinticHermite(
        waypoint[name], end[name], waypointVelocity, 0,
        1 - waypointPhase, (p - waypointPhase) / (1 - waypointPhase)
      );
    }
  }
  return out;
}

// Recover a CubePose from a world matrix, inverting `createWorldFromCubeMatrix`
// (translation followed by an intrinsic ZYX roll/pitch/yaw rotation).
function cubePoseFromMatrix(matrix: THREE.Matrix4): CubePose {
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, quaternion, scale);
  const euler = new THREE.Euler().setFromQuaternion(quaternion, 'ZYX');
  return {
    x: position.x,
    y: position.y,
    z: position.z,
    roll: euler.x,
    pitch: euler.y,
    yaw: euler.z
  };
}

// The sideways carry runs either as a straight Cartesian chord (shortest) or,
// when that chord would leave the annular sector, as a polar arc about the pan
// axis (radius and azimuth swept between the endpoints, which keeps the path
// inside the sector by construction).
type CarryMode = 'straight' | 'polar';

interface CarryPlan {
  mode: CarryMode;
  // Rigid cube→gripper transform captured at the grasp; the gripper follows the
  // cube through the carry so the held cube stays flush and lands on target.
  cubeFromGripper: THREE.Matrix4;
  graspCubePosition: THREE.Vector3;
  dropCubePosition: THREE.Vector3;
  graspCubeQuaternion: THREE.Quaternion;
  dropCubeQuaternion: THREE.Quaternion;
  graspRadius: number;
  dropRadius: number;
  graspAzimuth: number;
  dropAzimuth: number;
}

interface CarryPathPoint {
  travel: number;
  height: number;
}

interface CarryPathWaypoint extends CarryPathPoint {
  travelVelocity: number;
  heightVelocity: number;
}

interface ArcLengthSample {
  parameter: number;
  length: number;
}

const ARC_LENGTH_TABLES = new WeakMap<CarryPlan, ArcLengthSample[]>();

// Integral of smootherstep from 0 to t. It is used as the distance travelled
// while speed ramps smoothly from zero to cruise speed.
function smootherstepIntegral(t: number): number {
  const c = Math.min(1, Math.max(0, t));
  return c ** 6 - 3 * c ** 5 + 2.5 * c ** 4;
}

// Arc-length fraction at a playback phase: smooth acceleration over the first
// window, constant speed through the middle, and smooth deceleration at the end.
function timedArcFraction(phase: number): number {
  const p = Math.min(1, Math.max(0, phase));
  const ease = CARRY_EASE_FRACTION;
  const totalArea = 1 - ease;
  if (p < ease) {
    return ease * smootherstepIntegral(p / ease) / totalArea;
  }
  if (p <= 1 - ease) {
    return (ease * 0.5 + p - ease) / totalArea;
  }
  return 1 -
    ease * smootherstepIntegral((1 - p) / ease) / totalArea;
}

// Quintic Hermite interpolation with derivatives expressed against the
// geometry parameter. Matching position, velocity, and acceleration at each end makes
// adjacent segments C2. Waypoint acceleration is zero, while internal waypoint
// velocity remains nonzero so the carry flows through rather than pausing.
function quinticHermite(
  start: number,
  end: number,
  startVelocity: number,
  endVelocity: number,
  duration: number,
  u: number
): number {
  const v0 = startVelocity * duration;
  const v1 = endVelocity * duration;
  const delta = end - start - v0;
  const velocityDelta = v1 - v0;
  const c3 = 10 * delta - 4 * velocityDelta;
  const c4 = -15 * delta + 7 * velocityDelta;
  const c5 = 6 * delta - 3 * velocityDelta;
  return start + v0 * u + c3 * u ** 3 + c4 * u ** 4 + c5 * u ** 5;
}

// Side-view carry path through explicit C2 waypoints. The endpoint tangent
// velocities are purely vertical; the internal tangent velocities are purely
// horizontal, producing one smooth rounded ascent and descent.
function carryPath(graspZ: number, dropZ: number, parameter: number): CarryPathPoint {
  const p = Math.min(1, Math.max(0, parameter));
  const ascentVelocity = (CARRY_CRUISE_Z - graspZ) * 2;
  const descentVelocity = (dropZ - CARRY_CRUISE_Z) * 2;
  const cruiseVelocity = 1;
  const points: CarryPathWaypoint[] = [
    {
      travel: 0,
      height: graspZ,
      travelVelocity: 0,
      heightVelocity: ascentVelocity
    },
    {
      travel: CARRY_CORNER_TRAVEL,
      height: CARRY_CRUISE_Z,
      travelVelocity: cruiseVelocity,
      heightVelocity: 0
    },
    {
      travel: 1 - CARRY_CORNER_TRAVEL,
      height: CARRY_CRUISE_Z,
      travelVelocity: cruiseVelocity,
      heightVelocity: 0
    },
    {
      travel: 1,
      height: dropZ,
      travelVelocity: 0,
      heightVelocity: descentVelocity
    }
  ];
  const endIndex = CARRY_WAYPOINTS.findIndex(point => p <= point.phase);
  const i = Math.max(0, endIndex - 1);
  const startPhase = CARRY_WAYPOINTS[i].phase;
  const duration = CARRY_WAYPOINTS[i + 1].phase - startPhase;
  const u = (p - startPhase) / duration;
  const start = points[i];
  const end = points[i + 1];
  return {
    travel: quinticHermite(
      start.travel, end.travel,
      start.travelVelocity, end.travelVelocity,
      duration, u
    ),
    height: quinticHermite(
      start.height, end.height,
      start.heightVelocity, end.heightVelocity,
      duration, u
    )
  };
}

// World cube pose at a given geometry parameter. This defines shape only;
// playback timing is applied separately by `carryCubeMatrix`.
function carryGeometryMatrix(
  plan: CarryPlan, k: So101Kinematics, parameter: number
): THREE.Matrix4 {
  const path = carryPath(
    plan.graspCubePosition.z, plan.dropCubePosition.z, parameter
  );
  const travel = path.travel;
  let x: number;
  let y: number;
  if (plan.mode === 'straight') {
    x = plan.graspCubePosition.x +
      (plan.dropCubePosition.x - plan.graspCubePosition.x) * travel;
    y = plan.graspCubePosition.y +
      (plan.dropCubePosition.y - plan.graspCubePosition.y) * travel;
  } else {
    const radius = plan.graspRadius + (plan.dropRadius - plan.graspRadius) * travel;
    const azimuth =
      plan.graspAzimuth + (plan.dropAzimuth - plan.graspAzimuth) * travel;
    x = k.panAxis.x + radius * Math.cos(azimuth);
    y = k.panAxis.y + radius * Math.sin(azimuth);
  }
  const quaternion =
    plan.graspCubeQuaternion.clone().slerp(plan.dropCubeQuaternion, travel);
  return new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, path.height), quaternion, UNIT_SCALE
  );
}

function arcLengthTable(plan: CarryPlan, k: So101Kinematics): ArcLengthSample[] {
  const cached = ARC_LENGTH_TABLES.get(plan);
  if (cached) { return cached; }
  const table: ArcLengthSample[] = [{ parameter: 0, length: 0 }];
  let previous = new THREE.Vector3().setFromMatrixPosition(
    carryGeometryMatrix(plan, k, 0)
  );
  let length = 0;
  for (let i = 1; i <= CARRY_ARC_LENGTH_SAMPLES; i++) {
    const parameter = i / CARRY_ARC_LENGTH_SAMPLES;
    const position = new THREE.Vector3().setFromMatrixPosition(
      carryGeometryMatrix(plan, k, parameter)
    );
    length += position.distanceTo(previous);
    table.push({ parameter, length });
    previous = position;
  }
  ARC_LENGTH_TABLES.set(plan, table);
  return table;
}

function interpolateTable(
  table: ArcLengthSample[],
  value: number,
  input: 'parameter' | 'length',
  output: 'parameter' | 'length'
): number {
  const clamped = Math.min(table[table.length - 1][input], Math.max(0, value));
  const end = table.findIndex(sample => sample[input] >= clamped);
  const b = table[Math.max(1, end)];
  const a = table[Math.max(0, end - 1)];
  const span = b[input] - a[input];
  const alpha = span === 0 ? 0 : (clamped - a[input]) / span;
  return a[output] + (b[output] - a[output]) * alpha;
}

function parameterAtCarryPhase(
  plan: CarryPlan, k: So101Kinematics, phase: number
): number {
  const table = arcLengthTable(plan, k);
  const targetLength = timedArcFraction(phase) * table[table.length - 1].length;
  return interpolateTable(table, targetLength, 'length', 'parameter');
}

function carryPhaseAtParameter(
  plan: CarryPlan, k: So101Kinematics, parameter: number
): number {
  const table = arcLengthTable(plan, k);
  const arcFraction =
    interpolateTable(table, parameter, 'parameter', 'length') /
    table[table.length - 1].length;
  let low = 0;
  let high = 1;
  for (let i = 0; i < 24; i++) {
    const mid = (low + high) / 2;
    if (timedArcFraction(mid) < arcFraction) { low = mid; } else { high = mid; }
  }
  return (low + high) / 2;
}

// Traverse the geometric curve by arc length, with one global C2 ease-in/out.
// Speed therefore changes only at the beginning and end, not at waypoints.
function carryCubeMatrix(
  plan: CarryPlan, k: So101Kinematics, phase: number
): THREE.Matrix4 {
  return carryGeometryMatrix(plan, k, parameterAtCarryPhase(plan, k, phase));
}

// Plan the carry for a candidate grasp (face + elbow). The cube path is fixed
// (grasp pose → drop hover); we pick the path *mode*: prefer the straight chord,
// fall back to the polar arc. A mode is accepted only if the chosen elbow keeps
// the arm within joint limits across the *whole* sweep, not just the endpoints.
// That whole-path check is what prevents the wrist from being driven past its
// limit mid-carry (which the per-frame IK would otherwise resolve by silently
// falling back to a joint lerp, whipping the gripper).
function planCarry(
  k: So101Kinematics,
  face: CubeFace,
  elbow: 'up' | 'down',
  sourcePose: CubePose,
  targetPose: CubePose,
  targetHoverOffset: number
): CarryPlan | null {
  const graspGripperMatrix = graspMatrix(face, sourcePose);
  if (!graspGripperMatrix) { return null; }
  const inwardNormal = new THREE.Vector3(0, 0, 1).transformDirection(
    createWorldFromCubeContactMatrix(face, sourcePose)
  );
  const graspCubeMatrix =
    createWorldFromCubeMatrix(pushedCube(sourcePose, inwardNormal, SAFETY_MARGIN));
  const cubeFromGripper =
    graspCubeMatrix.clone().invert().multiply(graspGripperMatrix);
  const dropCubeMatrix = createWorldFromCubeMatrix({
    ...targetPose,
    z: targetPose.z + targetHoverOffset
  });
  const graspCubePosition =
    new THREE.Vector3().setFromMatrixPosition(graspCubeMatrix);
  const dropCubePosition =
    new THREE.Vector3().setFromMatrixPosition(dropCubeMatrix);
  const base = {
    cubeFromGripper,
    graspCubePosition,
    dropCubePosition,
    graspCubeQuaternion: new THREE.Quaternion().setFromRotationMatrix(graspCubeMatrix),
    dropCubeQuaternion: new THREE.Quaternion().setFromRotationMatrix(dropCubeMatrix),
    graspRadius: Math.hypot(
      graspCubePosition.x - k.panAxis.x, graspCubePosition.y - k.panAxis.y
    ),
    dropRadius: Math.hypot(
      dropCubePosition.x - k.panAxis.x, dropCubePosition.y - k.panAxis.y
    ),
    graspAzimuth: Math.atan2(
      graspCubePosition.y - k.panAxis.y, graspCubePosition.x - k.panAxis.x
    ),
    dropAzimuth: Math.atan2(
      dropCubePosition.y - k.panAxis.y, dropCubePosition.x - k.panAxis.x
    )
  };
  for (const mode of ['straight', 'polar'] as const) {
    const plan: CarryPlan = { mode, ...base };
    let feasible = true;
    for (let i = 0; i <= CARRY_SAMPLES; i++) {
      const gripperMatrix =
        carryGeometryMatrix(plan, k, i / CARRY_SAMPLES).multiply(cubeFromGripper);
      const result = solveSimpleGraspIk(k, gripperMatrix);
      if (!(result.type === 'success' &&
        result.branches.some(branch => branch.elbow === elbow))) {
        feasible = false;
        break;
      }
    }
    if (feasible) { return plan; }
  }
  return null;
}

// One point of the carry's height-over-time profile.
export interface CarryProfilePoint {
  phase: number;
  time: number;
  distance: number;
  height: number;
  waypoint?: string;
}

export interface Trajectory {
  duration: number;
  evaluate(t: number): TrajectoryFrame;
  // Sampled carry profile (stage 4): height against carry time.
  carryProfile(samples?: number): CarryProfilePoint[];
  // Carry phase [0, 1] at time `t`, or null when `t` is outside the carry.
  carryFraction(t: number): number | null;
}

export interface TrajectoryOptions {
  startFromAndReturnToRestPose?: boolean;
}

export function computeTrajectory(
  k: So101Kinematics,
  sourcePose: CubePose,
  targetPose: CubePose,
  options: TrajectoryOptions = {}
): Trajectory | null {
  // The gripper never lets go between the grasp and the drop, so the cube
  // arrives at the target in an orientation that is fully determined by which
  // physical face was grasped. The grasp therefore has to be chosen so the
  // *whole* motion works with one face and one elbow: the source hover and
  // grasp, the target predrop and postdrop hovers, AND every point of the
  // carry in between. Faces are tried in preference order; for each, an elbow
  // that solves the four keyframes is only accepted if `planCarry` can also
  // follow the carry without leaving the workspace or exceeding a joint limit.
  let hoverJoints: JointAngles | null = null;
  let graspJoints: JointAngles | null = null;
  let targetHoverJoints: JointAngles | null = null;
  let predropJoints: JointAngles | null = null;
  let postdropHoverJoints: JointAngles | null = null;
  let selectedFace: CubeFace | null = null;
  let selectedElbow: 'up' | 'down' | null = null;
  let carryPlan: CarryPlan | null = null;
  const sourceHoverOffset = SOURCE_HOVER_TIP_Z - sourcePose.z;
  const targetHoverOffset = PREDROP_HOVER_TIP_Z - targetPose.z;
  const postdropHoverOffset = POSTDROP_HOVER_TIP_Z - targetPose.z;
  for (const face of VERTICAL_FACES) {
    const hover = graspMatrix(face, sourcePose, sourceHoverOffset);
    const grasp = graspMatrix(face, sourcePose);
    const targetHover = graspMatrix(face, targetPose, targetHoverOffset);
    const postdropHover = graspMatrix(face, targetPose, postdropHoverOffset);
    if (!hover || !grasp || !targetHover || !postdropHover) { continue; }
    const hoverResult = solveSimpleGraspIk(k, hover);
    const graspResult = solveSimpleGraspIk(k, grasp);
    const targetHoverResult = solveSimpleGraspIk(k, targetHover);
    if (
      hoverResult.type === 'unreachable' ||
      graspResult.type === 'unreachable' ||
      targetHoverResult.type === 'unreachable'
    ) {
      continue;
    }
    // Prefer elbow-up, but accept elbow-down; the chosen elbow must solve all
    // three keyframes and yield a followable carry.
    for (const elbow of ['up', 'down'] as const) {
      const hoverBranch = hoverResult.branches.find(b => b.elbow === elbow);
      const graspBranch = graspResult.branches.find(b => b.elbow === elbow);
      const targetHoverBranch = targetHoverResult.branches.find(b => b.elbow === elbow);
      if (!hoverBranch || !graspBranch || !targetHoverBranch) { continue; }
      const plan = planCarry(
        k, face, elbow, sourcePose, targetPose, targetHoverOffset
      );
      if (!plan) { continue; }
      const predropMatrix =
        carryGeometryMatrix(plan, k, 1).multiply(plan.cubeFromGripper);
      const predropResult = solveSimpleGraspIk(k, predropMatrix);
      const predropBranch = predropResult.type === 'success'
        ? predropResult.branches.find(b => b.elbow === elbow)
        : undefined;
      const postdropHoverResult = solveSimpleGraspIk(k, postdropHover);
      const postdropHoverBranch = postdropHoverResult.type === 'success'
        ? postdropHoverResult.branches.find(b => b.elbow === elbow)
        : undefined;
      if (!predropBranch || !postdropHoverBranch) { continue; }
      hoverJoints = hoverBranch.joints;
      graspJoints = graspBranch.joints;
      targetHoverJoints = targetHoverBranch.joints;
      predropJoints = predropBranch.joints;
      postdropHoverJoints = postdropHoverBranch.joints;
      selectedFace = face;
      selectedElbow = elbow;
      carryPlan = plan;
      break;
    }
    if (carryPlan) { break; }
  }
  if (
    hoverJoints === null ||
    graspJoints === null ||
    targetHoverJoints === null ||
    predropJoints === null ||
    postdropHoverJoints === null ||
    selectedFace === null ||
    selectedElbow === null ||
    carryPlan === null
  ) {
    return null;
  }

  const stage1EndJoints = hoverJoints;
  const stage2EndJoints = graspJoints;
  const stage4EndJoints = predropJoints;
  const stage5EndJoints = postdropHoverJoints;
  const stage2End = STAGE1_DURATION + STAGE2_DURATION;
  const stage3End = stage2End + STAGE3_DURATION;
  const stage4End = stage3End + STAGE4_DURATION;
  const coreDuration = stage4End + STAGE5_DURATION;
  const startFromAndReturnToRestPose =
    options.startFromAndReturnToRestPose ?? false;
  const startOffset = startFromAndReturnToRestPose ? REST_PHASE_DURATION : 0;
  const restToHoverDuration = startOffset + STAGE1_DURATION;
  const postdropHoverTime = stage4End + POSTDROP_HOVER_DURATION;
  const hoverToRestDuration = RETURN_TO_NEUTRAL_DURATION + REST_PHASE_DURATION;
  const duration = coreDuration + startOffset +
    (startFromAndReturnToRestPose ? REST_PHASE_DURATION : 0);
  const releaseOpeningFraction =
    RELEASE_OPENING_ANGLE / (GRIPPER_OPEN - GRIPPER_CLOSED);
  const movementOpeningFraction =
    MOVEMENT_OPENING_ANGLE / (GRIPPER_OPEN - GRIPPER_CLOSED);
  const releaseTime = releaseOpeningFraction * POSTDROP_HOVER_DURATION;
  const movementStartTime =
    movementOpeningFraction * POSTDROP_HOVER_DURATION;
  const retreatSplineDuration = STAGE5_DURATION - movementStartTime;
  const hoverSplinePhase =
    (POSTDROP_HOVER_DURATION - movementStartTime) / retreatSplineDuration;
  const releasedCubePose = cubePoseFromMatrix(carryCubeMatrix(carryPlan, k, 1));

  // Direction pointing from the grasped face into the cube. The cube is pushed
  // the opposite way (toward the fixed jaw) as the gripper closes.
  const inwardNormal = new THREE.Vector3(0, 0, 1).transformDirection(
    createWorldFromCubeContactMatrix(selectedFace, sourcePose)
  );

  return {
    duration,
    evaluate(t: number): TrajectoryFrame {
      if (startFromAndReturnToRestPose && t <= restToHoverDuration) {
        const phase = t / restToHoverDuration;
        return {
          joints: restToHoverJoints(stage1EndJoints, phase),
          gripper: quinticBezier(
            REST_FRAME.gripper,
            REST_FRAME.gripper,
            NEUTRAL_FRAME.gripper,
            NEUTRAL_FRAME.gripper,
            GRIPPER_OPEN,
            GRIPPER_OPEN,
            phase
          ),
          sourceCube: sourcePose
        };
      }
      const coreTime = t - startOffset;
      if (startFromAndReturnToRestPose && coreTime >= postdropHoverTime) {
        const phase =
          (coreTime - postdropHoverTime) / hoverToRestDuration;
        return {
          joints: hoverToRestJoints(stage5EndJoints, phase),
          gripper: quinticBezier(
            GRIPPER_OPEN,
            GRIPPER_OPEN,
            NEUTRAL_FRAME.gripper,
            NEUTRAL_FRAME.gripper,
            REST_FRAME.gripper,
            REST_FRAME.gripper,
            phase
          ),
          sourceCube: targetPose
        };
      }
      t = coreTime;
      if (t >= stage4End) {
        // Stage 5: wait until the released cube clears the jaws, then follow a
        // single C2 joint spline through hover and directly back to neutral.
        const elapsed = Math.min(STAGE5_DURATION, t - stage4End);
        const openingPhase =
          Math.min(1, elapsed / POSTDROP_HOVER_DURATION);
        const movementPhase = Math.min(
          1, Math.max(0, (elapsed - movementStartTime) / retreatSplineDuration)
        );
        const joints = splineJointsThroughWaypoint(
          stage4EndJoints,
          stage5EndJoints,
          NEUTRAL_FRAME.joints,
          hoverSplinePhase,
          movementPhase
        );
        const gripper = elapsed <= POSTDROP_HOVER_DURATION
          ? GRIPPER_CLOSED +
            (GRIPPER_OPEN - GRIPPER_CLOSED) * openingPhase
          : GRIPPER_OPEN +
            (NEUTRAL_FRAME.gripper - GRIPPER_OPEN) *
            smoothstep(
              (elapsed - POSTDROP_HOVER_DURATION) / RETURN_TO_NEUTRAL_DURATION
            );
        return {
          joints,
          gripper,
          sourceCube: openingPhase < releaseOpeningFraction
            ? releasedCubePose
            : {
              ...targetPose,
              z: Math.max(
                targetPose.z,
                releasedCubePose.z -
                (releasedCubePose.z - targetPose.z) *
                ((t - stage4End - releaseTime) / DROP_DURATION) ** 2
              )
            }
        };
      }
      if (t >= stage3End) {
        // Stage 4: carry the grasped cube up and over to the drop hover, along
        // the path mode chosen by `planCarry` (straight when possible, polar
        // otherwise). The plan was validated across the whole sweep, so the
        // per-frame IK below resolves cleanly; the joint-lerp is only a defensive
        // fallback for numerical edge cases and should not normally run.
        const phase = (t - stage3End) / STAGE4_DURATION;
        const cubeMatrix = carryCubeMatrix(carryPlan, k, phase);
        const gripperMatrix = cubeMatrix.clone().multiply(carryPlan.cubeFromGripper);
        const result = solveSimpleGraspIk(k, gripperMatrix);
        const branch = result.type === 'success'
          ? result.branches.find(candidate => candidate.elbow === selectedElbow)
          : undefined;
        return {
          joints: branch?.joints ??
            lerpJoints(stage2EndJoints, stage4EndJoints, smoothstep(phase)),
          gripper: GRIPPER_CLOSED,
          sourceCube: cubePoseFromMatrix(cubeMatrix)
        };
      }
      if (t >= stage2End) {
        // Stage 3: hold the arm at grasp and close the gripper. Once the
        // moving jaw reaches the cube it shoves it flush against the fixed jaw.
        const alpha = smoothstep((t - stage2End) / STAGE3_DURATION);
        const gripper = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * alpha;
        const push = gripper < GRIPPER_CONTACT
          ? SAFETY_MARGIN *
            (GRIPPER_CONTACT - gripper) / (GRIPPER_CONTACT - GRIPPER_CLOSED)
          : 0;
        return {
          joints: stage2EndJoints,
          gripper,
          sourceCube: pushedCube(sourcePose, inwardNormal, push)
        };
      }
      if (t >= STAGE1_DURATION) {
        const alpha = smoothstep((t - STAGE1_DURATION) / STAGE2_DURATION);
        const matrix = graspMatrix(
          selectedFace,
          sourcePose,
          sourceHoverOffset * (1 - alpha)
        );
        const result = matrix ? solveSimpleGraspIk(k, matrix) : null;
        const branch = result?.type === 'success'
          ? result.branches.find(candidate => candidate.elbow === selectedElbow)
          : undefined;
        return {
          joints: branch?.joints ??
            lerpJoints(stage1EndJoints, stage2EndJoints, alpha),
          gripper: GRIPPER_OPEN,
          sourceCube: sourcePose
        };
      }
      const alpha = smoothstep(t / STAGE1_DURATION);
      return {
        joints: lerpJoints(NEUTRAL_FRAME.joints, stage1EndJoints, alpha),
        gripper: NEUTRAL_FRAME.gripper + (GRIPPER_OPEN - NEUTRAL_FRAME.gripper) * alpha,
        sourceCube: sourcePose
      };
    },
    carryProfile(samples = 64): CarryProfilePoint[] {
      const points: CarryProfilePoint[] = [];
      let previous: THREE.Vector3 | null = null;
      let distance = 0;
      const waypointPhases = CARRY_WAYPOINTS.map(point => ({
        phase: carryPhaseAtParameter(carryPlan, k, point.phase),
        label: point.label
      }));
      const phases = new Set<number>(waypointPhases.map(point => point.phase));
      for (let i = 0; i <= samples; i++) { phases.add(i / samples); }
      for (const phase of [...phases].sort((a, b) => a - b)) {
        const position = new THREE.Vector3().setFromMatrixPosition(
          carryCubeMatrix(carryPlan, k, phase)
        );
        if (previous) {
          distance += Math.hypot(position.x - previous.x, position.y - previous.y);
        }
        points.push({
          phase,
          time: phase * STAGE4_DURATION,
          distance,
          height: position.z,
          waypoint: waypointPhases.find(point => point.phase === phase)?.label
        });
        previous = position;
      }
      return points;
    },
    carryFraction(t: number): number | null {
      t -= startOffset;
      if (t < stage3End) { return null; }
      return Math.min(1, (t - stage3End) / STAGE4_DURATION);
    }
  };
}

// Source cube translated `push` metres toward the fixed jaw (opposite the
// face's inward normal). Faces are vertical, so this only moves x/y.
function pushedCube(
  pose: CubePose,
  inwardNormal: THREE.Vector3,
  push: number
): CubePose {
  if (push === 0) { return pose; }
  return {
    ...pose,
    x: pose.x - inwardNormal.x * push,
    y: pose.y - inwardNormal.y * push,
    z: pose.z - inwardNormal.z * push
  };
}
