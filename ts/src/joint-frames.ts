// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Convert joint values between the sim frame and the real frame.
//
// A port of py/src/pick_and_place/core/joint_frames.py, held to it by the
// fixtures in fixtures/parity. The trajectory and the simulator speak the *sim
// frame* (arm joints in radians, gripper a joint angle in radians); a policy's
// action and a follower's readback speak the *real frame* (arm joints in
// degrees, gripper a 0-100 position).
//
// The arm is a plain radians/degrees conversion. The gripper is not: it maps
// through jaw endpoints calibrated on the hardware, so a policy action that
// says "gripper 39.3" means an angle, not a percentage of the hinge's range.

// Observed follower gripper encoder endpoints, calibrated on the hardware.
export const GRIPPER_READBACK_CLOSED = 2.3;
export const GRIPPER_READBACK_OPEN = 98.5;
export const GRIPPER_RENDER_CLOSED_DEG = -10.0;
export const GRIPPER_RENDER_OPEN_DEG = 120.0;

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

/** Map a sim gripper joint angle (radians) to a follower 0-100 position. */
export function gripperAngleToPosition(angleRad: number): number {
  const angleDeg = (angleRad * 180) / Math.PI;
  const spanDeg = GRIPPER_RENDER_OPEN_DEG - GRIPPER_RENDER_CLOSED_DEG;
  const t = clamp((angleDeg - GRIPPER_RENDER_CLOSED_DEG) / spanDeg, 0, 1);
  return GRIPPER_READBACK_CLOSED + t * (GRIPPER_READBACK_OPEN - GRIPPER_READBACK_CLOSED);
}

/** Map a follower 0-100 position to a sim gripper joint angle (radians). */
export function gripperPositionToAngle(position: number): number {
  const spanEncoder = GRIPPER_READBACK_OPEN - GRIPPER_READBACK_CLOSED;
  const t = clamp((position - GRIPPER_READBACK_CLOSED) / spanEncoder, 0, 1);
  const spanDeg = GRIPPER_RENDER_OPEN_DEG - GRIPPER_RENDER_CLOSED_DEG;
  return ((GRIPPER_RENDER_CLOSED_DEG + t * spanDeg) * Math.PI) / 180;
}

/**
 * Convert sim-frame joints (radians, gripper included) into a real-frame
 * 6-vector: the shape a policy observation and a policy action both take.
 *
 * The conversion itself is done at full precision, as Python's is. Narrowing to
 * float32 is a separate step that belongs where the value becomes an
 * observation, not here.
 */
export function simFrameToReal(simJoints: ArrayLike<number>): Float64Array {
  const out = new Float64Array(6);
  for (let i = 0; i < 5; i += 1) {
    out[i] = (simJoints[i] * 180) / Math.PI;
  }
  out[5] = gripperAngleToPosition(simJoints[5]);
  return out;
}

/** Convert a real-frame 6-vector back into sim-frame joints (radians). */
export function realFrameToSim(realJoints: ArrayLike<number>): Float64Array {
  const out = new Float64Array(6);
  for (let i = 0; i < 5; i += 1) {
    out[i] = (realJoints[i] * Math.PI) / 180;
  }
  out[5] = gripperPositionToAngle(realJoints[5]);
  return out;
}
