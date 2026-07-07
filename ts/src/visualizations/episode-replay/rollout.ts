// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// Parser for the "PPRL" episode rollout format written by
// py/scripts/export_episode_rolls.py. Each frame is the sim's full qpos: the
// 6 arm/gripper joint angles (radians) followed by the cube's free-joint pose
// (pos[3] + quat[4], MuJoCo's w,x,y,z order) — the minimal state needed to
// drive the existing joint-hierarchy robot model frame by frame.
const MAGIC = 'PPRL';
const HEADER_BYTES = 4 + 4 * 4 + 2 * 4;

export interface Rollout {
  fps: number;
  nframes: number;
  nq: number;
  targetX: number;
  targetY: number;
  qpos: Float32Array;
}

export function parseRollout(buffer: ArrayBuffer): Rollout {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== MAGIC) {
    throw new Error(`Unexpected rollout magic header: ${magic}`);
  }

  let offset = 4;
  const version = view.getUint32(offset, true);
  offset += 4;
  if (version !== 1) {
    throw new Error(`Unsupported rollout version: ${version}`);
  }
  const fps = view.getUint32(offset, true);
  offset += 4;
  const nframes = view.getUint32(offset, true);
  offset += 4;
  const nq = view.getUint32(offset, true);
  offset += 4;
  const targetX = view.getFloat32(offset, true);
  offset += 4;
  const targetY = view.getFloat32(offset, true);
  offset += 4;

  if (offset !== HEADER_BYTES) {
    throw new Error(`Rollout header size mismatch: ${offset} !== ${HEADER_BYTES}`);
  }

  const qpos = new Float32Array(buffer, offset, nframes * nq);

  return { fps, nframes, nq, targetX, targetY, qpos };
}
