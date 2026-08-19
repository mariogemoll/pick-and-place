// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

// The policy environment, stepped in the browser by MuJoCo's WebAssembly build.
//
// A port of `PolicySimEnv` in py/src/pick_and_place/runtime/policy_sim.py, kept
// to the parts a learned policy actually goes through: put the arm and the cube
// somewhere, hand it a real-frame action, step the physics a tenth of a second,
// hand back what it can observe. Nothing renders here -- the scene is drawn
// separately from the web manifest -- and nothing samples a scenario, because
// on this page the cube and the target are wherever the reader dragged them.
//
// The scene is a compiled binary model rather than MJCF. MuJoCo's XML writer
// rounds every number to six significant figures; the binary carries the
// compiled model verbatim, so what steps here is what steps in Python.

import loadMujoco from '@mujoco/mujoco';

import { realFrameToSim, simFrameToReal } from '../../joint-frames';

export interface PolicySceneManifest {
  format: 'pick-and-place-policy-scene';
  version: number;
  mujocoVersion: string;
  model: string;
  timestep: number;
  simulationHz: number;
  policyHz: number;
  substeps: number;
  jointNames: string[];
  jointQposAdr: number[];
  ctrlIndex: number[];
  trackingBiasRad: number[];
  neutralJointsRad: number[];
  cubeQposAdr: number;
  cubeDofAdr: number;
  cubeHalfSize: number;
  dropZoneHalfSize: number;
}

/** A cube pose in the world frame; the quaternion is MuJoCo's (w, x, y, z). */
export interface CubePose {
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

export interface EpisodeSetup {
  cube: CubePose;
  targetXy: [number, number];
  /** Real-frame joints the arm is observed at, six values. */
  initialJointsReal: number[];
}

export interface PolicyEnvironment {
  readonly manifest: PolicySceneManifest;
  /** Seconds of simulated time one step covers. */
  readonly stepSeconds: number;
  reset(setup: EpisodeSetup): void;
  /** Advance one policy tick under a real-frame action, and observe the result. */
  step(actionReal: ArrayLike<number>): Float32Array;
  /** Real-frame joints, the observation a policy is shown. */
  observe(): Float32Array;
  /** Sim-frame joint angles in radians, which is what a viewer draws. */
  jointAnglesRad(): Float64Array;
  cubePose(): CubePose;
  targetXy(): [number, number];
  destroy(): void;
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

/** Fetch the scene's two files and build the environment from them. */
export async function loadPolicyEnvironment(baseUrl: string): Promise<PolicyEnvironment> {
  const manifest = (await (await fetch(`${baseUrl}.json`)).json()) as PolicySceneManifest;
  const response = await fetch(`${baseUrl.replace(/[^/]*$/, '')}${manifest.model}`);
  if (!response.ok) {
    throw new Error(`Unable to load ${manifest.model}: ${response.status}`);
  }
  return createPolicyEnvironment(manifest, new Uint8Array(await response.arrayBuffer()));
}

export async function createPolicyEnvironment(
  manifest: PolicySceneManifest,
  modelBytes: Uint8Array
): Promise<PolicyEnvironment> {
  const mujoco = await loadMujoco();

  // A compiled model is tied to the engine that wrote it. Refusing here names
  // the problem; loading anyway would produce a model that is wrong in ways
  // nothing on the page would surface.
  const engineVersion = mujoco.mj_versionString();
  if (engineVersion !== manifest.mujocoVersion) {
    throw new Error(
      `Scene was compiled by MuJoCo ${manifest.mujocoVersion} but this build is ` +
        `${engineVersion}. Re-export it with scripts/export_web_policy_scene.py.`
    );
  }

  const vfs = new mujoco.MjVFS();
  vfs.addBuffer(manifest.model, modelBytes);
  const model = mujoco.MjModel.from_binary_path(manifest.model, vfs);
  vfs.delete();

  // Python sets this after compiling, for the same reason the export leaves it
  // out: it does not survive being written at six significant figures.
  model.opt.timestep = manifest.timestep;
  const data = new mujoco.MjData(model);

  const jointCount = manifest.jointNames.length;
  // The bindings hand back live views into WebAssembly memory, typed loosely.
  // Naming them once here keeps every write below type-checked, and holding the
  // reference is exactly what the bindings intend: the view tracks the
  // simulation rather than snapshotting it.
  const ctrlRange = model.actuator_ctrlrange as Float64Array;
  const actuatorActAdr = model.actuator_actadr as Int32Array;
  const qpos = data.qpos as Float64Array;
  const qvel = data.qvel as Float64Array;
  const ctrl = data.ctrl as Float64Array;
  const activation = data.act as Float64Array;
  const trackingBias = Float64Array.from(manifest.trackingBiasRad);
  const target: [number, number] = [0, 0];

  function writeCtrl(values: ArrayLike<number>): void {
    for (let i = 0; i < jointCount; i += 1) {
      const actuator = manifest.ctrlIndex[i];
      ctrl[actuator] = clamp(
        values[i],
        ctrlRange[actuator * 2],
        ctrlRange[actuator * 2 + 1]
      );
    }
  }

  function clampedSimCommand(actionReal: ArrayLike<number>, bias: number): Float64Array {
    const sim = realFrameToSim(actionReal);
    const out = new Float64Array(jointCount);
    for (let i = 0; i < jointCount; i += 1) {
      out[i] = sim[i] + bias * trackingBias[i];
    }
    return out;
  }

  function reset(setup: EpisodeSetup): void {
    mujoco.mj_resetData(model, data);

    // The arm is *observed* at the initial pose, so the command that holds it
    // there is the one whose droop lands on it. Seeding ctrl with the pose
    // itself would start every episode drifting by the tracking bias.
    const truePose = clampedSimCommand(setup.initialJointsReal, 0);
    for (let i = 0; i < jointCount; i += 1) {
      const actuator = manifest.ctrlIndex[i];
      const clamped = clamp(truePose[i], ctrlRange[actuator * 2], ctrlRange[actuator * 2 + 1]);
      qpos[manifest.jointQposAdr[i]] = clamped;
      truePose[i] = clamped;
    }
    const holdCtrl = new Float64Array(jointCount);
    for (let i = 0; i < jointCount; i += 1) {
      holdCtrl[i] = truePose[i] - trackingBias[i];
    }
    writeCtrl(holdCtrl);
    // The position actuators filter ctrl through an activation state. Left at
    // zero it produces a startup transient that never happens in Python.
    for (let i = 0; i < jointCount; i += 1) {
      const actadr = actuatorActAdr[manifest.ctrlIndex[i]];
      if (actadr >= 0) {
        activation[actadr] = holdCtrl[i];
      }
    }

    const cubeAdr = manifest.cubeQposAdr;
    for (let i = 0; i < 3; i += 1) {
      qpos[cubeAdr + i] = setup.cube.position[i];
    }
    for (let i = 0; i < 4; i += 1) {
      qpos[cubeAdr + 3 + i] = setup.cube.quaternion[i];
    }
    for (let i = 0; i < 6; i += 1) {
      qvel[manifest.cubeDofAdr + i] = 0;
    }

    target[0] = setup.targetXy[0];
    target[1] = setup.targetXy[1];
    mujoco.mj_forward(model, data);
  }

  function step(actionReal: ArrayLike<number>): Float32Array {
    writeCtrl(clampedSimCommand(actionReal, 1));
    for (let i = 0; i < manifest.substeps; i += 1) {
      mujoco.mj_step(model, data);
    }
    return observe();
  }

  function jointAnglesRad(): Float64Array {
    const out = new Float64Array(jointCount);
    for (let i = 0; i < jointCount; i += 1) {
      out[i] = qpos[manifest.jointQposAdr[i]];
    }
    return out;
  }

  function observe(): Float32Array {
    // Narrowed here and only here: `sim_state_to_real` hands the policy a
    // float32 observation, so the browser has to lose the same bits Python
    // loses or the two are shown different numbers.
    return Float32Array.from(simFrameToReal(jointAnglesRad()));
  }

  function cubePose(): CubePose {
    const adr = manifest.cubeQposAdr;
    return {
      position: [qpos[adr], qpos[adr + 1], qpos[adr + 2]],
      quaternion: [
        qpos[adr + 3],
        qpos[adr + 4],
        qpos[adr + 5],
        qpos[adr + 6]
      ]
    };
  }

  return {
    manifest,
    stepSeconds: 1 / manifest.policyHz,
    reset,
    step,
    observe,
    jointAnglesRad,
    cubePose,
    targetXy: () => [target[0], target[1]],
    destroy: (): void => {
      data.delete();
      model.delete();
    }
  };
}
