#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fit arm joint zero offsets from hand-eye offset measurements.

Generalizes the shoulder-pan zero fit to the whole pitch chain. Perturbing a
revolute sim joint (world axis ``a``, anchor ``p``) by ``dtheta`` changes the
measured real-minus-sim cube delta by ``-dtheta * (a x (q - p))`` in world
coordinates (``q`` = cube position): moving the sim camera toward the real
camera's pose removes the offset the measurement sees. This script loads
measurement JSONs written by ``measure_hand_eye_offset``, replays each
measured frame's joint state through the sim model to get every joint's world
axis and anchor, and least-squares fits the zero offsets of shoulder_pan,
shoulder_lift, elbow_flex and wrist_flex jointly against the measured
world-frame deltas.

The measured episodes' exports may already carry joint corrections
(``joint_offsets_deg`` or the older ``pan_offset_deg`` in pairs.json); they are
re-applied when replaying joints, so the fitted values are residuals on top of
what the export already corrected. Fitted values are directly the amounts to
*add* to the sim joints via the exporter's ``--joint-offsets-deg`` (verified
empirically: exporting with the fitted offsets collapses the measured error;
the opposite sign doubles it).

A robust trim pass drops frames with grossly outlying residuals (bad tag
detections at range) before the final fit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_sim_real_pairs import _load_episode, _read_episode_rows, _read_info
from pick_and_place import build_scene
from pick_and_place.follower import JOINT_NAMES, real_frame_to_sim
from pick_and_place.joint_zero_fit import (
    FIT_JOINTS,
    JointZeroSample,
    build_columns,
    fit_robust,
    joint_ids,
)
from pick_and_place.scene import PICK_CUBE_HALF_SIZE


def _sim_joint_vector(real_joints: np.ndarray) -> np.ndarray:
    arm_rad, gripper_rad = real_frame_to_sim(real_joints)
    names = [name for name in JOINT_NAMES if name != "gripper"]
    return np.asarray([arm_rad[name] for name in names] + [gripper_rad], dtype=float)


def load_samples(measurement_paths: list[Path]) -> list[JointZeroSample]:
    spec = build_scene(include_environment=False)
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos_addrs = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    }
    ids = joint_ids(model)

    samples: list[JointZeroSample] = []
    dataset_cache: dict[tuple[str, int], np.ndarray] = {}
    for path in measurement_paths:
        with path.open() as f:
            summary = json.load(f)
        for episode in summary["episodes"]:
            episode_dir = Path(episode["name"])
            if not episode_dir.is_absolute():
                for base in (Path.cwd(), path.parent):
                    if (base / episode_dir / "pairs.json").is_file():
                        episode_dir = (base / episode_dir).resolve()
                        break
            with (episode_dir / "pairs.json").open() as f:
                index = json.load(f)
            applied_offsets = {
                name: math.radians(float(deg))
                for name, deg in (index.get("joint_offsets_deg") or {}).items()
            }
            if "pan_offset_deg" in index:
                applied_offsets["shoulder_pan"] = math.radians(float(index["pan_offset_deg"]))
            dataset_root = Path(index["dataset"])
            if not dataset_root.is_absolute():
                for base in (Path.cwd(), *episode_dir.parents):
                    candidate = base / dataset_root
                    if (candidate / "meta" / "info.json").is_file():
                        dataset_root = candidate.resolve()
                        break
                else:
                    raise FileNotFoundError(f"dataset {dataset_root} for {episode_dir}")
            episode_index = int(index["episode_index"])
            key = (str(dataset_root), episode_index)
            if key not in dataset_cache:
                info = _read_info(dataset_root)
                row = next(
                    r
                    for r in _read_episode_rows(dataset_root)
                    if int(r["episode_index"]) == episode_index
                )
                dataset_cache[key] = _load_episode(dataset_root, info, row).states
            states = dataset_cache[key]
            cube_by_frame = {fr["frame"]: fr for fr in index["frames"]}

            for record in episode["frames"]:
                if "delta_world_mm" not in record:
                    continue
                pair = cube_by_frame[record["frame"]]
                sim_joints = _sim_joint_vector(states[record["frame"]])
                for joint_name, offset in applied_offsets.items():
                    sim_joints[JOINT_NAMES.index(joint_name)] += offset
                for i, name in enumerate(JOINT_NAMES):
                    data.qpos[qpos_addrs[name]] = sim_joints[i]
                mujoco.mj_forward(model, data)
                cube = np.array(
                    [pair["cube"]["x"], pair["cube"]["y"], PICK_CUBE_HALF_SIZE], dtype=float
                )
                samples.append(
                    JointZeroSample(
                        delta=np.asarray(record["delta_world_mm"], dtype=float) / 1000.0,
                        columns=build_columns(data, ids, cube),
                        group=episode_dir.parent.name,
                    )
                )
    return samples


def _report(label: str, samples: list[JointZeroSample]) -> None:
    deltas = np.stack([s.delta for s in samples]) * 1000.0
    result = fit_robust(samples)
    residual = result.residual
    rms = np.sqrt((residual**2).sum(axis=1).mean()) * 1000.0
    rms_z = np.sqrt((residual[:, 2] ** 2).mean()) * 1000.0
    offsets = ", ".join(
        f"{name}={math.degrees(v):+.2f}deg" for name, v in zip(FIT_JOINTS, result.params)
    )
    print(
        f"{label}: n={len(samples)} kept={result.keep.sum()} "
        f"delta rms={np.sqrt((deltas**2).sum(axis=1).mean()):.1f}mm -> "
        f"residual rms={rms:.1f}mm (z {rms_z:.1f}mm)"
    )
    print(f"  {offsets}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurements", type=Path, nargs="+", help="measure_hand_eye_offset --output JSON(s)"
    )
    args = parser.parse_args()

    samples = load_samples(args.measurements)
    if not samples:
        raise SystemExit("no usable frames found")

    days = sorted({s.group for s in samples})
    for day in days:
        _report(day, [s for s in samples if s.group == day])
    if len(days) > 1:
        _report("ALL", samples)


if __name__ == "__main__":
    main()
