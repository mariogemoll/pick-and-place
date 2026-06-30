#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Open the MuJoCo viewer at recorded episodes' per-phase reset states.

The reverse curriculum resets the sim into a full ``qpos``/``qvel`` snapshot
drawn from a recorded episode, restoring the arm (and, mid-task, the grasped
cube) to the state at the start of a scripted phase. This restores that snapshot
and shows it live.

Point it at a single ``.npz`` to inspect one episode, or at a directory to step
through every episode's reset state for a phase — Enter (in the viewer or the
terminal) advances to the next episode, so you can eyeball the spread of, say,
every before-drop pose in a recorded pool. By default each frame is held static;
with ``--settle`` the recorded set point is re-commanded and physics steps, so a
mid-grasp reset can be checked for whether the held cube actually stays grasped
once MuJoCo recomputes contacts from the restored state.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.follower import JOINT_NAMES
from pick_and_place.geometry import CUBE_HALF_SIZE

# GLFW key codes the viewer reports for Return / keypad-Enter.
_ENTER_KEYS = frozenset({257, 335})


def _build_scene() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Compile one reusable scene shared across episodes.

    The cube is a free body (so any episode's 13-wide ``qpos`` restores into it)
    and the drop target is a mocap marker repositioned per episode at runtime, so
    a single model/data serves the whole pool without recompiling.
    """
    spec = build_scene(include_environment=True)
    marker = spec.worldbody.add_body(name="reset_target_marker")
    marker.mocap = True
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, 0.002),
        size=(CUBE_HALF_SIZE, CUBE_HALF_SIZE, 0.001),
        rgba=(0.0, 0.95, 0.35, 0.7),
        contype=0,
        conaffinity=0,
    )
    spec.body("pick_cube").add_freejoint()
    model = spec.compile()
    marker_mocapid = int(model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reset_target_marker")
    ])
    return model, mujoco.MjData(model), marker_mocapid


def _episode_paths(path: Path) -> list[Path]:
    if path.is_dir():
        paths = sorted(path.glob("episode_*.npz"))
        if not paths:
            raise SystemExit(f"no episode_*.npz found in {path}")
        return paths
    return [path]


def _watch_for_enter(advance: threading.Event) -> None:
    for _ in iter(sys.stdin.readline, ""):
        advance.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="an episode .npz, or a directory of episode_*.npz"
    )
    parser.add_argument(
        "--phase",
        default="release",
        help="phase whose start frame to restore (default: release = hover above drop)",
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="hold the recorded set point and step physics instead of freezing the frame",
    )
    args = parser.parse_args()

    paths = _episode_paths(args.path)
    model, data, marker_mocapid = _build_scene()
    actuator_id = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }

    def restore(record: np.lib.npyio.NpzFile) -> int:
        names = [str(n) for n in record["phase_names"]]
        if args.phase not in names:
            raise SystemExit(f"phase {args.phase!r} not in {names}")
        frame = int(record["phase_boundaries"][names.index(args.phase)])
        if record["qpos"].shape[1] != model.nq:
            raise SystemExit(
                f"qpos width {record['qpos'].shape[1]} != model.nq {model.nq}; scene mismatch"
            )
        # Clear any solver warm-start, contact, and time state left from the
        # previous episode's settling, so each scene starts physics fresh from
        # its own snapshot rather than continuing the last one.
        mujoco.mj_resetData(model, data)
        data.qpos[:] = record["qpos"][frame]
        data.qvel[:] = record["qvel"][frame]
        for name, value in zip(JOINT_NAMES, record["commanded"][frame]):
            data.ctrl[actuator_id[name]] = float(value)
        target = record["cube_target"]
        data.mocap_pos[marker_mocapid] = (float(target[0]), float(target[1]), 0.0)
        mujoco.mj_forward(model, data)
        return frame

    advance = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(target=_watch_for_enter, args=(advance,), daemon=True).start()
    if len(paths) > 1:
        print(
            f"{len(paths)} episodes; phase {args.phase!r}. "
            "Press Enter (viewer or terminal) for the next; close the viewer to stop."
        )

    def on_key(keycode: int) -> None:
        if keycode in _ENTER_KEYS:
            advance.set()

    period = float(model.opt.timestep)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.opt.geomgroup[4] = 1
        index = 0
        while viewer.is_running():
            record = np.load(paths[index], allow_pickle=True)
            if "phase_boundaries" not in record:
                raise SystemExit(
                    f"{paths[index].name} has no phase_boundaries; re-record with the "
                    "current record_episodes.py"
                )
            frame = restore(record)
            print(
                f"[{index + 1}/{len(paths)}] {paths[index].name}: "
                f"phase {args.phase!r} at frame {frame}/{len(record['qpos'])}, "
                f"success={bool(record['success'])}"
            )
            advance.clear()
            while viewer.is_running() and not advance.is_set():
                start = time.time()
                if args.settle:
                    mujoco.mj_step(model, data)
                viewer.sync()
                remaining = period - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
            index = (index + 1) % len(paths)


if __name__ == "__main__":
    main()
