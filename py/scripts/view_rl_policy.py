#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Watch a saved RL policy from recorded per-phase reset states.

This is the policy-action companion to ``view_reset_state.py``: point it at one
recorded episode or a directory of episodes, choose a phase such as ``release``,
and it restores each episode's exact phase-boundary snapshot before rolling the
policy forward in the live MuJoCo viewer. Press Enter to advance to the next
episode after a rollout finishes, or during a rollout to skip it.

Run with ``mjpython`` on macOS (the viewer needs the main thread):

    mjpython scripts/view_rl_policy.py out/episodes --phase release
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_and_place.rl import ReverseCurriculumEnv
from pick_and_place.rl.episode_pool import ResetSnapshot

_ENTER_KEYS = frozenset({257, 335})
_BUDGET_SLACK = 1.5
_MIN_BUDGET_STEPS = 20


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


def _snapshot(record: np.lib.npyio.NpzFile, path: Path, phase: str) -> ResetSnapshot:
    if "phase_boundaries" not in record:
        raise SystemExit(
            f"{path.name} has no phase_boundaries; re-record with current record_episodes.py"
        )
    names = [str(n) for n in record["phase_names"]]
    if phase not in names:
        raise SystemExit(f"phase {phase!r} not in {names}")
    frame = int(record["phase_boundaries"][names.index(phase)])
    return ResetSnapshot(
        qpos=np.asarray(record["qpos"][frame], dtype=np.float64).copy(),
        qvel=np.asarray(record["qvel"][frame], dtype=np.float64).copy(),
        ctrl=np.asarray(record["commanded"][frame], dtype=np.float64).copy(),
        target_xy=np.asarray(record["cube_target"][:2], dtype=np.float64).copy(),
        frame=frame,
        total_frames=int(record["qpos"].shape[0]),
        source=path,
    )


def _restore(env: ReverseCurriculumEnv, snapshot: ResetSnapshot) -> np.ndarray:
    env._restore(snapshot)
    remaining = snapshot.total_frames - snapshot.frame
    env._max_steps = max(_MIN_BUDGET_STEPS, math.ceil(remaining * _BUDGET_SLACK))
    env._step_count = 0
    return env._observation()


def _load_policy_env(
    checkpoint_dir: Path,
    pool: Path,
    phase: str,
) -> tuple[PPO | None, VecNormalize | None, ReverseCurriculumEnv]:
    """Create the RL env and, unless running scripted/random, its normalizer/model."""
    env = ReverseCurriculumEnv(pool, stage=0)
    env.phase = phase
    vec_env = DummyVecEnv([lambda: env])
    vecnormalize_path = checkpoint_dir / "vecnormalize.pkl"
    model_path = checkpoint_dir / "latest.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not vecnormalize_path.exists():
        raise FileNotFoundError(vecnormalize_path)
    vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(str(model_path), env=vec_env)
    return model, vec_env, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "out" / "episodes",
        help="an episode .npz, or a directory of episode_*.npz",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "rl" / "stage0",
        help="directory containing latest.zip and vecnormalize.pkl",
    )
    parser.add_argument(
        "--phase",
        default="release",
        help="phase boundary to restore before rollout (default: release)",
    )
    parser.add_argument(
        "--mode",
        choices=("policy", "scripted", "random"),
        default="policy",
        help="action source (default: policy)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample from the policy instead of deterministic mean actions",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="advance to the next episode immediately after each rollout",
    )
    args = parser.parse_args()

    paths = _episode_paths(args.path)
    if args.speed <= 0.0:
        parser.error("--speed must be positive")

    pool_path = args.path if args.path.is_dir() else args.path.parent

    if args.mode == "policy":
        model, vec_env, env = _load_policy_env(args.checkpoint_dir, pool_path, args.phase)
    else:
        model = None
        vec_env = None
        env = ReverseCurriculumEnv(pool_path, stage=0)
        env.phase = args.phase

    advance = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(target=_watch_for_enter, args=(advance,), daemon=True).start()

    def on_key(keycode: int) -> None:
        if keycode in _ENTER_KEYS:
            advance.set()

    step_seconds = env._sim_steps * float(env.model.opt.timestep) / args.speed
    print(
        f"{len(paths)} episode(s); phase {args.phase!r}; mode={args.mode}. "
        "Press Enter to skip/advance; close the viewer to stop."
    )

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.opt.geomgroup[4] = 1
        index = 0
        while viewer.is_running():
            with np.load(paths[index], allow_pickle=True) as record:
                snapshot = _snapshot(record, paths[index], args.phase)
                commanded = (
                    np.asarray(record["commanded"], dtype=np.float64)
                    if args.mode == "scripted"
                    else None
                )

            raw_obs = _restore(env, snapshot)
            obs = (
                vec_env.normalize_obs(raw_obs.reshape(1, -1))
                if vec_env is not None
                else raw_obs.reshape(1, -1)
            )
            viewer.sync()
            print(
                f"[{index + 1}/{len(paths)}] {paths[index].name}: "
                f"{args.phase}@{snapshot.frame}/{snapshot.total_frames}, "
                f"budget={env._max_steps}"
            )

            advance.clear()
            terminated = truncated = False
            step_info = {"success": False, "collision": False, "out_of_bounds": False}
            command_frame = snapshot.frame
            while viewer.is_running() and not advance.is_set() and not (
                terminated or truncated
            ):
                start = time.time()
                if model is not None:
                    action, _ = model.predict(obs, deterministic=not args.stochastic)
                    raw_obs, _, terminated, truncated, step_info = env.step(action[0])
                    obs = vec_env.normalize_obs(raw_obs.reshape(1, -1))
                elif commanded is not None:
                    command_frame += 1
                    action = commanded[min(command_frame, len(commanded) - 1)]
                    raw_obs, _, terminated, truncated, step_info = env.step(action)
                    obs = raw_obs.reshape(1, -1)
                else:
                    raw_obs, _, terminated, truncated, step_info = env.step(
                        env.action_space.sample()
                    )
                    obs = raw_obs.reshape(1, -1)
                viewer.sync()
                remaining = step_seconds - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)

            outcome = (
                "SUCCESS" if step_info["success"]
                else "collision" if step_info["collision"]
                else "out-of-bounds" if step_info["out_of_bounds"]
                else "skipped" if advance.is_set() and not (terminated or truncated)
                else "timeout"
            )
            print(f"[{index + 1}/{len(paths)}] -> {outcome}")

            if not args.auto_advance and viewer.is_running() and not advance.is_set():
                print("Press Enter for next episode.")
                while viewer.is_running() and not advance.is_set():
                    viewer.sync()
                    time.sleep(0.03)

            index = (index + 1) % len(paths)

    if vec_env is not None:
        vec_env.close()
    else:
        env.close()


if __name__ == "__main__":
    main()
