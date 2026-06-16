#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a trained PPO policy in the MuJoCo viewer against any gym env.

Generic viewer for the standalone RL envs (e.g. ``rl.hover_env:ApproachToHoverEnv``
or ``rl.lift_env:LiftCubeEnv``): pass the env class as a ``module:Class`` spec and
a PPO checkpoint, and the policy is rolled out live with its paired
``VecNormalize`` stats. Each episode resets automatically when it ends. Press
Ctrl-C to quit.

This does not handle the curriculum / 31-dim ``PickPlaceEnv`` contract — see
``view_rl_policy.py`` for that.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import select
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _load_env_class(spec: str) -> type:
    """Resolve a ``module.path:ClassName`` spec to the gym env class."""
    module_path, _, class_name = spec.partition(":")
    if not class_name:
        raise SystemExit(f"--env must be 'module.path:ClassName', got {spec!r}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _resolve_vec_normalize(checkpoint: Path, explicit: Path | None) -> Path:
    """Find the vec_normalize paired with a checkpoint, falling back to checkpoints/."""
    if explicit is not None:
        return explicit
    # best_vecnormalize.pkl saved alongside best_model.zip by _SaveVecNormalizeOnBest
    candidate = checkpoint.parent / "best_vecnormalize.pkl"
    if candidate.exists():
        return candidate
    # fall back to the latest periodic checkpoint pkl in the same run
    ckpt_dir = checkpoint.parent / "checkpoints"
    pkls = sorted(ckpt_dir.glob("*_vecnormalize_*.pkl"))
    if pkls:
        return pkls[-1]
    raise FileNotFoundError(
        f"No vec_normalize found for {checkpoint}. Pass --vec-normalize explicitly."
    )


# GLFW key codes for the main and keypad ENTER keys (MuJoCo's viewer uses GLFW).
_ENTER_KEYS = (257, 335)


def _enter_pressed() -> bool:
    """True if the user has pressed ENTER in the terminal (a line waits on stdin)."""
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()  # consume the line so it doesn't trigger again
        return True
    return False


def _format_info(info: dict) -> str:
    """Render an env info dict as a compact one-line summary."""
    parts = []
    for key, value in info.items():
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return "  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        required=True,
        help="env class as 'module.path:ClassName', e.g. pick_and_place.rl.lift_env:LiftCubeEnv",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="PPO model .zip")
    parser.add_argument("--vec-normalize", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    parser.add_argument("--fps", type=float, default=50.0, help="viewer playback rate")
    args = parser.parse_args()

    env_cls = _load_env_class(args.env)
    vec_normalize_path = _resolve_vec_normalize(args.checkpoint, args.vec_normalize)
    print(f"env:           {args.env}")
    print(f"checkpoint:    {args.checkpoint}")
    print(f"vec_normalize: {vec_normalize_path}")

    # Pressing ENTER in the viewer window flips this; polled in the rollout loop.
    abort = {"requested": False}

    def _on_key(keycode: int) -> None:
        if keycode in _ENTER_KEYS:
            abort["requested"] = True

    # Render env is not wrapped in VecNormalize — we normalise obs manually.
    # Forward the key handler only if the env's viewer supports it.
    raw_kwargs = {"render_mode": "human"}
    if "key_callback" in inspect.signature(env_cls).parameters:
        raw_kwargs["key_callback"] = _on_key
    raw_env = env_cls(**raw_kwargs)
    vec_env = DummyVecEnv([lambda: env_cls()])
    norm_env = VecNormalize.load(str(vec_normalize_path), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = PPO.load(args.checkpoint, device="auto")

    step_period = 1.0 / args.fps
    ep = 0
    try:
        while args.episodes == 0 or ep < args.episodes:
            obs_raw, _ = raw_env.reset()
            print(f"\nepisode {ep + 1}  (press ENTER to abort and start a new one)")

            done = False
            aborted = False
            abort["requested"] = False
            info: dict = {}
            while not done:
                if abort["requested"] or _enter_pressed():
                    aborted = True
                    break
                # Normalise obs the same way the training env did.
                obs_norm = norm_env.normalize_obs(obs_raw[np.newaxis])
                action, _ = model.predict(obs_norm, deterministic=True)
                obs_raw, _, term, trunc, info = raw_env.step(action[0])
                raw_env.render()
                done = term or trunc
                time.sleep(step_period)

            print(f"  {_format_info(info)}{'  (aborted)' if aborted else ''}")
            ep += 1
    except KeyboardInterrupt:
        pass
    finally:
        raw_env.close()
        norm_env.close()


if __name__ == "__main__":
    main()
