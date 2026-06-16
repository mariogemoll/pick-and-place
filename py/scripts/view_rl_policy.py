#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Watch an RL (PPO) policy drive the arm in the MuJoCo viewer.

The RL counterpart to ``view_policy.py`` (which only knows the BC checkpoint
format and the separate 21-dim ``il/observations.py`` contract). Loads any PPO
checkpoint plus its paired ``VecNormalize`` stats trained on ``PickPlaceEnv``
(the frozen 31-dim RL contract in ``pick_and_place.rl.contract``) and rolls it
out live. ``PickPlaceEnv`` and its observation/action contract don't care
whether the checkpoint came from ``scripts/train_curriculum.py`` or any other
training run — only the obs/action shapes and the ``VecNormalize`` stats need to
match, so this takes a checkpoint path, not a curriculum/stage name.

Unlike ``view_rl_env.py`` (the generic viewer for the standalone hover/lift
envs), this drives the actual pick-and-place env.

Each episode resets automatically when it ends. Press Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_and_place.rl.pick_place_env import CONTROL_HZ, PickPlaceEnv


def _resolve_vec_normalize(checkpoint: Path, explicit: Path | None) -> Path:
    """Find the VecNormalize stats paired with a checkpoint's standard name.

    ``train_curriculum.py`` and ``train_rl_hover.py`` both write
    ``best_model.zip`` / ``best_vecnormalize.pkl`` (SB3's ``EvalCallback``
    naming) and ``model_final``-stem / ``vec_normalize_final.pkl`` pairs.
    """
    if explicit is not None:
        return explicit
    if checkpoint.name == "best_model.zip":
        candidate = checkpoint.parent / "best_vecnormalize.pkl"
    elif checkpoint.stem == "model_final":
        candidate = checkpoint.parent / "vec_normalize_final.pkl"
    else:
        raise FileNotFoundError(
            f"Can't infer VecNormalize stats for {checkpoint}; pass --vec-normalize explicitly."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"No VecNormalize stats found at {candidate}.")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="PPO model .zip")
    parser.add_argument(
        "--vec-normalize", type=Path, default=None, help="explicit VecNormalize .pkl"
    )
    parser.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    vec_normalize_path = _resolve_vec_normalize(args.checkpoint, args.vec_normalize)

    print(f"checkpoint:    {args.checkpoint}")
    print(f"vec_normalize: {vec_normalize_path}")

    # Render env is not wrapped in VecNormalize — we normalise obs manually.
    raw_env = PickPlaceEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: PickPlaceEnv()])
    norm_env = VecNormalize.load(str(vec_normalize_path), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = PPO.load(args.checkpoint, device="auto")

    ep = 0
    try:
        while args.episodes == 0 or ep < args.episodes:
            obs_raw, info = raw_env.reset()
            print(f"\nepisode {ep + 1}  source={info['source']}  target={info['target']}")

            done = False
            while not done:
                # Normalise obs the same way the training env did.
                obs_norm = norm_env.normalize_obs(obs_raw[np.newaxis])
                action, _ = model.predict(obs_norm, deterministic=True)
                obs_raw, _, term, trunc, info = raw_env.step(action[0])
                raw_env.render()
                done = term or trunc
                time.sleep(1.0 / CONTROL_HZ)

            print(
                f"  tip_to_hover={info['tip_to_hover'] * 1000:.1f} mm"
                f"  cube_to_target_xy={info['cube_to_target_xy'] * 1000:.1f} mm"
                f"  yaw_error={np.degrees(info['yaw_error']):.1f} deg"
                f"  grasped={info['grasped']}  placed={info['placed']}"
                f"  success={info['success']}  collisions={info['n_collisions']}"
            )
            ep += 1
    except KeyboardInterrupt:
        pass
    finally:
        raw_env.close()
        norm_env.close()


if __name__ == "__main__":
    main()
