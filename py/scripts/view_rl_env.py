#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Watch the reverse-curriculum env live in the MuJoCo viewer.

Resets the env at a curriculum stage and drives it forward, episode after
episode, in real time so the reset distribution and the rollouts are eyeballable.

Two action sources:

* ``scripted`` (default) replays the source episode's recorded commanded set
  points from the reset frame onward — a faithful demonstrator, so you see an
  actual pick-and-place finish from wherever the stage reset landed.
* ``random`` samples the action space, which mostly flails and collides — useful
  only to sanity-check that failures terminate.

Run with ``mjpython`` on macOS (the viewer needs the main thread):

    mjpython scripts/view_rl_env.py --stage 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from pick_and_place.rl import CURRICULUM_PHASES, EpisodePool, ReverseCurriculumEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "episodes",
        help="directory of recorded episode_*.npz (default: py/out/episodes)",
    )
    parser.add_argument("--stage", type=int, default=0, help="curriculum stage to view")
    parser.add_argument(
        "--mode",
        choices=("scripted", "random"),
        default="scripted",
        help="action source: replay the demo, or sample random actions",
    )
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    args = parser.parse_args()

    pool = EpisodePool(args.pool)
    env = ReverseCurriculumEnv(pool, stage=args.stage)
    print(
        f"stage {args.stage} (phase {CURRICULUM_PHASES[args.stage]!r}), "
        f"{args.mode} actions; close the viewer to stop."
    )

    # The env steps the sim by this many seconds per env-step; sleep that long so
    # playback runs at wall-clock speed.
    step_seconds = env._sim_steps * float(env.model.opt.timestep)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            _, info = env.reset(seed=args.seed + episode)
            commanded = (
                np.load(args.pool / info["source"], allow_pickle=True)["commanded"]
                if args.mode == "scripted"
                else None
            )
            frame = info["reset_frame"]
            viewer.sync()
            print(
                f"[ep {episode}] {info['source']} reset@{frame} "
                f"budget={info['max_steps']}"
            )

            terminated = truncated = False
            while viewer.is_running() and not (terminated or truncated):
                start = time.time()
                if commanded is not None:
                    frame += 1
                    action = commanded[min(frame, len(commanded) - 1)]
                else:
                    action = env.action_space.sample()
                _, _, terminated, truncated, step_info = env.step(action)
                viewer.sync()
                remaining = step_seconds - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)

            outcome = (
                "SUCCESS" if step_info["success"]
                else "collision" if step_info["collision"]
                else "out-of-bounds" if step_info["out_of_bounds"]
                else "timeout"
            )
            print(f"[ep {episode}] -> {outcome}")
            episode += 1


if __name__ == "__main__":
    main()
