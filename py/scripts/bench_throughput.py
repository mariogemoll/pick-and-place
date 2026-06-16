#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Profile where PPO time goes on PickPlaceEnv: env-step (MuJoCo, CPU) vs PPO update.

Answers "would a GPU help?" empirically. With a ~150k-param MLP the update is
tiny; this measures the env-step / update split and scans n_envs to find the
core sweet spot on this machine.

    python scripts/bench_throughput.py
    python scripts/bench_throughput.py --device cpu --n-envs 2,4,8,10,16
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from pick_and_place.rl.pick_place_env import PickPlaceEnv

# Match the curriculum's PPO/policy config so the numbers transfer.
_N_STEPS = 2048
_POLICY_KWARGS = dict(net_arch=[256, 256])
_PPO_KWARGS = dict(
    learning_rate=3.0e-4,
    n_steps=_N_STEPS,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


def _bench_env_only(n_envs: int, steps: int) -> float:
    """Steps/sec of pure env stepping (no policy), n_envs in parallel processes."""
    env = make_vec_env(PickPlaceEnv, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv)
    env.reset()
    acts = np.stack([env.action_space.sample() for _ in range(n_envs)])
    # Warmup (JIT-y caches, process spin-up).
    for _ in range(20):
        env.step(acts)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(acts)
    dt = time.perf_counter() - t0
    env.close()
    return n_envs * steps / dt


def _bench_ppo(n_envs: int, device: str) -> tuple[float, float, float]:
    """One full PPO iteration: returns (rollout_s, update_s, total_steps/s)."""
    env = make_vec_env(PickPlaceEnv, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", env, device=device, policy_kwargs=_POLICY_KWARGS, verbose=0, **_PPO_KWARGS
    )

    # Warmup one short learn so torch/allocator caches are hot.
    model.learn(total_timesteps=n_envs * 64)

    def _t(_locals, _globals):
        return True

    # Time a rollout (collect n_steps) then an update directly.
    model.policy.set_training_mode(False)
    model._last_obs = model.env.reset()
    model._last_episode_starts = np.ones((n_envs,), dtype=bool)

    t0 = time.perf_counter()
    model.collect_rollouts(
        model.env, callback=model._init_callback(_t), rollout_buffer=model.rollout_buffer,
        n_rollout_steps=_N_STEPS,
    )
    rollout_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.train()
    update_s = time.perf_counter() - t0

    env.close()
    steps_per_s = (n_envs * _N_STEPS) / (rollout_s + update_s)
    return rollout_s, update_s, steps_per_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", default="1,4,8,10,12", help="comma-separated env counts")
    parser.add_argument("--device", default="cpu", help="cpu|mps|cuda|auto for the PPO update")
    parser.add_argument("--env-steps", type=int, default=200, help="steps/worker for env-only bench")
    args = parser.parse_args()

    counts = [int(c) for c in args.n_envs.split(",") if c.strip()]
    print(f"cores: {os.cpu_count()}   device(update): {args.device}\n")

    print("== Pure env throughput (MuJoCo, no policy) ==")
    print(f"{'n_envs':>7} {'steps/s':>12} {'per-core':>12}")
    for n in counts:
        sps = _bench_env_only(n, args.env_steps)
        print(f"{n:>7} {sps:>12.0f} {sps / n:>12.0f}")

    print("\n== Full PPO iteration (rollout vs update) ==")
    print(f"{'n_envs':>7} {'rollout_s':>10} {'update_s':>10} {'update%':>8} {'steps/s':>10}")
    for n in counts:
        r, u, sps = _bench_ppo(n, args.device)
        pct = 100 * u / (r + u)
        print(f"{n:>7} {r:>10.2f} {u:>10.2f} {pct:>7.1f}% {sps:>10.0f}")


if __name__ == "__main__":
    main()
