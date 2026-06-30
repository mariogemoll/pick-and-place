#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train one reverse-curriculum stage with Stable-Baselines3 PPO.

This is the stage-0 plumbing smoke test from RL_REVERSE_CURRICULUM.txt, but the
script accepts any stage so later curriculum work can reuse the same training
entry point. It saves the policy and VecNormalize state together; resume/eval
must load both files because the policy was trained against normalized
observations. Exact bit-for-bit resume reproducibility is not guaranteed once
VecNormalize statistics continue updating, but the saved pair is the right
operational checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pick_and_place.rl import CURRICULUM_PHASES, ReverseCurriculumEnv


DEFAULT_TIMESTEPS = 100_000


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize statistics alongside periodic model checkpoints."""

    def __init__(self, save_freq: int, save_path: Path) -> None:
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.training_env.save(str(self.save_path))
        return True


def _make_env(pool: Path, stage: int, phase_fraction: float, seed: int, rank: int):
    def factory():
        env = ReverseCurriculumEnv(pool, stage=stage, phase_fraction=phase_fraction)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return factory


def _make_vec_env(
    pool: Path,
    *,
    stage: int,
    phase_fraction: float,
    seed: int,
    n_envs: int,
) -> VecNormalize:
    env = DummyVecEnv(
        [_make_env(pool, stage, phase_fraction, seed, rank) for rank in range(n_envs)]
    )
    return VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)


def _load_or_create_model(
    args: argparse.Namespace,
    env: VecNormalize,
    model_path: Path,
    vecnormalize_path: Path,
) -> PPO:
    if not args.resume:
        return PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(args.out_dir / "tensorboard"),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
        )

    if not model_path.exists() or not vecnormalize_path.exists():
        raise FileNotFoundError(
            f"--resume requested, but missing {model_path.name} or {vecnormalize_path.name}"
        )
    print(f"resuming from {model_path}")
    return PPO.load(str(model_path), env=env, seed=args.seed, print_system_info=True)


def _evaluate(
    model: PPO,
    vecnormalize_path: Path,
    pool: Path,
    *,
    stage: int,
    phase_fraction: float,
    seed: int,
    episodes: int,
) -> tuple[int, float]:
    env = DummyVecEnv([_make_env(pool, stage, phase_fraction, seed, 10_000)])
    env = VecNormalize.load(str(vecnormalize_path), env)
    env.training = False
    env.norm_reward = False

    successes = 0
    returns: list[float] = []
    for episode in range(episodes):
        obs = env.reset()
        done = np.array([False])
        total = 0.0
        last_info = {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total += float(reward[0])
            last_info = infos[0]
        successes += bool(last_info.get("success", False))
        returns.append(total)
        print(
            f"eval {episode:03d}: return={total:.1f} "
            f"{'SUCCESS' if last_info.get('success') else 'miss'}"
        )
    env.close()
    return successes, float(np.mean(returns)) if returns else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "episodes",
        help="directory of recorded episode_*.npz (default: py/out/episodes)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "rl" / "stage0",
        help="checkpoint directory (default: py/out/rl/stage0)",
    )
    parser.add_argument("--stage", type=int, default=0, help="curriculum stage to train")
    parser.add_argument(
        "--phase-fraction",
        type=float,
        default=0.0,
        help="start this fraction into the selected phase (escape hatch; default 0)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"PPO environment steps to train (default {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument("--n-envs", type=int, default=1, help="parallel DummyVecEnv envs")
    parser.add_argument("--seed", type=int, default=0, help="training RNG seed")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument("--gamma", type=float, default=0.995, help="PPO discount factor")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="save checkpoint every this many callback steps (default 10000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="deterministic eval episodes after training (default 20)",
    )
    parser.add_argument("--resume", action="store_true", help="resume from latest.zip")
    args = parser.parse_args()

    if not 0 <= args.stage < len(CURRICULUM_PHASES):
        parser.error(f"--stage must be in 0..{len(CURRICULUM_PHASES) - 1}")
    if args.n_envs < 1:
        parser.error("--n-envs must be at least 1")
    if not 0.0 <= args.phase_fraction < 1.0:
        parser.error("--phase-fraction must be in [0, 1)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "latest.zip"
    vecnormalize_path = args.out_dir / "vecnormalize.pkl"

    env = _make_vec_env(
        args.pool,
        stage=args.stage,
        phase_fraction=args.phase_fraction,
        seed=args.seed,
        n_envs=args.n_envs,
    )
    if args.resume:
        env = VecNormalize.load(str(vecnormalize_path), env.venv)
        env.training = True
        env.norm_reward = False

    phase = CURRICULUM_PHASES[args.stage]
    print(
        f"training stage {args.stage} ({phase!r}) for {args.timesteps} steps "
        f"with {args.n_envs} env(s)"
    )
    model = _load_or_create_model(args, env, model_path, vecnormalize_path)
    callbacks: list[BaseCallback] = []
    if args.checkpoint_freq > 0:
        callbacks.extend(
            [
                CheckpointCallback(
                    save_freq=args.checkpoint_freq,
                    save_path=str(args.out_dir / "checkpoints"),
                    name_prefix=f"stage{args.stage}_ppo",
                    save_replay_buffer=False,
                    save_vecnormalize=True,
                ),
                SaveVecNormalizeCallback(args.checkpoint_freq, vecnormalize_path),
            ]
        )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
        tb_log_name=f"stage{args.stage}",
    )

    model.save(str(model_path))
    env.save(str(vecnormalize_path))
    env.close()
    print(f"saved {model_path}")
    print(f"saved {vecnormalize_path}")

    if args.eval_episodes > 0:
        successes, mean_return = _evaluate(
            model,
            vecnormalize_path,
            args.pool,
            stage=args.stage,
            phase_fraction=args.phase_fraction,
            seed=args.seed + 100_000,
            episodes=args.eval_episodes,
        )
        print(
            f"\neval: {successes}/{args.eval_episodes} successes, "
            f"mean return {mean_return:.3f}"
        )


if __name__ == "__main__":
    main()
