#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train a PPO policy for the approach-to-hover sub-task.

DEPRECATED — reference / smoke test only. Trains the throwaway hover milestone
env (``rl.hover_env``), which is **not** on the frozen 31-dim RL contract and
**not** part of the curriculum. Kept only to smoke-test the SB3 loop; the
curriculum trains ``PickPlaceEnv`` via the (forthcoming) curriculum runner. See
``docs/rl-curriculum-roadmap.md``.

Uses Stable-Baselines3 with observation/reward normalisation (VecNormalize) and
subprocess parallelism (SubprocVecEnv) for throughput.  The eval env is kept
separate from the training env and its normalisation stats are synced each
rollout so that reported mean rewards are on the same scale.

Outputs (in --out directory):
    checkpoints/           periodic model + VecNormalize snapshots
    best_model.zip         checkpoint with highest mean eval reward
    vec_normalize.pkl      final normalisation stats
    tensorboard/           TensorBoard event files
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from pick_and_place.rl.hover_env import ApproachToHoverEnv

_RUNS_ROOT = Path(__file__).resolve().parents[1] / "out" / "rl_hover"


class _SaveVecNormalizeOnBest(BaseCallback):
    """Save the training VecNormalize alongside best_model.zip whenever EvalCallback finds a new best."""

    def __init__(self, save_path: Path) -> None:
        super().__init__()
        self._save_path = save_path

    def _on_step(self) -> bool:
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(self._save_path / "best_vecnormalize.pkl"))
        return True


class _SyncNormStats(BaseCallback):
    """Copy obs/ret running stats from the training env to the eval env each rollout.

    Without this the eval env's normaliser diverges from training's and the
    reported mean rewards are on a different scale.
    """

    def __init__(self, eval_env: VecNormalize) -> None:
        super().__init__()
        self._eval_env = eval_env

    def _on_rollout_end(self) -> None:
        self._eval_env.obs_rms = self.training_env.obs_rms
        self._eval_env.ret_rms = self.training_env.ret_rms

    def _on_step(self) -> bool:
        return True


class _SuccessRateCallback(BaseCallback):
    """Log success rate from episode info dicts to TensorBoard."""

    def __init__(self, log_freq: int = 10_000) -> None:
        super().__init__()
        self._log_freq = log_freq
        self._successes: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "success" in info:
                self._successes.append(info["success"])
        if self.num_timesteps % self._log_freq == 0 and self._successes:
            rate = float(np.mean(self._successes[-1000:]))
            self.logger.record("rollout/success_rate", rate)
            self.logger.dump(self.num_timesteps)
        return True


def resolve_vecnormalize_path(resume: Path) -> Path:
    """Find the VecNormalize .pkl saved alongside a resume checkpoint.

    Resuming must reload the *parent's* running obs/reward stats; building a
    fresh ``VecNormalize`` instead resets ``obs_rms``/``ret_rms`` to zero and
    forces them to re-converge on already-trained weights — a faithfulness bug.

    Maps a model zip back to its sibling stats file for the three names this
    script writes:

    * ``<prefix>_<N>_steps.zip`` (CheckpointCallback) ->
      ``<prefix>_vecnormalize_<N>_steps.pkl``
    * ``best_model.zip`` (EvalCallback) -> ``best_vecnormalize.pkl``
    * ``rl_hover_final.zip`` -> ``vec_normalize_final.pkl``

    Returns the first candidate that exists; raises ``FileNotFoundError`` with
    a hint to pass ``--resume-vecnormalize`` explicitly otherwise.
    """
    candidates: list[Path] = []
    step_match = re.fullmatch(r"(.+)_(\d+)_steps", resume.stem)
    if step_match:
        prefix, steps = step_match.groups()
        candidates.append(resume.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl"))
    if resume.stem == "best_model":
        candidates.append(resume.with_name("best_vecnormalize.pkl"))
    if resume.stem == "rl_hover_final":
        candidates.append(resume.with_name("vec_normalize_final.pkl"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the VecNormalize stats for resume checkpoint {resume}. "
        f"Looked for: {[str(c) for c in candidates] or 'no known naming match'}. "
        "Pass --resume-vecnormalize to point at the parent's .pkl explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="parallel training envs")
    parser.add_argument("--run-name", default=None, help="run subdirectory name (default: timestamp)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None, help="resume from checkpoint zip")
    parser.add_argument(
        "--resume-vecnormalize",
        type=Path,
        default=None,
        help="VecNormalize .pkl to resume stats from (default: auto-derived from --resume)",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _RUNS_ROOT / run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out}")

    # --- training env ---
    train_env = make_vec_env(
        ApproachToHoverEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    if args.resume:
        # Resuming finetunes trained weights, so the normaliser must carry the
        # parent's running stats forward — a fresh VecNormalize would reset
        # obs_rms/ret_rms to zero and re-converge under the loaded policy.
        vec_path = args.resume_vecnormalize or resolve_vecnormalize_path(args.resume)
        print(f"Loading VecNormalize stats from {vec_path}")
        train_env = VecNormalize.load(str(vec_path), train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- eval env (separate normaliser, synced from training) ---
    eval_env = make_vec_env(
        ApproachToHoverEnv,
        n_envs=1,
        seed=args.seed + 9999,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )

    callbacks = [
        _SyncNormStats(eval_env),
        _SuccessRateCallback(log_freq=10_000),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=str(out / "checkpoints"),
            name_prefix="rl_hover",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            callback_on_new_best=_SaveVecNormalizeOnBest(out),
            best_model_save_path=str(out),
            log_path=str(out / "eval"),
            eval_freq=max(20_000 // args.n_envs, 1),
            n_eval_episodes=20,
            verbose=1,
        ),
    ]

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, device=args.device)
        model.set_env(train_env)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256]},
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(out / "tensorboard"),
            device=args.device,
        )

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=args.resume is None,
        progress_bar=True,
    )

    model.save(str(out / "rl_hover_final"))
    train_env.save(str(out / "vec_normalize_final.pkl"))
    print(f"\nSaved final model and normalisation stats to {out}")


if __name__ == "__main__":
    main()
