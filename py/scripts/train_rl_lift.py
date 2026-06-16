#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Train a PPO policy for the 'lift the cube' sub-task.

Trains :class:`pick_and_place.rl.lift_env.LiftCubeEnv` — the arm must reach the
cube, grasp it and lift it off the floor. The env shares the approach-to-hover
milestone's 23-dim observation and 6-dim delta-action layout, so the usual
starting point is a checkpoint trained by ``train_rl_hover``: pass it via
``--init-from`` to reload its policy weights *and* VecNormalize stats and finetune
on the lift reward. Without ``--init-from`` it trains from scratch.

Uses Stable-Baselines3 with observation/reward normalisation (VecNormalize) and
subprocess parallelism (SubprocVecEnv) for throughput. The eval env is kept
separate from the training env and its normalisation stats are synced each
rollout so reported mean rewards are on the same scale.

Outputs (in the run directory):
    checkpoints/           periodic model + VecNormalize snapshots
    best_model.zip         checkpoint with highest mean eval reward
    rl_lift_final.zip      final policy
    vec_normalize_final.pkl final normalisation stats
    tensorboard/           TensorBoard event files

The ``--reward`` mode selects the (unshaped) objective: ``lift`` (cube height,
the default) or ``move`` (cube displacement from its reset pose) — a denser
contact/bootstrap rung that a lift run can then warm-start from.

Examples
--------
    # From scratch:
    python scripts/train_rl_lift.py

    # Warm-started from a hover run (the common case):
    python scripts/train_rl_lift.py --init-from out/rl_hover/<run>

    # The move -> lift chain:
    python scripts/train_rl_lift.py --reward move --init-from out/rl_hover/<run> --out out/rl_lift/move
    python scripts/train_rl_lift.py --reward lift --init-from out/rl_lift/move --out out/rl_lift/lift
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

from pick_and_place.rl.lift_env import CONTROL_HZ, MAX_DELTA, MAX_STEPS, LiftCubeEnv

_RUNS_ROOT = Path(__file__).resolve().parents[1] / "out" / "rl_lift"


class _SaveVecNormalizeOnBest(BaseCallback):
    """Save the training VecNormalize alongside best_model.zip on each new best."""

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
    """Log lift success rate from episode info dicts to TensorBoard."""

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


def resolve_checkpoint(path: Path, vec_override: Path | None) -> tuple[Path, Path]:
    """Resolve a checkpoint to an explicit ``(model.zip, vec_normalize.pkl)`` pair.

    ``path`` may be a run directory (use its ``best_model.zip`` + matching
    ``best_vecnormalize.pkl``, else the ``*_final.zip`` + ``vec_normalize_final.pkl``)
    or a model ``.zip``. For a zip the sibling VecNormalize is auto-located by name
    for the three names these training scripts write:

    * ``<prefix>_<N>_steps.zip`` (CheckpointCallback) ->
      ``<prefix>_vecnormalize_<N>_steps.pkl``
    * ``best_model.zip`` (EvalCallback) -> ``best_vecnormalize.pkl``
    * ``rl_hover_final.zip`` / ``rl_lift_final.zip`` -> ``vec_normalize_final.pkl``

    Pass ``vec_override`` to point at the stats file explicitly.
    """
    if path.is_dir():
        best = path / "best_model.zip"
        if best.exists():
            return best, path / "best_vecnormalize.pkl"
        finals = sorted(path.glob("*_final.zip"))
        if finals:
            return finals[0], path / "vec_normalize_final.pkl"
        raise FileNotFoundError(f"No best_model.zip or *_final.zip found in {path}")

    if vec_override is not None:
        return path, vec_override

    candidates: list[Path] = []
    step_match = re.fullmatch(r"(.+)_(\d+)_steps", path.stem)
    if step_match:
        prefix, steps = step_match.groups()
        candidates.append(path.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl"))
    if path.stem == "best_model":
        candidates.append(path.with_name("best_vecnormalize.pkl"))
    if path.stem.endswith("_final"):
        candidates.append(path.with_name("vec_normalize_final.pkl"))

    for candidate in candidates:
        if candidate.exists():
            return path, candidate
    raise FileNotFoundError(
        f"Could not locate the VecNormalize stats for checkpoint {path}. "
        f"Looked for: {[str(c) for c in candidates] or 'no known naming match'}. "
        "Pass --init-vecnormalize to point at the .pkl explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="parallel training envs")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory for this run (default: out/rl_lift/<run-name>)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="run subdirectory under out/rl_lift (default: timestamp); ignored if --out is given",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm-start checkpoint (a run dir or a model .zip) — reloads its policy "
        "weights and VecNormalize stats, then finetunes on the lift reward. Usually a "
        "train_rl_hover run.",
    )
    parser.add_argument(
        "--init-vecnormalize",
        type=Path,
        default=None,
        help="VecNormalize .pkl for --init-from (default: auto-derived from the checkpoint)",
    )
    parser.add_argument(
        "--reward",
        choices=LiftCubeEnv.REWARD_MODES,
        default="lift",
        help="reward mode: 'lift' (cube height) or 'move' (cube displacement from "
        "reset pose, a denser contact/bootstrap rung)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy coefficient — the exploration vs exploitation knob. Higher "
        "keeps the action distribution wider (more exploration). Applied to both "
        "from-scratch and --init-from runs (overriding the checkpoint's value).",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=CONTROL_HZ,
        help="control decisions per second. Lower (e.g. 10) means each action covers "
        "more sim time, so exploration travels further per decision.",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=MAX_DELTA,
        help="largest per-decision joint change (rad). A sim-training exploration knob "
        "only — real-hardware safety is a deploy-time action down-scale, so do not keep "
        "this small 'for the real robot' during training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="episode horizon in control steps. Lower --control-hz with the default "
        "horizon makes episodes longer in wall-clock task time; shorten this to keep "
        "them ~a few seconds.",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    if args.out is not None:
        if args.run_name is not None:
            parser.error("pass either --out or --run-name, not both")
        out = args.out
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _RUNS_ROOT / run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out}")
    print(f"Reward mode: {args.reward}  |  ent_coef: {args.ent_coef}")
    print(
        f"Action regime: control_hz={args.control_hz:g}  max_delta={args.max_delta:g}  "
        f"max_steps={args.max_steps}"
    )

    init_model: Path | None = None
    init_vec: Path | None = None
    if args.init_from is not None:
        init_model, init_vec = resolve_checkpoint(args.init_from, args.init_vecnormalize)
        print(f"Warm-starting from {init_model}\n           stats from {init_vec}")
    elif args.init_vecnormalize is not None:
        parser.error("--init-vecnormalize is only valid together with --init-from")

    env_kwargs = {
        "reward_mode": args.reward,
        "control_hz": args.control_hz,
        "max_delta": args.max_delta,
        "max_steps": args.max_steps,
    }

    # --- training env ---
    train_env = make_vec_env(
        LiftCubeEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )
    if init_vec is not None:
        # Warm-starting finetunes trained weights, so the normaliser must carry the
        # parent's running stats forward — a fresh VecNormalize would reset
        # obs_rms/ret_rms to zero and re-converge under the loaded policy.
        train_env = VecNormalize.load(str(init_vec), train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- eval env (separate normaliser, synced from training) ---
    eval_env = make_vec_env(
        LiftCubeEnv,
        n_envs=1,
        seed=args.seed + 9999,
        env_kwargs=env_kwargs,
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
            name_prefix="rl_lift",
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

    if init_model is not None:
        # Warm-start: reuse the parent's policy weights but start this run's
        # timestep counter fresh (the lift reward is a different objective).
        model = PPO.load(init_model, env=train_env, device=args.device)
        model.set_env(train_env)
        model.tensorboard_log = str(out / "tensorboard")
        # PPO.load restores the parent's ent_coef, so override it explicitly —
        # otherwise the warm-started run silently inherits the hover checkpoint's
        # (low, exploitative) entropy on a new objective, which is exactly the
        # case where you want to dial exploration back up.
        model.ent_coef = args.ent_coef
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
            ent_coef=args.ent_coef,
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
        reset_num_timesteps=True,
        progress_bar=True,
    )

    model.save(str(out / "rl_lift_final"))
    train_env.save(str(out / "vec_normalize_final.pkl"))
    print(f"\nSaved final model and normalisation stats to {out}")


if __name__ == "__main__":
    main()
