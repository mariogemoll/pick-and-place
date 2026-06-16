#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Run a YAML curriculum stage on ``PickPlaceEnv``.

Build-order step 4 of ``docs/rl-curriculum-roadmap.md``: the reproducible driver
for the curriculum. Each stage is the *same* :class:`PickPlaceEnv` with a
different reward weighting. By default a stage trains **from scratch**; to
finetune one chapter on top of an earlier one, pass ``--init-from`` the earlier
stage's run dir (or a checkpoint ``.zip``), which reloads its policy weights
**and** its VecNormalize stats (a fresh normaliser would reset
``obs_rms``/``ret_rms`` and re-converge under already-trained weights).

The spec format, deep-merge of ``defaults`` into stages, deterministic run dirs
and provenance all live in :mod:`pick_and_place.rl.curriculum`; this script owns
the Stable-Baselines3 training loop. Per stage it writes (in the stage run dir):

    checkpoints/           periodic model + VecNormalize snapshots
    best_model.zip         checkpoint with highest mean eval reward
    model_final.zip        final policy (what later stages --init-from)
    vec_normalize_final.pkl final normalisation stats (reloaded by --init-from)
    tensorboard/           TensorBoard event files
    provenance.json        config + git SHA + seed + versions + timestamp

Examples
--------
    python scripts/train_curriculum.py --curriculum curricula/pick_place.yaml \\
        --stages ch1_reach

    # Finetune the next chapter on top of the previous stage's run dir:
    python scripts/train_curriculum.py --curriculum curricula/pick_place.yaml \\
        --stages ch2_grasp --init-from out/rl_curriculum/pick_place/ch1_reach
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from pick_and_place.rl import curriculum as cur
from pick_and_place.rl.pick_place_env import DEFAULT_WEIGHTS, PickPlaceEnv

_DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "out" / "rl_curriculum"


class _SaveVecNormalizeOnBest(BaseCallback):
    """Save the training VecNormalize alongside best_model.zip on each new best."""

    def __init__(self, save_path: Path) -> None:
        super().__init__()
        self._save_path = save_path

    def _on_step(self) -> bool:
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(self._save_path / cur.BEST_VECNORMALIZE_NAME))
        return True


class _SyncNormStats(BaseCallback):
    """Copy obs/ret running stats from the training env to the eval env each rollout.

    Without this the eval normaliser diverges from training's and reported mean
    rewards are on a different scale.
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


def _validate_reward_weights(stage: cur.StageSpec) -> None:
    """Fail fast on a reward-weight key the env would silently ignore.

    ``weighted_reward`` treats an unknown weight as a no-op (missing-weight = 0),
    so a typo'd reward term would quietly train the wrong objective — exactly the
    kind of irreproducibility this runner exists to prevent.
    """
    weights = stage.env_kwargs.get("reward_weights", {})
    unknown = set(weights) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(
            f"Stage {stage.name!r} sets unknown reward weight(s) {sorted(unknown)}; "
            f"valid terms are {sorted(DEFAULT_WEIGHTS)}"
        )


def _make_train_env(stage: cur.StageSpec, vec_path: Path | None) -> VecNormalize:
    env = make_vec_env(
        PickPlaceEnv,
        n_envs=stage.n_envs,
        seed=stage.seed,
        env_kwargs=stage.env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )
    if vec_path is not None:
        if not vec_path.exists():
            raise FileNotFoundError(f"VecNormalize stats to resume not found at {vec_path}")
        print(f"  Resuming VecNormalize stats from {vec_path}")
        env = VecNormalize.load(str(vec_path), env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def _make_eval_env(stage: cur.StageSpec) -> VecNormalize:
    env = make_vec_env(
        PickPlaceEnv,
        n_envs=1,
        seed=stage.seed + 9999,
        env_kwargs=stage.env_kwargs,
        vec_env_cls=DummyVecEnv,
    )
    return VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)


def _build_model(
    stage: cur.StageSpec,
    train_env: VecNormalize,
    init_model: Path | None,
    tb_dir: Path,
    device: str,
) -> PPO:
    if init_model is not None:
        if not init_model.exists():
            raise FileNotFoundError(f"Model to resume not found at {init_model}")
        print(f"  Resuming policy weights from {init_model}")
        model = PPO.load(init_model, env=train_env, device=device)
        model.set_env(train_env)
        # PPO.load doesn't carry a tensorboard_log; point it at this stage's dir
        # so resumed stages log alongside from-scratch ones.
        model.tensorboard_log = str(tb_dir)
        return model

    ppo_kwargs = dict(stage.ppo)
    policy = ppo_kwargs.pop("policy", "MlpPolicy")
    return PPO(
        policy,
        train_env,
        seed=stage.seed,
        verbose=1,
        tensorboard_log=str(tb_dir),
        device=device,
        **ppo_kwargs,
    )


def _resolve_init(init_from: Path, vec_override: Path | None) -> tuple[Path, Path]:
    """Resolve ``--init-from`` to an explicit ``(model.zip, vec_normalize.pkl)`` pair.

    ``init_from`` may be a run directory (use its ``model_final.zip`` +
    ``vec_normalize_final.pkl``) or a model ``.zip``. For a model zip the matching
    VecNormalize snapshot is auto-located by name — ``model_final.zip`` pairs with
    ``vec_normalize_final.pkl``; a ``<prefix>_<N>_steps.zip`` checkpoint pairs with
    ``<prefix>_vecnormalize_<N>_steps.pkl`` — unless ``--init-vecnormalize`` is given.
    """
    if init_from.is_dir():
        return cur.final_model_path(init_from), cur.final_vecnormalize_path(init_from)

    model_path = init_from
    if vec_override is not None:
        return model_path, vec_override

    name = model_path.name
    if name == f"{cur.FINAL_MODEL_STEM}.zip":
        vec_name = cur.FINAL_VECNORMALIZE_NAME
    else:
        m = re.fullmatch(r"(.+)_(\d+)_steps\.zip", name)
        if not m:
            raise ValueError(
                f"Cannot infer the VecNormalize file for {model_path}; "
                f"pass --init-vecnormalize explicitly."
            )
        vec_name = f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl"
    return model_path, model_path.with_name(vec_name)


def run_stage(
    spec: cur.CurriculumSpec,
    stage: cur.StageSpec,
    runs_root: Path,
    device: str,
    overwrite: bool,
    init_from: tuple[Path, Path] | None = None,
) -> None:
    run_dir = cur.stage_run_dir(runs_root, spec.name, stage.name)
    final_model = cur.final_model_path(run_dir)

    if final_model.exists() and not overwrite:
        print(f"[{stage.name}] already trained ({final_model}); skipping (use --overwrite to redo)")
        return

    # With --init-from the stage finetunes the given checkpoint's weights +
    # VecNormalize stats; otherwise it trains from scratch. Either way the
    # timestep counter starts fresh (each stage is its own run).
    if init_from is not None:
        init_model, init_vec = init_from
        source_desc = f"--init-from {init_model}"
    else:
        init_model = init_vec = None
        source_desc = "(from scratch)"
    reset_num_timesteps = True

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Stage {stage.name} ===")
    print(f"  Run directory: {run_dir}")
    print(f"  Init from: {source_desc}")

    # Provenance first, so the run dir is self-describing even if training is
    # interrupted before the final model is written.
    provenance = cur.build_provenance(
        stage,
        spec,
        versions={"stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__},
    )
    cur.write_provenance(run_dir, provenance)

    train_env = _make_train_env(stage, init_vec)
    eval_env = _make_eval_env(stage)

    callbacks = [
        _SyncNormStats(eval_env),
        _SuccessRateCallback(log_freq=10_000),
        CheckpointCallback(
            save_freq=max(stage.checkpoint_freq // stage.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=stage.name,
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            callback_on_new_best=_SaveVecNormalizeOnBest(run_dir),
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=max(stage.eval_freq // stage.n_envs, 1),
            n_eval_episodes=stage.eval_episodes,
            verbose=1,
        ),
    ]

    model = _build_model(stage, train_env, init_model, run_dir / "tensorboard", device)
    model.learn(
        total_timesteps=stage.total_steps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True,
    )

    model.save(str(run_dir / cur.FINAL_MODEL_STEM))
    train_env.save(str(cur.final_vecnormalize_path(run_dir)))
    train_env.close()
    eval_env.close()
    print(f"  Saved final model + stats to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, required=True, help="curriculum YAML spec")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=_DEFAULT_RUNS_ROOT,
        help="root for run directories (default: py/out/rl_curriculum)",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="comma-separated subset of stage names to run (default: all, in order)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="retrain stages whose final model already exists"
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="checkpoint to finetune from (a run dir, or a model .zip) — reloads its "
        "policy weights and VecNormalize stats. The stage still supplies the reward "
        "function. Requires --stages to select exactly one stage.",
    )
    parser.add_argument(
        "--init-vecnormalize",
        type=Path,
        default=None,
        help="explicit VecNormalize .pkl for --init-from (default: inferred from the "
        "model filename)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="auto|cpu|cuda|mps. Defaults to cpu: the policy is a ~150k-param MLP, so a "
        "GPU's launch/transfer overhead makes the PPO update slower than CPU (benchmark "
        "it with scripts/bench_throughput.py). Override only if the env is ported to a "
        "GPU-parallel sim like MJX.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="override every selected stage's total_steps (e.g. for a smoke test). "
        "Combine with --runs-root to avoid clobbering real run artifacts.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="override every selected stage's n_envs (e.g. for a smoke test)",
    )
    args = parser.parse_args()

    spec = cur.load_curriculum(args.curriculum)
    print(f"Curriculum {spec.name!r}: {len(spec.stages)} stage(s) from {args.curriculum}")

    selected: set[str] | None = None
    if args.stages:
        selected = {s.strip() for s in args.stages.split(",") if s.strip()}
        known = {s.name for s in spec.stages}
        unknown = selected - known
        if unknown:
            parser.error(f"Unknown stage(s) {sorted(unknown)}; known: {sorted(known)}")

    init_from: tuple[Path, Path] | None = None
    if args.init_from is not None:
        if selected is None or len(selected) != 1:
            parser.error("--init-from requires --stages to select exactly one stage")
        init_from = _resolve_init(args.init_from, args.init_vecnormalize)
    elif args.init_vecnormalize is not None:
        parser.error("--init-vecnormalize is only valid together with --init-from")

    for stage in spec.stages:
        _validate_reward_weights(stage)

    for stage in spec.stages:
        if selected is not None and stage.name not in selected:
            continue
        if args.total_steps is not None:
            stage = dataclasses.replace(stage, total_steps=args.total_steps)
        if args.n_envs is not None:
            stage = dataclasses.replace(stage, n_envs=args.n_envs)
        run_stage(spec, stage, args.runs_root, args.device, args.overwrite, init_from=init_from)


if __name__ == "__main__":
    main()
