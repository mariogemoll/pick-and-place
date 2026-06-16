# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Declarative curriculum spec + provenance — build-order step 4.

The pick-place policy is trained as a sequence of chapters
(position -> orientation -> regrasp -> robustness, see
``docs/rl-curriculum-roadmap.md``). Every chapter is the *same*
:class:`~pick_and_place.rl.pick_place_env.PickPlaceEnv` with a different reward
weighting. Each stage is its own training run; to finetune one chapter on top of
another, the operator points ``scripts/train_curriculum.py --init-from`` at the
earlier stage's run dir.

This module is the **declarative** half of the runner: it parses a YAML
curriculum spec into resolved :class:`StageSpec` records and owns the
*reproducibility* concerns — deterministic stage-named run directories and the
git-SHA / config / seed provenance written next to every run. It is deliberately
free of Stable-Baselines3 / Gymnasium / MuJoCo imports so the spec logic is
unit-testable without the training stack; the heavy SB3 loop lives in
``scripts/train_curriculum.py``.

Spec format
-----------
A curriculum is one YAML document::

    name: pick_place
    defaults:                 # merged into (and overridden by) every stage
      seed: 0
      n_envs: 8
      total_steps: 2000000
      ppo: {learning_rate: 3.0e-4, ...}
      env_kwargs:
        reward_weights: {reach: 1.0, ..., yaw: 0.0}
    stages:
      - name: ch1_position
      - name: ch2_orientation
        env_kwargs:
          reward_weights: {yaw: 2.0}   # a chapter is a weight *delta*

``defaults`` and each stage are **deep-merged** (nested dicts like ``ppo`` and
``env_kwargs.reward_weights`` merge key-by-key), so a stage expresses only the
deltas from the defaults — mirroring the roadmap's "a chapter is a weight
change, not a fork".
"""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# The repo root, derived from this file's location (…/py/src/pick_and_place/rl/).
# Used as the default directory for git provenance queries.
REPO_ROOT = Path(__file__).resolve().parents[4]

# Built-in fallbacks so a minimal spec trains something sane; a spec's
# ``defaults`` (and then each stage) override these by deep-merge.
BUILTIN_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "n_envs": 8,
    "total_steps": 2_000_000,
    "eval_episodes": 20,
    "eval_freq": 20_000,
    "checkpoint_freq": 50_000,
    "ppo": {},
    "env_kwargs": {},
}

# Deterministic artifact names written into each stage's run dir. ``--init-from``
# resolves a stage's resume source purely from a run dir + these names — no
# globbing, no timestamp guessing (unlike the legacy hover trainer's regex
# resolver).
FINAL_MODEL_STEM = "model_final"
FINAL_VECNORMALIZE_NAME = "vec_normalize_final.pkl"
BEST_MODEL_NAME = "best_model.zip"
BEST_VECNORMALIZE_NAME = "best_vecnormalize.pkl"
PROVENANCE_NAME = "provenance.json"


# ----------------------------------------------------------------------------
# Resolved spec
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """One fully-resolved curriculum stage (defaults already merged in).

    ``ppo`` is the dict of :class:`stable_baselines3.PPO` constructor kwargs and
    ``env_kwargs`` the :class:`PickPlaceEnv` constructor kwargs (typically just
    ``reward_weights``).
    """

    name: str
    seed: int
    n_envs: int
    total_steps: int
    eval_episodes: int
    eval_freq: int
    checkpoint_freq: int
    ppo: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumSpec:
    """A parsed curriculum: a name and an ordered chain of resolved stages."""

    name: str
    stages: tuple[StageSpec, ...]
    source_path: Path | None = None

    def stage(self, name: str) -> StageSpec:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"No stage named {name!r} in curriculum {self.name!r}")


# ----------------------------------------------------------------------------
# Parsing / merging
# ----------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` into a fresh dict.

    Nested dicts merge key-by-key (so a stage can override one PPO hyperparam or
    one reward weight without restating the rest); any non-dict value, or a
    dict replacing a non-dict (and vice versa), is taken wholesale from
    ``override``. Neither argument is mutated.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_stage(merged: dict[str, Any], name: str) -> StageSpec:
    """Build a :class:`StageSpec` from a fully-merged config dict."""
    known = {
        "seed",
        "n_envs",
        "total_steps",
        "eval_episodes",
        "eval_freq",
        "checkpoint_freq",
        "ppo",
        "env_kwargs",
    }
    unknown = set(merged) - known
    if unknown:
        raise ValueError(
            f"Stage {name!r} has unknown config key(s) {sorted(unknown)}; "
            f"expected a subset of {sorted(known)}"
        )
    return StageSpec(
        name=name,
        seed=int(merged["seed"]),
        n_envs=int(merged["n_envs"]),
        total_steps=int(merged["total_steps"]),
        eval_episodes=int(merged["eval_episodes"]),
        eval_freq=int(merged["eval_freq"]),
        checkpoint_freq=int(merged["checkpoint_freq"]),
        ppo=dict(merged["ppo"]),
        env_kwargs=dict(merged["env_kwargs"]),
    )


def parse_curriculum(doc: dict[str, Any], source_path: Path | None = None) -> CurriculumSpec:
    """Validate and resolve a parsed-YAML curriculum document.

    Raises :class:`ValueError` on a malformed spec: a missing/blank curriculum
    name, an empty or non-list ``stages``, a stage without a name, or a
    duplicate stage name.
    """
    if not isinstance(doc, dict):
        raise ValueError("Curriculum spec must be a YAML mapping at the top level")

    name = doc.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Curriculum spec needs a non-empty 'name'")

    defaults = doc.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping if present")

    raw_stages = doc.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("Curriculum spec needs a non-empty 'stages' list")

    base = deep_merge(BUILTIN_DEFAULTS, defaults)

    stages: list[StageSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise ValueError(f"Stage #{index} must be a mapping")
        stage_name = raw.get("name")
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError(f"Stage #{index} needs a non-empty 'name'")
        if stage_name in seen:
            raise ValueError(f"Duplicate stage name {stage_name!r}")

        overrides = {k: v for k, v in raw.items() if k != "name"}
        merged = deep_merge(base, overrides)
        stages.append(_resolve_stage(merged, stage_name))
        seen.add(stage_name)

    return CurriculumSpec(name=name, stages=tuple(stages), source_path=source_path)


def load_curriculum(path: str | Path) -> CurriculumSpec:
    """Load and resolve a curriculum spec from a YAML file."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    return parse_curriculum(doc, source_path=path)


# ----------------------------------------------------------------------------
# Run directories / artifact resolution
# ----------------------------------------------------------------------------


def stage_run_dir(runs_root: str | Path, curriculum_name: str, stage_name: str) -> Path:
    """Deterministic, stage-named run dir: ``<root>/<curriculum>/<stage>``.

    Deterministic (no timestamp) so ``--init-from`` can find a stage's artifacts
    by name and a re-run lands in the same place; provenance records the git SHA
    / seed that produced each run.
    """
    return Path(runs_root) / curriculum_name / stage_name


def final_model_path(run_dir: str | Path) -> Path:
    """Path to a stage's final policy zip (``model_final.zip``)."""
    return Path(run_dir) / f"{FINAL_MODEL_STEM}.zip"


def final_vecnormalize_path(run_dir: str | Path) -> Path:
    """Path to a stage's final VecNormalize stats (``vec_normalize_final.pkl``)."""
    return Path(run_dir) / FINAL_VECNORMALIZE_NAME


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------


def git_provenance(repo_dir: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit SHA and a working-tree-dirty flag for ``repo_dir``.

    Returns ``{"sha": None, "dirty": None}`` when git is unavailable or the
    directory is not a repo, so provenance writing never fails the run.
    """

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def build_provenance(
    stage: StageSpec,
    curriculum: CurriculumSpec,
    git: dict[str, Any] | None = None,
    versions: dict[str, str] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Assemble the provenance record written next to a stage's run.

    Captures the roadmap's required trio — **config + git SHA + seed** — plus the
    curriculum/stage identity, library versions and a UTC timestamp, so a run dir
    is self-describing and re-runnable.
    """
    return {
        "curriculum": curriculum.name,
        "curriculum_path": str(curriculum.source_path) if curriculum.source_path else None,
        "stage": stage.name,
        "seed": stage.seed,
        "git": git if git is not None else git_provenance(),
        "versions": versions or {},
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_envs": stage.n_envs,
            "total_steps": stage.total_steps,
            "eval_episodes": stage.eval_episodes,
            "eval_freq": stage.eval_freq,
            "checkpoint_freq": stage.checkpoint_freq,
            "ppo": stage.ppo,
            "env_kwargs": stage.env_kwargs,
        },
    }


def write_provenance(run_dir: str | Path, provenance: dict[str, Any]) -> Path:
    """Write the provenance record as pretty JSON into ``run_dir``."""
    path = Path(run_dir) / PROVENANCE_NAME
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return path
