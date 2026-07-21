# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Policy-agnostic scenario, oracle, result, and artifact primitives.

This module deliberately has no policy or MuJoCo dependency. The visual policy
environment supplies ground-truth :class:`TaskState` values to the oracle while
passing only camera images and proprioception to its controller.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from pick_and_place.geometry import CUBE_HALF_SIZE

SCENARIO_MANIFEST_VERSION = 1


def _number_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only JSON values") from exc


@dataclass(frozen=True)
class EvaluationScenario:
    """One fully materialized simulator reset in hardware-frame units."""

    scenario_id: str
    group: str
    workspace_region: str
    seed: int
    source_position_m: tuple[float, float, float]
    source_orientation_wxyz: tuple[float, float, float, float]
    target_position_m: tuple[float, float, float]
    initial_robot_state_real: tuple[float, float, float, float, float, float]
    domain_randomization_preset: str | None
    domain_randomization_sample: dict[str, Any]
    miscalibration_sample: dict[str, Any]
    control_hz: float
    max_steps: int

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be nonempty")
        if not self.group or not self.workspace_region:
            raise ValueError("group and workspace_region must be nonempty")
        if self.control_hz <= 0.0 or not math.isfinite(self.control_hz):
            raise ValueError("control_hz must be a positive finite number")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        quaternion_norm = math.sqrt(sum(value * value for value in self.source_orientation_wxyz))
        if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-6):
            raise ValueError("source_orientation_wxyz must be a unit quaternion")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EvaluationScenario":
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "invalid scenario fields; "
                f"missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)}"
            )
        preset = payload["domain_randomization_preset"]
        if preset is not None and not isinstance(preset, str):
            raise ValueError("domain_randomization_preset must be a string or null")
        return cls(
            scenario_id=str(payload["scenario_id"]),
            group=str(payload["group"]),
            workspace_region=str(payload["workspace_region"]),
            seed=int(payload["seed"]),
            source_position_m=_number_tuple(payload["source_position_m"], 3, "source_position_m"),
            source_orientation_wxyz=_number_tuple(
                payload["source_orientation_wxyz"], 4, "source_orientation_wxyz"
            ),
            target_position_m=_number_tuple(payload["target_position_m"], 3, "target_position_m"),
            initial_robot_state_real=_number_tuple(
                payload["initial_robot_state_real"], 6, "initial_robot_state_real"
            ),
            domain_randomization_preset=preset,
            domain_randomization_sample=_json_mapping(
                payload["domain_randomization_sample"], "domain_randomization_sample"
            ),
            miscalibration_sample=_json_mapping(
                payload["miscalibration_sample"], "miscalibration_sample"
            ),
            control_hz=float(payload["control_hz"]),
            max_steps=int(payload["max_steps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioManifest:
    schema_version: int
    suite: str
    scenarios: tuple[EvaluationScenario, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCENARIO_MANIFEST_VERSION:
            raise ValueError(
                f"unsupported scenario manifest version {self.schema_version}; "
                f"expected {SCENARIO_MANIFEST_VERSION}"
            )
        if not self.suite:
            raise ValueError("suite must be nonempty")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if not scenario_ids:
            raise ValueError("a scenario manifest must contain at least one scenario")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_id values must be unique within a manifest")

    @classmethod
    def load(cls, path: Path | str) -> "ScenarioManifest":
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "suite",
            "scenarios",
        }:
            raise ValueError("manifest must contain exactly schema_version, suite, and scenarios")
        raw_scenarios = payload["scenarios"]
        if not isinstance(raw_scenarios, list):
            raise ValueError("scenarios must be an array")
        return cls(
            schema_version=int(payload["schema_version"]),
            suite=str(payload["suite"]),
            scenarios=tuple(EvaluationScenario.from_dict(item) for item in raw_scenarios),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True)
class TaskOracleConfig:
    """Centralized full-task thresholds, in meters, seconds, and radians."""

    success_xy_tolerance_m: float = 0.04
    resting_height_m: float = CUBE_HALF_SIZE
    resting_height_tolerance_m: float = 0.01
    settled_linear_speed_m_s: float = 0.02
    settled_angular_speed_rad_s: float = 0.2
    success_confirmation_s: float = 0.5
    lift_clearance_m: float = 0.02
    stable_carry_confirmation_s: float = 0.2
    target_while_held_xy_tolerance_m: float = 0.06


@dataclass(frozen=True)
class TaskState:
    """Privileged simulator facts consumed by the oracle, never by a policy."""

    cube_position_m: tuple[float, float, float]
    cube_linear_velocity_m_s: tuple[float, float, float]
    cube_angular_velocity_rad_s: tuple[float, float, float]
    target_xy_m: tuple[float, float]
    robot_cube_contact: bool = False
    grasped: bool = False
    gripper_open: bool = False
    unexpected_collision: bool = False
    out_of_bounds: bool = False

    @property
    def xy_error_m(self) -> float:
        return math.hypot(
            self.cube_position_m[0] - self.target_xy_m[0],
            self.cube_position_m[1] - self.target_xy_m[1],
        )


@dataclass(frozen=True)
class TaskMilestones:
    pickup_contact_attempted: bool = False
    cube_lifted: bool = False
    stable_carry: bool = False
    target_reached_while_holding: bool = False
    cube_released: bool = False
    cube_settled: bool = False
    successful_placement: bool = False


@dataclass(frozen=True)
class FailureFlags:
    missed_pickup: bool = False
    unstable_or_lost_grasp: bool = False
    early_release: bool = False
    off_target_placement: bool = False
    unexpected_collision: bool = False
    cube_out_of_bounds: bool = False
    timeout: bool = False


class TaskSuccessOracle:
    """Stateful settled-placement oracle and milestone tracker."""

    def __init__(self, config: TaskOracleConfig | None = None) -> None:
        self.config = config or TaskOracleConfig()
        self.reset()

    def reset(self) -> None:
        self._milestones = TaskMilestones()
        self._collision = False
        self._out_of_bounds = False
        self._early_release = False
        self._lost_grasp = False
        self._was_grasped = False
        self._success_dwell_s = 0.0
        self._stable_carry_dwell_s = 0.0
        self._success_time_s: float | None = None
        self._elapsed_s = 0.0

    @property
    def milestones(self) -> TaskMilestones:
        return self._milestones

    @property
    def success(self) -> bool:
        return self._milestones.successful_placement

    @property
    def success_time_s(self) -> float | None:
        return self._success_time_s

    def _settled(self, state: TaskState) -> bool:
        config = self.config
        at_resting_height = (
            abs(state.cube_position_m[2] - config.resting_height_m)
            <= config.resting_height_tolerance_m
        )
        linear_speed = math.sqrt(sum(value * value for value in state.cube_linear_velocity_m_s))
        angular_speed = math.sqrt(sum(value * value for value in state.cube_angular_velocity_rad_s))
        return (
            at_resting_height
            and linear_speed < config.settled_linear_speed_m_s
            and angular_speed < config.settled_angular_speed_rad_s
        )

    def update(self, state: TaskState, step_duration_s: float) -> bool:
        if step_duration_s <= 0.0 or not math.isfinite(step_duration_s):
            raise ValueError("step_duration_s must be a positive finite number")
        if self.success:
            return True

        self._elapsed_s += step_duration_s
        self._collision |= state.unexpected_collision
        self._out_of_bounds |= state.out_of_bounds
        settled = self._settled(state)
        lifted = state.cube_position_m[2] >= (
            self.config.resting_height_m + self.config.lift_clearance_m
        )

        if state.grasped and lifted:
            self._stable_carry_dwell_s += step_duration_s
        else:
            self._stable_carry_dwell_s = 0.0

        released_now = self._was_grasped and not state.grasped
        if released_now:
            if not state.gripper_open:
                self._lost_grasp = True
            if not self._milestones.target_reached_while_holding:
                self._early_release = True

        target_reached = (
            state.grasped
            and state.xy_error_m <= self.config.target_while_held_xy_tolerance_m
        )
        lifted_during_episode = self._milestones.cube_lifted or lifted
        settled_after_lift = settled and lifted_during_episode
        placement_candidate = (
            settled
            and state.xy_error_m <= self.config.success_xy_tolerance_m
            and not self._collision
            and not self._out_of_bounds
        )
        self._success_dwell_s = (
            self._success_dwell_s + step_duration_s if placement_candidate else 0.0
        )
        successful = self._success_dwell_s >= self.config.success_confirmation_s

        self._milestones = TaskMilestones(
            pickup_contact_attempted=(
                self._milestones.pickup_contact_attempted or state.robot_cube_contact
            ),
            cube_lifted=lifted_during_episode,
            stable_carry=(
                self._milestones.stable_carry
                or self._stable_carry_dwell_s >= self.config.stable_carry_confirmation_s
            ),
            target_reached_while_holding=(
                self._milestones.target_reached_while_holding or target_reached
            ),
            cube_released=self._milestones.cube_released or released_now,
            cube_settled=self._milestones.cube_settled or settled_after_lift,
            successful_placement=successful,
        )
        self._was_grasped = state.grasped
        if successful:
            self._success_time_s = self._elapsed_s
        return successful

    def failure_flags(self, *, timed_out: bool) -> FailureFlags:
        return FailureFlags(
            missed_pickup=not self._milestones.pickup_contact_attempted,
            unstable_or_lost_grasp=self._lost_grasp,
            early_release=self._early_release,
            off_target_placement=(
                self._milestones.cube_settled and not self._milestones.successful_placement
            ),
            unexpected_collision=self._collision,
            cube_out_of_bounds=self._out_of_bounds,
            timeout=timed_out and not self._milestones.successful_placement,
        )


@dataclass(frozen=True)
class EpisodeResult:
    scenario_id: str
    group: str
    workspace_region: str
    success: bool
    milestones: TaskMilestones
    failures: FailureFlags
    final_xy_error_m: float
    control_steps: int
    simulated_time_s: float
    time_to_success_s: float | None
    controller_failure: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return [lower, upper]


def _rate(results: Sequence[EpisodeResult], attribute: str, nested: str) -> float:
    if not results:
        return 0.0
    return sum(bool(getattr(getattr(result, nested), attribute)) for result in results) / len(results)


def _aggregate(results: Sequence[EpisodeResult], *, include_strata: bool) -> dict[str, Any]:
    total = len(results)
    successes = sum(result.success for result in results)
    errors = [result.final_xy_error_m for result in results]
    success_times = [
        result.time_to_success_s for result in results if result.time_to_success_s is not None
    ]
    controller_failure_codes = [
        result.controller_failure["code"]
        for result in results
        if result.controller_failure is not None
    ]
    summary: dict[str, Any] = {
        "episode_count": total,
        "success_count": successes,
        "success_rate": successes / total if total else 0.0,
        "success_rate_95ci_wilson": _wilson_interval(successes, total),
        "milestone_rates": {
            field.name: _rate(results, field.name, "milestones") for field in fields(TaskMilestones)
        },
        "failure_rates": {
            field.name: _rate(results, field.name, "failures") for field in fields(FailureFlags)
        },
        "controller_failures": {
            "count": len(controller_failure_codes),
            "rate": len(controller_failure_codes) / total if total else 0.0,
            "by_code": {
                code: controller_failure_codes.count(code)
                for code in sorted(set(controller_failure_codes))
            },
        },
        "final_xy_error_m": {
            "median": median(errors) if errors else None,
            "p90": float(np.percentile(errors, 90)) if errors else None,
            "p95": float(np.percentile(errors, 95)) if errors else None,
        },
        "control_steps_median": median(result.control_steps for result in results) if results else None,
        "time_to_success_s_median": median(success_times) if success_times else None,
    }
    if include_strata:
        summary["by_group"] = {
            value: _aggregate([result for result in results if result.group == value], include_strata=False)
            for value in sorted({result.group for result in results})
        }
        summary["by_workspace_region"] = {
            value: _aggregate(
                [result for result in results if result.workspace_region == value],
                include_strata=False,
            )
            for value in sorted({result.workspace_region for result in results})
        }
    return summary


def aggregate_episode_results(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    return _aggregate(results, include_strata=True)


def write_evaluation_artifacts(
    output_dir: Path | str,
    run: Mapping[str, Any],
    results: Sequence[EpisodeResult],
) -> dict[str, Any]:
    """Create a new self-contained evaluation result directory."""
    output_path = Path(output_dir)
    if output_path.exists():
        unexpected = {path.name for path in output_path.iterdir()} - {"videos"}
        if unexpected:
            raise FileExistsError(f"evaluation output already contains {sorted(unexpected)}")
    else:
        output_path.mkdir(parents=True)
    run_payload = _json_mapping(dict(run), "run")
    summary = aggregate_episode_results(results)
    (output_path / "run.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n")
    with (output_path / "episodes.jsonl").open("w") as stream:
        for result in results:
            stream.write(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _update_hash_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)


def fingerprint_checkpoint(checkpoint: str) -> dict[str, Any]:
    """Hash a local checkpoint's contents, or its unresolved repository ID."""
    path = Path(checkpoint).expanduser()
    if path.is_file():
        digest = hashlib.sha256()
        _update_hash_from_file(digest, path)
        return {"kind": "file", "sha256": digest.hexdigest(), "file_count": 1}
    if path.is_dir():
        digest = hashlib.sha256()
        checkpoint_files = sorted(item for item in path.rglob("*") if item.is_file())
        for item in checkpoint_files:
            relative = item.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            _update_hash_from_file(digest, item)
        return {
            "kind": "directory",
            "sha256": digest.hexdigest(),
            "file_count": len(checkpoint_files),
        }
    digest = hashlib.sha256(checkpoint.encode()).hexdigest()
    return {
        "kind": "repository_id",
        "sha256": digest,
        "warning": "fingerprint covers the identifier, not downloaded checkpoint contents",
    }


def git_provenance(repository_root: Path | str) -> dict[str, Any]:
    """Return the current Git revision and dirty status without mutating Git."""

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def package_versions(package_names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
