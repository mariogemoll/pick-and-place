# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reconstruct simulator resets from recorded LeRobot episodes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.physics import NOMINAL
from pick_and_place.scripted.scenario_sampling import workspace_region
from pick_and_place.policies.policy_evaluation import EvaluationScenario
from pick_and_place.spec.workspace import CUBE_HALF_SIZE

STATE_FEATURE = "observation.state"
ENVIRONMENT_FEATURE = "observation.environment_state"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def recorded_episode_scenario(
    dataset_root: str | Path,
    episode_index: int,
    *,
    control_hz: float,
    max_steps: int | None = None,
) -> EvaluationScenario:
    """Build the exact initial simulator state stored for one episode."""
    if episode_index < 0:
        raise ValueError("episode_index cannot be negative")
    dataset_root = Path(dataset_root).resolve()
    info = _read_json(dataset_root / "meta" / "info.json")
    source_fps = int(info.get("fps", 0))
    frame_stride = source_fps / control_hz
    if source_fps < 1 or not frame_stride.is_integer():
        raise ValueError(f"source fps {source_fps} must be a positive multiple of {control_hz:g}")

    metadata_paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"no episode metadata found under {dataset_root}")
    metadata = pa.concat_tables([pq.read_table(path) for path in metadata_paths])
    matching = metadata.filter(pc.equal(metadata["episode_index"], episode_index))
    if matching.num_rows != 1:
        raise ValueError(
            f"expected one metadata row for episode {episode_index}, found {matching.num_rows}"
        )
    row = matching.to_pylist()[0]
    data_path = dataset_root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )
    frames = pq.read_table(
        data_path,
        columns=["frame_index", "episode_index", STATE_FEATURE, ENVIRONMENT_FEATURE],
        filters=[("episode_index", "=", episode_index)],
    ).sort_by("frame_index")
    if frames.num_rows != int(row["length"]):
        raise ValueError(
            f"episode {episode_index} metadata says {row['length']} frames but data has "
            f"{frames.num_rows}"
        )
    first = frames.slice(0, 1).to_pylist()[0]
    robot_state = tuple(float(value) for value in first[STATE_FEATURE])
    environment = tuple(float(value) for value in first[ENVIRONMENT_FEATURE])
    if len(robot_state) != 6 or len(environment) < 7:
        raise ValueError("recorded episode must begin with 6 robot and 7 environment values")

    source_position = environment[:3]
    source_orientation = environment[3:7]
    source = CubePose(*source_position)
    episode_steps = math.ceil(int(row["length"]) / int(frame_stride))
    return EvaluationScenario(
        scenario_id=f"recorded-{episode_index:06d}",
        group="recorded_episode",
        workspace_region=workspace_region(source),
        seed=episode_index,
        source_position_m=source_position,
        source_orientation_wxyz=source_orientation,
        target_position_m=(float(row["target_x"]), float(row["target_y"]), CUBE_HALF_SIZE),
        initial_robot_state_real=robot_state,
        domain_randomization_preset=None,
        domain_randomization_sample={"enabled": False},
        miscalibration_sample={
            "joint_offsets_deg": {},
            "pan_jitter": None,
            "cube_belief_error": [0.0, 0.0, 0.0, 0.0],
            "target_belief_error": [0.0, 0.0],
        },
        control_hz=control_hz,
        max_steps=max_steps if max_steps is not None else episode_steps,
        target_plate_yaw_rad=float(row["target_plate_yaw"]),
        physics_sample=asdict(NOMINAL),
    )
