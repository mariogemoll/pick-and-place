# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
from pathlib import Path

import numpy as np

from pick_and_place.spec.workspace import CUBE_REST_Z
import pyarrow as pa
import pyarrow.parquet as pq

from pick_and_place.runtime.recorded_scenes import recorded_episode_scenario


def _write_episode(root: Path) -> None:
    metadata_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    metadata_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        )
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [4],
                "length": [4],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "target_x": [0.2],
                "target_y": [0.1],
                "target_plate_yaw": [0.3],
            }
        ),
        metadata_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "frame_index": [0, 1, 2, 3],
                "episode_index": [4, 4, 4, 4],
                "observation.state": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 4,
                "observation.environment_state": [
                    [0.4, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]
                ]
                * 4,
            }
        ),
        data_dir / "file-000.parquet",
    )


def test_recorded_episode_scenario_uses_first_recorded_state(tmp_path: Path) -> None:
    _write_episode(tmp_path)

    scenario = recorded_episode_scenario(tmp_path, 4, control_hz=10)

    np.testing.assert_allclose(scenario.source_position_m, [0.4, 0.0, 0.015])
    np.testing.assert_allclose(scenario.source_orientation_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(scenario.target_position_m, [0.2, 0.1, CUBE_REST_Z])
    np.testing.assert_allclose(scenario.initial_robot_state_real, [1, 2, 3, 4, 5, 6])
    assert scenario.max_steps == 2
    assert scenario.target_plate_yaw_rad == 0.3
