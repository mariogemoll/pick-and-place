# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Training resets drawn from the evaluation distribution.

A policy is never trained on the frozen benchmark scenes. Training and
evaluation both draw from :mod:`pick_and_place.planning.scenario_sampling`, so the
distribution is shared by construction, but the seed streams are disjoint:
manifests use small bases (1701 for ``canonical_100_v1``/``dr_100_v1``, 2701 for the perturbation
smoke suite) and training starts at :data:`TRAINING_SEED_BASE`.

Scenes carry no domain randomization and no miscalibration, matching the
blue-cube dataset the policy was pretrained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pick_and_place.core.geometry import cube_quat_from_pose
from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.policies.policy_evaluation import EvaluationScenario
from pick_and_place.planning.scenario_sampling import sample_scene, workspace_region
from pick_and_place.spec.robot import GRIPPER_OPEN, NEUTRAL_ARM_JOINTS

# Far above every manifest seed base in config/evaluation/, so a training scene
# can never coincide with a benchmark scene.
TRAINING_SEED_BASE = 5_000_000
# The DPPO policy runs at 10 Hz, the rate its 10 Hz dataset was decimated to.
# 150 ticks is 15 s, the same episode budget the 30 Hz manifests allow (450
# steps), so a training episode is neither longer nor shorter than an eval one.
TRAINING_CONTROL_HZ = 10.0
TRAINING_MAX_STEPS = 150
# The neutral start every recorded episode and closed-loop rollout begins from,
# in the real frame the scenario records: the spec's neutral arm pose with the
# gripper already open, which is where a trajectory's approach phase begins.
NEUTRAL_ROBOT_STATE_REAL = tuple(
    float(v) for v in sim_frame_to_real(NEUTRAL_ARM_JOINTS, GRIPPER_OPEN)
)


def training_scenario(
    index: int,
    *,
    seed_base: int = TRAINING_SEED_BASE,
    control_hz: float = TRAINING_CONTROL_HZ,
    max_steps: int = TRAINING_MAX_STEPS,
) -> EvaluationScenario:
    """Materialize training scene ``index`` of the seed stream at ``seed_base``."""
    seed = seed_base + index
    scene = sample_scene(np.random.default_rng(seed))
    source, target = scene.source, scene.target
    return EvaluationScenario(
        scenario_id=f"dppo-train-{index:06d}",
        group="dppo_training",
        workspace_region=workspace_region(source),
        seed=seed,
        source_position_m=(source.x, source.y, source.z),
        source_orientation_wxyz=tuple(float(value) for value in cube_quat_from_pose(source)),
        target_position_m=(target.x, target.y, target.z),
        initial_robot_state_real=NEUTRAL_ROBOT_STATE_REAL,
        domain_randomization_preset=None,
        domain_randomization_sample={"enabled": False},
        miscalibration_sample={"joint_offsets_deg": {}},
        control_hz=control_hz,
        max_steps=max_steps,
        target_plate_yaw_rad=scene.plate_yaw_rad,
    )


@dataclass
class SceneStream:
    """An endless, resumable sequence of training scenes for one worker.

    Each worker owns a stride of the shared stream (``offset``, ``stride``), so
    ``n_envs`` workers never draw the same scene, and the scenes a run visited
    are reproducible from the run's seed base and worker count alone.
    """

    offset: int
    stride: int
    seed_base: int = TRAINING_SEED_BASE
    control_hz: float = TRAINING_CONTROL_HZ
    max_steps: int = TRAINING_MAX_STEPS
    _drawn: int = 0

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if not 0 <= self.offset < self.stride:
            raise ValueError("offset must be in [0, stride)")

    @property
    def next_index(self) -> int:
        return self.offset + self._drawn * self.stride

    def next(self) -> EvaluationScenario:
        scenario = training_scenario(
            self.next_index,
            seed_base=self.seed_base,
            control_hz=self.control_hz,
            max_steps=self.max_steps,
        )
        self._drawn += 1
        return scenario
