# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The seam between a controller and the world it drives.

The rest of the runtime is covered through the loops that use it; what is
asserted here is the part the interface exists to hide, and where the two
implementations genuinely differ — which frame commands and readback live in,
and whether a sighting is new.
"""

import math

import numpy as np
import pytest

from pick_and_place.core.geometry import CubePose
from pick_and_place.core.joint_frames import follower_clamp_limits
from pick_and_place.core.miscalibration import MiscalibrationModel
from pick_and_place.plant.interface import NOTHING_SEEN, Sighting
from pick_and_place.plant.real import RealPlant
from pick_and_place.plant.sim import SimPlant
from pick_and_place.runtime.believed_frame import BelievedFrame
from pick_and_place.runtime.episodes import prepare_episode
from pick_and_place.sim.model import get_joint
from pick_and_place.spec.robot import ARM_JOINT_NAMES, HARDWARE_SIMULATION_HZ


@pytest.fixture(scope="module")
def episode():
    built = prepare_episode(np.random.default_rng(0), max_attempts=40)
    built.model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    return built


def _sim_plant(episode, draw=None) -> SimPlant:
    belief = BelievedFrame(episode.model, episode.data, draw, episode.data.time)
    return SimPlant(
        episode.model,
        episode.data,
        belief=belief,
        actuator_id=episode.actuator_id,
        robot_geom_ids=episode.robot_geom_ids,
        env_geom_ids=episode.env_geom_ids,
        kinematics=episode.kinematics,
        substeps_per_tick=round(HARDWARE_SIMULATION_HZ / 30),
    )


def test_a_sighting_is_only_usable_when_it_is_both_solved_and_new():
    assert not NOTHING_SEEN.usable
    assert not Sighting(pose=CubePose(0.3, 0.0, 0.015), fresh=False).usable
    assert not Sighting(pose=None, fresh=True).usable
    assert Sighting(pose=CubePose(0.3, 0.0, 0.015), fresh=True).usable


def test_the_sim_plant_commands_the_true_frame_and_reports_the_believed_one(episode):
    """A servo commanded theta rests at theta + offset and reports theta back."""
    draw = MiscalibrationModel().sample(np.random.default_rng(1000))
    plant = _sim_plant(episode, draw)
    commanded_offsets = draw.offsets_rad(0.0)
    assert any(abs(value) > 1e-3 for value in commanded_offsets.values())

    plant.step({name: 0.1 for name in ARM_JOINT_NAMES}, 0.5)

    for name in ARM_JOINT_NAMES:
        ctrl = episode.data.ctrl[episode.actuator_id[name]]
        assert ctrl == pytest.approx(0.1 + commanded_offsets.get(name, 0.0))

    # The readback subtracts the offsets again, so the controller sees the frame
    # it planned in. Read at the tick's own time: the pan zero wanders, so the
    # offset in effect now is not the one that went in a tick ago.
    measured_arm, _ = plant.measured()
    offsets = plant.belief.offsets_rad()
    for name in ARM_JOINT_NAMES:
        true = get_joint(episode.model, episode.data, name)
        assert measured_arm[name] == pytest.approx(true - offsets.get(name, 0.0))
    assert offsets["shoulder_pan"] != commanded_offsets["shoulder_pan"]


def test_without_a_draw_the_sim_plant_is_the_identity(episode):
    plant = _sim_plant(episode)
    joints = {name: 0.2 for name in ARM_JOINT_NAMES}

    plant.step(joints, 0.5)

    for name in ARM_JOINT_NAMES:
        assert episode.data.ctrl[episode.actuator_id[name]] == pytest.approx(0.2)
    measured_arm, _ = plant.measured()
    for name in ARM_JOINT_NAMES:
        assert measured_arm[name] == pytest.approx(get_joint(episode.model, episode.data, name))


def test_the_sim_plants_command_comes_back_in_the_real_frame(episode):
    plant = _sim_plant(episode)

    commanded = plant.step({name: math.radians(10.0) for name in ARM_JOINT_NAMES}, 0.5)

    assert commanded[: len(ARM_JOINT_NAMES)] == pytest.approx(10.0)


def test_a_sim_sighting_without_a_camera_sees_nothing(episode):
    assert _sim_plant(episode).sighting(CubePose(0.3, 0.0, 0.015)) is NOTHING_SEEN


class _StubEstimate:
    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.source = CubePose(x=0.3, y=0.0, z=0.015)


class _StubServo:
    """A detector thread that keeps answering with whatever it last managed."""

    def __init__(self, frame_ids: list[int]) -> None:
        self.frame_ids = list(frame_ids)
        self.reader = None
        self.undistort_map = None

    def begin_phase(self, active: bool) -> tuple[int, int]:
        return -1, -1

    def sample(self, camera_position, camera_rotation):
        frame_id = self.frame_ids.pop(0)
        return (None if frame_id is None else _StubEstimate(frame_id)), None

    def close(self) -> None:
        pass


class _StubFollower:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_action(self, action: dict) -> None:
        self.sent.append(action)

    def get_observation(self) -> dict:
        return {}


def _real_plant(episode, servo=None) -> RealPlant:
    low, high = follower_clamp_limits(episode.kinematics)
    return RealPlant(
        episode.model,
        episode.data,
        follower=_StubFollower(),
        actuator_id=episode.actuator_id,
        robot_geom_ids=episode.robot_geom_ids,
        env_geom_ids=episode.env_geom_ids,
        kinematics=episode.kinematics,
        substeps_per_tick=round(HARDWARE_SIMULATION_HZ / 30),
        clamp_low=low,
        clamp_high=high,
        wrist_camera_id=0,
        servo=servo,
    )


def test_the_real_plant_reports_a_repeated_detection_as_stale(episode):
    """A detector on its own thread answers twice with the same solve.

    Folding it in twice would let one detection pull the grasp further than it
    should, which is why freshness is decided here and not in the loop.
    """
    plant = _real_plant(episode, _StubServo([7, 7, 8, None]))
    believed = CubePose(0.3, 0.0, 0.015)

    assert plant.sighting(believed).usable
    assert not plant.sighting(believed).usable
    assert plant.sighting(believed).usable
    assert plant.sighting(believed) is NOTHING_SEEN


def test_the_real_plant_sends_a_clamped_real_frame_command(episode):
    plant = _real_plant(episode)

    commanded = plant.step({name: math.radians(10.0) for name in ARM_JOINT_NAMES}, 0.5)

    assert commanded[: len(ARM_JOINT_NAMES)] == pytest.approx(10.0)
    assert plant.follower.sent[-1] == {
        f"{name}.pos": pytest.approx(value)
        for name, value in zip(
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
            commanded,
            strict=True,
        )
    }


def test_a_partial_observation_keeps_the_commanded_value(episode):
    """The stub follower reports nothing, so every joint falls back to its command."""
    plant = _real_plant(episode)

    commanded = plant.step({name: math.radians(10.0) for name in ARM_JOINT_NAMES}, 0.5)

    np.testing.assert_allclose(plant.readback(), commanded)
