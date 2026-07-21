# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json

import numpy as np
import pytest

from pick_and_place.physical_rig import PhysicalRig, require_joint_zero_offsets


def test_joint_zero_calibration_is_required_by_default(tmp_path):
    with pytest.raises(RuntimeError, match="missing required joint-zero"):
        require_joint_zero_offsets(tmp_path / "missing.json")


def test_uncalibrated_debug_override_is_explicit(tmp_path):
    assert require_joint_zero_offsets(
        tmp_path / "missing.json", allow_uncalibrated=True
    ) == {}


def test_joint_zero_calibration_loads_latest_offsets(tmp_path):
    path = tmp_path / "zeros.json"
    path.write_text(json.dumps({"latest": {"offsets_deg": {"shoulder_pan": 1.25}}}))
    assert require_joint_zero_offsets(path) == {"shoulder_pan": 1.25}


def test_rig_releases_torque_and_resources_if_parking_fails():
    events = []

    class Bus:
        def disable_torque(self):
            events.append("torque")

    class Follower:
        bus = Bus()

        def disconnect(self):
            events.append("follower")

    class Camera:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(self.name)

    def fail_park():
        events.append("park")
        raise RuntimeError("park failed")

    rig = PhysicalRig(
        follower=Follower(),
        overhead=Camera("overhead"),
        wrist=Camera("wrist"),
        clamp_low=np.zeros(6),
        clamp_high=np.ones(6),
        joint_zero_offsets={},
        park_action=fail_park,
    )

    with pytest.raises(RuntimeError, match="park failed"):
        rig.park_and_release()
    assert events == ["park", "torque", "follower", "overhead", "wrist"]
