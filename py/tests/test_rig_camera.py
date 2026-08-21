# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""A solved rig camera converts to a randomization draw, and back exactly.

The round-trip is the whole claim: if applying the derived jitter does not land
the camera precisely where calibration solved it, then expressing a rig as a
draw is lossy, and a policy would be rolled out somewhere subtly other than the
place it was trained.
"""

import json
import math

import numpy as np
import pytest

from pick_and_place.sim.camera_pose_envelope import (
    CameraJitter,
    overhead_pose_filter,
    set_camera_jitter,
    snapshot_camera,
)
from pick_and_place.sim.domain_randomization import DomainRandomizationPreset
from pick_and_place.runtime.policy_sim import build_policy_sim_model
from pick_and_place.variants.scene import OVERHEAD_CAMERA
from pick_and_place.variants.rig_camera import (
    envelope_usage,
    jitter_payload,
    rig_camera_jitter,
)

# The overhead camera as pick-and-place's own rig solved it on 2026-06-17,
# recorded here so the round-trip runs on a real calibration rather than a
# plausible-looking one. The sidecar it came from is gitignored by design.
SOLVED_POS = (-0.005826346685220326, 0.00480986962778549, -0.02612838252977453)
SOLVED_QUAT = (
    0.9998816406749875,
    -0.012564085589208728,
    -0.006696581846603056,
    -0.005831310828133577,
)
SOLVED_FOVY_DEG = 47.20442035229455
# The same solve's own bookkeeping, which the derived jitter must reproduce.
RECORDED_DELTA_MM = 9.131
RECORDED_DELTA_DEG = 1.763


@pytest.fixture(scope="module")
def overhead_base():
    model, _ = build_policy_sim_model(480, 640)
    return model, snapshot_camera(model, OVERHEAD_CAMERA)


def test_jitter_reproduces_the_solves_own_recorded_deltas(overhead_base):
    _, base = overhead_base
    jitter = rig_camera_jitter(
        base, solved_pos=SOLVED_POS, solved_quat=SOLVED_QUAT, solved_fovy_deg=SOLVED_FOVY_DEG
    )

    position_mm = np.linalg.norm(jitter.position_m) * 1000.0
    rotation = np.linalg.norm(jitter.rotation_deg)
    assert position_mm == pytest.approx(RECORDED_DELTA_MM, abs=0.001)
    # An xyz Euler decomposition's norm is not the geodesic angle in general,
    # but at this magnitude the two agree to well under the recorded precision.
    assert rotation == pytest.approx(RECORDED_DELTA_DEG, abs=0.01)


def test_applying_the_jitter_lands_on_the_solved_pose(overhead_base):
    """The round trip: solved pose -> jitter -> model, unchanged."""
    model, base = overhead_base
    jitter = rig_camera_jitter(
        base, solved_pos=SOLVED_POS, solved_quat=SOLVED_QUAT, solved_fovy_deg=SOLVED_FOVY_DEG
    )
    try:
        set_camera_jitter(model, base, jitter)

        assert model.cam_pos[base.camera] == pytest.approx(SOLVED_POS, abs=1e-12)
        # A quaternion and its negation are the same rotation.
        landed = np.asarray(model.cam_quat[base.camera], dtype=float)
        expected = np.asarray(SOLVED_QUAT, dtype=float)
        if np.dot(landed, expected) < 0:
            landed = -landed
        assert landed == pytest.approx(expected, abs=1e-12)
        assert float(model.cam_fovy[base.camera]) == pytest.approx(SOLVED_FOVY_DEG, abs=1e-9)
    finally:
        set_camera_jitter(model, base, None)


def test_restoring_puts_the_authored_pose_back(overhead_base):
    model, base = overhead_base
    authored_pos = np.array(model.cam_pos[base.camera], copy=True)
    authored_fovy = float(model.cam_fovy[base.camera])

    jitter = rig_camera_jitter(
        base, solved_pos=SOLVED_POS, solved_quat=SOLVED_QUAT, solved_fovy_deg=SOLVED_FOVY_DEG
    )
    set_camera_jitter(model, base, jitter)
    set_camera_jitter(model, base, None)

    assert model.cam_pos[base.camera] == pytest.approx(authored_pos, abs=1e-12)
    assert float(model.cam_fovy[base.camera]) == pytest.approx(authored_fovy, abs=1e-12)


def test_an_unsolved_camera_is_the_identity_jitter(overhead_base):
    """Handing back the authored pose must produce no displacement at all."""
    _, base = overhead_base
    jitter = rig_camera_jitter(
        base, solved_pos=base.pos, solved_quat=base.quat, solved_fovy_deg=base.fovy
    )
    assert jitter.position_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)
    assert jitter.rotation_deg == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert jitter.focal_scale == pytest.approx(1.0, abs=1e-15)


def test_the_projects_rig_sits_inside_the_act_mild_envelope(overhead_base):
    """The claim build_policy_sim_model's docstring makes, as a check.

    It says the randomization envelope is "wide enough to cover an ordinary
    rig's deviation" from the authored pose. This is that assertion against the
    one rig the project has, and it is what makes "my rig" a point randomization
    already samples rather than a separate world.
    """
    _, base = overhead_base
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    preset = DomainRandomizationPreset.load(
        repository_root / "config" / "domain_randomization" / "act_mild_v1.json"
    )
    jitter = rig_camera_jitter(
        base, solved_pos=SOLVED_POS, solved_quat=SOLVED_QUAT, solved_fovy_deg=SOLVED_FOVY_DEG
    )

    usage = envelope_usage(jitter, preset)
    assert usage.inside, f"the rig left the envelope: worst axis at {usage.worst:.1%}"

    # Inside the box is not enough: the preset also rejects poses that would
    # lose a workspace-frame tag off the sensor, so the reachable set is
    # narrower than the box.
    accepted = overhead_pose_filter().accepts(
        np.asarray(jitter.position_m),
        np.asarray(jitter.rotation_deg),
        preset.scalars["overhead_camera_frame_tag_margin_px"],
        focal_scale=jitter.focal_scale,
    )
    assert accepted, "the rig pose is in the box but the tag-visibility filter rejects it"


def test_focal_scale_inverts_the_fovy_the_applier_computes(overhead_base):
    """Pin the relationship the two sides have to agree on.

    set_camera_jitter divides tan(fovy/2) by focal_scale. Getting the direction
    wrong changes every pixel and nothing raises, which is the failure this
    project has already been bitten by for image size and appearance.
    """
    _, base = overhead_base
    for target_fovy in (base.fovy * 0.95, base.fovy, base.fovy * 1.05):
        jitter = rig_camera_jitter(
            base, solved_pos=base.pos, solved_quat=base.quat, solved_fovy_deg=target_fovy
        )
        half = math.radians(base.fovy) / 2.0
        applied = math.degrees(2.0 * math.atan(math.tan(half) / jitter.focal_scale))
        assert applied == pytest.approx(target_fovy, abs=1e-9)


def test_payload_is_json_round_trippable(overhead_base):
    """It has to survive being written into a scenario and read back."""
    _, base = overhead_base
    jitter = rig_camera_jitter(
        base, solved_pos=SOLVED_POS, solved_quat=SOLVED_QUAT, solved_fovy_deg=SOLVED_FOVY_DEG
    )
    payload = json.loads(json.dumps(jitter_payload(jitter)))
    restored = CameraJitter(
        position_m=tuple(payload["overhead_camera_position_m"]),
        rotation_deg=tuple(payload["overhead_camera_rotation_deg"]),
        focal_scale=payload["overhead_camera_focal_scale"],
    )
    assert restored == jitter
