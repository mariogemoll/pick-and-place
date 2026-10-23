# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Tests for mimic derivation in the generic robot exporter."""

from __future__ import annotations

import mujoco
import pytest

from pick_and_place.cli.export_generic_robot import derive_joint_mimics, equality_joint_mimics

# A palm with two independently actuated fingers. Each finger's tip follows its
# base through a joint equality, and the thumb tip follows the thumb's middle
# joint, which in turn follows the thumb base (a two-level chain).
TWO_FINGER_HAND = """
<mujoco>
  <worldbody>
    <body name="palm">
      <geom type="box" size="0.05 0.05 0.01"/>
      <body name="index_base" pos="0.03 0 0">
        <joint name="index" axis="0 1 0" range="0 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.008"/>
        <body name="index_tip" pos="0 0 0.04">
          <joint name="index_tip" axis="0 1 0" range="0 2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.007"/>
        </body>
      </body>
      <body name="thumb_base" pos="-0.03 0 0">
        <joint name="thumb" axis="0 1 0" range="0 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.008"/>
        <body name="thumb_mid" pos="0 0 0.03">
          <joint name="thumb_mid" axis="0 1 0" range="0 2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.03" size="0.007"/>
          <body name="thumb_tip" pos="0 0 0.03">
            <joint name="thumb_tip" axis="0 1 0" range="0 3"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.02" size="0.006"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="index_tip" joint2="index" polycoef="0.1 1.2 0 0 0"/>
    <joint joint1="thumb_mid" joint2="thumb" polycoef="0 1.5 0 0 0"/>
    <joint joint1="thumb_tip" joint2="thumb_mid" polycoef="0.2 0.5 0 0 0"/>
  </equality>
  <actuator>
    <position joint="index" kp="5" ctrlrange="0 1.5"/>
    <position joint="thumb" kp="5" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def hand() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(TWO_FINGER_HAND)


def test_equality_mimics_are_exact_and_chains_fold_to_the_root(hand: mujoco.MjModel) -> None:
    mimics = equality_joint_mimics(hand)
    assert mimics["index_tip"] == {"joint": "index", "multiplier": pytest.approx(1.2), "offset": pytest.approx(0.1)}
    assert mimics["thumb_mid"] == {"joint": "thumb", "multiplier": pytest.approx(1.5), "offset": pytest.approx(0.0)}
    # thumb_tip = 0.2 + 0.5 * thumb_mid = 0.2 + 0.5 * (1.5 * thumb)
    assert mimics["thumb_tip"] == {"joint": "thumb", "multiplier": pytest.approx(0.75), "offset": pytest.approx(0.2)}


def test_fingers_are_not_fitted_against_other_fingers_actuators(hand: mujoco.MjModel) -> None:
    mimics = derive_joint_mimics(hand)
    assert set(mimics) == {"index_tip", "thumb_mid", "thumb_tip"}
    assert mimics["index_tip"]["joint"] == "index"
    assert mimics["thumb_tip"]["joint"] == "thumb"
