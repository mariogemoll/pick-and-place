# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Cube, contact, and grasp transforms.

The "simple grasp" pose keeps the gripper vertical (roll axis up, jaws closing
horizontally onto a vertical cube face).

The canonical grasp pose generalizes that same contact geometry: the jaw-closing
axis is snapped to one of the cube's four side faces, while the approach may
tilt in the radial/vertical plane so the same grasp works throughout the broader
floor workspace.

Naming note: ``grasp`` here is the gripper pose *at* the cube (open, ready to
close) and a raised ``grasp`` (positive ``z_offset``) is the "hover". In
canonical grasp terminology the raised pose is the *pre-grasp / approach* pose
and the at-cube pose is the *grasp* pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pick_and_place import transforms as tf
from pick_and_place.transforms import Mat4, Vec3

# Fixed-jaw tip collision box (the innermost box of the fixed-jaw model).
_TIP_BOX_POS = np.array((-0.01189, -0.00015, -0.099363))
_TIP_BOX_SIZE = np.array((0.004, 0.00545, 0.005063))

CUBE_HALF_SIZE = 0.015
SAFETY_MARGIN = 0.01
_MARKER_SURFACE_OFFSET = 0.00001

WORLD_UP = np.array((0.0, 0.0, 1.0))
_VERTICAL_TOLERANCE = 1e-9

# Jaw contact point: center of the tip box's inner face. The y component is
# deliberately zeroed so it stays in the pan-axis plane (which the IK relies on).
JAW_CONTACT_POSITION = np.array(
    (_TIP_BOX_POS[0] + _TIP_BOX_SIZE[0] + _MARKER_SURFACE_OFFSET, 0.0, _TIP_BOX_POS[2])
)

# Jaw vertical offset from the vertical center of the cube (in meters).
# A negative value shifts the grip lower on the cube (e.g., -0.005 grips 5mm lower).
GRIP_Z_OFFSET = -0.005

# Cube contact point: center of the +x face, nudged out by the marker offset,
# and shifted vertically by GRIP_Z_OFFSET.
CUBE_CONTACT_POSITION = np.array((CUBE_HALF_SIZE + _MARKER_SURFACE_OFFSET, 0.0, GRIP_Z_OFFSET))

# IK position target: jaw contact projected onto the wrist-roll axis. The offset
# from contact to here runs along gripper x, so wrist roll leaves it invariant.
GRIPPER_TARGET_POSITION = np.array((0.0, 0.0, _TIP_BOX_POS[2]))

CubeFace = str  # one of '+x', '-x', '+y', '-y', '+z', '-z'

VERTICAL_FACES: tuple[CubeFace, ...] = ("+x", "-x", "+y", "-y")

CANONICAL_PREGRASP_DISTANCE = 0.045
CANONICAL_FACE_OFFSET = CUBE_HALF_SIZE + SAFETY_MARGIN + JAW_CONTACT_POSITION[0]


@dataclass(frozen=True)
class CubePose:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


def world_from_cube(pose: CubePose) -> Mat4:
    return tf.translation(pose.x, pose.y, pose.z) @ tf.rotation_zyx(
        pose.roll, pose.pitch, pose.yaw
    )


def _cube_face_rotation(face: CubeFace) -> Mat4:
    if face == "+x":
        return tf.identity()
    if face == "+y":
        return tf.rot_z(-np.pi / 2)
    if face == "-x":
        return tf.rot_z(np.pi)
    if face == "-y":
        return tf.rot_z(np.pi / 2)
    if face == "+z":
        return tf.rot_y(np.pi / 2)
    if face == "-z":
        return tf.rot_y(-np.pi / 2)
    raise ValueError(f"Unknown cube face: {face}")


def _cube_from_contact() -> Mat4:
    # Point the target frame into the cube so the surfaces are flush.
    out = tf.rot_y(-np.pi / 2)
    out[:3, 3] = CUBE_CONTACT_POSITION
    return out


def _gripper_from_contact() -> Mat4:
    out = tf.rot_y(np.pi / 2)
    out[:3, 3] = JAW_CONTACT_POSITION
    return out


def world_from_cube_contact(face: CubeFace, pose: CubePose) -> Mat4:
    return world_from_cube(pose) @ _cube_face_rotation(face) @ _cube_from_contact()


def simple_grasp_matrix(face: CubeFace, pose: CubePose) -> Mat4 | None:
    """World-from-gripper for the vertical grasp on ``face``, or ``None`` when
    the face is not vertical for this pose."""
    world_from_contact = world_from_cube_contact(face, pose)
    inward_normal = tf.transform_direction(world_from_contact, WORLD_UP)
    if abs(float(np.dot(inward_normal, WORLD_UP))) > _VERTICAL_TOLERANCE:
        return None

    gripper_y = np.cross(WORLD_UP, inward_normal)
    world_from_gripper = tf.make_basis(inward_normal, gripper_y, WORLD_UP)

    cube_contact_position = tf.get_position(world_from_contact)
    jaw_contact_position = cube_contact_position - inward_normal * SAFETY_MARGIN
    jaw_offset = world_from_gripper[:3, :3] @ JAW_CONTACT_POSITION
    return tf.with_position(world_from_gripper, jaw_contact_position - jaw_offset)


def grasp_matrix(face: CubeFace, pose: CubePose, z_offset: float = 0.0) -> Mat4 | None:
    """``simple_grasp_matrix`` shifted up along world z by ``z_offset``."""
    grasp = simple_grasp_matrix(face, pose)
    if grasp is None:
        return None
    pos = tf.get_position(grasp)
    return tf.with_position(grasp, pos + np.array((0.0, 0.0, z_offset)))


def canonical_grasp_matrix(
    pose: CubePose,
    closing_azimuth: float,
    approach: Vec3,
) -> Mat4:
    """World-from-gripper for the full-range canonical grasp.

    ``approach`` is the unit world direction from wrist to cube target.
    ``closing_azimuth`` is the horizontal jaw-closing direction.
    """
    z_axis = -np.asarray(approach, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.array((np.cos(closing_azimuth), np.sin(closing_azimuth), 0.0))
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    matrix = tf.make_basis(x_axis, y_axis, z_axis)
    target = np.array((pose.x, pose.y, pose.z)) - x_axis * CANONICAL_FACE_OFFSET
    gripper_offset = matrix[:3, :3] @ GRIPPER_TARGET_POSITION
    return tf.with_position(matrix, target - gripper_offset)


def canonical_pregrasp_matrix(
    grasp: Mat4,
    approach: Vec3,
    distance: float = CANONICAL_PREGRASP_DISTANCE,
) -> Mat4:
    """Back off from the canonical contact grasp along the approach line."""
    return tf.with_position(
        grasp,
        tf.get_position(grasp) - np.asarray(approach, dtype=np.float64) * distance,
    )
