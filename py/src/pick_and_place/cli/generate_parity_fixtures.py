# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate the cross-language parity fixtures in ``fixtures/parity/``.

Python and TypeScript each carry their own implementation of the arm's
kinematics, the closed-form IK, the grasp geometry and the canonical grasp
search. Both are unit-tested; on their own neither proves the two agree. These
fixtures are the shared oracle: Python writes them, ``py/tests/test_parity.py``
checks Python still reproduces them, and the tests in ``ts/src/parity/`` check
TypeScript reproduces them too. Whichever side drifts, one of those tests fails.

Python is the source of truth: it drives the real arm and generates every
demonstration. TypeScript follows.

Run from ``py/``::

    MUJOCO_GL=egl python scripts/generate_parity_fixtures.py

Committed output, so regenerate deliberately and review the diff.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.paths import REPO_ROOT
from pick_and_place.core import transforms as tf
from pick_and_place.core.joint_frames import (
    gripper_angle_to_position,
    gripper_position_to_angle,
    sim_frame_to_real,
)
from pick_and_place.core.geometry import (
    CANONICAL_PREGRASP_DISTANCE,
    CubeFace,
    CubePose,
    canonical_grasp_matrix,
    canonical_pregrasp_matrix,
    grasp_matrix,
    world_from_cube,
    world_from_cube_contact,
)
from pick_and_place.core.ik import solve_simple_grasp_ik
from pick_and_place.core.kinematics import So101Kinematics
from pick_and_place.scripted.grasp import GraspChoice, grasp_candidates
from pick_and_place.scripted.motion import _timed_arc_fraction, smoothstep
from pick_and_place.sim.derive_kinematics import derive_kinematics
from pick_and_place.sim.model import build_model
from pick_and_place.spec.robot import ARM_JOINT_NAMES, GRIPPER_OPEN
from pick_and_place.spec.workspace import CUBE_HALF_SIZE


#: Where the committed fixtures live, relative to the repository root.
FIXTURE_DIR = REPO_ROOT / "fixtures" / "parity"

#: Significant digits kept when writing a float. Well below any tolerance the
#: consumers assert at, and short enough to keep every fixture inside the
#: repository's 40 KB per-file ceiling.
_SIGNIFICANT_DIGITS = 12

_LICENSE = {
    "SPDX-FileCopyrightText": "2026 Mario Gemoll",
    "SPDX-License-Identifier": "0BSD",
}

#: Cube z for a cube resting on the floor.
_GROUND_Z = CUBE_HALF_SIZE

_ALL_FACES: tuple[CubeFace, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")


def _round(value: Any) -> float:
    return float(f"{float(value):.{_SIGNIFICANT_DIGITS}g}")


def _vec(values: Any) -> list[float]:
    return [_round(value) for value in np.asarray(values).reshape(-1)]


def _mat(matrix: tf.Mat4) -> list[float]:
    """A 4x4 as 16 row-major numbers — the order ``THREE.Matrix4.set`` takes."""
    return _vec(np.asarray(matrix, dtype=np.float64))


def _joints(joints: dict[str, float]) -> dict[str, float]:
    return {name: _round(joints[name]) for name in ARM_JOINT_NAMES}


def _pose(pose: CubePose) -> dict[str, float]:
    return {
        "x": _round(pose.x),
        "y": _round(pose.y),
        "z": _round(pose.z),
        "roll": _round(pose.roll),
        "pitch": _round(pose.pitch),
        "yaw": _round(pose.yaw),
    }


def _document(description: str, **payload: Any) -> dict[str, Any]:
    return {
        **_LICENSE,
        "description": description,
        "generator": "pap generate-parity-fixtures",
        **payload,
    }


# --------------------------------------------------------------------------- #
# The poses every fixture is built from.                                       #
# --------------------------------------------------------------------------- #


def _cube_poses() -> tuple[CubePose, ...]:
    """A deterministic spread of cube poses over and around the pickup sector.

    Radii and azimuths walk the reachable band from just inside the inner limit
    to just past the outer one, and the yaws are chosen so no two cases share a
    face-snapping remainder. The last few sit outside the sector on purpose, so
    the fixtures pin the "no grasp here" answer as firmly as the reachable ones.
    """
    radii = (0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.42)
    azimuths = (-95.0, -62.0, -31.0, -7.0, 11.0, 38.0, 66.0, 88.0, 99.0)
    yaws = (0.0, 17.0, 33.0, -12.0, 45.0, -44.0, 71.0, -80.0, 100.0)
    poses = [
        CubePose(
            x=0.0388353 + radius * math.cos(math.radians(azimuth)),
            y=radius * math.sin(math.radians(azimuth)),
            z=_GROUND_Z,
            yaw=math.radians(yaw),
        )
        for radius, azimuth, yaw in zip(radii, azimuths, yaws)
    ]
    # Outside the sector: too close, too far, and past the azimuth limit.
    poses += [
        CubePose(x=0.0388353 + 0.05, y=0.0, z=_GROUND_Z),
        CubePose(x=0.0388353 + 0.60, y=0.0, z=_GROUND_Z),
        CubePose(
            x=0.0388353 + 0.25 * math.cos(math.radians(140.0)),
            y=0.25 * math.sin(math.radians(140.0)),
            z=_GROUND_Z,
            yaw=math.radians(20.0),
        ),
    ]
    return tuple(poses)


# --------------------------------------------------------------------------- #
# Fixtures.                                                                    #
# --------------------------------------------------------------------------- #


def build_kinematics_fixture(k: So101Kinematics) -> dict[str, Any]:
    """The arm as both sides measure it off the model.

    This is the root of every other fixture: if the two derivations disagree
    here, nothing downstream can agree either.
    """
    return _document(
        "SO-101 kinematics derived from the compiled model. TypeScript derives "
        "the same numbers from ts/public/so101.json.",
        kinematics={
            "panAxis": _vec(k.pan_axis[:2]),
            "shoulderLift": {
                "radial": _round(k.shoulder_lift_radial),
                "height": _round(k.shoulder_lift_height),
            },
            "upperArm": {
                "radial": _round(k.upper_arm.radial),
                "height": _round(k.upper_arm.height),
                "length": _round(k.upper_arm.length),
            },
            "lowerArm": {
                "radial": _round(k.lower_arm.radial),
                "height": _round(k.lower_arm.height),
                "length": _round(k.lower_arm.length),
            },
            "toolLength": _round(k.tool_length),
            "wristRollZeroTwist": _round(k.wrist_roll_zero_twist),
            "jointLimits": {
                name: {
                    "min": _round(k.joint_limits[name].min),
                    "max": _round(k.joint_limits[name].max),
                }
                for name in ARM_JOINT_NAMES
            },
        },
    )


def build_geometry_fixture() -> dict[str, Any]:
    """Cube, contact and grasp transforms — no kinematics involved.

    Pure frame algebra, so it isolates a disagreement in the transform
    conventions from one in the arm measurement or the IK.
    """
    poses = (
        CubePose(x=0.22, y=0.0, z=_GROUND_Z),
        CubePose(x=0.18, y=-0.13, z=_GROUND_Z, yaw=math.radians(31.0)),
        CubePose(x=-0.05, y=0.26, z=_GROUND_Z, yaw=math.radians(-77.0)),
        CubePose(x=0.30, y=0.10, z=0.08, roll=math.radians(9.0), yaw=math.radians(12.0)),
    )
    contacts = [
        {
            "pose": _pose(pose),
            "face": face,
            "worldFromCube": _mat(world_from_cube(pose)),
            "worldFromCubeContact": _mat(world_from_cube_contact(face, pose)),
            "simpleGrasp": (
                None if (matrix := grasp_matrix(face, pose)) is None else _mat(matrix)
            ),
        }
        for pose in poses
        for face in _ALL_FACES
    ]

    canonical = []
    for pose in poses:
        azimuth = math.atan2(pose.y, pose.x - 0.0388353)
        for pitch_deg in (90.0, 62.0, 16.0):
            pitch = math.radians(pitch_deg)
            horizontal = math.cos(pitch)
            approach = np.array(
                (
                    math.cos(azimuth) * horizontal,
                    math.sin(azimuth) * horizontal,
                    -math.sin(pitch),
                )
            )
            for closing_deg in (0.0, 90.0):
                closing = azimuth + math.radians(closing_deg)
                grasp = canonical_grasp_matrix(pose, closing, approach)
                canonical.append(
                    {
                        "pose": _pose(pose),
                        "closingAzimuth": _round(closing),
                        "approach": _vec(approach),
                        "grasp": _mat(grasp),
                        "pregrasp": _mat(
                            canonical_pregrasp_matrix(
                                grasp, approach, CANONICAL_PREGRASP_DISTANCE
                            )
                        ),
                    }
                )

    return _document(
        "Cube, contact and grasp transforms as 16 row-major numbers each.",
        pregraspDistance=_round(CANONICAL_PREGRASP_DISTANCE),
        contactCases=contacts,
        canonicalGraspCases=canonical,
    )


def _ik_pose_matrices() -> Iterator[tuple[str, tf.Mat4]]:
    """Gripper poses spanning what the IK has to answer for.

    Reachable in-plane poses of both kinds, poses past the arm's reach, and
    poses whose approach leaves the arm's vertical plane — the last group is
    unreachable for a 5-DOF arm, and a solver that answers anything but "no"
    for them is silently projecting the request back in-plane.
    """
    for pose in _cube_poses()[:5]:
        for face in ("+x", "-x", "+y", "-y"):
            for z_offset in (0.0, 0.03):
                matrix = grasp_matrix(face, pose, z_offset)
                if matrix is not None:
                    yield f"simple {face} z+{z_offset:g} @ ({pose.x:.3f}, {pose.y:.3f})", matrix

    for pose in _cube_poses()[:5]:
        azimuth = math.atan2(pose.y, pose.x - 0.0388353)
        for pitch_deg in (90.0, 54.0, 20.0):
            pitch = math.radians(pitch_deg)
            horizontal = math.cos(pitch)
            approach = np.array(
                (
                    math.cos(azimuth) * horizontal,
                    math.sin(azimuth) * horizontal,
                    -math.sin(pitch),
                )
            )
            grasp = canonical_grasp_matrix(pose, azimuth + math.pi / 2.0, approach)
            yield f"canonical pitch {pitch_deg:g} @ ({pose.x:.3f}, {pose.y:.3f})", grasp
            yield (
                f"pregrasp pitch {pitch_deg:g} @ ({pose.x:.3f}, {pose.y:.3f})",
                canonical_pregrasp_matrix(grasp, approach, CANONICAL_PREGRASP_DISTANCE),
            )

    # Out of reach: far beyond the annulus, and buried under the floor.
    far = grasp_matrix("+x", CubePose(x=0.95, y=0.0, z=_GROUND_Z))
    assert far is not None
    yield "out of reach (radius 0.95)", far
    under = grasp_matrix("+x", CubePose(x=0.22, y=0.0, z=-0.30))
    assert under is not None
    yield "below the floor", under

    # Out of the arm's vertical plane: tilt a reachable grasp about the radial
    # axis. The larger tilts are well within the 2R annulus and the joint
    # limits, so nothing but an explicit out-of-plane check rejects them — which
    # is the point, since a 5-DOF arm cannot actually strike these poses.
    base = grasp_matrix("+x", CubePose(x=0.0388353 + 0.18, y=0.0, z=_GROUND_Z))
    assert base is not None
    for tilt_deg in (2.0, 10.0, 45.0, 60.0):
        tilted = base.copy()
        tilted[:3, :3] = tf.rot_x(math.radians(tilt_deg))[:3, :3] @ base[:3, :3]
        yield f"approach {tilt_deg:g} deg out of plane", tilted


def build_simple_ik_fixture(k: So101Kinematics) -> dict[str, Any]:
    """Every branch the closed-form IK returns for each pose."""
    cases = []
    for label, matrix in _ik_pose_matrices():
        branches = solve_simple_grasp_ik(k, matrix)
        cases.append(
            {
                "label": label,
                "worldFromGripper": _mat(matrix),
                "branches": [
                    {"elbow": branch.elbow, "joints": _joints(branch.joints)}
                    for branch in branches
                ],
            }
        )
    return _document(
        "solve_simple_grasp_ik over reachable, out-of-reach and out-of-plane "
        "gripper poses. An empty branch list means unreachable.",
        cases=cases,
    )


def build_forward_kinematics_fixture(k: So101Kinematics) -> dict[str, Any]:
    """Where the gripper's IK target ends up for a given joint set.

    Python answers with the closed-form planar chain in
    :meth:`So101Kinematics.tip_position`; TypeScript walks the model's body
    tree. Two independent derivations, so agreement here is worth more than the
    round trip through the IK that produced most of these joint sets.
    """
    cases = []
    for label, matrix in _ik_pose_matrices():
        for branch in solve_simple_grasp_ik(k, matrix):
            cases.append(
                {
                    "label": f"{label} [{branch.elbow}]",
                    "joints": _joints(branch.joints),
                    "tip": _vec(k.tip_position(branch.joints)),
                }
            )
    return _document(
        "Forward kinematics: arm joints to the world position of the gripper's "
        "IK target. Python solves the planar chain in closed form, which tracks "
        "the model's own kinematics to well under a millimetre rather than "
        "exactly — assert accordingly.",
        cases=cases,
    )


def _candidate_summary(choice: GraspChoice) -> dict[str, Any]:
    return {
        "face": choice.face,
        "elbow": choice.elbow,
        "pitch": _round(choice.pitch),
        "rollOffset": _round(choice.roll_offset),
    }


def build_grasp_fixture(k: So101Kinematics) -> dict[str, Any]:
    """The canonical grasp search: what it picks, and the order it offers.

    The search is a preference ordering, not just a predicate, so the fixture
    records the head of the candidate stream as well as the winner. A port that
    returns a reachable-but-differently-ranked grasp is still a port that will
    diverge on the next cube.
    """
    cases = []
    for pose in _cube_poses():
        choice = next(grasp_candidates(k, pose), None)
        prefix = [
            _candidate_summary(candidate)
            for _, candidate in zip(range(6), grasp_candidates(k, pose))
        ]
        cases.append(
            {
                "pose": _pose(pose),
                "candidatePrefix": prefix,
                "selected": (
                    None
                    if choice is None
                    else {
                        **_candidate_summary(choice),
                        "closingAzimuth": _round(choice.closing_azimuth),
                        "cameraOutward": _round(choice.camera_outward),
                        "inwardNormal": _vec(choice.inward_normal),
                        "hoverJoints": _joints(choice.hover_joints),
                        "graspJoints": _joints(choice.grasp_joints),
                        "liftJoints": _joints(choice.lift_joints),
                        "hoverMatrix": _mat(choice.hover_matrix),
                        "graspMatrix": _mat(choice.grasp_matrix),
                        "liftMatrix": _mat(choice.lift_matrix),
                    }
                ),
            }
        )
    return _document(
        "Canonical grasp selection. `selected` is null where no grasp exists; "
        "`candidatePrefix` is the head of the candidate stream, which must "
        "agree in order and not merely in membership.",
        cases=cases,
    )


def build_easing_fixture() -> dict[str, Any]:
    """The easing curves both sides time their moves with.

    The trajectories themselves have diverged — Python plans the physical
    eight-phase motion from a canonical grasp, TypeScript animates a five-stage
    illustrative one from a vertical grasp — but they still share the curves
    that shape every move, and those are cheap to hold together.
    """
    samples = [i / 64.0 for i in range(-2, 67)]
    return _document(
        "Shared easing curves, sampled past both ends to pin the clamping.",
        smoothstep=[{"t": _round(t), "value": _round(smoothstep(t))} for t in samples],
        timedArcFraction=[
            {"phase": _round(t), "value": _round(_timed_arc_fraction(t))}
            for t in samples
        ],
    )


def build_joint_frames_fixture() -> dict[str, Any]:
    """The sim-frame/real-frame conversion, which the browser policy page runs on.

    A learned policy emits real-frame joints and is shown real-frame joints, so
    every action crossing into a simulator goes through this and every
    observation comes back out of it. The arm half is a unit conversion; the
    gripper half is a calibrated map with clamped ends, which is the part worth
    pinning. Sampled past both endpoints so the clamping is covered.
    """
    angles_rad = [math.radians(d) for d in range(-30, 141, 5)]
    positions = [p / 4.0 for p in range(-40, 441, 5)]
    arm_cases = [
        (0.0, 0.0, 0.0, 0.0, -math.pi / 2),
        (0.1, -0.6, 0.9, 0.4, 0.0),
        (-1.9, 1.2, -1.4, 0.75, 2.6),
    ]
    return _document(
        "Sim-frame/real-frame joint conversion, including the calibrated gripper map.",
        gripperAngleToPosition=[
            {"angleRad": _round(a), "position": _round(gripper_angle_to_position(a))}
            for a in angles_rad
        ],
        gripperPositionToAngle=[
            {"position": _round(p), "angleRad": _round(gripper_position_to_angle(p))}
            for p in positions
        ],
        simFrameToReal=[
            {
                "simJoints": [_round(v) for v in (*arm, gripper)],
                "realJoints": [
                    _round(v)
                    for v in sim_frame_to_real(
                        dict(zip(ARM_JOINT_NAMES, arm, strict=True)), gripper
                    )
                ],
            }
            for arm in arm_cases
            for gripper in (0.0, 0.1, GRIPPER_OPEN, 2.0)
        ],
    )


def build_fixtures(k: So101Kinematics) -> dict[str, dict[str, Any]]:
    return {
        "kinematics.json": build_kinematics_fixture(k),
        "geometry.json": build_geometry_fixture(),
        "simple_ik.json": build_simple_ik_fixture(k),
        "forward_kinematics.json": build_forward_kinematics_fixture(k),
        "grasp.json": build_grasp_fixture(k),
        "easing.json": build_easing_fixture(),
        "joint_frames.json": build_joint_frames_fixture(),
    }


def kinematics() -> So101Kinematics:
    """The arm measured off a freshly compiled, unrandomized scene."""
    model, _ = build_model(CubePose(x=0.22, y=0.0, z=_GROUND_Z))
    return derive_kinematics(model)


def serialize(fixture: dict[str, Any]) -> str:
    """JSON with one line per case.

    Neither of the usual settings suits a committed fixture: fully indented
    output runs to hundreds of kilobytes, and a single compact line makes every
    regeneration look like a total rewrite. One line per case keeps the files
    small and the diffs readable.
    """
    entries = []
    for key, value in fixture.items():
        if isinstance(value, list):
            items = ",\n".join(
                "  " + json.dumps(item, separators=(",", ":")) for item in value
            )
            entries.append(f" {json.dumps(key)}: [\n{items}\n ]")
        else:
            body = json.dumps(value, indent=1).replace("\n", "\n ")
            entries.append(f" {json.dumps(key)}: {body}")
    return "{\n" + ",\n".join(entries) + "\n}\n"


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the parity fixture generator."""
    parser = SuggestingArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=FIXTURE_DIR,
        help="where to write the fixtures (default: the committed fixtures/parity)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Write the fixtures TypeScript is checked against."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fixture in build_fixtures(kinematics()).items():
        path = args.output_dir / name
        path.write_text(serialize(fixture), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size} bytes)")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
