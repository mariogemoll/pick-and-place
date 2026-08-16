# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The trajectory artifact: what it stores, and what it refuses to store.

The format is the contract between generating an episode and rendering it, so
what is asserted here is that a round trip is lossless, that a malformed episode
is rejected at the point it is built rather than at the point something tries to
render it, and that nothing derivable is stored twice.
"""

import json
import zipfile

import numpy as np
import pytest

from pick_and_place.core.geometry import CubePose
from pick_and_place.data.trajectory_artifact import (
    ARTIFACT_VERSION,
    EpisodeFacts,
    MiscalibrationRecord,
    TrajectoryArtifact,
    TrajectoryFrames,
    TrajectoryWriter,
    load_trajectory,
    render_environment_fingerprint,
    save_trajectory,
)
from pick_and_place.core.miscalibration import MiscalibrationModel
from pick_and_place.spec.workspace import CUBE_HALF_SIZE


def _writer(cube_xy=((0.30, 0.00), (0.25, 0.05)), phases=("approach", "carry")) -> TrajectoryWriter:
    """Two ticks in two phases, with the arm and cube in different places in each."""
    writer = TrajectoryWriter()
    for index, ((x, y), phase) in enumerate(zip(cube_xy, phases, strict=True)):
        writer.record(
            phase_name=phase,
            true_state=np.full(6, float(index) + 0.5),
            believed_state=np.full(6, float(index)),
            action=np.full(6, float(index) + 1.0),
            true_cube_pose=np.array([x, y, CUBE_HALF_SIZE, 1.0, 0.0, 0.0, 0.0]),
            believed_cube_pose=CubePose(x=x + 0.006, y=y, z=CUBE_HALF_SIZE),
            wrist_sighting=CubePose(x=x, y=y, z=CUBE_HALF_SIZE, yaw=0.2) if index else None,
        )
    return writer


def _facts(writer: TrajectoryWriter, **overrides) -> EpisodeFacts:
    defaults = {
        "target_xy": (0.25, 0.05),
        "target_plate_yaw": 0.4,
        "verdict": "success",
        "phase_spans": writer.spans,
        "fingerprint": {"mujoco_version": "3.11.0"},
        "seed": 7,
        "episode_index": 42,
        "miscalibration": None,
    }
    return EpisodeFacts(**{**defaults, **overrides})


def _artifact(**overrides) -> TrajectoryArtifact:
    writer = _writer()
    return TrajectoryArtifact(frames=writer.frames(), facts=_facts(writer, **overrides))


def test_a_round_trip_returns_every_column_unchanged(tmp_path):
    artifact = _artifact()
    path = tmp_path / "trajectory.npz"

    save_trajectory(path, artifact)
    loaded = load_trajectory(path)

    for name in ("true_state", "believed_state", "action", "true_cube_pose"):
        np.testing.assert_array_equal(
            getattr(loaded.frames, name), getattr(artifact.frames, name)
        )
    assert loaded.facts == artifact.facts


def test_the_two_frames_are_stored_separately(tmp_path):
    """The reason the format exists: the training label and the render pose differ."""
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, _artifact())

    frames = load_trajectory(path).frames

    assert np.all(frames.true_state != frames.believed_state)


def test_an_absent_wrist_sighting_reads_back_as_nothing_seen(tmp_path):
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, _artifact())

    frames = load_trajectory(path).frames

    assert list(frames.sighted) == [False, True]
    assert np.all(np.isnan(frames.wrist_sighting[0]))
    assert frames.wrist_sighting[1][5] == pytest.approx(0.2)


def test_the_spans_come_from_the_phase_names_the_ticks_carried():
    writer = _writer(phases=("approach", "approach"))
    assert [(span.name, span.start_frame) for span in writer.spans] == [("approach", 0)]

    writer = _writer()
    assert [(span.name, span.start_frame) for span in writer.spans] == [
        ("approach", 0),
        ("carry", 1),
    ]


def test_a_column_of_the_wrong_width_is_refused_when_the_frames_are_built():
    with pytest.raises(ValueError, match="true_cube_pose must be"):
        TrajectoryFrames(
            true_state=np.zeros((2, 6)),
            believed_state=np.zeros((2, 6)),
            action=np.zeros((2, 6)),
            true_cube_pose=np.zeros((2, 6)),
            believed_cube_pose=np.zeros((2, 6)),
            wrist_sighting=np.zeros((2, 6)),
        )


def test_columns_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match="holds 1 frames against 2"):
        TrajectoryFrames(
            true_state=np.zeros((2, 6)),
            believed_state=np.zeros((1, 6)),
            action=np.zeros((2, 6)),
            true_cube_pose=np.zeros((2, 7)),
            believed_cube_pose=np.zeros((2, 6)),
            wrist_sighting=np.zeros((2, 6)),
        )


def test_the_placement_error_is_derived_from_the_last_frame():
    """Not stored, so it cannot disagree with the trajectory it summarizes."""
    artifact = _artifact(target_xy=(0.20, 0.05))

    error = artifact.placement_error()

    assert error.cube_xyz[:2] == pytest.approx((0.25, 0.05))
    assert error.dx == pytest.approx(0.05)
    assert error.xy == pytest.approx(0.05)


def test_a_miscalibration_draw_survives_the_round_trip(tmp_path):
    draw = MiscalibrationModel().sample(np.random.default_rng(0))
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, _artifact(miscalibration=MiscalibrationRecord.of(draw)))

    record = load_trajectory(path).facts.miscalibration

    assert record.base_offsets_deg == pytest.approx(draw.base_offsets_deg)
    assert record.cube_belief_error == pytest.approx(draw.cube_belief_error)
    assert record.target_belief_error == pytest.approx(draw.target_belief_error)


def test_an_artifact_from_a_future_schema_is_refused(tmp_path):
    """A silently misread column is worse than a failed load."""
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, _artifact())
    _rewrite_metadata(path, lambda facts: {**facts, "artifact_version": ARTIFACT_VERSION + 1})

    with pytest.raises(ValueError, match="cannot be read by this build"):
        load_trajectory(path)


def test_the_fingerprint_names_what_moves_a_pixel_outside_the_artifact():
    fingerprint = render_environment_fingerprint(render_hw=(720, 1280), image_hw=(96, 96))

    assert fingerprint["mujoco_version"]
    assert fingerprint["render_hw"] == [720, 1280]
    assert fingerprint["image_hw"] == [96, 96]
    # An episode's own fingerprint has no render pass attached to it.
    assert "render_hw" not in render_environment_fingerprint()


def _rewrite_metadata(path, transform) -> None:
    """Rewrite an artifact's JSON member, leaving its arrays alone."""
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    facts = transform(json.loads(bytes(arrays.pop("episode")).decode()))
    arrays["episode"] = np.frombuffer(json.dumps(facts).encode(), dtype=np.uint8)
    with path.open("wb") as file:
        np.savez(file, **arrays)


def test_the_archive_is_stored_uncompressed(tmp_path):
    """So a reader can map an episode's arrays instead of decompressing them."""
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, _artifact())

    with zipfile.ZipFile(path) as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {zipfile.ZIP_STORED}


def test_facts_load_without_reading_the_frames(tmp_path):
    """Selecting a master reads one small JSON per episode, not every trajectory."""
    from pick_and_place.data.trajectory_artifact import load_facts

    path = tmp_path / "trajectory.npz"
    save_trajectory(path, _artifact())

    assert load_facts(path) == load_trajectory(path).facts


def test_the_retry_budget_keeps_single_recoveries_and_drops_the_flailing_tail(tmp_path):
    """A re-pick shows up as the grasp phase running again, not as a phase of its own."""
    from pick_and_place.data.sim_dataset_staging import (
        episodes_within_retry_budget,
        grasp_attempts,
    )

    def staged(name: str, phases: tuple[str, ...]):
        root = tmp_path / name
        root.mkdir()
        writer = _writer(cube_xy=tuple((0.30, 0.01 * i) for i in range(len(phases))), phases=phases)
        save_trajectory(
            root / "trajectory.npz",
            TrajectoryArtifact(frames=writer.frames(), facts=_facts(writer)),
        )
        return root

    clean = staged("ep000000", ("approach", "grasp", "carry"))
    recovered = staged("ep000001", ("approach", "grasp", "approach", "grasp", "carry"))
    flailing = staged("ep000002", ("grasp", "approach") * 5)

    assert grasp_attempts(clean) == 1
    assert grasp_attempts(recovered) == 2
    assert grasp_attempts(flailing) == 5

    assert episodes_within_retry_budget([clean, recovered, flailing], 2) == [clean, recovered]
    assert episodes_within_retry_budget([clean, recovered, flailing], 1) == [clean]
