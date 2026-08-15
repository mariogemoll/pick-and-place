# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Recording a whole episode from the simulation, against doubles for the dataset.

``record_episode`` is the sim twin of ``execute_episode``, and the better tested
of the two: nothing in it runs on a thread of its own, so both of its playback
modes are deterministic and can be pinned exactly.

- **Feedforward** — no miscalibration draw. The vetted plan is played open loop,
  no camera is rendered, and a whole episode costs a fraction of a second. This
  is the path every plain sim recording takes.
- **Closed loop** — with a draw, commands live in the believed frame while
  physics runs in the true one, the descent runs the AprilTag visual servo
  against wrist renders, and each unfused phase boundary replans from the
  believed readback. It needs a working offscreen GL and a few seconds per
  episode, and it is where the interesting branches are.

The dataset is a spy rather than a LeRobotDataset and the cameras are a stub rig
where the rendered pixels are not what is under test — what is asserted is the
row stream, the phase spans, and the decisions the loop makes.
"""

import contextlib
import io
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pick_and_place.core.joint_frames import sim_frame_to_real
from pick_and_place.core.miscalibration import MiscalibrationModel
from pick_and_place.runtime.episodes import prepare_episode
from pick_and_place.rollout.sim import record_episode
from pick_and_place.rollout.sim_dataset import CUBE_POSE_STATE_NAMES
from pick_and_place.spec.robot import ARM_JOINT_NAMES, CONTROL_HZ, HARDWARE_SIMULATION_HZ

#: Real speed. The pick only holds at 1.0 — faster playback outruns the position
#: servos and drops the cube, which still records but covers everything after the
#: grasp with a run that is not carrying anything.
SPEED = 1.0

#: Frame size the stub rig reports. Small and square, like a recorded dataset's.
FRAME_SHAPE = (128, 128, 3)

PLANNED_PHASES = (
    "approach",
    "descent",
    "grasp",
    "lift",
    "carry",
    "drop_descent",
    "release",
    "retreat",
)


class StubRig:
    """A ``SimCameraRig`` stand-in that answers with a fixed frame, without rendering.

    The recorder only ever asks the rig for its output size and for two images
    per tick, so this is the whole of the interface. What the sim looks like is
    covered where the renderer is, not here.
    """

    def __init__(self, shape: tuple[int, int, int] = FRAME_SHAPE) -> None:
        self.height, self.width = shape[0], shape[1]
        self.frame = np.zeros(shape, dtype=np.uint8)
        self.captures = 0

    def capture(self, data):
        self.captures += 1
        return self.frame, self.frame


class RecordingSpy:
    """A ``RecordingSession`` stand-in that keeps every row it is handed."""

    task = "pick up the cube and place it on the target"

    def __init__(self, dropped: int = 0) -> None:
        self.dataset = None
        self.created: tuple | None = None
        self.rows: list[dict] = []
        self.dropped = dropped

    def create_dataset(self, wrist_shape, overhead_shape, environment_state_names=None) -> None:
        self.created = (tuple(wrist_shape), tuple(overhead_shape), tuple(environment_state_names))
        self.dataset = self

    def add_frame(self, frame: dict) -> None:
        self.rows.append(frame)

    def dropped_frame_count(self) -> int:
        return self.dropped

    def states(self) -> np.ndarray:
        return np.array([row["observation.state"] for row in self.rows])

    def actions(self) -> np.ndarray:
        return np.array([row["action"] for row in self.rows])

    def cube_poses(self) -> np.ndarray:
        return np.array([row["observation.environment_state"] for row in self.rows])


class CountingViewer:
    """A passive viewer that only counts its syncs."""

    def __init__(self) -> None:
        self.syncs = 0

    def sync(self) -> None:
        self.syncs += 1


def _episode(seed: int = 0, *, closed_loop: bool = False, draw_seed: int = 1000):
    """A prepared episode at the substep rate the recorder requires.

    Every production caller gets that rate from ``build_recording_scene``, which
    sets it while compiling the persistent scene; a plain ``prepare_episode``
    leaves the model at the MuJoCo default.
    """
    draw = (
        MiscalibrationModel().sample(np.random.default_rng(draw_seed)) if closed_loop else None
    )
    episode = prepare_episode(
        np.random.default_rng(seed),
        max_attempts=40,
        include_environment=closed_loop,
        miscalibration=draw,
    )
    episode.model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    return episode


def _run(episode, **kwargs):
    """Record ``episode`` into a spy, returning ``(result, recording, rig)``."""
    recording = RecordingSpy(dropped=kwargs.pop("dropped", 0))
    rig = StubRig()
    result = record_episode(
        episode, recording=recording, rig=rig, speed=kwargs.pop("speed", SPEED), **kwargs
    )
    return result, recording, rig


@pytest.fixture(scope="module")
def feedforward_run():
    """One plain recorded episode, shared by the assertions that only read it."""
    episode = _episode()
    result, recording, rig = _run(episode, verbose=False)
    return SimpleNamespace(episode=episode, result=result, recording=recording, rig=rig)


# --- feedforward playback ---------------------------------------------------


def test_a_prepared_episode_records_to_completion(feedforward_run) -> None:
    assert feedforward_run.result.status == "success"
    assert len(feedforward_run.recording.rows) > 0


def test_every_phase_of_the_plan_appears_once_in_the_spans(feedforward_run) -> None:
    """The spans come from the controller, so they are the phases as executed."""
    spans = feedforward_run.result.phase_spans
    assert tuple(span.name for span in spans) == PLANNED_PHASES
    starts = [span.start_frame for span in spans]
    assert starts[0] == 0
    assert starts == sorted(starts)
    assert starts[-1] < len(feedforward_run.recording.rows)


def test_one_row_and_one_capture_per_control_tick(feedforward_run) -> None:
    """A tick is a row is a pair of frames — the invariant the video rests on."""
    frames = len(feedforward_run.recording.rows)
    assert feedforward_run.rig.captures == frames
    # Played at real speed, so the rows are the trajectory's duration in ticks —
    # plus one per phase, because a phase's last tick is captured and then ends
    # the phase without stepping physics again.
    stepped = round(feedforward_run.episode.data.time * CONTROL_HZ)
    assert frames == stepped + len(feedforward_run.result.phase_spans)


def test_the_dataset_is_created_from_the_rigs_output_shape(feedforward_run) -> None:
    assert feedforward_run.recording.created == (
        FRAME_SHAPE,
        FRAME_SHAPE,
        CUBE_POSE_STATE_NAMES,
    )


def test_rows_carry_both_camera_images_and_the_task(feedforward_run) -> None:
    row = feedforward_run.recording.rows[0]
    assert row["observation.images.wrist"].shape == FRAME_SHAPE
    assert row["observation.images.overhead"].shape == FRAME_SHAPE
    assert row["task"] == feedforward_run.recording.task


def test_state_and_action_are_the_real_frame_and_observed_before_acting(
    feedforward_run,
) -> None:
    """The first row pairs the untouched start pose with the command about to be sent."""
    episode = feedforward_run.episode
    expected_state = sim_frame_to_real(episode.start_joints, episode.start_gripper)
    first = feedforward_run.recording.rows[0]
    np.testing.assert_allclose(first["observation.state"], expected_state, atol=1e-4)
    frame = episode.trajectory.phases[0].evaluate(0.0)
    np.testing.assert_allclose(
        first["action"], sim_frame_to_real(frame.joints, frame.gripper), atol=1e-4
    )
    # Degrees, not radians: everything is comparable to a real recording.
    assert np.abs(feedforward_run.recording.states()[:, :5]).max() > 2 * np.pi


def test_the_true_cube_pose_is_stored_with_every_frame(feedforward_run) -> None:
    """Privileged ground truth: position plus wxyz quaternion, valid in every phase."""
    poses = feedforward_run.recording.cube_poses()
    assert poses.shape == (len(feedforward_run.recording.rows), len(CUBE_POSE_STATE_NAMES))
    source = feedforward_run.episode.source
    assert poses[0][:2] == pytest.approx([source.x, source.y], abs=1e-3)
    assert np.linalg.norm(poses[:, 3:], axis=1) == pytest.approx(1.0, abs=1e-3)
    # The cube is carried: it does not stay where it started.
    assert np.linalg.norm(poses[-1][:2] - poses[0][:2]) > 0.05


def test_the_viewer_is_synced_once_per_tick() -> None:
    viewer = CountingViewer()
    result, recording, _ = _run(_episode(), viewer=viewer, verbose=False)
    assert result.status == "success"
    # Once per tick that steps physics: the tick that ends a phase is captured
    # and then breaks out before stepping, so it draws nothing new.
    assert viewer.syncs == len(recording.rows) - len(result.phase_spans)


def test_the_run_is_reproducible() -> None:
    """Bit-identical rows run to run, which is what makes this a refactoring oracle."""
    first = _run(_episode(seed=1), verbose=False)[1]
    second = _run(_episode(seed=1), verbose=False)[1]
    assert np.array_equal(first.states(), second.states())
    assert np.array_equal(first.actions(), second.actions())
    assert np.array_equal(first.cube_poses(), second.cube_poses())


def test_playback_without_a_recording_still_runs_the_episode() -> None:
    """The sim viewer's path: physics and the closed loop run, no images are written.

    The trajectory artifact is produced regardless, because it costs a few dozen
    floats per tick and is what an episode's pixels can be made from later.
    """
    episode = _episode()
    result = record_episode(episode, speed=SPEED, verbose=False)
    assert result.status == "success"
    assert result.phase_spans[0].name == "approach"
    assert len(result.frames) > 0
    assert episode.data.time > 1.0


def test_should_stop_ends_the_episode_where_it_was_asked_to() -> None:
    ticks = 0

    def should_stop() -> bool:
        nonlocal ticks
        ticks += 1
        return ticks > 20

    result, recording, _ = _run(_episode(), should_stop=should_stop, verbose=False)
    assert result.status == "stopped"
    assert len(recording.rows) == 20
    # The spans recorded so far come back, so a partial run is still interpretable.
    assert result.phase_spans[0].name == "approach"


def test_realtime_paces_the_loop_to_the_control_rate() -> None:
    """Live viewing sleeps out the rest of each tick; recording runs unpaced."""
    import time

    episode = _episode()
    started = time.monotonic()
    result = record_episode(
        episode, speed=8.0, realtime=True, should_stop=_stop_after(15), verbose=False
    )
    assert result.status == "stopped"
    # Not every one of the 15 ticks sleeps — the tick that ends a phase leaves
    # the loop before the pacing sleep — so this is a floor, not the exact cost.
    assert time.monotonic() - started >= 10 / CONTROL_HZ


def _stop_after(ticks: int):
    seen = 0

    def should_stop() -> bool:
        nonlocal seen
        seen += 1
        return seen > ticks

    return should_stop


# --- refusals ---------------------------------------------------------------


def test_speed_must_be_positive() -> None:
    with pytest.raises(ValueError, match="speed must be positive"):
        record_episode(_episode(), speed=0.0)


def test_a_recording_without_a_rig_is_refused() -> None:
    with pytest.raises(ValueError, match="recording and rig must be provided together"):
        record_episode(_episode(), recording=RecordingSpy())


def test_a_rig_without_a_recording_is_refused() -> None:
    with pytest.raises(ValueError, match="recording and rig must be provided together"):
        record_episode(_episode(), rig=StubRig())


def test_the_mixed_wrist_view_needs_a_closed_loop_episode() -> None:
    with pytest.raises(ValueError, match="show_wrist_mixed requires a miscalibration"):
        record_episode(_episode(), show_wrist_mixed=True)


def test_a_model_that_cannot_hit_the_control_rate_is_rejected() -> None:
    """The tick must be a whole number of substeps, or state and action desync."""
    episode = _episode()
    episode.model.opt.timestep = 1.0 / (HARDWARE_SIMULATION_HZ + 1)
    with pytest.raises(ValueError, match="cannot produce"):
        record_episode(episode, speed=SPEED)


def test_a_dropped_encoder_frame_fails_instead_of_saving_a_desynced_episode() -> None:
    """A short video against full-length rows is a corrupt episode; refuse to write it."""
    with pytest.raises(RuntimeError, match="dropped 1 frame"):
        _run(_episode(), dropped=1, verbose=False)


# --- closed loop ------------------------------------------------------------


@pytest.fixture(scope="module")
def closed_loop_run():
    """One miscalibrated episode, servoed down onto the cube and replanned.

    Module scoped because it renders the wrist camera every descent tick and is
    the most expensive run here. ``capsys`` cannot reach a fixture this wide, so
    the narration is captured directly.
    """
    episode = _episode(closed_loop=True)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result, recording, rig = _run(episode, verbose=True)
    return SimpleNamespace(
        episode=episode,
        result=result,
        recording=recording,
        rig=rig,
        log=log.getvalue().splitlines(),
    )


def test_a_miscalibrated_episode_servos_down_and_completes(closed_loop_run) -> None:
    assert closed_loop_run.result.status == "success"
    assert tuple(span.name for span in closed_loop_run.result.phase_spans) == PLANNED_PHASES


def test_the_descent_reaches_the_true_cube_despite_the_belief_error(
    closed_loop_run,
) -> None:
    """What the servo is for: the plan aims at the believed pose, the jaws land on the real one."""
    episode = closed_loop_run.episode
    believed = np.array([episode.believed_source.x, episode.believed_source.y])
    true = np.array([episode.source.x, episode.source.y])
    assert np.linalg.norm(believed - true) > 1e-3
    # The cube left its start pose, which only happens if it was actually picked up.
    poses = closed_loop_run.recording.cube_poses()
    assert np.linalg.norm(poses[-1][:2] - poses[0][:2]) > 0.05


def test_the_rows_stay_believed_while_the_actuators_get_the_true_frame(
    closed_loop_run,
) -> None:
    """A servo commanded ``theta`` rests at ``theta + offset``; the readback undoes it.

    So the recorded ``observation.state`` is the believed pose a real arm would
    report, and the first row is the planned start pose exactly — even though
    physics is holding the arm several degrees away from it.
    """
    episode = closed_loop_run.episode
    offsets = episode.miscalibration.offsets_deg(0.0)
    assert any(abs(value) > 0.1 for value in offsets.values())
    start_state = sim_frame_to_real(episode.start_joints, episode.start_gripper)
    np.testing.assert_allclose(closed_loop_run.recording.states()[0], start_state, atol=1e-4)


def test_the_artifact_keeps_the_true_arm_pose_the_dataset_drops(closed_loop_run) -> None:
    """The whole reason the artifact exists.

    ``observation.state`` is the believed readback, so a re-render driven by it
    would put the arm several degrees from where physics held it. The artifact
    stores both, and the difference between them is the offset in effect that
    tick — which is not a constant, because the pan zero wanders over the
    episode. Nothing else records that wander, so nothing else can undo it.
    """
    frames = closed_loop_run.result.frames
    np.testing.assert_allclose(
        frames.believed_state, closed_loop_run.recording.states(), atol=1e-4
    )
    offsets = frames.true_state[:, :5] - frames.believed_state[:, :5]
    assert np.abs(offsets).max() > 0.1
    # The pan offset moves within the episode; the others are drawn once and hold.
    pan = offsets[:, 0]
    assert pan.max() - pan.min() > 0.1
    assert np.abs(offsets[:, 1:] - offsets[0, 1:]).max() < 1e-4


def test_the_artifact_records_what_the_expert_believed_about_the_cube(
    closed_loop_run,
) -> None:
    """Kept for analysis: the believed pose is why the expert aimed where it did."""
    frames = closed_loop_run.result.frames
    episode = closed_loop_run.episode
    believed_start = frames.believed_cube_pose[0]
    assert believed_start[:2] == pytest.approx(
        [episode.believed_source.x, episode.believed_source.y], abs=1e-6
    )
    # It is a belief, so it differs from the true pose the same frame records.
    assert np.linalg.norm(believed_start[:2] - frames.true_cube_pose[0][:2]) > 1e-3
    # The descent's servo sees the cube, and its sightings land near the true pose.
    assert frames.sighted.any()
    seen = frames.wrist_sighting[frames.sighted]
    truth = frames.true_cube_pose[frames.sighted]
    assert np.abs(seen[:, :2] - truth[:, :2]).max() < 0.05


def test_the_actuators_are_commanded_the_believed_action_plus_the_offsets() -> None:
    """The other half of the same fact, read off ``data.ctrl`` mid-phase.

    Stopping between ticks leaves the actuators holding exactly what the last
    recorded row's action became on its way into physics.
    """
    episode = _episode(closed_loop=True)
    result, recording, _ = _run(episode, should_stop=_stop_after(20), verbose=False)
    assert result.status == "stopped"

    # The row was written, and the actuators loaded, at the tick before the last
    # step; the drawn pan offset wanders with time, so it is read at that instant.
    offsets = episode.miscalibration.offsets_rad(episode.data.time - 1.0 / CONTROL_HZ)
    action = recording.actions()[-1]
    for index, name in enumerate(ARM_JOINT_NAMES):
        commanded = episode.data.ctrl[episode.actuator_id[name]]
        believed = math.radians(action[index])
        assert commanded == pytest.approx(believed + offsets.get(name, 0.0), abs=1e-4)


def test_unfused_phase_boundaries_replan_from_the_believed_readback(
    closed_loop_run,
) -> None:
    """Only the boundaries the hardware executor checkpoints at are replanned.

    Approach flows into the servo descent, the descent rebuilds grasp and lift as
    one contact-critical section, and carry/drop_descent and drop_descent/release
    fly from the locked plan. What is left is the two boundaries below.
    """
    replanned = [
        line.removeprefix("Replanning remaining trajectory after ").rstrip(".")
        for line in closed_loop_run.log
        if line.startswith("Replanning remaining trajectory after ")
    ]
    assert replanned == ["lift", "release"]
    # A replan that changed the plan's shape would show up here as a repeated or
    # missing phase; it does not.
    assert [span.name for span in closed_loop_run.result.phase_spans] == list(PLANNED_PHASES)


def test_a_descent_that_never_sees_the_cube_retries_then_asks_for_a_restart(
    monkeypatch, capsys
) -> None:
    """Blind descent: back up to pregrasp, come in again, and give up rather than close blind."""
    from pick_and_place.runtime import sim_wrist_servo

    monkeypatch.setattr(sim_wrist_servo, "detect_cube_faces", lambda rgb, detector: [])
    result, recording, _ = _run(_episode(closed_loop=True), speed=4.0, verbose=True)
    assert result.status == "restart"
    out = capsys.readouterr().out
    assert "backing up to pregrasp" in out
    assert "without a cube detection" in out
    # The frames recorded up to the abort are still described by the spans.
    assert [span.name for span in result.phase_spans] == ["approach", "descent"]
    assert len(recording.rows) > 0


def test_a_checkpoint_with_no_clean_replan_asks_for_a_restart(monkeypatch, capsys) -> None:
    from pick_and_place.rollout import checkpoint

    monkeypatch.setattr(checkpoint, "replan_remaining_candidates", lambda *a, **k: iter(()))
    result, _, _ = _run(_episode(closed_loop=True), verbose=True)
    assert result.status == "restart"
    assert "No clean replan after" in capsys.readouterr().out


def test_the_mixed_wrist_view_blends_the_believed_world_under_the_true_one(
    monkeypatch,
) -> None:
    """The sim analog of the hardware overlay: the gap between the layers is the miscalibration."""
    import cv2

    shown: list[np.ndarray] = []
    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append(image))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    result = record_episode(
        _episode(closed_loop=True),
        speed=8.0,
        show_wrist_mixed=True,
        should_stop=_stop_after(12),
        verbose=False,
    )
    assert result.status == "stopped"
    # One window per tick, from the first tick on — the overlay is not descent-only.
    assert len(shown) == 12
    assert shown[0].ndim == 3 and shown[0].shape[2] == 3
