# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Driving a whole episode through the executor, against doubles for the rig.

The hardware execution path was taken to be untestable because it opens four USB
cameras and streams to a real arm. Neither is reachable from here, but neither is
required.

Two levels of double are used, and they buy different guarantees:

- With ``recording=None`` and ``wrist_camera=None`` the executor needs only a
  follower that accepts actions and a viewer that says it is running. That path
  is **deterministic** — the commanded joint stream is bit-identical run to run,
  which is what makes it a refactoring oracle: pin it once, then diff future
  runs against the pinned stream.
- With a looping fake capture and a recording spy, the wrist-servo and recording
  branches come under test too. Those run their own threads at rates unrelated
  to the control loop, so they are **not** bit-reproducible; the assertions are
  about behavior — that the descent consumes estimates, that a blind descent
  retries and gives up, that exactly one frame per tick reaches the dataset, and
  that an episode is committed or discarded according to how it ended.
"""

import contextlib
import io
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from pick_and_place.core.joint_frames import (
    action_to_joints,
    follower_clamp_limits,
    joints_to_action,
    sim_frame_to_real,
)
from pick_and_place.runtime.episodes import prepare_episode
from pick_and_place.rollout.real import execute_episode
from pick_and_place.runtime.ramp import RAMP_DURATION
from pick_and_place.spec.robot import CONTROL_HZ, HARDWARE_SIMULATION_HZ

WRIST_INTRINSICS = Path(__file__).parent / "fixtures" / "wrist_camera_intrinsics.json"
#: The executor opens the wrist camera at a fixed size and builds its undistort
#: map for it, so the double has to produce frames of that shape.
WRIST_FRAME_SHAPE = (720, 1280, 3)

# Compresses a ~9 s trajectory into ~1.6 s of ticks. The commanded end pose is
# the same at 4x; only how finely the trajectory is sampled changes.
SPEED = 8.0
#: Ticks the executor spends easing the arm onto the start pose before the first
#: phase. It checks the viewer once per tick, so a test that wants to close the
#: window *during* the trajectory has to outlast this.
RAMP_TICKS = round(RAMP_DURATION * CONTROL_HZ)
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


class FakeFollower:
    """Accepts every command and reports it straight back as the measured pose.

    Perfect tracking, so any command the executor sees on readback is one it sent.
    """

    def __init__(self, start: np.ndarray) -> None:
        self.pose = np.asarray(start, dtype=float)
        self.commands: list[np.ndarray] = []

    def get_observation(self) -> dict[str, float]:
        return joints_to_action(self.pose)

    def send_action(self, action: dict[str, float]) -> None:
        self.pose = action_to_joints(action, self.pose)
        self.commands.append(self.pose.copy())


class StubViewer:
    """Always running, never draws."""

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass


class ClosingViewer:
    """Reports itself closed from the ``at`` th check onward, as a window close would."""

    def __init__(self, at: int) -> None:
        self.at = at
        self.checks = 0

    def is_running(self) -> bool:
        self.checks += 1
        return self.checks < self.at

    def sync(self) -> None:
        pass


class LoopingCapture:
    """A ``VideoCapture`` stand-in that re-serves one image at a fixed rate.

    The rig's cameras free-run and nothing in the executor waits for a particular
    frame — every consumer samples whatever is newest — so a double only has to
    keep producing at a plausible rate. Unlike ``test_frame_reader``'s
    ``FakeCapture``, which releases frames one at a time to pin the reader in an
    exact state, this one just runs.
    """

    def __init__(self, shape: tuple[int, int, int], fps: float = 60.0) -> None:
        self.image = np.zeros(shape, dtype=np.uint8)
        self.period = 1.0 / fps
        self.released = False
        self.requested_fps: float | None = None

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(self.period)
        return True, self.image

    def isOpened(self) -> bool:
        return not self.released

    def set(self, prop: int, value: float) -> bool:
        self.requested_fps = value
        return True

    def get(self, prop: int) -> float:
        # Report the rate that was asked for, so the executor's mismatch warning
        # stays quiet and the log the tests read holds only phase lines.
        return float(CONTROL_HZ)

    def release(self) -> None:
        self.released = True


class RecordingSpy:
    """A ``RecordingSession`` stand-in that remembers what it was asked to write."""

    task = "pick up the cube and place it on the target"

    def __init__(self, dropped: int = 0) -> None:
        self.initialized = False
        self.created_with: tuple | None = None
        self.frames: list[dict] = []
        self.live_frames: dict[str, int] = {}
        self.servo_overlays: list[dict] = []
        self.lifecycle: list[str] = []
        self.saved_metadata: dict | None = None
        self.saved = False
        self.discarded = False
        self._dropped = dropped
        self._lock = threading.Lock()

    def create_dataset(self, wrist_shape, overhead_shape, workspace_shape) -> None:
        self.created_with = (wrist_shape, overhead_shape, workspace_shape)
        self.initialized = True

    def record_frame(self, frame, *, sim_qpos, wall_t, servo_active, servo_source) -> None:
        with self._lock:
            self.frames.append(
                {
                    "keys": tuple(sorted(frame)),
                    "wall_t": wall_t,
                    "servo_active": servo_active,
                    "servo_source": servo_source,
                }
            )

    def has_pending_frames(self) -> bool:
        return bool(self.frames)

    def discard_episode(self) -> None:
        self.discarded = True

    def record_live_frame(self, name: str, bgr: np.ndarray, captured_at: float) -> None:
        with self._lock:
            self.live_frames[name] = self.live_frames.get(name, 0) + 1

    def record_visual_servo_overlay(self, captured_at: float, primitives: dict) -> None:
        with self._lock:
            self.servo_overlays.append(primitives)

    def start_live_capture(self, wall_t: float) -> None:
        self.lifecycle.append("start_live")

    def stop_live_capture(self) -> None:
        self.lifecycle.append("stop_live")

    def start_audio_capture(self) -> None:
        if "start_audio" not in self.lifecycle:
            self.lifecycle.append("start_audio")

    def stop_audio_capture(self) -> None:
        self.lifecycle.append("stop_audio")

    def dropped_frame_count(self) -> int:
        return self._dropped

    def save_episode(self, episode_metadata: dict | None = None) -> None:
        self.saved = True
        self.saved_metadata = episode_metadata


class StubCubeTracker:
    """Answers every frame with one fixed world-frame pose, or with nothing.

    Standing in for ``CubeTracker`` keeps the AprilTag detector — and the need for
    a frame that actually contains tags — out of the test. What the executor cares
    about is only whether an estimate arrives.
    """

    def __init__(self, pose=None) -> None:
        self.detector = object()
        self.pose = pose
        self.updates = 0
        self.camera_poses: list[tuple[np.ndarray, np.ndarray]] = []

    def update(self, detections, camera_matrix, camera_position, camera_rotation, *, dist=None):
        self.updates += 1
        self.camera_poses.append((camera_position, camera_rotation))
        if self.pose is None:
            return None
        yaw = self.pose.yaw
        rotation = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return SimpleNamespace(
            position=np.array([self.pose.x, self.pose.y, self.pose.z]), rotation=rotation
        )


def _detection(tag_id: int):
    """A ``detect_cube_faces`` result, carrying only what the executor reads off it."""
    return SimpleNamespace(
        tag_id=tag_id, corners=[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
    )


def _install_camera_doubles(monkeypatch, tracker: StubCubeTracker, detections=()):
    """Point the executor's wrist-camera open path at doubles, and hand back the capture.

    Three seams are replaced: opening the device, constructing the tracker, and
    the tag detector the servo worker runs per frame. Intrinsics are left real —
    a committed fixture file goes through the actual loader, so the undistort map
    the worker remaps with is one OpenCV built.
    """
    from pick_and_place.perception import cube_detection
    from pick_and_place.runtime import wrist_servo

    capture = LoopingCapture(WRIST_FRAME_SHAPE)
    monkeypatch.setattr(wrist_servo, "open_capture", lambda *a, **k: capture)
    monkeypatch.setattr(cube_detection, "CubeTracker", lambda **kwargs: tracker)
    monkeypatch.setattr(cube_detection, "detect_cube_faces", lambda rgb, det: list(detections))
    return capture


def _episode(seed: int = 0, free_grasp: bool = False):
    episode = prepare_episode(
        np.random.default_rng(seed), max_attempts=40, free_grasp=free_grasp
    )
    # The hardware path steps the sim in whole control ticks, so the model has to
    # run at the hardware substep rate. Every caller sets this after preparing.
    episode.model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    return episode


class Run(NamedTuple):
    """What one executor run leaves behind that a test can look at."""

    status: str
    follower: "FakeFollower"
    #: Phase names in the order they were entered, from the printed log. This is
    #: the only external evidence of the control flow, and it is what
    #: distinguishes a straight playback from one that replanned.
    executed: list[str]
    log: str

    @property
    def control_ticks(self) -> int:
        """Ticks of the control loop, from the tracking summary printed at teardown.

        Not the same as the number of commands the follower saw: the ramp onto the
        start pose sends commands too, and those are not part of the trajectory.
        """
        match = re.search(r"\((\d+) samples over ", self.log)
        assert match is not None, f"no tracking summary in the log:\n{self.log}"
        return int(match.group(1))


def _run(episode, viewer=None, **kwargs) -> Run:
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = execute_episode(
            episode,
            follower=follower,
            viewer=viewer if viewer is not None else StubViewer(),
            speed=SPEED,
            **kwargs,
        )
    log = out.getvalue()
    executed = [
        line.split(": ", 1)[1]
        for line in log.splitlines()
        if line.startswith("Executing phase: ")
    ]
    return Run(status, follower, executed, log)


@pytest.fixture(scope="module")
def completed_run():
    """One full episode, shared by the assertions that only read its result."""
    episode = _episode()
    return episode, _run(episode)


def test_a_prepared_episode_runs_to_completion(completed_run) -> None:
    _, run = completed_run

    assert run.status == "success"
    assert len(run.follower.commands) > 0


def test_every_phase_of_the_plan_is_executed(completed_run) -> None:
    """Each planned phase is entered, in order. A checkpoint replan may re-enter
    one, so the executed log is a supersequence rather than an exact match."""
    episode, run = completed_run

    assert tuple(phase.name for phase in episode.trajectory.phases) == PLANNED_PHASES
    remaining = list(run.executed)
    for name in PLANNED_PHASES:
        assert name in remaining, f"phase {name} never executed: {run.executed}"
        remaining = remaining[remaining.index(name) + 1 :]


def test_the_arm_is_driven_the_whole_way(completed_run) -> None:
    """A run that ramped on and then stalled would still return success, so assert
    the commands actually move and end somewhere other than they started."""
    _, run = completed_run
    commands = np.asarray(run.follower.commands)

    assert not np.allclose(commands[0], commands[-1])


def test_no_command_leaves_the_follower_limits(completed_run) -> None:
    """Every command goes through ``clamp_and_warn``; nothing may reach the servos
    outside their range even when the planner asks for it."""
    episode, run = completed_run
    low, high = follower_clamp_limits(episode.kinematics)
    commands = np.asarray(run.follower.commands)

    assert np.all(commands >= low - 1e-9)
    assert np.all(commands <= high + 1e-9)


def test_the_run_is_reproducible() -> None:
    """The oracle a refactor is checked against: same seed, same command stream,
    bit for bit. Wall-clock pacing must not reach the commanded values."""
    first = _run(_episode())
    second = _run(_episode())

    assert first.status == second.status
    np.testing.assert_array_equal(
        np.asarray(first.follower.commands), np.asarray(second.follower.commands)
    )


def test_closing_the_viewer_during_the_ramp_never_starts_the_trajectory() -> None:
    """The ramp onto the start pose is 60 ticks and checks the window every one, so
    a close there aborts before the first phase — commands were sent, none of them
    the trajectory's."""
    run = _run(_episode(), viewer=ClosingViewer(at=RAMP_TICKS - 20))

    assert run.status == "restart"
    assert len(run.follower.commands) > 0
    assert run.executed == []


def test_closing_the_viewer_mid_episode_asks_the_caller_to_restart() -> None:
    """A closed window aborts the trajectory, and an aborted trajectory is not a
    success — the caller has to re-home rather than record it."""
    run = _run(_episode(), viewer=ClosingViewer(at=RAMP_TICKS + 20))

    assert run.status == "restart"
    assert run.executed != []
    assert 0 < run.control_ticks < 20


def test_speed_must_be_positive() -> None:
    episode = _episode()
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))

    with pytest.raises(ValueError, match="speed must be positive"):
        execute_episode(episode, follower=follower, viewer=StubViewer(), speed=0.0)


def test_recording_rest_to_rest_needs_a_recording() -> None:
    """The flag only widens what gets written, so asking for it without a dataset
    is a caller mistake rather than a no-op."""
    episode = _episode()
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))

    with pytest.raises(ValueError, match="record_rest_to_rest requires recording"):
        execute_episode(
            episode,
            follower=follower,
            viewer=StubViewer(),
            speed=SPEED,
            record_rest_to_rest=True,
        )


def test_a_model_that_cannot_hit_the_control_rate_is_rejected() -> None:
    """``prepare_episode`` leaves the model at the sim timestep, which does not
    divide the control period. Running anyway would silently drift the sim clock
    away from the tick clock, so the executor refuses."""
    episode = prepare_episode(np.random.default_rng(0), max_attempts=40)
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))

    with pytest.raises(ValueError, match="cannot produce"):
        execute_episode(episode, follower=follower, viewer=StubViewer(), speed=SPEED)


# --- The wrist-servo branch -------------------------------------------------
#
# Reached only when a wrist camera is open. The worker runs on its own thread at
# the camera's rate, so how many estimates a given tick sees is not fixed; these
# pin what the branch does, not how many times it does it.


def _servo_run(episode, tracker, detections=(), monkeypatch=None, **kwargs):
    _install_camera_doubles(monkeypatch, tracker, detections)
    return _run(
        episode,
        wrist_camera="0",
        wrist_intrinsics=str(WRIST_INTRINSICS),
        **kwargs,
    )


def test_the_descent_servo_consumes_estimates_and_tracks_the_wrist_camera(monkeypatch) -> None:
    """During the descent the loop publishes the wrist camera's live sim pose and
    the worker answers with cube estimates. Both directions have to be live: a
    stale camera pose would map the detection to the wrong world point."""
    episode = _episode()
    tracker = StubCubeTracker(pose=episode.source)

    run = _servo_run(episode, tracker, detections=[_detection(0)], monkeypatch=monkeypatch)

    assert run.status == "success"
    assert "descent" in run.executed
    assert tracker.updates > 0
    # Every published pose is a real camera frame, not the zero default.
    assert all(pos is not None and rot.shape == (3, 3) for pos, rot in tracker.camera_poses)
    assert any(np.any(pos != 0.0) for pos, _ in tracker.camera_poses)


def test_a_descent_that_never_sees_the_cube_retries_then_asks_for_a_restart(monkeypatch) -> None:
    """The jaws can hide every tag on the way down. The executor backs up to
    pregrasp and retries a bounded number of times, then gives up rather than
    closing blind on a cube it cannot see."""
    episode = _episode()
    tracker = StubCubeTracker(pose=None)

    run = _servo_run(episode, tracker, monkeypatch=monkeypatch)

    assert run.status == "restart"
    assert "backing up to pregrasp and retrying" in run.log
    assert "without a cube detection" in run.log


def test_no_wrist_camera_leaves_the_descent_feedforward() -> None:
    """The same episode with no camera runs the descent open-loop and still
    completes — the servo is a correction, not a precondition."""
    run = _run(_episode())

    assert run.status == "success"
    assert "descent" in run.executed


# --- The recording branch ---------------------------------------------------


def _recorded_run(episode, monkeypatch, *, tracker=None, viewer=None, spy=None, **kwargs):
    """Run with a wrist and overhead camera and a recording spy attached."""
    tracker = tracker if tracker is not None else StubCubeTracker(pose=episode.source)
    _install_camera_doubles(monkeypatch, tracker, [_detection(0)])
    spy = spy if spy is not None else RecordingSpy()
    overhead = LoopingCapture((480, 640, 3))
    run = _run(
        episode,
        viewer=viewer,
        recording=spy,
        overhead_camera_cap=overhead,
        wrist_camera="0",
        wrist_intrinsics=str(WRIST_INTRINSICS),
        **kwargs,
    )
    return run, spy, overhead


def test_a_recorded_episode_writes_one_frame_per_tick_and_is_committed(monkeypatch) -> None:
    """The dataset row count is the contract: one frame per camera per control
    tick, in order, and the episode is saved only once the writer has drained."""
    episode = _episode()

    run, spy, _ = _recorded_run(episode, monkeypatch)

    assert run.status == "success"
    assert spy.saved and not spy.discarded
    assert len(spy.frames) == run.control_ticks
    assert all(
        frame["keys"] == ("action", "observation.images.overhead", "observation.images.wrist",
                          "observation.state", "task")
        for frame in spy.frames
    )
    # Wall clocks are stamped on the control loop, so they only ever advance.
    stamps = [frame["wall_t"] for frame in spy.frames]
    assert stamps == sorted(stamps)


def test_recording_creates_the_dataset_from_the_first_real_frame_shapes(monkeypatch) -> None:
    """Feature shapes come from frames that actually arrived, so a camera that
    ignored the requested resolution still produces a self-consistent dataset."""
    episode = _episode()

    _, spy, _ = _recorded_run(episode, monkeypatch)

    wrist_shape, overhead_shape, workspace_shape = spy.created_with
    assert wrist_shape == WRIST_FRAME_SHAPE
    assert overhead_shape == (480, 640, 3)
    assert workspace_shape is None


def test_a_workspace_camera_adds_a_third_stream(monkeypatch) -> None:
    episode = _episode()
    workspace = LoopingCapture((240, 320, 3))

    _, spy, _ = _recorded_run(episode, monkeypatch, workspace_camera_cap=workspace)

    assert spy.created_with[2] == (240, 320, 3)
    assert all("observation.images.workspace" in frame["keys"] for frame in spy.frames)
    assert spy.live_frames.get("workspace", 0) > 0


def test_an_aborted_recording_is_discarded_rather_than_committed(monkeypatch) -> None:
    """A closed viewer ends the run as ``restart``. Half an episode in the dataset
    is worse than none, so it is dropped."""
    episode = _episode()

    run, spy, _ = _recorded_run(
        episode, monkeypatch, viewer=ClosingViewer(at=RAMP_TICKS + 20)
    )

    assert run.status == "restart"
    assert spy.frames, "the run has to reach the control loop for there to be anything to drop"
    assert spy.discarded and not spy.saved


def test_a_dropped_encoder_frame_fails_instead_of_saving_a_desynced_episode(
    monkeypatch,
) -> None:
    """A drop desyncs the video from the recorded rows. That is unrecoverable
    after the fact, so it has to fail before the episode is committed."""
    episode = _episode()

    with pytest.raises(RuntimeError, match="dropped 1 frame"):
        _recorded_run(episode, monkeypatch, spy=RecordingSpy(dropped=1))


def test_recording_needs_both_the_wrist_and_overhead_cameras(monkeypatch) -> None:
    """Two of the three streams are part of the dataset schema, so a missing one
    is a caller mistake rather than a narrower recording."""
    episode = _episode()
    tracker = StubCubeTracker(pose=episode.source)
    _install_camera_doubles(monkeypatch, tracker)
    follower = FakeFollower(sim_frame_to_real(episode.start_joints, episode.start_gripper))

    with pytest.raises(RuntimeError, match="wrist and overhead cameras"):
        with contextlib.redirect_stdout(io.StringIO()):
            execute_episode(
                episode,
                follower=follower,
                viewer=StubViewer(),
                speed=SPEED,
                recording=RecordingSpy(),
                wrist_camera="0",
                wrist_intrinsics=str(WRIST_INTRINSICS),
            )


def test_the_cameras_full_rate_is_logged_alongside_the_tick_sampled_frames(
    monkeypatch,
) -> None:
    """``record_live_frame`` fires per capture, not per tick, which is how the
    continuous native-rate footage is kept while the dataset samples at 30 Hz."""
    episode = _episode()

    _, spy, _ = _recorded_run(episode, monkeypatch)

    assert spy.live_frames.get("wrist", 0) > 0
    assert spy.live_frames.get("overhead", 0) > 0
    assert spy.lifecycle.count("start_live") == 1
    assert spy.lifecycle[-1] in ("stop_live", "stop_audio")


def test_rest_to_rest_recording_brackets_the_episode_with_the_parking_moves(
    monkeypatch,
) -> None:
    """The wider recording includes the ramp on from REST and the return to it, so
    it holds strictly more frames than the trajectory alone."""
    episode = _episode()
    narrow_spy = RecordingSpy()
    _recorded_run(episode, monkeypatch, spy=narrow_spy)

    wide_episode = _episode()
    wide_spy = RecordingSpy()
    run, _, _ = _recorded_run(
        wide_episode, monkeypatch, spy=wide_spy, record_rest_to_rest=True
    )

    assert run.status == "success"
    assert len(wide_spy.frames) > len(narrow_spy.frames)


def test_solved_frames_are_reported_to_the_recording_without_a_preview_window(
    monkeypatch,
) -> None:
    """The overlay log is what the replay video draws the wrist tile from, so it
    has to hold the frames the servo *solved* — which are the interesting ones.
    Those used to be reported only when a preview window happened to be open,
    because the wireframe was a by-product of drawing it."""
    episode = _episode()

    _, spy, _ = _recorded_run(episode, monkeypatch)

    assert spy.servo_overlays, "no overlays recorded at all"
    solved = [o for o in spy.servo_overlays if "cube_edges" in o]
    assert solved, "solved frames reported no cube wireframe"
    assert all(len(o["cube_edges"]) == 12 for o in solved)
    assert all("tags" in o for o in spy.servo_overlays)


def test_unsolved_frames_report_their_tags_and_no_wireframe(monkeypatch) -> None:
    """A frame whose tags were not enough to fix a pose still says what it saw —
    that is how a missed detection is told apart from a missing frame."""
    episode = _episode()

    _, spy, _ = _recorded_run(episode, monkeypatch, tracker=StubCubeTracker(pose=None))

    assert spy.servo_overlays
    assert all("cube_edges" not in o for o in spy.servo_overlays)


def test_the_caller_keeps_the_cameras_it_opened(monkeypatch) -> None:
    """The executor opens the wrist camera and releases it; the overhead capture
    belongs to the caller across episodes and must survive."""
    episode = _episode()

    _, _, overhead = _recorded_run(episode, monkeypatch)

    assert not overhead.released
