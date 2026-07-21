# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import numpy as np

from pick_and_place.policy_controllers import OVERHEAD_FEATURE, STATE_FEATURE, WRIST_FEATURE
from pick_and_place.policy_real import PhysicalPolicyTick
from pick_and_place.policy_recording import PolicyRecordingSession


class StubSession:
    fps = 30.0

    def __init__(self):
        self.initialized = False
        self.created = None
        self.frames = []
        self.saved = 0
        self.saved_metadata = None

    def create_dataset(self, wrist_shape, overhead_shape, workspace_shape=None):
        self.created = (wrist_shape, overhead_shape, workspace_shape)
        self.initialized = True

    def record_frame(self, frame, **kwargs):
        self.frames.append((frame, kwargs))

    def save_episode(self, metadata=None):
        self.saved += 1
        self.saved_metadata = metadata

    def has_pending_frames(self):
        return bool(self.frames)

    def discard_episode(self):
        self.frames.clear()

    def finalize(self):
        pass


def test_policy_recording_keeps_images_state_and_action_on_one_tick():
    session = StubSession()
    recording = PolicyRecordingSession(session, "test task")
    observation = {
        STATE_FEATURE: np.arange(6, dtype=np.float32),
        OVERHEAD_FEATURE: np.zeros((480, 640, 3), dtype=np.uint8),
        WRIST_FEATURE: np.ones((480, 640, 3), dtype=np.uint8),
    }
    tick = PhysicalPolicyTick(
        index=1,
        scheduled_at=2.0,
        observed_at=2.01,
        observation=observation,
        requested_action=np.ones(6),
        command=np.full(6, 0.5),
        clamped=False,
        slew_limited=True,
    )

    recording.record_tick(tick)
    recording.commit()

    assert session.created == ((480, 640, 3), (480, 640, 3), None)
    frame, extras = session.frames[0]
    assert frame[OVERHEAD_FEATURE] is observation[OVERHEAD_FEATURE]
    assert frame[WRIST_FEATURE] is observation[WRIST_FEATURE]
    np.testing.assert_array_equal(frame["action"], tick.command)
    assert frame[STATE_FEATURE].dtype == np.float32
    assert frame["action"].dtype == np.float32
    assert extras["wall_t"] == 0.0
    assert session.saved == 1
    assert session.saved_metadata is None


def test_policy_recording_adds_workspace_frame_to_same_tick():
    session = StubSession()
    workspace = np.full((480, 640, 3), 2, dtype=np.uint8)
    recording = PolicyRecordingSession(session, "test task", workspace_rgb=lambda: workspace)
    observation = {
        STATE_FEATURE: np.arange(6, dtype=np.float32),
        OVERHEAD_FEATURE: np.zeros((480, 640, 3), dtype=np.uint8),
        WRIST_FEATURE: np.ones((480, 640, 3), dtype=np.uint8),
    }
    tick = PhysicalPolicyTick(
        index=1,
        scheduled_at=2.0,
        observed_at=2.01,
        observation=observation,
        requested_action=np.ones(6),
        command=np.full(6, 0.5),
        clamped=False,
        slew_limited=False,
    )

    recording.record_tick(tick)

    assert session.created == (
        (480, 640, 3),
        (480, 640, 3),
        (480, 640, 3),
    )
    frame, _ = session.frames[0]
    assert frame["observation.images.workspace"] is workspace


def test_policy_recording_commits_episode_metadata():
    session = StubSession()
    metadata = {"cube_start_x": 0.25, "target_x": 0.30}
    recording = PolicyRecordingSession(
        session,
        "test task",
        episode_metadata=lambda: metadata,
    )

    recording.commit()

    assert session.saved_metadata == metadata
