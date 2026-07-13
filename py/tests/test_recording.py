# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pathlib import Path

from pick_and_place.recording import RecordingSession


def test_recording_session_adds_custom_metadata_to_episode():
    saved_metadata = []

    class FakeMeta:
        def save_episode(self, *args):
            saved_metadata.append(args[-1])

    class FakeDataset:
        def __init__(self):
            self.meta = FakeMeta()

        def save_episode(self):
            self.meta.save_episode(0, 1, ["pick cube"], {}, {"base": "value"})

    recording = RecordingSession("test/recording", Path("/tmp/recording"), "pick cube", 30.0)
    recording.dataset = FakeDataset()

    recording.save_episode({"placement_success": True})

    assert saved_metadata == [{"base": "value", "placement_success": True}]
