# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import sys
import types
from pathlib import Path

from pick_and_place import recording as recording_module
from pick_and_place.recording import RecordingSession


def test_ffmpeg_encoder_reports_stay_quiet_after_lerobot_restores_callback(monkeypatch):
    levels = []
    logging = types.SimpleNamespace(
        ERROR=16,
        set_level=levels.append,
        restore_default_callback=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(logging=logging))

    recording_module._silence_ffmpeg_encoder_reports()
    logging.restore_default_callback()

    assert levels == [logging.ERROR, logging.ERROR]


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
