# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pick_and_place.core.geometry import CubePose

from pick_and_place.variants.render import assert_rerenderable
from pick_and_place.variants.video import (
    DEFAULT_X264_CRF,
    DEFAULT_X264_KEYINT,
    STATS_PIXEL_STRIDE,
    ImageStatsAccumulator,
    VideoWriter,
    encode_decode,
    evenly_spaced,
    rewrite_image_stats,
    video_frame_count,
    x264_settings,
)
from pick_and_place.data.trajectory_artifact import (
    ARTIFACT_FILENAME,
    EpisodeFacts,
    TrajectoryArtifact,
    TrajectoryWriter,
    save_trajectory,
)
from pick_and_place.variants.appearance import (
    CUBE_COLOURS,
    SceneAppearance,
    SceneAppearanceOverride,
)
from pick_and_place.runtime.sim_recorder import build_recording_scene

RERENDER_PATH = Path(__file__).parents[1] / "scripts" / "rerender_episodes.py"

X264_OPTIONS = (
    b"x264 - core 165 - H.264/MPEG-4 AVC codec - options: cabac=1 ref=3 "
    b"deblock=1:0:0 8x8dct=1 keyint=2 keyint_min=1 scenecut=40 rc=crf crf=30.0 qcomp=0.60"
)


def _rerender_module():
    spec = importlib.util.spec_from_file_location("rerender_episodes", RERENDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so its dataclasses can resolve their own
    # module's annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episode_metadata(root: Path, columns: dict[str, list]) -> Path:
    path = root / "meta" / "episodes" / "chunk-000"
    path.mkdir(parents=True)
    pq.write_table(pa.table(columns), path / "file-000.parquet")
    return root


def _with_artifact(root: Path) -> Path:
    """Give a stub episode the trajectory artifact a real recording would write."""
    writer = TrajectoryWriter()
    writer.record(
        phase_name="approach",
        true_state=np.zeros(6),
        believed_state=np.zeros(6),
        action=np.zeros(6),
        true_cube_pose=np.array([0.3, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]),
        believed_cube_pose=CubePose(x=0.3, y=0.0, z=0.015),
        wrist_sighting=None,
    )
    save_trajectory(
        root / ARTIFACT_FILENAME,
        TrajectoryArtifact(
            frames=writer.frames(),
            facts=EpisodeFacts(
                target_xy=(0.25, 0.05),
                target_plate_yaw=0.0,
                verdict="success",
                phase_spans=writer.spans,
                fingerprint={},
            ),
        ),
    )
    return root


def test_x264_settings_are_read_from_the_video_the_recording_produced():
    """The re-encode has to match the recording, not the encoder's current default."""
    settings = x264_settings_from_bytes(X264_OPTIONS)

    assert settings.crf == 30.0
    assert settings.keyint == 2


def x264_settings_from_bytes(payload: bytes, tmp: Path | None = None):
    path = (tmp or Path(__import__("tempfile").mkdtemp())) / "video.mp4"
    path.write_bytes(b"\x00" * 64 + payload)
    return x264_settings(path)


def test_x264_settings_fall_back_when_the_options_string_is_absent(tmp_path):
    settings = x264_settings_from_bytes(b"no encoder metadata here", tmp_path)

    assert settings.crf == DEFAULT_X264_CRF
    assert settings.keyint == DEFAULT_X264_KEYINT


def test_video_writer_keeps_one_frame_per_recorded_row(tmp_path):
    """A video shorter than the parquet rows silently desyncs a whole episode."""
    path = tmp_path / "clip.mp4"
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for _ in range(7)]

    with VideoWriter(
        path, width=64, height=48, fps=30, settings=x264_settings_from_bytes(X264_OPTIONS, tmp_path)
    ) as writer:
        for frame in frames:
            writer.write(frame)

    assert video_frame_count(path) == len(frames)


def test_encode_decode_reports_the_codec_noise_floor():
    """A perfect re-render still differs from a recorded video by this much."""
    frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (40, 120, 200)]

    decoded = encode_decode(frames, fps=30, settings=x264_settings_from_bytes(X264_OPTIONS))

    assert len(decoded) == len(frames)
    for original, round_tripped in zip(frames, decoded, strict=True):
        assert np.abs(original.astype(int) - round_tripped.astype(int)).mean() < 5.0


def test_image_stats_match_the_recorded_sampling_stride():
    accumulator = ImageStatsAccumulator()
    frames = 4
    for _ in range(frames):
        accumulator.add(np.full((720, 960, 3), 128, dtype=np.uint8))

    stats = accumulator.result()

    expected = frames * (720 // STATS_PIXEL_STRIDE) * (960 // STATS_PIXEL_STRIDE)
    assert stats["count"] == [expected]
    assert np.asarray(stats["mean"]).ravel() == pytest.approx([128 / 255.0] * 3)
    assert set(stats) == {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}


def test_only_the_camera_statistics_are_rewritten(tmp_path):
    """The re-render changes pixels only, so every other recorded stat must survive."""
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps(
            {
                "observation.state": {"mean": [1.0]},
                "observation.images.wrist": {"mean": [[[0.5]]] * 3},
            }
        )
    )

    rewrite_image_stats(path, {"observation.images.wrist": {"mean": [[[0.25]]] * 3}})

    stats = json.loads(path.read_text())
    assert stats["observation.state"] == {"mean": [1.0]}
    assert stats["observation.images.wrist"] == {"mean": [[[0.25]]] * 3}


def test_episodes_without_a_trajectory_artifact_are_refused(tmp_path):
    """observation.state is the believed readback; the true pose lives only in the artifact."""
    root = _episode_metadata(
        tmp_path / "ep000000",
        {"episode_index": [0], "injected_offset_shoulder_pan_deg": [1.5]},
    )

    with pytest.raises(ValueError, match=ARTIFACT_FILENAME):
        assert_rerenderable(root)


def test_domain_randomized_episodes_are_accepted(tmp_path):
    """Their appearance draw is a look, and looks are what a variant pass replaces."""
    root = _with_artifact(
        _episode_metadata(
            tmp_path / "ep000001", {"episode_index": [1], "domain_sample_json": ["{}"]}
        )
    )

    assert_rerenderable(root)


def test_miscalibrated_episodes_carrying_an_artifact_are_accepted(tmp_path):
    """The whole point of the artifact: a drawn episode can be re-rendered."""
    root = _with_artifact(
        _episode_metadata(
            tmp_path / "ep000002",
            {"episode_index": [2], "injected_offset_shoulder_pan_deg": [1.5]},
        )
    )

    assert_rerenderable(root)


def test_plain_episodes_are_accepted(tmp_path):
    root = _with_artifact(
        _episode_metadata(tmp_path / "ep000003", {"episode_index": [3], "target_x": [0.3]})
    )

    assert_rerenderable(root)


def test_evenly_spaced_covers_both_ends():
    assert evenly_spaced(10, 3) == [0, 4, 9]
    assert evenly_spaced(4, 0) == [0, 1, 2, 3]
    assert evenly_spaced(3, 99) == [0, 1, 2]


def test_as_recorded_appearance_changes_nothing():
    assert SceneAppearance().is_as_recorded
    assert not SceneAppearance(cube="blue").is_as_recorded
    with pytest.raises(ValueError, match="unknown cube appearance"):
        SceneAppearance(cube="chartreuse")


def test_appearance_override_restores_the_recorded_colours():
    """Rendering several variants of one frame must not leak paint between them."""
    model, _ = build_recording_scene(render_width=64, render_height=64)
    override = SceneAppearanceOverride(model)
    cube_geom = override._cube_geom
    recorded_rgba = tuple(model.geom_rgba[cube_geom])
    recorded_matid = int(model.geom_matid[cube_geom])

    override.apply(SceneAppearance(cube="blue"))
    assert tuple(model.geom_rgba[cube_geom])[:3] == CUBE_COLOURS["blue"]
    assert model.geom_matid[cube_geom] == -1  # the AprilTag material is detached

    override.apply(SceneAppearance())
    assert tuple(model.geom_rgba[cube_geom]) == recorded_rgba
    assert int(model.geom_matid[cube_geom]) == recorded_matid


def test_recorded_appearance_survives_many_episodes_of_variants():
    """"As recorded" must mean the compiled scene, not the last variant painted.

    Rendering variants episode after episode from one model would otherwise
    drift: episode 2's "as recorded" would restore episode 1's last paint, and
    every variant would silently converge on whichever was applied last.
    """
    model, _ = build_recording_scene(render_width=64, render_height=64)
    override = SceneAppearanceOverride(model)
    floor_geom = override._floor_geom
    cube_geom = override._cube_geom
    recorded_floor = tuple(model.geom_rgba[floor_geom])
    recorded_cube = tuple(model.geom_rgba[cube_geom])

    for _ in range(3):  # stand in for successive episodes
        override.refresh_plate_baseline()
        for appearance in (
            SceneAppearance(),
            SceneAppearance(cube="blue"),
            SceneAppearance(floor="dark-gray", target="white"),
        ):
            override.apply(appearance)

    override.apply(SceneAppearance())
    assert tuple(model.geom_rgba[floor_geom]) == recorded_floor
    assert tuple(model.geom_rgba[cube_geom]) == recorded_cube


def test_variant_specs_name_their_own_staging_root(tmp_path):
    module = _rerender_module()

    preset = module.parse_variant("blue-cube")
    assert preset.appearance == SceneAppearance(cube="blue")
    assert module.variant_staging_root(preset, tmp_path).name == "blue-cube_episodes"

    ad_hoc = module.parse_variant("cube=blue,floor=dark-gray")
    assert ad_hoc.appearance == SceneAppearance(floor="dark-gray", cube="blue")
    assert ad_hoc.name == "cube-blue_floor-dark-gray"
