# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pick_and_place import sim_dataset_staging as staging
from pick_and_place.sim_recorder import resize_and_center_crop


RECORD_SIM_PATH = Path(__file__).parents[1] / "scripts" / "pick_and_place" / "record_sim.py"


def _record_sim_module():
    spec = importlib.util.spec_from_file_location("record_sim", RECORD_SIM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode_dir(root: Path, name: str, *, complete: bool) -> Path:
    """Create a per-episode dataset dir; ``complete`` writes the finalize marker."""
    path = root / name
    (path / "meta").mkdir(parents=True)
    if complete:
        (path / "meta" / "info.json").write_text("{}")
    return path


def test_only_finalized_episode_dirs_are_merged(tmp_path):
    """A killed worker leaves a dir with no ``info.json``; it must be skipped.

    This is the property that makes a worker kill cost one episode instead of
    every episode that worker had banked: LeRobot only writes ``info.json``
    when the parquet writers are closed, so its presence marks a readable
    dataset.
    """
    module = _record_sim_module()
    _episode_dir(tmp_path, "ep000000", complete=True)
    _episode_dir(tmp_path, "ep000001", complete=False)  # killed mid-episode
    _episode_dir(tmp_path, "ep000002", complete=True)

    found = module.find_episode_datasets(tmp_path)

    assert [path.name for path in found] == ["ep000000", "ep000002"]


def test_episode_dirs_merge_in_global_index_order(tmp_path):
    """Merge order must follow the episode index, not worker completion order.

    Workers pull from a shared queue, so they finish out of order; ordering by
    index keeps a merged dataset's episode order reproducible.
    """
    module = _record_sim_module()
    for name in ("ep000010", "ep000002", "ep000001"):
        _episode_dir(tmp_path, name, complete=True)

    found = module.find_episode_datasets(tmp_path)

    assert [path.name for path in found] == ["ep000001", "ep000002", "ep000010"]


def test_find_episode_datasets_tolerates_a_missing_root(tmp_path):
    module = _record_sim_module()
    assert module.find_episode_datasets(tmp_path / "never_created") == []


def test_merge_episodes_passes_roots_through_in_order(tmp_path, monkeypatch):
    roots = [_episode_dir(tmp_path, f"ep00000{i}", complete=True) for i in range(3)]
    captured = {}

    def fake_aggregate(**kwargs):
        captured.update(kwargs)

    lerobot = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    aggregate = types.ModuleType("lerobot.datasets.aggregate")
    aggregate.aggregate_datasets = fake_aggregate
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.aggregate", aggregate)

    staging.merge_episodes(
        roots,
        output_root=tmp_path / "merged",
        output_repo_id="test/merged",
        keep_episodes=True,
    )

    assert captured["roots"] == roots
    assert captured["aggr_root"] == tmp_path / "merged"
    # keep_episodes=True must leave the staged episodes on disk.
    assert all(root.exists() for root in roots)


def test_merge_episodes_removes_staged_dirs_unless_kept(tmp_path, monkeypatch):
    roots = [_episode_dir(tmp_path, f"ep00000{i}", complete=True) for i in range(2)]

    lerobot = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    aggregate = types.ModuleType("lerobot.datasets.aggregate")
    aggregate.aggregate_datasets = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.aggregate", aggregate)

    staging.merge_episodes(
        roots,
        output_root=tmp_path / "merged",
        output_repo_id="test/merged",
        keep_episodes=False,
    )

    assert not any(root.exists() for root in roots)


def test_merge_episodes_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="empty episode selection"):
        staging.merge_episodes(
            [],
            output_root=Path("unused"),
            output_repo_id="test/merged",
            keep_episodes=False,
        )


def test_next_episode_index_advances_past_complete_and_partial_dirs(tmp_path):
    _episode_dir(tmp_path, "ep000003", complete=True)
    _episode_dir(tmp_path, "ep000007", complete=False)
    (tmp_path / "notes").mkdir()

    assert staging.next_episode_index(tmp_path) == 8


def test_top_up_requires_the_same_collection_configuration(tmp_path):
    config = {
        "seed": 42,
        "source_xy": (0.1, 0.2),
        "domain_randomization": {"sha256": "abc"},
    }
    staging.ensure_collection_config(tmp_path, config)
    staging.ensure_collection_config(tmp_path, config.copy())

    with pytest.raises(ValueError, match="seed"):
        staging.ensure_collection_config(
            tmp_path,
            {"seed": 43, "domain_randomization": {"sha256": "abc"}},
        )


def test_successful_episode_selection_uses_recorded_placement(tmp_path, monkeypatch):
    roots = [_episode_dir(tmp_path, f"ep00000{i}", complete=True) for i in range(3)]

    def fake_episodes(root):
        error = {"ep000000": 0.01, "ep000001": 0.06, "ep000002": 0.02}[root.name]
        return pd.DataFrame(
            {
                "placement_detected": [root.name != "ep000002"],
                "cube_end_x": [error],
                "cube_end_y": [0.0],
                "target_x": [0.0],
                "target_y": [0.0],
            }
        )

    monkeypatch.setattr(staging, "load_all_episodes", fake_episodes)

    assert staging.successful_episode_datasets(roots) == [roots[0]]


def test_episode_rng_depends_only_on_root_seed_and_global_episode():
    module = _record_sim_module()

    first = module._episode_rng(17, 6).integers(2**31, size=4)
    repeated = module._episode_rng(17, 6).integers(2**31, size=4)
    neighboring = module._episode_rng(17, 7).integers(2**31, size=4)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, neighboring)


def test_queue_order_does_not_change_what_each_episode_records():
    """Which worker pulls which index must not affect the episode's content.

    Under a shared queue, episodes are claimed in nondeterministic order. Both
    per-episode streams key off the global index alone, so an arbitrary
    interleaving must still reproduce the sequential streams exactly.
    """
    module = _record_sim_module()
    sequential = [module._episode_rng(23, index).integers(2**31) for index in range(10)]

    scrambled_order = [7, 0, 3, 9, 1, 8, 2, 6, 4, 5]
    out_of_order = {
        index: module._episode_rng(23, index).integers(2**31) for index in scrambled_order
    }

    assert [out_of_order[index] for index in range(10)] == sequential

    domain_sequential = [module._domain_seed(23, index) for index in range(10)]
    domain_scrambled = {index: module._domain_seed(23, index) for index in scrambled_order}
    assert [domain_scrambled[index] for index in range(10)] == domain_sequential


def test_a_requeued_episode_reproduces_the_same_draw():
    """The watchdog requeues a killed episode; the retry must be identical.

    Otherwise a wedge would silently change what episode index N contains,
    breaking the index-addressability a resume depends on.
    """
    module = _record_sim_module()

    first_attempt = module._episode_rng(11, 42).integers(2**31, size=4)
    after_requeue = module._episode_rng(11, 42).integers(2**31, size=4)

    np.testing.assert_array_equal(first_attempt, after_requeue)
    assert module._domain_seed(11, 42) == module._domain_seed(11, 42)


def test_resuming_at_an_offset_extends_the_run_instead_of_repeating_it():
    """``--first-episode`` must continue the interrupted run's seed stream.

    A resume is only worth anything if the episodes it records are ones the
    interrupted run had not reached. Both per-episode streams key off the global
    index, so offsetting the resume past the last index the original run reached
    has to yield episodes disjoint from everything already banked.
    """
    module = _record_sim_module()

    banked = [module._episode_rng(0, index).integers(2**31) for index in range(300)]
    resumed = [module._episode_rng(0, 300 + index).integers(2**31) for index in range(50)]
    assert not set(banked) & set(resumed)

    # The domain-randomization stream is keyed the same way, so it carries the
    # same guarantee -- otherwise a resume would repeat appearances already banked.
    banked_domain = {module._domain_seed(0, index) for index in range(300)}
    resumed_domain = {module._domain_seed(0, 300 + index) for index in range(50)}
    assert not banked_domain & resumed_domain


def test_recording_defaults_supersample_saved_frames():
    module = _record_sim_module()
    parameters = inspect.signature(module.run_recording).parameters

    assert parameters["image_width"].default == 960
    assert parameters["image_height"].default == 720
    assert parameters["render_width"].default == 1920
    assert parameters["render_height"].default == 1080


def test_recording_render_quality_focuses_a_larger_shadow_map():
    module = _record_sim_module()
    model = types.SimpleNamespace(
        vis=types.SimpleNamespace(
            quality=types.SimpleNamespace(shadowsize=4096, offsamples=4),
            map=types.SimpleNamespace(shadowscale=0.6),
        )
    )

    module._configure_render_quality(model)

    assert model.vis.quality.shadowsize == 8192
    assert model.vis.quality.offsamples == 8
    assert model.vis.map.shadowscale == 0.4


def test_resize_and_center_crop_downsamples_then_removes_the_sides():
    image = np.zeros((108, 192, 3), dtype=np.uint8)
    image[:, :16] = (255, 0, 0)
    image[:, -16:] = (0, 0, 255)

    result = resize_and_center_crop(image, 48, 64)

    assert result.shape == (48, 64, 3)
    assert result.max() == 0


def test_watchdog_flags_only_workers_past_the_deadline():
    module = _record_sim_module()
    now = 1000.0
    status = {
        0: (5, now - 10.0),    # healthy, well inside the limit
        1: (7, now - 400.0),   # wedged
        2: (9, now - 300.0),   # exactly at the limit, not past it
    }

    wedged = module.find_wedged_workers(status, [0, 1, 2], now=now, episode_timeout=300.0)

    assert [(wid, ep) for wid, ep, _ in wedged] == [(1, 7)]


def test_watchdog_never_kills_a_worker_between_episodes():
    """An idle worker reports ``None``; killing it would loop forever.

    Once the queue drains, workers sit idle before exiting. If idleness counted
    against the deadline the pool would kill and respawn workers indefinitely.
    """
    module = _record_sim_module()
    now = 1000.0
    status = {0: (None, now - 99999.0), 1: (None, now - 5.0)}

    assert module.find_wedged_workers(status, [0, 1], now=now, episode_timeout=300.0) == []


def test_watchdog_tolerates_a_worker_that_has_not_reported_yet():
    """A just-spawned worker may have no status entry; that is not a wedge."""
    module = _record_sim_module()
    now = 1000.0

    assert module.find_wedged_workers({}, [0, 1], now=now, episode_timeout=300.0) == []


def test_watchdog_reports_every_wedged_worker_not_just_the_first():
    """The doc records runs losing two workers at once."""
    module = _record_sim_module()
    now = 1000.0
    status = {0: (1, now - 900.0), 1: (2, now - 10.0), 2: (3, now - 600.0)}

    wedged = module.find_wedged_workers(status, [0, 1, 2], now=now, episode_timeout=300.0)

    assert [(wid, ep) for wid, ep, _ in wedged] == [(0, 1), (2, 3)]


def test_episode_timeout_default_leaves_room_for_resampling():
    """Nominal episode is ~35 s; the limit must not clip a slow-but-healthy one."""
    module = _record_sim_module()

    assert module.DEFAULT_EPISODE_TIMEOUT >= 35.0 * 5


def test_vcodec_defaults_to_software_h264():
    """`auto` probes for a HW encoder and silently picks the ~4x slower path."""
    module = _record_sim_module()
    parser = [
        line for line in inspect.getsource(module.main).splitlines() if '"--vcodec"' in line
    ]

    assert parser, "expected a --vcodec argument"
    assert 'default="h264"' in inspect.getsource(module.main)


def test_wedged_episode_is_abandoned_once_retries_run_out():
    """Unbounded requeuing would spin forever on a deterministically bad index."""
    module = _record_sim_module()
    attempts = {}

    decisions = [module.claim_retry(attempts, 42, 1) for _ in range(3)]

    assert decisions == [True, False, False]


def test_zero_retries_marks_a_wedged_episode_failed_immediately():
    module = _record_sim_module()
    attempts = {}

    assert module.claim_retry(attempts, 7, 0) is False


def test_retry_budget_is_tracked_per_episode():
    """One bad index must not consume another index's retry budget."""
    module = _record_sim_module()
    attempts = {}

    assert module.claim_retry(attempts, 1, 1) is True
    assert module.claim_retry(attempts, 2, 1) is True
    assert module.claim_retry(attempts, 1, 1) is False
    assert module.claim_retry(attempts, 2, 1) is False
