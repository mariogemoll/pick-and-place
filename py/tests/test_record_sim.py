# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np

from pick_and_place.sim_recorder import resize_and_center_crop


RECORD_SIM_PATH = Path(__file__).parents[1] / "scripts" / "pick_and_place" / "record_sim.py"


def _record_sim_module():
    spec = importlib.util.spec_from_file_location("record_sim", RECORD_SIM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_shards_uses_each_nonempty_shard_once(tmp_path, monkeypatch):
    module = _record_sim_module()
    roots = [tmp_path / "shard0", tmp_path / "shard1"]
    for root in roots:
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}")

    opened_roots = []

    class FakeDataset:
        def __init__(self, repo_id, root):
            opened_roots.append(root)
            self.meta = types.SimpleNamespace(total_episodes=1)

    merged = types.SimpleNamespace(meta=types.SimpleNamespace(total_episodes=2, total_frames=10))

    def fake_merge(datasets, **kwargs):
        assert [dataset.meta.total_episodes for dataset in datasets] == [1, 1]
        assert kwargs["output_dir"] == tmp_path / "merged"
        return merged

    lerobot = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    dataset_tools = types.ModuleType("lerobot.datasets.dataset_tools")
    dataset_tools.merge_datasets = fake_merge
    lerobot_dataset = types.ModuleType("lerobot.datasets.lerobot_dataset")
    lerobot_dataset.LeRobotDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.dataset_tools", dataset_tools)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", lerobot_dataset)

    module.merge_shards(
        [{"dataset_root": root, "repo_id": f"test/shard{i}"} for i, root in enumerate(roots)],
        output_root=tmp_path / "merged",
        output_repo_id="test/merged",
        keep_shards=True,
    )

    assert opened_roots == roots


def test_episode_rng_depends_only_on_root_seed_and_global_episode():
    module = _record_sim_module()

    first = module._episode_rng(17, 6).integers(2**31, size=4)
    repeated = module._episode_rng(17, 6).integers(2**31, size=4)
    neighboring = module._episode_rng(17, 7).integers(2**31, size=4)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, neighboring)


def test_shard_ranges_preserve_global_episode_rng_streams():
    module = _record_sim_module()
    counts = module._split(10, 3)

    first_episode = 0
    shard_streams = []
    for count in counts:
        shard_streams.extend(
            module._episode_rng(23, first_episode + index).integers(2**31)
            for index in range(count)
        )
        first_episode += count

    sequential_stream = [module._episode_rng(23, index).integers(2**31) for index in range(10)]
    assert shard_streams == sequential_stream


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
