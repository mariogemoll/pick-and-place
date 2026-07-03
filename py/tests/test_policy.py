# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pick_and_place.policy import (
    DEFAULT_INSTRUCTION,
    OVERHEAD_FEATURE,
    WRIST_FEATURE,
    make_policy,
    resolve_checkpoint_cameras,
)


def test_default_instruction_matches_training_dataset():
    assert DEFAULT_INSTRUCTION == "Pick up the cube and place it at the target."


def test_make_policy_dispatches_by_type_and_keys_cameras_as_asked(monkeypatch):
    from lerobot.policies import factory
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig()

    class DummyPolicy:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        PreTrainedConfig, "from_pretrained", classmethod(lambda cls, checkpoint: config)
    )
    monkeypatch.setattr(
        SmolVLAPolicy,
        "from_pretrained",
        classmethod(lambda cls, checkpoint, *, config: DummyPolicy()),
    )
    monkeypatch.setattr(factory, "make_pre_post_processors", lambda **kwargs: (None, None))

    # SmolVLA keys its cameras camera1/camera2; the loader builds exactly those
    # keys, overhead first, dispatching the class from the config's type.
    keys = ("observation.images.camera1", "observation.images.camera2")
    make_policy("checkpoint", (512, 512), keys, "cpu")

    assert list(config.image_features) == list(keys)


def test_resolve_checkpoint_cameras_reads_keys_and_size_from_config(monkeypatch):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig()
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    }
    monkeypatch.setattr(
        PreTrainedConfig, "from_pretrained", classmethod(lambda cls, checkpoint: config)
    )

    hw, keys = resolve_checkpoint_cameras("checkpoint")
    assert hw == (512, 512)
    assert keys == ("observation.images.camera1", "observation.images.camera2")


def test_resolve_checkpoint_cameras_falls_back_for_base_checkpoint(monkeypatch):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig()  # base ships no input_features
    monkeypatch.setattr(
        PreTrainedConfig, "from_pretrained", classmethod(lambda cls, checkpoint: config)
    )

    hw, keys = resolve_checkpoint_cameras("checkpoint", override_hw=(480, 640))
    assert hw == (480, 640)
    assert keys == (OVERHEAD_FEATURE, WRIST_FEATURE)
