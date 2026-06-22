# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pick_and_place.vla import (
    DEFAULT_INSTRUCTION,
    OVERHEAD_FEATURE,
    WRIST_FEATURE,
    make_policy,
)


def test_default_instruction_matches_training_dataset():
    assert DEFAULT_INSTRUCTION == "Pick up the cube and place it at the target."


def test_make_policy_preserves_training_camera_order(monkeypatch):
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

    make_policy("checkpoint", (512, 512), (512, 512), "cpu")

    assert list(config.image_features) == [OVERHEAD_FEATURE, WRIST_FEATURE]
