# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Shared plumbing for running a SmolVLA policy on this robot, sim or real.

A LeRobot policy expects a fixed observation contract: a proprioceptive state
vector, one image per camera keyed by name, and a language instruction. The
state and action are in the *real (hardware) frame* the dataset was recorded in
— arm joints in degrees, gripper as a 0-100 position — which is why a sim run
converts at its boundaries while a hardware run feeds the follower's readings
straight through.

SmolVLA keys cameras by their name in ``input_features``, so the training
``--rename_map`` and eval must agree on which physical camera fills each slot.
Following SmolVLA's convention that the main/overview camera comes first, the
overhead view is ``camera1`` and the wrist is ``camera2``.
"""

from __future__ import annotations

from pick_and_place.follower import JOINT_NAMES

OVERHEAD_FEATURE = "observation.images.camera1"
WRIST_FEATURE = "observation.images.camera2"
DEFAULT_CHECKPOINT = "lerobot/smolvla_base"
DEFAULT_INSTRUCTION = "Pick up the cube and place it at the target."


def select_device(requested: str):
    """Resolve ``auto`` to the best available torch device, or honor an explicit one."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_policy(
    checkpoint: str,
    wrist_hw: tuple[int, int],
    overhead_hw: tuple[int, int],
    device,
):
    """Load a SmolVLA checkpoint with feature specs for our 6-DOF arm and two
    cameras, plus its pre/post-processors.

    The saved config is loaded first so architectural settings and image-feature
    order remain identical to training. State/action and camera shapes are then
    specialized to this robot. SmolVLA pads state/action to fixed internal widths
    and resizes every camera image to its own square input.
    The normalization stats come from the checkpoint's own saved processor (the
    base ships its pretraining stats; a fine-tune saves the project dataset's),
    which is why the dataset stays in raw physical units.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    n_joints = len(JOINT_NAMES)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(n_joints,)),
        # Image order is part of SmolVLA's input contract. Preserve the order
        # used by training: camera1 (overhead), then camera2 (wrist).
        OVERHEAD_FEATURE: PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, overhead_hw[0], overhead_hw[1])
        ),
        WRIST_FEATURE: PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, wrist_hw[0], wrist_hw[1])
        ),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(n_joints,)),
    }
    config.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor
