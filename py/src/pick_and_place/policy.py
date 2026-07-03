# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Load a LeRobot policy and run it closed-loop on this robot, sim or real.

A LeRobot policy expects a fixed observation contract: a proprioceptive state
vector, one image per camera keyed by name, and a language instruction. The
state and action are in the *real (hardware) frame* the dataset was recorded in
-- arm joints in degrees, gripper as a 0-100 position -- which is why a sim run
converts at its boundaries while a hardware run feeds the follower's readings
straight through.

Both the camera keys and the image resolution are read from the checkpoint's
saved config, because they are properties of how the policy was trained: a
SmolVLA fine-tune keys its cameras ``camera1``/``camera2`` (SmolVLA's naming
convention, applied via the training rename map) and may use a square input,
while an ACT model trained on the native dataset keys them ``overhead``/``wrist``
at the dataset resolution. The concrete policy class is likewise resolved from
the config's ``type``, so the same loader serves ACT, SmolVLA, or any other
LeRobot policy -- the checkpoint carries its own architecture and normalization
stats, and the dataset stays in raw physical units.
"""

from __future__ import annotations

from pick_and_place.follower import JOINT_NAMES

# Camera keys used only as a fallback for an un-finetuned base checkpoint, which
# pins none of its own. A fine-tune's own keys (e.g. SmolVLA's camera1/camera2)
# take precedence. The overhead view is first, matching the recorded order.
OVERHEAD_FEATURE = "observation.images.overhead"
WRIST_FEATURE = "observation.images.wrist"
DEFAULT_CHECKPOINT = "lerobot/smolvla_base"
DEFAULT_INSTRUCTION = "Pick up the cube and place it at the target."
# Fallback image size (height, width) when the checkpoint doesn't pin one, i.e.
# the dataset's native resolution.
DEFAULT_IMAGE_HW = (480, 640)


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


def resolve_checkpoint_cameras(
    checkpoint: str, override_hw: tuple[int, int] | None = None
) -> tuple[tuple[int, int], tuple[str, str]]:
    """Read the (overhead, wrist) camera keys and image size the checkpoint was
    trained on, so the runner feeds the policy exactly what it learned on.

    A fine-tune's saved config lists its two image features in recorded order
    (overhead first); their names and shape are used verbatim -- ``camera1``/
    ``camera2`` at whatever resolution for a SmolVLA fine-tune, ``overhead``/
    ``wrist`` at the dataset resolution for an ACT model. An un-finetuned base
    checkpoint pins neither, so the native keys and ``DEFAULT_IMAGE_HW`` are used.
    ``override_hw`` forces the resolution while still taking the keys from the
    checkpoint.
    """
    from lerobot.configs.types import FeatureType
    from lerobot.configs.policies import PreTrainedConfig
    import lerobot.policies  # noqa: F401  registers policy choice types (act, smolvla, ...) with draccus

    config = PreTrainedConfig.from_pretrained(checkpoint)
    visual = [
        (name, feature)
        for name, feature in (getattr(config, "input_features", None) or {}).items()
        if getattr(feature, "type", None) == FeatureType.VISUAL and len(feature.shape) == 3
    ]
    if len(visual) == 2:
        names = [name for name, _ in visual]
        overhead = next((n for n in names if "overhead" in n), None)
        wrist = next((n for n in names if "wrist" in n), None)
        if overhead is not None and wrist is not None:
            # ACT keeps the dataset's descriptive names, so match by name -- the
            # order features happen to appear in the config is not reliable.
            keys = (overhead, wrist)
        else:
            # SmolVLA-style renamed keys (camera1/camera2) carry no view in the
            # name; fall back to recorded order, which puts overhead first.
            keys = (names[0], names[1])
    else:
        keys = (OVERHEAD_FEATURE, WRIST_FEATURE)
    if override_hw is not None:
        hw = override_hw
    elif visual:
        shape = visual[0][1].shape
        hw = (int(shape[1]), int(shape[2]))
    else:
        hw = DEFAULT_IMAGE_HW
    return hw, keys


def make_policy(
    checkpoint: str,
    image_hw: tuple[int, int],
    image_keys: tuple[str, str],
    device,
):
    """Load a LeRobot policy checkpoint with feature specs for our 6-DOF arm and
    two cameras, plus its pre/post-processors.

    The saved config is loaded first so architectural settings stay identical to
    training, then state/action and the two camera shapes are specialized to this
    robot. The concrete policy class is looked up from the config's ``type``, so
    ACT, SmolVLA, and friends all load through this one path. ``image_hw`` and
    ``image_keys`` are the (height, width) and (overhead, wrist) feature names the
    cameras are fed under; take both from :func:`resolve_checkpoint_cameras` so
    they match what the policy learned on. Normalization stats come from the
    checkpoint's own saved processor, which is why the dataset stays in raw
    physical units.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    n_joints = len(JOINT_NAMES)
    height, width = image_hw
    config = PreTrainedConfig.from_pretrained(checkpoint)

    # The checkpoint's own image-feature order must be preserved: ACT stacks
    # camera tokens in config.input_features order, so that order has to match
    # training (an ACT config may list wrist before overhead). A fine-tune
    # already lists its two cameras in trained order; only a base checkpoint,
    # which pins no image features, needs the fallback keys from image_keys.
    # Either way the runner feeds each frame under its key by name, so the
    # stack order here and the frame-to-key mapping there stay independent.
    existing = [
        name
        for name, feature in (getattr(config, "input_features", None) or {}).items()
        if getattr(feature, "type", None) == FeatureType.VISUAL and len(feature.shape) == 3
    ]
    camera_keys = existing if len(existing) == 2 else list(image_keys)

    input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(n_joints,))}
    for name in camera_keys:
        input_features[name] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
    config.input_features = input_features
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(n_joints,)),
    }
    config.device = str(device)

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(checkpoint, config=config)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor
