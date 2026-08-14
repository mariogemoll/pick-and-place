# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Reuse SmolVLA's frozen vision tower output instead of recomputing it every epoch.

SmolVLA finetuning here freezes the whole VLM (``freeze_vision_encoder`` and
``train_expert_only`` are both true), so ``embed_image`` -- SigLIP over 1024
patches per camera, then the pixel-shuffle connector -- is a pure function of the
pixels. It is also 59% of a training step at batch 64. A run of N epochs pays for
it N times over and gets the same 64x960 token block back each time.

This module pays for it once. ``write_prefix_cache`` streams the dataset through
the tower and stores the tokens in a memory-mapped file; ``CachedPrefixDataset``
serves them in place of pixels, and ``patch_policy_for_cached_prefix`` teaches the
policy to accept them. Video is never decoded during training, so the run also
stops depending on how many cores the host really gave the container.

The cache is exact only when the images are not augmented, because a photometric
transform changes the pixels the tower sees. ``variants`` is the way to keep
augmentation: store several independently augmented draws per frame and pick one
at random per read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Cached tokens travel as an observation feature so that lerobot's batch/transition
# conversion carries them through: `batch_to_transition` keeps every key under
# `observation.` and drops everything else it does not recognize. The prefix must
# not be `observation.images.`, because `AddBatchDimensionObservationStep` treats a
# 3D tensor under that prefix as an unbatched image and would grow a leading axis.
EMBEDDING_PREFIX = "observation.prefix_embedding."

CACHE_MANIFEST_NAME = "prefix_cache.json"
CACHE_ARRAY_NAME = "prefix_cache.npy"


def embedding_key(camera_key: str) -> str:
    """Name the cached-token feature that stands in for one camera's pixels."""
    return EMBEDDING_PREFIX + camera_key.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class PrefixCacheSpec:
    """What one cache file holds, written beside it so a reader can check itself."""

    camera_keys: tuple[str, ...]
    num_frames: int
    variants: int
    num_tokens: int
    width: int
    dtype: str
    augmented: bool
    dataset_fingerprint: str

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_frames,
            self.variants,
            len(self.camera_keys),
            self.num_tokens,
            self.width,
        )

    def to_json(self) -> str:
        payload = {
            "camera_keys": list(self.camera_keys),
            "num_frames": self.num_frames,
            "variants": self.variants,
            "num_tokens": self.num_tokens,
            "width": self.width,
            "dtype": self.dtype,
            "augmented": self.augmented,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> PrefixCacheSpec:
        payload = json.loads(text)
        return cls(
            camera_keys=tuple(payload["camera_keys"]),
            num_frames=int(payload["num_frames"]),
            variants=int(payload["variants"]),
            num_tokens=int(payload["num_tokens"]),
            width=int(payload["width"]),
            dtype=str(payload["dtype"]),
            augmented=bool(payload["augmented"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
        )


def dataset_fingerprint(dataset: LeRobotDataset) -> str:
    """Identify the dataset a cache was built from, cheaply and without reading video."""
    meta = dataset.meta
    return f"{meta.total_episodes}ep-{meta.total_frames}f-{meta.fps}fps"


def cache_nbytes(spec: PrefixCacheSpec) -> int:
    return int(np.prod(spec.shape)) * _numpy_dtype(spec.dtype).itemsize


def _numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "bfloat16":
        # numpy has no bfloat16, so the file holds the raw 16 bits and the reader
        # views them back as bfloat16. Storing float16 instead would lose the
        # exponent range the tower's activations actually use.
        return np.dtype(np.uint16)
    return np.dtype(dtype)


def _torch_dtype(dtype: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _to_storage(tokens: torch.Tensor, dtype: str) -> np.ndarray:
    tokens = tokens.to(_torch_dtype(dtype)).cpu()
    if dtype == "bfloat16":
        return tokens.view(torch.uint16).numpy()
    return tokens.numpy()


def _from_storage(block: np.ndarray, dtype: str) -> torch.Tensor:
    # Copied out of the mapping: torch refuses to wrap a read-only array without
    # warning, and the block is 120 KiB against a step that moves megabytes.
    tensor = torch.from_numpy(np.array(block))
    if dtype == "bfloat16":
        return tensor.view(torch.bfloat16)
    return tensor


def tower_tokens(policy: SmolVLAPolicy, images: list[torch.Tensor]) -> torch.Tensor:
    """Run the frozen tower over one batch of prepared images.

    ``images`` is what ``SmolVLAPolicy.prepare_images`` returns: one [B, 3, H, W]
    tensor per camera, already resized and scaled into SigLIP's [-1, 1]. The result
    is [B, cameras, tokens, width], which is exactly what ``embed_prefix`` would
    have computed inline.
    """
    embed_image = policy.model.vlm_with_expert.embed_image
    with torch.no_grad():
        return torch.stack([embed_image(image) for image in images], dim=1)


def write_prefix_cache(
    policy: SmolVLAPolicy,
    dataset: LeRobotDataset,
    destination: Path,
    *,
    batch_size: int = 64,
    num_workers: int = 8,
    variants: int = 1,
    dtype: str = "bfloat16",
    device: torch.device | None = None,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
    progress: Callable[[int, int], None] | None = None,
) -> PrefixCacheSpec:
    """Stream ``dataset`` through the frozen tower and store the tokens under ``destination``.

    One pass per variant. With ``variants`` above one the dataset's own image
    transforms are expected to be enabled, so each pass sees a different draw and
    training can pick between them; with one variant an augmented dataset would
    freeze a single random draw per frame, which is a different experiment and is
    recorded in the manifest as ``augmented``.
    """
    from torch.utils.data import DataLoader

    if variants < 1:
        raise ValueError(f"variants must be positive, got {variants}")

    device = device or next(policy.parameters()).device
    camera_keys = tuple(key for key in policy.config.image_features)
    if not camera_keys:
        raise ValueError("policy declares no image features, so there is nothing to cache")

    probe = _probe_tower(policy, device, autocast_dtype)
    spec = PrefixCacheSpec(
        camera_keys=camera_keys,
        num_frames=dataset.num_frames,
        variants=variants,
        num_tokens=probe[0],
        width=probe[1],
        dtype=dtype,
        augmented=dataset.image_transforms is not None,
        dataset_fingerprint=dataset_fingerprint(dataset),
    )

    destination.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(
        destination / CACHE_ARRAY_NAME,
        mode="w+",
        dtype=_numpy_dtype(dtype),
        shape=spec.shape,
    )

    # The dataset is read in index order and the actions are not needed, so the
    # loader only has to keep up with the tower. Shuffling would cost random video
    # seeks for nothing.
    written = 0
    total = spec.num_frames * variants
    for variant in range(variants):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        offset = 0
        for batch in loader:
            images = _prepare_images_on_device(policy, batch, camera_keys, device)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                tokens = tower_tokens(policy, images)
            block = _to_storage(tokens, dtype)
            array[offset : offset + block.shape[0], variant] = block
            offset += block.shape[0]
            written += block.shape[0]
            if progress is not None:
                progress(written, total)
        if offset != spec.num_frames:
            raise RuntimeError(f"expected {spec.num_frames} frames, wrote {offset}")

    array.flush()
    del array
    (destination / CACHE_MANIFEST_NAME).write_text(spec.to_json())
    return spec


def _probe_tower(
    policy: SmolVLAPolicy,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[int, int]:
    """Ask the tower for its own token count and width rather than deriving them.

    Deliberately a blank image rather than a real frame. Reading one would decode
    video in this process, and `DatasetReader._query_videos` says in as many words
    not to do that when workers are coming: lerobot keeps open torchcodec decoders
    in a module-level cache, and a forked worker inheriting one fails with
    "Could not push packet to decoder: Invalid data found when processing input".
    Only the output shape is wanted here, and blank pixels carry it.
    """
    if policy.config.resize_imgs_with_padding is not None:
        height, width = policy.config.resize_imgs_with_padding
    else:
        height, width = next(iter(policy.config.image_features.values())).shape[1:]
    blank = torch.zeros(1, 3, height, width, device=device)
    with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        tokens = tower_tokens(policy, [blank])
    return int(tokens.shape[2]), int(tokens.shape[3])


def _prepare_images_on_device(
    policy: SmolVLAPolicy,
    batch: dict[str, Any],
    camera_keys: tuple[str, ...],
    device: torch.device,
) -> list[torch.Tensor]:
    """Apply SmolVLA's own image preprocessing, on the device the tower lives on."""
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    images = []
    for key in camera_keys:
        image = batch[key]
        image = image[:, -1] if image.ndim == 5 else image
        image = image.to(device, non_blocking=True)
        if policy.config.resize_imgs_with_padding is not None:
            image = resize_with_pad(image, *policy.config.resize_imgs_with_padding, pad_value=0)
        images.append(image * 2.0 - 1.0)
    return images


class CachedPrefixDataset(torch.utils.data.Dataset):
    """A LeRobot dataset that serves cached tower tokens and never decodes video.

    Everything except the camera streams comes from the wrapped dataset unchanged,
    so the action chunk, the padding flags, the task string and the sampling
    behaviour are lerobot's rather than a reimplementation of them.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        cache_dir: Path,
        *,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.spec = PrefixCacheSpec.from_json((cache_dir / CACHE_MANIFEST_NAME).read_text())
        if self.spec.num_frames != dataset.num_frames:
            raise ValueError(
                f"cache holds {self.spec.num_frames} frames but the dataset has "
                f"{dataset.num_frames}; rebuild it against this dataset"
            )
        if self.spec.dataset_fingerprint != dataset_fingerprint(dataset):
            raise ValueError(
                f"cache was built from {self.spec.dataset_fingerprint}, not "
                f"{dataset_fingerprint(dataset)}"
            )
        if dataset.image_transforms is not None:
            # Silently inert otherwise: the transforms run on pixels this dataset
            # never reads. Augmentation belongs in the cache, as several variants.
            raise ValueError(
                "image transforms are enabled but cached training never decodes an "
                "image. Build the cache with --augment --variants N instead, and "
                "train with dataset.image_transforms.enable=false."
            )
        self._cache_dir = cache_dir
        self._array: np.memmap | None = None
        self._seed = seed
        self._keys = tuple(embedding_key(key) for key in self.spec.camera_keys)
        _disable_video_decoding(dataset)

    # Forwarded so that lerobot's trainer, sampler and logging see a dataset.
    @property
    def meta(self):  # noqa: ANN201 - mirrors LeRobotDataset
        return self.dataset.meta

    @property
    def num_frames(self) -> int:
        return self.dataset.num_frames

    @property
    def num_episodes(self) -> int:
        return self.dataset.num_episodes

    @property
    def fps(self) -> int:
        return self.dataset.fps

    def __len__(self) -> int:
        return len(self.dataset)

    def _open(self) -> np.memmap:
        # Opened lazily so that each dataloader worker gets its own mapping rather
        # than inheriting a file object across the fork.
        if self._array is None:
            self._array = np.load(self._cache_dir / CACHE_ARRAY_NAME, mmap_mode="r")
        return self._array

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        array = self._open()
        if self.spec.variants == 1:
            variant = 0
        else:
            # A per-item draw, not a per-epoch one: two reads of the same frame in
            # one run should be able to disagree, which is the whole point of
            # having stored several.
            variant = int(torch.randint(self.spec.variants, (1,)).item())
        block = _from_storage(array[index, variant], self.spec.dtype)
        for position, key in enumerate(self._keys):
            item[key] = block[position]
        for camera_key in self.spec.camera_keys:
            item.pop(camera_key, None)
        return item


def _disable_video_decoding(dataset: LeRobotDataset) -> None:
    """Stop the wrapped dataset from decoding frames nobody is going to look at.

    lerobot decides this inside ``DatasetReader.get_item`` from
    ``meta.video_keys``, with no flag to turn it off, so the reader's video query
    is replaced with an empty answer. Patched on the class rather than the
    instance so that a forked dataloader worker inherits it.
    """
    from lerobot.datasets.dataset_reader import DatasetReader

    if not getattr(DatasetReader, "_pap_video_query_patched", False):
        original = DatasetReader._query_videos

        def _query_videos(self, query_timestamps, ep_idx):  # noqa: ANN001, ANN202
            if getattr(self, "_pap_skip_video", False):
                return {}
            return original(self, query_timestamps, ep_idx)

        DatasetReader._query_videos = _query_videos
        DatasetReader._pap_video_query_patched = True

    reader = dataset._ensure_reader()
    reader._pap_skip_video = True


def patch_policy_for_cached_prefix(policy: SmolVLAPolicy) -> None:
    """Teach one policy instance to read tower tokens out of the batch.

    Two methods move: ``prepare_images`` picks up the cached feature instead of
    pixels, and ``embed_image`` becomes the identity, so ``embed_prefix`` and
    everything below it run unchanged and the loss is the one the uncached path
    would have produced.
    """
    from lerobot.utils.constants import OBS_IMAGES  # noqa: F401  documents the key space

    keys = [embedding_key(key) for key in policy.config.image_features]
    if not keys:
        raise ValueError("policy declares no image features")

    def prepare_images(batch: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError(
                f"cached prefix tokens missing from the batch: {missing}. "
                "Wrap the dataset in CachedPrefixDataset, or train without --cached-prefix."
            )
        images = [batch[key] for key in keys]
        masks = [
            torch.ones(image.shape[0], dtype=torch.bool, device=image.device) for image in images
        ]
        return images, masks

    policy.prepare_images = prepare_images
    policy.model.vlm_with_expert.embed_image = _identity_embed_image


def _identity_embed_image(tokens: torch.Tensor) -> torch.Tensor:
    return tokens


def iter_cache_blocks(cache_dir: Path) -> Iterator[tuple[int, torch.Tensor]]:
    """Read a cache back frame by frame, for verification against a live tower."""
    spec = PrefixCacheSpec.from_json((cache_dir / CACHE_MANIFEST_NAME).read_text())
    array = np.load(cache_dir / CACHE_ARRAY_NAME, mmap_mode="r")
    for index in range(spec.num_frames):
        yield index, _from_storage(array[index, 0], spec.dtype)
