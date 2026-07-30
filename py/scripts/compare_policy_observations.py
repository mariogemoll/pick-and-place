#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Diff the images a closed-loop run feeds a policy against the ones it trained on.

A diffusion policy that imitates its training chunks almost exactly and still
does nothing useful in closed loop is usually not looking at the images it was
trained on. This command puts the two side by side at the tensor the policy
actually consumes: 96x96 uint8, cameras concatenated the way the exporter wrote
them.

Three images are produced per frame, all from the same recorded ground truth
(arm joints and cube pose from the source episode, drop-zone marker at the
episode's target):

``train``
    the exported training tensor itself, read straight out of ``train.npz`` --
    what the policy saw during training, H.264 loss and all.
``rerender``
    re-rendered here through ``build_recording_scene``, the path the training
    videos were produced by. Its difference from ``train`` is the codec's own
    loss plus any machine-to-machine render difference.
``closed-loop``
    rendered through ``run_policy_sim``'s scene, the path that feeds the policy
    at rollout time.

The comparison that matters is ``closed-loop`` against ``train``: it is the
distribution shift the policy actually experiences, and it is only meaningful
next to the ``rerender`` difference, which measures the same frame's
unavoidable noise floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pick_and_place.camera_extrinsics import load_local_camera_extrinsics
from pick_and_place.camera_intrinsics import load_local_camera_intrinsics
from pick_and_place.episode_rerender import EpisodeRenderer
from pick_and_place.episodes import set_joint
from pick_and_place.follower import ARM_JOINT_NAMES, real_frame_to_sim
from pick_and_place.paper_detection import DROP_ZONE_HALF_SIZE, place_paper_target_marker
from pick_and_place.policy_sim import build_policy_sim_model
from pick_and_place.scene_appearance import SceneAppearanceOverride, parse_appearance
from pick_and_place.scene_visibility import load_episode_truth, video_render_hw
from pick_and_place.sim_dataset_staging import episode_index, find_episode_datasets
from pick_and_place.sim_recorder import (
    OVERHEAD_CAMERA,
    WRIST_CAMERA,
    configure_render_quality,
    resize_and_center_crop,
)
from pick_and_place.workspace_overlays import is_cube_drop_allowed

CAMERAS = (OVERHEAD_CAMERA, WRIST_CAMERA)
# The exporter concatenates cameras in camera_features order, three channels each.
CHANNELS_PER_CAMERA = 3


@dataclass(frozen=True)
class FrameImages:
    """One frame's 96x96 RGB image from each of the three paths, per camera."""

    train: dict[str, np.ndarray]
    rerender: dict[str, np.ndarray]
    closed_loop: dict[str, np.ndarray]


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute per-pixel difference in grey levels."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def train_frame_images(images: np.ndarray, index: int) -> dict[str, np.ndarray]:
    """Split one stitched NCHW training row back into per-camera HWC RGB."""
    row = images[index]
    return {
        camera: row[i * CHANNELS_PER_CAMERA : (i + 1) * CHANNELS_PER_CAMERA].transpose(1, 2, 0)
        for i, camera in enumerate(CAMERAS)
    }


def episode_offsets(traj_lengths: np.ndarray) -> np.ndarray:
    """Start index of each episode in the stitched training arrays."""
    return np.concatenate([[0], np.cumsum(traj_lengths)[:-1]]).astype(np.int64)


class ClosedLoopRenderer:
    """Render recorded ground truth through the scene ``run_policy_sim`` builds.

    Mirrors :class:`EpisodeRenderer`'s interface so the two paths can be driven
    from the same loop and differ only where the scene and renderer differ.
    """

    def __init__(
        self,
        *,
        render_hw: tuple[int, int],
        image_hw: tuple[int, int],
        recording_quality: bool = False,
    ) -> None:
        render_height, render_width = render_hw
        self.image_hw = image_hw
        self.model, self.data = build_policy_sim_model(render_height, render_width)
        if recording_quality:
            configure_render_quality(self.model)
        self._renderer = mujoco.Renderer(
            self.model, height=render_height, width=render_width
        )
        self.appearance = SceneAppearanceOverride(self.model)
        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[cube_body]])

    def set_episode(self, target_xy: tuple[float, float], target_plate_yaw: float) -> None:
        place_paper_target_marker(
            self.model,
            target_xy,
            target_plate_yaw,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(*target_xy),
            alpha=1.0,
        )
        mujoco.mj_forward(self.model, self.data)
        self.appearance.refresh_plate_baseline()

    def set_frame(self, state_real: np.ndarray, cube_pose: np.ndarray) -> None:
        arm_rad, gripper_rad = real_frame_to_sim(np.asarray(state_real, dtype=np.float64))
        for name in ARM_JOINT_NAMES:
            set_joint(self.model, self.data, name, arm_rad[name])
        set_joint(self.model, self.data, "gripper", gripper_rad)
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_pose
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera)
        return resize_and_center_crop(self._renderer.render(), *self.image_hw)

    def close(self) -> None:
        self._renderer.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--episodes-root",
        type=Path,
        required=True,
        help="staged source episodes holding ep*/ with per-frame ground truth",
    )
    parser.add_argument(
        "--train-npz",
        type=Path,
        required=True,
        help="train.npz written by the Diffusion Policy dataset export",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="export.json beside train.npz (default: <train-npz parent>/export.json)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="episode index, in both the staged root and the export (default: 0)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=12,
        help="frames to compare, evenly spaced over the episode (default: 12)",
    )
    parser.add_argument(
        "--appearance",
        default="blue-cube",
        help="scene appearance the training videos were rendered with (default: blue-cube)",
    )
    parser.add_argument(
        "--recording-quality",
        action="store_true",
        help=(
            "give the closed-loop scene the recording pipeline's shadow-map and "
            "offscreen-sampling settings, isolating them as the cause of a difference"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="directory for the report and side-by-side panels (default: no files written)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    export_json = args.export_json or args.train_npz.parent / "export.json"
    with export_json.open() as file:
        export = json.load(file)
    image_size = int(export["image_size"][0])
    frame_stride = int(export["frame_stride"])
    _, appearance = parse_appearance(args.appearance)

    datasets = find_episode_datasets(args.episodes_root)
    staged = {episode_index(root): root for root in datasets}
    if args.episode not in staged:
        raise SystemExit(f"{args.episodes_root} has no episode {args.episode}")
    episode_root = staged[args.episode]
    truth = load_episode_truth(episode_root)
    image_hw = video_render_hw(episode_root)
    render_hw = tuple(export.get("source_render_hw", (1080, 1920)))

    bundle = np.load(args.train_npz)
    offsets = episode_offsets(bundle["traj_lengths"])
    start = int(offsets[args.episode])
    length = int(bundle["traj_lengths"][args.episode])

    # The exporter keeps every Nth source frame, counted from each episode start.
    keep = np.arange(0, len(truth.states), frame_stride)[:length]
    picks = np.unique(np.linspace(0, len(keep) - 1, args.frames).round().astype(int))

    rerender = EpisodeRenderer(render_hw=render_hw, image_hw=image_hw)
    rerender.set_episode(truth.target_xy, truth.target_plate_yaw)
    rerender.appearance.apply(appearance)

    closed_loop = ClosedLoopRenderer(
        render_hw=render_hw,
        image_hw=(image_size, image_size),
        recording_quality=args.recording_quality,
    )
    closed_loop.set_episode(truth.target_xy, truth.target_plate_yaw)
    closed_loop.appearance.apply(appearance)

    print(
        f"{episode_root.name}: {len(truth.states)} source frames, {length} exported at "
        f"stride {frame_stride}; comparing {len(picks)} of them at {image_size}x{image_size}."
    )

    per_camera: dict[str, dict[str, list[float]]] = {
        camera: {"rerender": [], "closed_loop": []} for camera in CAMERAS
    }
    frames: list[FrameImages] = []
    for pick in picks:
        source_frame = int(keep[pick])
        state = truth.states[source_frame]
        cube_pose = truth.cube_poses[source_frame]

        rerender.set_frame(state, cube_pose)
        closed_loop.set_frame(state, cube_pose)

        train_images = train_frame_images(bundle["images"], start + int(pick))
        rerender_images = {
            camera: resize_and_center_crop(
                rerender.rig.render(rerender.data, camera), image_size, image_size
            )
            for camera in CAMERAS
        }
        closed_loop_images = {camera: closed_loop.render(camera) for camera in CAMERAS}
        frames.append(FrameImages(train_images, rerender_images, closed_loop_images))

        for camera in CAMERAS:
            per_camera[camera]["rerender"].append(
                mean_abs_diff(rerender_images[camera], train_images[camera])
            )
            per_camera[camera]["closed_loop"].append(
                mean_abs_diff(closed_loop_images[camera], train_images[camera])
            )

    report: dict[str, object] = {
        "episode": episode_root.name,
        "appearance": args.appearance,
        "image_size": image_size,
        "frames_compared": len(picks),
        "source_frames": [int(keep[pick]) for pick in picks],
        "cameras": {},
    }
    for camera in CAMERAS:
        rerender_diffs = np.asarray(per_camera[camera]["rerender"])
        closed_loop_diffs = np.asarray(per_camera[camera]["closed_loop"])
        report["cameras"][camera] = {
            "rerender_vs_train_mean": float(rerender_diffs.mean()),
            "rerender_vs_train_max": float(rerender_diffs.max()),
            "closed_loop_vs_train_mean": float(closed_loop_diffs.mean()),
            "closed_loop_vs_train_max": float(closed_loop_diffs.max()),
            "ratio": float(closed_loop_diffs.mean() / max(rerender_diffs.mean(), 1e-6)),
        }
        print(
            f"{camera}: closed-loop {closed_loop_diffs.mean():.2f} grey levels vs train "
            f"(re-render floor {rerender_diffs.mean():.2f}, "
            f"{closed_loop_diffs.mean() / max(rerender_diffs.mean(), 1e-6):.1f}x), "
            f"worst frame {closed_loop_diffs.max():.2f}"
        )

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        for pick, images in zip(picks, frames, strict=True):
            for camera in CAMERAS:
                panel = np.concatenate(
                    [
                        images.train[camera],
                        images.rerender[camera],
                        images.closed_loop[camera],
                    ],
                    axis=1,
                )
                name = camera.rsplit(".", maxsplit=1)[-1]
                imageio.imwrite(
                    args.output / f"{episode_root.name}-{name}-f{int(keep[pick]):04d}.png",
                    panel,
                )
        with (args.output / "observation_comparison.json").open("w") as file:
            json.dump(report, file, indent=1, sort_keys=True)
        print(f"\nPanels (train | re-render | closed-loop) -> {args.output}")

    rerender.rig.close()
    closed_loop.close()


if __name__ == "__main__":
    main()
