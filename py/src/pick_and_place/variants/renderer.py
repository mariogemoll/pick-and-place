# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Put a recorded episode back in front of a camera, under a different scene.

The renderer is deliberately dumb about behavior. It takes poses and paints
them: the arm where physics actually held it, the cube where physics actually
had it, the drop plate where the episode put it. Nothing here plans, detects or
steps physics, because the trajectory already exists — which is what makes an
appearance a free variable and a variant pass cheap.

Reproducing the recording pipeline rather than merely resembling it is the whole
job, so the scene comes from the function the recorder itself builds through and
the images go through the same camera rig at the same resolution. One thing that
looks like appearance is not: the wrist camera's physical mount displacement was
drawn before the episode ran, so a faithful re-render has to put the camera back
on the mount it was actually on.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from pick_and_place.core.camera_calibration import load_local_camera_intrinsics
from pick_and_place.core.joint_frames import real_frame_to_sim
from pick_and_place.core.workspace_bounds import is_cube_drop_allowed
from pick_and_place.data.trajectory_artifact import EpisodeFacts, WristCameraMount
from pick_and_place.runtime.sim_recorder import WRIST_CAMERA, SimCameraRig, build_recording_scene
from pick_and_place.sim.camera_pose_envelope import (
    CameraBase,
    CameraJitter,
    set_camera_jitter,
    snapshot_camera,
)
from pick_and_place.core.appearance import AppearanceDraw
from pick_and_place.sim.domain_randomization import ProceduralAppearance
from pick_and_place.sim.model import cube_qpos_address, set_joint
from pick_and_place.sim.paper_target_marker import place_paper_target_marker
from pick_and_place.spec.robot import ARM_JOINT_NAMES
from pick_and_place.spec.workspace import DROP_ZONE_HALF_SIZE
from pick_and_place.variants.appearance import SceneAppearance, SceneAppearanceOverride
from pick_and_place.variants.scene import AppearanceRandomizer


class VariantRenderer:
    """Replays trajectory artifacts through the recording camera pipeline."""

    def __init__(
        self,
        *,
        render_hw: tuple[int, int],
        image_hw: tuple[int, int],
        background_panorama: Path | str | np.ndarray | None = None,
        table_texture: Path | str | np.ndarray | None = None,
    ) -> None:
        render_height, render_width = render_hw
        image_height, image_width = image_hw
        self.model, self.data = build_recording_scene(
            render_width=render_width,
            render_height=render_height,
            background_panorama=background_panorama,
            table_texture=table_texture,
        )
        self.rig = SimCameraRig(
            self.model,
            load_local_camera_intrinsics(),
            width=image_width,
            height=image_height,
            render_width=render_width,
            render_height=render_height,
        )
        self.appearance = SceneAppearanceOverride(self.model)
        self.randomizer = AppearanceRandomizer(self.model)
        self._cube_qpos_adr = cube_qpos_address(self.model)
        self._wrist_base: CameraBase = snapshot_camera(self.model, WRIST_CAMERA)
        self._postprocess: AppearanceDraw | None = None

    def set_camera_jitter(self, jitter: CameraJitter | None) -> None:
        """Apply an overhead-camera pose+focal jitter, or restore the authored pose."""
        self.randomizer.set_camera_jitter(jitter)

    def set_scene_texture(self, appearance: ProceduralAppearance | None) -> None:
        """Apply a background/table appearance, or restore the one built at construction.

        A no-op unless the renderer was built with ``background_panorama`` or
        ``table_texture`` (the finite-floor scene) — there is nothing to vary
        under the infinite groundplane every plain recording has used so far.
        """
        if not self.randomizer.texture_ids:
            return
        self.randomizer.set_scene_texture(appearance)
        self.rig.reload_textures(self.randomizer.texture_ids)

    def set_appearance_draw(self, sample: AppearanceDraw | None) -> None:
        """Apply a whole appearance draw: lights, materials, viewpoint and textures.

        The sample's camera-response half is kept for :meth:`capture` rather than
        applied here, since it acts on pixels after the render.
        """
        if sample is None:
            self.randomizer.reset()
        else:
            self.randomizer.apply(sample)
        if self.randomizer.texture_ids:
            self.rig.reload_textures(self.randomizer.texture_ids)
        self._postprocess = sample

    def set_wrist_camera_mount(self, mount: WristCameraMount | None) -> None:
        """Put the wrist camera back on the mount the episode was recorded with.

        Not appearance, despite happening at render time: the displacement was
        drawn before the trajectory existed and the expert servoed through it, so
        rendering from the nominal mount would show a wrist view the recording
        never had.
        """
        set_camera_jitter(
            self.model,
            self._wrist_base,
            None
            if mount is None
            else CameraJitter(
                position_m=mount.position_m, rotation_deg=mount.rotation_deg
            ),
        )

    def set_episode(
        self,
        facts: EpisodeFacts,
        *,
        camera_jitter: CameraJitter | None = None,
        scene_texture: ProceduralAppearance | None = None,
        appearance_draw: AppearanceDraw | None = None,
    ) -> None:
        """Set up everything that holds for a whole episode.

        Called once per variant, not once per frame: the drop plate does not move
        and neither does the scene's paint, so a variant pass restyles the scene
        here and then runs every frame through it.

        With no ``appearance_draw`` the episode's *recorded* look is restored, so
        that rendering an episode changes nothing it was not asked to change.
        For a plain recording that is the compiled scene; for a randomized one it
        is the draw that recording was made under, which is what lets it be
        verified against its own video rather than only replaced.
        """
        # The base look, then the narrower draws layered on top of it, so a
        # viewpoint jitter still applies over a full appearance draw.
        base = facts.recorded_appearance if appearance_draw is None else appearance_draw
        self.set_appearance_draw(base)
        if camera_jitter is not None:
            self.set_camera_jitter(camera_jitter)
        if scene_texture is not None:
            self.set_scene_texture(scene_texture)
        self.set_wrist_camera_mount(facts.wrist_camera_mount)
        place_paper_target_marker(
            self.model,
            facts.target_xy,
            facts.target_plate_yaw,
            (DROP_ZONE_HALF_SIZE, DROP_ZONE_HALF_SIZE),
            usable=is_cube_drop_allowed(*facts.target_xy),
            alpha=1.0,
        )
        if base is not None:
            # The plate is placed after the bulk paint, so its tint has to follow it.
            self.randomizer.tint_episode_markers()
        mujoco.mj_forward(self.model, self.data)
        # The placement decides this episode's plate colour, which is what an
        # appearance leaving the target unset must restore.
        self.appearance.refresh_plate_baseline()

    def set_frame(self, true_state: np.ndarray, cube_pose: np.ndarray) -> None:
        """Pose the arm and cube at one recorded frame's ground truth."""
        arm_rad, gripper_rad = real_frame_to_sim(np.asarray(true_state, dtype=np.float64))
        for name in ARM_JOINT_NAMES:
            set_joint(self.model, self.data, name, arm_rad[name])
        set_joint(self.model, self.data, "gripper", gripper_rad)
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_pose
        mujoco.mj_forward(self.model, self.data)

    def capture(self, appearance: SceneAppearance) -> dict[str, np.ndarray]:
        """Render both cameras under ``appearance``, keyed by dataset feature."""
        self.appearance.apply(appearance)
        wrist, overhead = self.rig.capture(self.data)
        if self._postprocess is not None:
            wrist = self.randomizer.postprocess(wrist)
            overhead = self.randomizer.postprocess(overhead)
        return {
            "observation.images.wrist": wrist,
            "observation.images.overhead": overhead,
        }

    def close(self) -> None:
        self.rig.close()
