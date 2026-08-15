# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Everything written out about one recorded simulation episode.

A finished episode leaves two records, and they are not the same record. The
dataset metadata row describes it to whatever will *train* on it: where the cube
started, where it landed, which phases ran, whether the grasp was deliberately
fumbled. The trajectory artifact describes it to whatever will *render* it
again: the true arm and cube poses, the drop plate, the camera mount, the look
its pixels were made under.

They are built together here because they come from the same handful of values,
and keeping them apart in the recorder is how one of them ends up missing a
field the other has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pick_and_place.core.geometry import PlacementError
from pick_and_place.core.grasp_perturbation import GraspPerturbation
from pick_and_place.core.miscalibration import MiscalibrationDraw
from pick_and_place.core.task_phases import phase_spans_json
from pick_and_place.data.dataset_metadata import cube_pose_metadata, placement_error_metadata
from pick_and_place.data.trajectory_artifact import (
    ARTIFACT_FILENAME,
    EpisodeFacts,
    MiscalibrationRecord,
    TrajectoryArtifact,
    WristCameraMount,
    save_trajectory,
)
from pick_and_place.runtime.episodes import Episode
from pick_and_place.runtime.sim_recorder import RecordEpisodeResult
from pick_and_place.sim.domain_randomization import DomainSample


def miscalibration_metadata(draw: MiscalibrationDraw) -> dict[str, float]:
    """Dataset metadata recording the injected draw (believed-vs-true errors)."""
    metadata: dict[str, float] = {
        f"injected_offset_{name}_deg": float(value)
        for name, value in draw.base_offsets_deg.items()
    }
    dx, dy, dz, dyaw = draw.cube_belief_error
    tx, ty = draw.target_belief_error
    metadata.update(
        {
            "injected_cube_belief_dx": float(dx),
            "injected_cube_belief_dy": float(dy),
            "injected_cube_belief_dz": float(dz),
            "injected_cube_belief_dyaw": float(dyaw),
            "injected_target_belief_dx": float(tx),
            "injected_target_belief_dy": float(ty),
        }
    )
    return metadata


def episode_metadata(
    episode: Episode,
    result: RecordEpisodeResult,
    error: PlacementError,
    *,
    target_plate_yaw: float,
    orientation_index: int,
    perturbation: GraspPerturbation | None,
    draw: MiscalibrationDraw | None,
    sample: DomainSample | None,
    preset_name: str | None = None,
    domain_seed: int | None = None,
) -> dict[str, Any]:
    """The dataset's episode row: what a consumer needs to filter and interpret it."""
    metadata = cube_pose_metadata(episode.source, episode.target)
    metadata.update(placement_error_metadata(error, detected=True))
    metadata["target_plate_yaw"] = float(target_plate_yaw)
    metadata["phase_spans"] = phase_spans_json(result.phase_spans)
    metadata["cube_start_roll"] = float(episode.source.roll)
    metadata["cube_start_pitch"] = float(episode.source.pitch)
    metadata["cube_orientation_index"] = orientation_index
    # Recorded on every episode, perturbed or not, so the fraction can be swept
    # later by filtering the dataset instead of regenerating it -- and so a
    # downstream reader can tell a clean episode from a recovered one, which the
    # frames alone do not reveal.
    metadata["grasp_perturbation_kind"] = (
        perturbation.kind if perturbation is not None else "none"
    )
    if perturbation is not None:
        metadata.update(
            {
                f"grasp_perturbation_{key}": value
                for key, value in perturbation.as_metadata().items()
                if key != "kind"
            }
        )
    if draw is not None:
        metadata.update(
            {
                "believed_cube_start_x": float(episode.believed_source.x),
                "believed_cube_start_y": float(episode.believed_source.y),
                "believed_cube_start_yaw": float(episode.believed_source.yaw),
                "believed_target_x": float(episode.believed_target.x),
                "believed_target_y": float(episode.believed_target.y),
            }
        )
        metadata.update(miscalibration_metadata(draw))
    if sample is not None:
        metadata.update(
            {
                "source_domain": "sim",
                "domain_preset": preset_name,
                "domain_seed": domain_seed,
                "domain_sample_json": sample.metadata_json(),
            }
        )
    return metadata


def save_episode_artifact(
    episode_root: Path,
    episode: Episode,
    result: RecordEpisodeResult,
    *,
    target_plate_yaw: float,
    draw: MiscalibrationDraw | None,
    sample: DomainSample | None,
    seed: int | None,
    episode_index: int,
    fingerprint: dict[str, Any],
) -> None:
    """Write the episode's trajectory artifact beside its dataset.

    Written for every episode rather than only for runs that expect to re-render
    later: it is the only record the pixels can be made from again, and it costs
    a few dozen floats per tick.
    """
    save_trajectory(
        episode_root / ARTIFACT_FILENAME,
        TrajectoryArtifact(
            frames=result.frames,
            facts=EpisodeFacts(
                target_xy=(float(episode.target.x), float(episode.target.y)),
                target_plate_yaw=float(target_plate_yaw),
                verdict=result.status,
                phase_spans=result.phase_spans,
                fingerprint=fingerprint,
                seed=seed,
                episode_index=episode_index,
                miscalibration=MiscalibrationRecord.of(draw),
                wrist_camera_mount=(
                    None
                    if sample is None
                    else WristCameraMount(
                        position_m=sample.wrist_camera_position_m,
                        rotation_deg=sample.wrist_camera_rotation_deg,
                    )
                ),
                recorded_appearance=None if sample is None else sample.appearance(),
            ),
        ),
    )
