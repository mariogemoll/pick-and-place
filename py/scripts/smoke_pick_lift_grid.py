# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Smoke-test canonical pick-and-lift over a sampled floor grid in MuJoCo."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place import build_scene
from pick_and_place.episodes import (
    PICKUP_YAW_DEVIATION,
    _build_model,
    build_geom_sets,
    is_unexpected,
    pickup_yaw_from_azimuth,
    scan_contacts,
    set_cube_pose,
    set_joint,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.trajectory import (
    ApproachPhase,
    DescentPhase,
    GRIPPER_OPEN,
    GraspPhase,
    LiftPhase,
    MAX_CANONICAL_GRASP_RADIUS,
    MIN_CANONICAL_GRASP_RADIUS,
    NEUTRAL_ARM_JOINTS,
    Trajectory,
    grasp_candidates,
)
from pick_and_place.workspace_overlays import (
    CANONICAL_PICKUP_OVERLAY,
    CUBE_PLACEMENT_OVERLAY,
    PAN_AXIS,
    is_cube_drop_allowed,
    is_cube_pickup_allowed,
)


@dataclass(frozen=True)
class PickLiftResult:
    pose: CubePose
    status: str
    face: str = ""
    elbow: str = ""
    start_z: float = CUBE_HALF_SIZE
    end_z: float = CUBE_HALF_SIZE
    collision_time: float | None = None
    collision_geom1: str = ""
    collision_geom2: str = ""


_STATUS_RGBA = {
    "lifted": (0.0, 0.82, 0.32, 0.78),
    "collision": (1.0, 0.64, 0.0, 0.9),
    "not-lifted": (1.0, 0.12, 0.04, 0.9),
    "no-ik": (0.1, 0.35, 1.0, 0.9),
}
_STATUS_Z = {
    "lifted": 0.004,
    "collision": 0.007,
    "not-lifted": 0.009,
    "no-ik": 0.014,
}


def _actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }


def _cube_body_id(model: mujoco.MjModel) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
    if body_id < 0:
        raise RuntimeError("model has no pick_cube body")
    return body_id


def _grid_poses(
    grid: int,
    yaw_count: int,
    *,
    placement_edges: bool,
    radius_margin: float,
    azimuth_margin: float,
    radius_min: float | None,
    radius_max: float | None,
    azimuth_min: float | None,
    azimuth_max: float | None,
    yaw_min: float,
    yaw_max: float,
) -> list[CubePose]:
    if yaw_count < 1:
        raise ValueError("yaw_count must be at least 1")
    if placement_edges:
        r_min = CUBE_PLACEMENT_OVERLAY.inner_radius
        r_max = CUBE_PLACEMENT_OVERLAY.outer_radius
        az_min = CUBE_PLACEMENT_OVERLAY.azimuth_min
        az_max = CUBE_PLACEMENT_OVERLAY.azimuth_max
        is_allowed = is_cube_drop_allowed
    else:
        r_min = MIN_CANONICAL_GRASP_RADIUS
        r_max = MAX_CANONICAL_GRASP_RADIUS
        az_min = CANONICAL_PICKUP_OVERLAY.azimuth_min
        az_max = CANONICAL_PICKUP_OVERLAY.azimuth_max
        is_allowed = is_cube_pickup_allowed
    r_min = max(0.0, r_min - radius_margin)
    r_max += radius_margin
    if radius_min is not None:
        r_min = radius_min
    if radius_max is not None:
        r_max = radius_max
    az_min -= azimuth_margin
    az_max += azimuth_margin
    if azimuth_min is not None:
        az_min = azimuth_min
    if azimuth_max is not None:
        az_max = azimuth_max
    radii = np.linspace(r_min, r_max, grid)
    azimuths = np.linspace(az_min, az_max, grid)
    yaw_deviations = (
        np.array((0.0,))
        if yaw_count == 1
        else np.linspace(yaw_min, yaw_max, yaw_count)
    )
    clip_to_overlay = (
        radius_margin <= 0.0
        and azimuth_margin <= 0.0
        and radius_min is None
        and radius_max is None
        and azimuth_min is None
        and azimuth_max is None
    )
    poses: list[CubePose] = []
    for radius in radii:
        for azimuth in azimuths:
            x = PAN_AXIS[0] + float(radius) * math.cos(float(azimuth))
            y = PAN_AXIS[1] + float(radius) * math.sin(float(azimuth))
            if clip_to_overlay and not is_allowed(x, y):
                continue
            for deviation in yaw_deviations:
                yaw = pickup_yaw_from_azimuth(float(azimuth), float(deviation))
                poses.append(CubePose(x=x, y=y, z=CUBE_HALF_SIZE, yaw=yaw))
    return poses


def _pick_lift_trajectory(kinematics, source: CubePose) -> Trajectory | None:
    grasp = next(grasp_candidates(kinematics, source), None)
    if grasp is None:
        return None
    phases = (
        ApproachPhase(kinematics, dict(NEUTRAL_ARM_JOINTS), GRIPPER_OPEN, grasp.hover_joints),
        DescentPhase(kinematics, grasp),
        GraspPhase(grasp.grasp_joints),
        LiftPhase(kinematics, grasp.grasp_joints, grasp.lift_joints),
    )
    return Trajectory(
        phases=phases,
        source=source,
        grasp=grasp,
        start_joints=dict(NEUTRAL_ARM_JOINTS),
        start_gripper=GRIPPER_OPEN,
    )


def _run_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: dict[str, int],
    cube_body_id: int,
    kinematics,
    pose: CubePose,
    *,
    robot_geom_ids: set[int],
    env_geom_ids: set[int],
    lift_threshold: float,
) -> PickLiftResult:
    set_cube_pose(model, data, pose)
    for name, value in NEUTRAL_ARM_JOINTS.items():
        set_joint(model, data, name, value)
        data.ctrl[actuator_id[name]] = value

    set_joint(model, data, "gripper", GRIPPER_OPEN)
    data.ctrl[actuator_id["gripper"]] = GRIPPER_OPEN
    data.time = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    trajectory = _pick_lift_trajectory(kinematics, pose)
    if trajectory is None or trajectory.grasp is None:
        return PickLiftResult(pose=pose, status="no-ik")

    start_z = float(data.xpos[cube_body_id][2])
    first_collision: tuple[float, str, str] | None = None
    while data.time < trajectory.duration:
        frame = trajectory.evaluate(data.time)
        for name, value in frame.joints.items():
            data.ctrl[actuator_id[name]] = value
        data.ctrl[actuator_id["gripper"]] = frame.gripper
        mujoco.mj_step(model, data)
        if first_collision is None:
            for n1, n2 in scan_contacts(model, data, robot_geom_ids, env_geom_ids):
                if is_unexpected(n1, n2):
                    first_collision = (float(data.time), n1, n2)
                    break

    end_z = float(data.xpos[cube_body_id][2])
    lifted = end_z >= start_z + lift_threshold
    status = "lifted" if lifted else "not-lifted"
    if first_collision is not None:
        status = "collision"
    return PickLiftResult(
        pose=pose,
        status=status,
        face=trajectory.grasp.face,
        elbow=trajectory.grasp.elbow,
        start_z=start_z,
        end_z=end_z,
        collision_time=None if first_collision is None else first_collision[0],
        collision_geom1="" if first_collision is None else first_collision[1],
        collision_geom2="" if first_collision is None else first_collision[2],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=5, help="radial/azimuth samples per axis")
    parser.add_argument(
        "--yaw-count",
        type=int,
        default=4,
        help="cube yaw deviation samples per grid point around azimuth",
    )
    parser.add_argument(
        "--placement-edges",
        action="store_true",
        help="probe the full placement overlay instead of the canonical pick-lift envelope",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "probe beyond the conservative pickup envelope: uses the full placement "
            "overlay while keeping the square-cube yaw window"
        ),
    )
    parser.add_argument(
        "--radius-margin-mm",
        type=float,
        default=0.0,
        help="extra radial margin outside the selected overlay; disables placement clipping",
    )
    parser.add_argument(
        "--azimuth-margin-deg",
        type=float,
        default=0.0,
        help="extra azimuth margin outside the selected overlay; disables placement clipping",
    )
    parser.add_argument(
        "--yaw-deviation-deg",
        type=float,
        default=None,
        help="maximum yaw deviation around azimuth; default 45 because square yaw repeats after 90 degrees",
    )
    parser.add_argument(
        "--radius-min-mm",
        type=float,
        default=None,
        help="minimum radius from the pan axis; overrides the selected overlay minimum",
    )
    parser.add_argument(
        "--radius-max-mm",
        type=float,
        default=None,
        help="maximum radius from the pan axis; overrides the selected overlay maximum",
    )
    parser.add_argument(
        "--azimuth-min-deg",
        type=float,
        default=None,
        help="minimum azimuth around the pan axis; overrides the selected overlay minimum",
    )
    parser.add_argument(
        "--azimuth-max-deg",
        type=float,
        default=None,
        help="maximum azimuth around the pan axis; overrides the selected overlay maximum",
    )
    parser.add_argument(
        "--yaw-min-deg",
        type=float,
        default=None,
        help="minimum yaw deviation around azimuth; overrides --yaw-deviation-deg",
    )
    parser.add_argument(
        "--yaw-max-deg",
        type=float,
        default=None,
        help="maximum yaw deviation around azimuth; overrides --yaw-deviation-deg",
    )
    parser.add_argument(
        "--lift-threshold-mm",
        type=float,
        default=20.0,
        help="minimum cube rise counted as a successful lift",
    )
    parser.add_argument("--limit", type=int, default=0, help="optional cap on tested poses")
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="only print and visualize samples that failed to lift",
    )
    parser.add_argument(
        "--only-status",
        choices=tuple(_STATUS_RGBA),
        default=None,
        help="only print and visualize samples with this exact status",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="open a MuJoCo viewer and replay the actual grasp for every sampled pose",
    )
    parser.add_argument(
        "--visualize-speed",
        type=float,
        default=1.0,
        help="trajectory playback speed in the visualization (1.0 = nominal)",
    )
    parser.add_argument(
        "--visualize-pause",
        type=float,
        default=0.6,
        help="seconds to pause between visualized samples",
    )
    parser.add_argument(
        "--visualize-shuffle-seed",
        type=int,
        default=None,
        help="seed for randomized visualization order; omit for a fresh order each run",
    )
    args = parser.parse_args()
    if args.yaw_count < 1:
        parser.error("--yaw-count must be at least 1")
    if args.radius_margin_mm < 0.0:
        parser.error("--radius-margin-mm cannot be negative")
    if args.azimuth_margin_deg < 0.0:
        parser.error("--azimuth-margin-deg cannot be negative")

    if (
        args.radius_min_mm is not None
        and args.radius_max_mm is not None
        and args.radius_min_mm > args.radius_max_mm
    ):
        parser.error("--radius-min-mm cannot be greater than --radius-max-mm")
    if (
        args.azimuth_min_deg is not None
        and args.azimuth_max_deg is not None
        and args.azimuth_min_deg > args.azimuth_max_deg
    ):
        parser.error("--azimuth-min-deg cannot be greater than --azimuth-max-deg")

    yaw_deviation_deg = (
        math.degrees(PICKUP_YAW_DEVIATION)
        if args.yaw_deviation_deg is None
        else args.yaw_deviation_deg
    )
    if yaw_deviation_deg < 0.0:
        parser.error("--yaw-deviation-deg cannot be negative")
    yaw_min_deg = -yaw_deviation_deg if args.yaw_min_deg is None else args.yaw_min_deg
    yaw_max_deg = yaw_deviation_deg if args.yaw_max_deg is None else args.yaw_max_deg
    if yaw_min_deg > yaw_max_deg:
        parser.error("--yaw-min-deg cannot be greater than --yaw-max-deg")

    poses = _grid_poses(
        args.grid,
        args.yaw_count,
        placement_edges=args.placement_edges or args.extended,
        radius_margin=args.radius_margin_mm / 1000.0,
        azimuth_margin=math.radians(args.azimuth_margin_deg),
        radius_min=None if args.radius_min_mm is None else args.radius_min_mm / 1000.0,
        radius_max=None if args.radius_max_mm is None else args.radius_max_mm / 1000.0,
        azimuth_min=None if args.azimuth_min_deg is None else math.radians(args.azimuth_min_deg),
        azimuth_max=None if args.azimuth_max_deg is None else math.radians(args.azimuth_max_deg),
        yaw_min=math.radians(yaw_min_deg),
        yaw_max=math.radians(yaw_max_deg),
    )
    if args.limit > 0:
        poses = poses[: args.limit]
    if not poses:
        print("No poses sampled.")
        return 2

    model, data = _build_model(poses[0])
    kinematics = derive_kinematics(model)
    actuator_id = _actuator_ids(model)
    cube_body_id = _cube_body_id(model)
    robot_geom_ids, env_geom_ids = build_geom_sets(model)

    results = [
        _run_pose(
            model,
            data,
            actuator_id,
            cube_body_id,
            kinematics,
            pose,
            robot_geom_ids=robot_geom_ids,
            env_geom_ids=env_geom_ids,
            lift_threshold=args.lift_threshold_mm / 1000.0,
        )
        for pose in poses
    ]
    counts = {
        status: sum(1 for result in results if result.status == status)
        for status in sorted({r.status for r in results})
    }
    print(f"tested={len(results)} " + " ".join(f"{key}={value}" for key, value in counts.items()))

    failures = [result for result in results if result.status != "lifted"]
    selected_results = results
    if args.only_status is not None:
        selected_results = [result for result in results if result.status == args.only_status]
        print(f"showing_{args.only_status}={len(selected_results)}")
    elif args.only_failures:
        selected_results = failures
        print(f"showing_failures={len(selected_results)}")

    printed_results = selected_results if args.only_failures or args.only_status else failures
    for result in printed_results[:20]:
        print(
            f"{result.status}: x={result.pose.x:.4f} y={result.pose.y:.4f} "
            f"yaw={math.degrees(result.pose.yaw):.1f}deg "
            f"z={result.start_z:.4f}->{result.end_z:.4f}"
            + (
                ""
                if result.collision_time is None
                else (
                    f" collision t={result.collision_time:.3f}s "
                    f"{result.collision_geom1}<->{result.collision_geom2}"
                )
            )
        )
    if len(printed_results) > 20:
        print(f"... {len(printed_results) - 20} more samples omitted")

    if args.visualize:
        visualized_results = selected_results
        if not visualized_results:
            print("No samples to visualize.")
            return 1 if failures else 0
        rng = np.random.default_rng(args.visualize_shuffle_seed)
        visualized_results = [
            visualized_results[index]
            for index in rng.permutation(len(visualized_results))
        ]
        _visualize_results(
            visualized_results,
            speed=args.visualize_speed,
            pause_seconds=args.visualize_pause,
        )
    return 1 if failures else 0


def _add_sample_markers(spec: mujoco.MjSpec, results: list[PickLiftResult]) -> None:
    """Add non-colliding floor markers for every sampled pick-lift pose."""
    for index, result in enumerate(results):
        rgba = _STATUS_RGBA.get(result.status, (1.0, 0.85, 0.0, 0.85))
        z = _STATUS_Z.get(result.status, 0.004)
        yaw = result.pose.yaw
        tick_len = CUBE_HALF_SIZE * 1.25
        tick_start = CUBE_HALF_SIZE * 0.35
        start_x = result.pose.x + tick_start * math.cos(yaw)
        start_y = result.pose.y + tick_start * math.sin(yaw)
        end_x = result.pose.x + tick_len * math.cos(yaw)
        end_y = result.pose.y + tick_len * math.sin(yaw)

        spec.worldbody.add_geom(
            name=f"pick_lift_sample_center_{index:04d}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(result.pose.x, result.pose.y, z),
            size=(0.0032, 0.0012),
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_geom(
            name=f"pick_lift_sample_yaw_{index:04d}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(start_x, start_y, z + 0.002, end_x, end_y, z + 0.002),
            size=(0.0014,),
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )


def _reset_visualized_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_id: dict[str, int],
    pose: CubePose,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.time = 0.0
    set_cube_pose(model, data, pose)
    for name, value in NEUTRAL_ARM_JOINTS.items():
        set_joint(model, data, name, value)
        data.ctrl[actuator_id[name]] = value
    set_joint(model, data, "gripper", GRIPPER_OPEN)
    data.ctrl[actuator_id["gripper"]] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)


def _sync_for(viewer, model: mujoco.MjModel, seconds: float, should_skip) -> bool:
    deadline = time.time() + seconds
    while viewer.is_running() and time.time() < deadline:
        if should_skip():
            break
        step_start = time.time()
        viewer.sync()
        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)
    return viewer.is_running()


def _visualize_results(
    results: list[PickLiftResult],
    *,
    speed: float,
    pause_seconds: float,
) -> None:
    """Replay the sampled pick-lift attempts in a passive MuJoCo viewer."""
    first = results[0].pose
    spec = build_scene(include_environment=True)
    cube = spec.body("pick_cube")
    cube.pos = (first.x, first.y, first.z)
    half_yaw = first.yaw / 2.0
    cube.quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    cube.add_freejoint()
    _add_sample_markers(spec, results)
    model = spec.compile()
    data = mujoco.MjData(model)
    kinematics = derive_kinematics(model)
    actuator_id = _actuator_ids(model)
    speed = max(speed, 1e-6)
    pause_seconds = max(pause_seconds, 0.0)

    print(
        "Visualizing actual grasps. Floor markers: "
        "green=lifted orange=collision red=not-lifted blue=no-ik"
    )
    print("Press Enter in the viewer to skip to the next sample.")
    pending_skip = {"flag": False}
    glfw_key_enter = 257
    glfw_key_kp_enter = 335

    def key_callback(keycode: int) -> None:
        if keycode in (glfw_key_enter, glfw_key_kp_enter):
            pending_skip["flag"] = True

    def should_skip() -> bool:
        if pending_skip["flag"]:
            pending_skip["flag"] = False
            return True
        return False

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            for index, result in enumerate(results, start=1):
                if not viewer.is_running():
                    break

                _reset_visualized_pose(model, data, actuator_id, result.pose)
                trajectory = _pick_lift_trajectory(kinematics, result.pose)
                print(
                    f"sample {index}/{len(results)} {result.status}: "
                    f"x={result.pose.x:.4f} y={result.pose.y:.4f} "
                    f"yaw={math.degrees(result.pose.yaw):.1f}deg"
                    + (
                        ""
                        if result.collision_time is None
                        else (
                            f" collision t={result.collision_time:.3f}s "
                            f"{result.collision_geom1}<->{result.collision_geom2}"
                        )
                    )
                )

                if trajectory is None:
                    if not _sync_for(viewer, model, pause_seconds, should_skip):
                        break
                    continue

                playback_start = time.time()
                while viewer.is_running():
                    if should_skip():
                        break
                    step_start = time.time()
                    traj_t = (time.time() - playback_start) * speed
                    if traj_t > trajectory.duration:
                        break
                    frame = trajectory.evaluate(traj_t)
                    for name, value in frame.joints.items():
                        data.ctrl[actuator_id[name]] = value
                    data.ctrl[actuator_id["gripper"]] = frame.gripper
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    remaining = model.opt.timestep - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)

                if not _sync_for(viewer, model, pause_seconds, should_skip):
                    break


if __name__ == "__main__":
    sys.exit(main())
