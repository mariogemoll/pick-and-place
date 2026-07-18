# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Session-start joint-zero calibration ("errors du jour").

The four arm joint zeros (shoulder_pan / shoulder_lift / elbow_flex /
wrist_flex) drift day to day, so the scripted pipeline's open-loop approach is
off by up to ~15 mm at reach. This routine measures them on the robot in a
couple of minutes: it localizes the cube with the overhead camera, drives the
wrist camera through a short look-at orbit around it, detects the cube from the
wrist at each pose, and least-squares fits the zeros with the shared core in
:mod:`pick_and_place.joint_zero_fit`.

A 5-DOF arm can only view one cube position from a single vertical plane at one
azimuth (``shoulder_pan`` is pinned to the cube's azimuth by the look-at
constraint), which is the collinear regime the offline fit had to avoid. So the
loop needs the cube seen from several positions spread in radius/azimuth; it
keeps adding positions until the fitted zeros' 1-sigma uncertainty is below
target. Placements need not be exact — the overhead camera re-localizes the cube
wherever it lands.

This module is the report-only stage: cube variety comes from the operator, who
is prompted (with a text instruction naming the emptiest reachable bin) whenever
more data is needed. Autonomous relocation is layered on top later.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

from pick_and_place.camera_compare import load_intrinsics
from pick_and_place.cube_detection import (
    _CV_TO_MJ,
    cube_pose_to_world,
    estimate_cube_pose,
    make_cube_detector,
)
from pick_and_place.episodes import (
    build_geom_sets,
    make_carry_collision_checker,
    sample_hunt_pose,
)
from pick_and_place.follower import (
    ARM_JOINT_NAMES,
    JOINT_NAMES,
    action_to_joints,
    joints_to_action,
    real_frame_to_sim,
    sim_frame_to_real,
)
from pick_and_place.geometry import (
    CUBE_HALF_SIZE,
    CubePose,
    canonical_grasp_matrix,
    canonical_pregrasp_matrix,
)
from pick_and_place.ik import solve_simple_grasp_ik
from pick_and_place.joint_zero_fit import (
    FIT_JOINTS,
    FitResult,
    JointZeroSample,
    build_columns,
    fit_robust,
    joint_ids,
    params_to_offsets_deg,
)
from pick_and_place.kinematics import So101Kinematics
from pick_and_place.overhead_detection import track_cube
from pick_and_place.trajectory import GRIPPER_OPEN
from pick_and_place.workspace_overlays import (
    CANONICAL_PICKUP_OVERLAY as PICKUP,
    is_cube_pickup_allowed,
)

CONTROL_HZ = 30.0
WRIST_WIDTH = 1280
WRIST_HEIGHT = 720
# Reject wrist detections further than this (bad tags at range) — matches
# measure_hand_eye_offset's gate.
MAX_CUBE_DISTANCE = 0.5


@dataclass
class CalibrationConfig:
    # Look-at orbit around one cube position: a swept grid of approach pitch (deg,
    # 90 = top-down) crossed with camera standoff (m). Which combinations are both
    # IK-reachable and frame the cube depends on radius, so the grid is filtered
    # per position (see lookat_poses) rather than fixed.
    pitch_min_deg: float = 45.0
    pitch_max_deg: float = 125.0
    pitch_step_deg: float = 5.0
    standoffs_m: tuple[float, ...] = (0.05, 0.08, 0.11, 0.14)
    frame_margin_frac: float = 0.12  # keep the cube this far inside the wrist frame
    max_poses_per_position: int = 9  # cap arm moves per cube position, spread over pitch
    # Per-joint 1-sigma target (deg); stop once every joint is under its target.
    std_target_deg: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.5,
            "shoulder_lift": 0.7,
            "elbow_flex": 0.7,
            "wrist_flex": 0.7,
        }
    )
    # Radius spread across positions is what conditions the parallel-axis
    # lift/elbow/wrist_flex split, so collect several spread positions in a single
    # session (results are not accumulated across runs).
    min_positions: int = 6
    max_positions: int = 9
    min_samples: int = 12
    settle_s: float = 0.4
    move_duration_s: float = 1.2
    frames_per_pose: int = 6
    flush_frames: int = 10  # drop stale buffered frames captured during the arm move
    gripper_rad: float = GRIPPER_OPEN  # jaws open so they stay out of the wrist view
    cube_search_timeout_s: float = 8.0
    # Wide shoulder-pan search poses to try when the cube isn't visible (the arm
    # may be parked over it after a relocation), swinging the arm clear of the
    # overhead view between tries.
    hunt_tries: int = 5


def _smoothstep(t: float) -> float:
    c = min(1.0, max(0.0, t))
    return c * c * (3.0 - 2.0 * c)


def _approach_vector(azimuth: float, pitch: float) -> np.ndarray:
    """Unit wrist->cube approach at ``pitch`` above horizontal in the ``azimuth`` plane."""
    horizontal = math.cos(pitch)
    return np.array(
        (math.cos(azimuth) * horizontal, math.sin(azimuth) * horizontal, -math.sin(pitch))
    )


def _cube_in_wrist_frame(
    data: mujoco.MjData,
    wrist_cam_id: int,
    wrist_camera_matrix: np.ndarray,
    cube_world: np.ndarray,
    margin_px: tuple[float, float],
) -> bool:
    """Whether ``cube_world`` projects inside the wrist frame (with margin).

    ``data`` must be forwarded at the candidate pose. Inverts the same camera
    convention :func:`cube_pose_to_world` uses (MuJoCo camera pose, OpenCV pixel
    projection through the rectified matrix).
    """
    cam_pos = data.cam_xpos[wrist_cam_id]
    cam_rot = data.cam_xmat[wrist_cam_id].reshape(3, 3)
    point_cv = _CV_TO_MJ.T @ (cam_rot.T @ (cube_world - cam_pos))
    if point_cv[2] <= 0.0:
        return False
    pixel = wrist_camera_matrix @ point_cv
    u, v = pixel[0] / pixel[2], pixel[1] / pixel[2]
    mx, my = margin_px
    width = 2.0 * wrist_camera_matrix[0, 2]
    height = 2.0 * wrist_camera_matrix[1, 2]
    return mx <= u <= width - mx and my <= v <= height - my


def lookat_poses(
    kinematics: So101Kinematics,
    cube: CubePose,
    config: CalibrationConfig,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addrs: dict[str, int],
    wrist_cam_id: int,
    wrist_camera_matrix: np.ndarray,
    collision_free: Callable[[dict[str, float]], bool],
) -> list[dict[str, float]]:
    """Sim-frame arm-joint set points that frame ``cube`` in the wrist camera.

    Reuses the pickup grasp/pregrasp geometry (the same poses the descent servo
    relies on): for a swept grid of approach pitch and camera standoff it keeps an
    IK-feasible branch (elbow-up preferred) only when forward kinematics puts the
    cube inside the wrist frame and ``collision_free`` clears the configuration.
    The collision screen matters at short cube-to-base radius, where a top-down
    look-at can fold the wrist into the upper arm. Which pitches survive depends
    on radius, so the filter adapts automatically. Results are capped at
    ``max_poses_per_position``, spread across the surviving pitch range.
    """
    dx = cube.x - kinematics.pan_axis[0]
    dy = cube.y - kinematics.pan_axis[1]
    azimuth = math.atan2(dy, dx)
    # Roll the gripper a quarter turn off the radial grasp so the side-mounted
    # wrist camera faces the cube. The -pi/2 sign keeps the camera upright (roll
    # ~= -90 deg, matching NEUTRAL/REST); +pi/2 frames the cube just as well but
    # carries the camera inverted.
    closing_azimuth = azimuth - math.pi / 2.0
    cube_world = np.array([cube.x, cube.y, cube.z], dtype=float)
    margin_px = (
        config.frame_margin_frac * 2.0 * wrist_camera_matrix[0, 2],
        config.frame_margin_frac * 2.0 * wrist_camera_matrix[1, 2],
    )

    n_pitch = int(round((config.pitch_max_deg - config.pitch_min_deg) / config.pitch_step_deg))
    framed: list[tuple[float, dict[str, float]]] = []
    for i in range(n_pitch + 1):
        pitch_deg = config.pitch_min_deg + i * config.pitch_step_deg
        approach = _approach_vector(azimuth, math.radians(pitch_deg))
        grasp = canonical_grasp_matrix(cube, closing_azimuth, approach)
        for distance in config.standoffs_m:
            hover = canonical_pregrasp_matrix(grasp, approach, distance)
            branches = solve_simple_grasp_ik(kinematics, hover)
            if not branches:
                continue
            branch = next((b for b in branches if b.elbow == "up"), branches[0])
            if not collision_free(branch.joints):
                continue
            for name in ARM_JOINT_NAMES:
                data.qpos[qpos_addrs[name]] = branch.joints[name]
            data.qpos[qpos_addrs["gripper"]] = config.gripper_rad
            mujoco.mj_forward(model, data)
            if _cube_in_wrist_frame(data, wrist_cam_id, wrist_camera_matrix, cube_world, margin_px):
                framed.append((pitch_deg, branch.joints))

    if len(framed) <= config.max_poses_per_position:
        return [joints for _, joints in framed]
    # Evenly stride the pitch-sorted list so the kept poses span the range.
    framed.sort(key=lambda item: item[0])
    idx = np.linspace(0, len(framed) - 1, config.max_poses_per_position)
    return [framed[int(round(j))][1] for j in idx]


def _move_arm_to(
    follower,
    target_sim: dict[str, float],
    gripper_rad: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addrs: dict[str, int],
    viewer,
    config: CalibrationConfig,
) -> None:
    """Smoothstep the real arm onto a sim-frame set point, mirroring it in the viewer."""
    target_real = sim_frame_to_real(target_sim, gripper_rad)
    current = action_to_joints(follower.get_observation(), target_real)
    delta = target_real - current
    steps = max(1, round(config.move_duration_s * CONTROL_HZ))
    period = 1.0 / CONTROL_HZ
    for i in range(1, steps + 1):
        if viewer is not None and not viewer.is_running():
            return
        step_start = time.time()
        interp = current + _smoothstep(i / steps) * delta
        follower.send_action(joints_to_action(interp))
        arm_rad, grip_rad = real_frame_to_sim(interp)
        for name in ARM_JOINT_NAMES:
            data.qpos[qpos_addrs[name]] = arm_rad[name]
        data.qpos[qpos_addrs["gripper"]] = grip_rad
        mujoco.mj_forward(model, data)
        if viewer is not None:
            viewer.sync()
        remaining = period - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)


def _measure_pose(
    follower,
    wrist_cap,
    detector,
    wrist_camera_matrix: np.ndarray,
    undistort_map,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addrs: dict[str, int],
    ids: dict[str, int],
    wrist_cam_id: int,
    cube_overhead_world: np.ndarray,
    group: str,
    config: CalibrationConfig,
) -> JointZeroSample | None:
    """Detect the cube in the wrist view and form one hand-eye sample.

    The arm is already parked at the look-at pose. Uses the follower's joint
    readback (the servos' actual position) for forward kinematics, exactly as the
    offline fit uses the recorded ``observation.state``: the wrist-camera pose it
    predicts, minus where the overhead camera says the cube truly is, is the
    joint-zero error the fit removes.
    """
    arm_rad, grip_rad = real_frame_to_sim(action_to_joints(follower.get_observation(), np.zeros(6)))
    for name in ARM_JOINT_NAMES:
        data.qpos[qpos_addrs[name]] = arm_rad[name]
    data.qpos[qpos_addrs["gripper"]] = grip_rad
    mujoco.mj_forward(model, data)
    cam_pos = data.cam_xpos[wrist_cam_id].copy()
    cam_rot = data.cam_xmat[wrist_cam_id].reshape(3, 3).copy()

    # Flush the buffered frames so we solve what the arm sees now, not frames
    # captured while it was still moving.
    for _ in range(config.flush_frames):
        wrist_cap.read()
    best: object | None = None
    for _ in range(config.frames_per_pose):
        ok, bgr = wrist_cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
        estimate = estimate_cube_pose(rgb, detector, wrist_camera_matrix)
        if estimate is None:
            continue
        if float(np.linalg.norm(estimate.position)) > MAX_CUBE_DISTANCE:
            continue
        if best is None or estimate.reproj_px < best.reproj_px:
            best = estimate
    if best is None:
        return None

    _, cube_world_wrist = cube_pose_to_world(best, cam_pos, cam_rot)
    delta = np.asarray(cube_world_wrist, dtype=float) - cube_overhead_world
    return JointZeroSample(
        delta=delta, columns=build_columns(data, ids, cube_overhead_world), group=group
    )


# Radius/azimuth binning over the pickup zone, used to spread cube positions and
# to name the emptiest reachable bin when prompting the operator.
_RADIUS_EDGES = (PICKUP.inner_radius, 0.22, 0.32, PICKUP.outer_radius)
_AZIMUTH_EDGES_DEG = (-90.0, -30.0, 30.0, 90.0)


def _bin_of(kinematics: So101Kinematics, cube: CubePose) -> tuple[int, int]:
    dx = cube.x - kinematics.pan_axis[0]
    dy = cube.y - kinematics.pan_axis[1]
    radius = math.hypot(dx, dy)
    azimuth_deg = math.degrees(math.atan2(dy, dx))
    r_bin = sum(radius >= e for e in _RADIUS_EDGES[1:])
    a_bin = sum(azimuth_deg >= e for e in _AZIMUTH_EDGES_DEG[1:-1])
    return r_bin, a_bin


def _suggest_bin(
    visited: set[tuple[int, int]], kinematics: So101Kinematics
) -> tuple[int, int] | None:
    """Next reachable cube bin, spread as far as possible from those visited.

    Radius separation is weighted more heavily than azimuth because radius spread
    is what conditions the shoulder_lift / elbow_flex / wrist_flex split (their
    rotation axes are parallel, so a single radius leaves that split ill-posed);
    azimuth mainly fixes shoulder_pan, which is already well-observed. Returns
    None when no reachable bin is left.
    """
    n_r = len(_RADIUS_EDGES) - 1
    n_a = len(_AZIMUTH_EDGES_DEG) - 1
    candidates = [
        (r, a)
        for r in range(n_r)
        for a in range(n_a)
        if (r, a) not in visited and _bin_center_pose(kinematics, (r, a)) is not None
    ]
    if not candidates:
        return None
    if not visited:
        # No anchor yet: start at the outer radius for a long-lever first sample.
        return max(candidates, key=lambda b: b[0])

    def spread(b: tuple[int, int]) -> int:
        return min(2 * abs(b[0] - v[0]) + abs(b[1] - v[1]) for v in visited)

    return max(candidates, key=spread)


def _describe_bin(bin_: tuple[int, int]) -> str:
    r_bin, a_bin = bin_
    r_mid = 0.5 * (_RADIUS_EDGES[r_bin] + _RADIUS_EDGES[r_bin + 1])
    a_lo, a_hi = _AZIMUTH_EDGES_DEG[a_bin], _AZIMUTH_EDGES_DEG[a_bin + 1]
    a_mid = 0.5 * (a_lo + a_hi)
    side = "center" if abs(a_mid) < 15 else ("left" if a_mid > 0 else "right")
    return f"~{r_mid * 100:.0f} cm from the base, {side} (azimuth {a_lo:+.0f}..{a_hi:+.0f} deg)"


def _bin_center_pose(kinematics: So101Kinematics, bin_: tuple[int, int]) -> CubePose | None:
    """A representative pickup-zone cube pose at the center of ``bin_`` (or None)."""
    r_bin, a_bin = bin_
    r = 0.5 * (_RADIUS_EDGES[r_bin] + _RADIUS_EDGES[r_bin + 1])
    a = math.radians(0.5 * (_AZIMUTH_EDGES_DEG[a_bin] + _AZIMUTH_EDGES_DEG[a_bin + 1]))
    x = kinematics.pan_axis[0] + r * math.cos(a)
    y = kinematics.pan_axis[1] + r * math.sin(a)
    if not is_cube_pickup_allowed(x, y):
        return None
    return CubePose(x=x, y=y, z=CUBE_HALF_SIZE)


def _draw_operator_cue(
    viewer, current: CubePose | None, desired: CubePose | None
) -> bool:
    """Draw ghost cubes at the current (red) and/or desired (green) positions.

    Either pose may be ``None`` to omit that ghost (e.g. the cube is not yet
    detected, or there is no target). Uses the passive viewer's user scene;
    returns False for a headless viewer so the caller falls back to text.
    """
    scn = getattr(viewer, "user_scn", None)
    if scn is None:
        return False
    size = np.full(3, CUBE_HALF_SIZE, dtype=float)
    mat = np.eye(3, dtype=float).flatten()
    scn.ngeom = 0
    for pose, rgba in (
        (current, (0.9, 0.1, 0.1, 0.5)),
        (desired, (0.1, 0.9, 0.1, 0.5)),
    ):
        if pose is None:
            continue
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            size,
            np.array([pose.x, pose.y, pose.z], dtype=float),
            mat,
            np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1
    viewer.sync()
    return True


def _look_for_cube(
    follower,
    overhead_cap,
    camera_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addrs: dict[str, int],
    viewer,
    config: CalibrationConfig,
    rng: np.random.Generator,
) -> CubePose | None:
    """Locate the cube overhead, panning the arm clear of the camera between tries.

    After the arm relocates the cube (or parks afterward) it can sit between the
    fixed overhead camera and the cube, so a single look fails even though the
    cube is right there. Like the scripted run, retry from fresh wide
    shoulder-pan search poses that swing the arm out of the way. Returns an
    in-zone cube pose, or None if none is found within ``hunt_tries``.
    """
    for attempt in range(config.hunt_tries):
        if viewer is not None and not viewer.is_running():
            return None
        if attempt > 0:
            arm, grip = sample_hunt_pose(rng)
            print(f"Look {attempt + 1}/{config.hunt_tries}: panning to clear the overhead view...")
            _move_arm_to(follower, arm, grip, model, data, qpos_addrs, viewer, config)
            time.sleep(0.5)  # let the camera settle
        cube = track_cube(overhead_cap, camera_name, model, data, config.cube_search_timeout_s)
        if cube is not None:
            return cube
    return None


def _make_overhead_tracker(
    model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, overhead_cap
) -> Callable[[], CubePose | None]:
    """Build a fast single-frame overhead cube locator for live viewer feedback.

    Captures the (world-fixed) overhead camera pose and intrinsics once, then
    returns a callable that reads one frame and returns the cube's world XY (or
    ``None`` if not seen this frame) — no printing, no timeout, unlike
    :func:`track_cube`.
    """
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()
    intrinsics_path = LOCAL_CAMERA_INTRINSICS_DIR / f"{camera_name}.json"
    camera_matrix, undistort_map = load_intrinsics(intrinsics_path, 1920, 1080, cv2)
    detector = make_cube_detector()

    def track() -> CubePose | None:
        ok, bgr = overhead_cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if undistort_map is not None:
            rgb = cv2.remap(rgb, *undistort_map, cv2.INTER_LINEAR)
        estimate = estimate_cube_pose(rgb, detector, camera_matrix)
        if estimate is None:
            return None
        _, position = cube_pose_to_world(estimate, cam_pos, cam_rot)
        return CubePose(x=float(position[0]), y=float(position[1]), z=CUBE_HALF_SIZE)

    return track


def _prompt_with_live_cube(
    viewer,
    message: str,
    *,
    desired: CubePose | None,
    track: Callable[[], CubePose | None],
    prompt,
) -> bool:
    """Prompt the operator while live-tracking the cube in the viewer.

    The blocking ``prompt`` runs on a worker thread; on the calling thread we
    keep locating the cube overhead and redrawing it (red) — chasing the desired
    position (green) when one is given — until the operator answers. Falls back
    to a plain prompt for a headless viewer. Returns True on 'q'.
    """
    scn = getattr(viewer, "user_scn", None) if viewer is not None else None
    if scn is None:
        return prompt(message).strip() == "q"

    result: dict[str, str] = {}
    done = threading.Event()

    def _ask() -> None:
        try:
            result["value"] = prompt(message)
        finally:
            done.set()

    worker = threading.Thread(target=_ask, daemon=True)
    worker.start()
    last: CubePose | None = None
    while not done.is_set():
        if not viewer.is_running():
            break
        cube = track()
        if cube is not None:
            last = cube
        _draw_operator_cue(viewer, last, desired)
        time.sleep(0.05)
    worker.join()
    _clear_operator_cue(viewer)
    return result.get("value", "").strip() == "q"


def _clear_operator_cue(viewer) -> None:
    scn = getattr(viewer, "user_scn", None)
    if scn is not None:
        scn.ngeom = 0
        viewer.sync()


def _std_ok(result: FitResult, config: CalibrationConfig) -> bool:
    return all(result.std_deg[name] <= config.std_target_deg[name] for name in FIT_JOINTS)


def _print_fit(result: FitResult, n_samples: int) -> None:
    offsets = params_to_offsets_deg(result.params)
    rms = math.sqrt((result.residual**2).sum(axis=1).mean()) * 1000.0
    parts = ", ".join(
        f"{name}={offsets[name]:+.2f}°±{result.std_deg[name]:.2f}" for name in FIT_JOINTS
    )
    print(f"  fit (n={n_samples}, kept={int(result.keep.sum())}, residual {rms:.1f} mm): {parts}")


@dataclass
class CalibrationResult:
    offsets_deg: dict[str, float]
    fit: FitResult
    samples: list[JointZeroSample]
    positions: int


def run_session_calibration(
    *,
    follower,
    overhead_cap,
    wrist_cap,
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    kinematics: So101Kinematics,
    camera_name: str,
    wrist_intrinsics_path,
    config: CalibrationConfig | None = None,
    relocate_cube: Callable[[CubePose, CubePose], bool] | None = None,
    prompt=input,
) -> CalibrationResult:
    """Run the session-start joint-zero calibration.

    Drives the already-connected ``follower`` and open cameras; ``viewer`` may be
    a live MuJoCo viewer or ``None``. When ``relocate_cube(source, target)`` is
    given, the routine moves the cube to each new position itself (returning True
    on success) and only falls back to prompting the operator — with a 3D cue in
    the viewer — when it fails or is absent. Returns the fitted offsets (exporter
    sign, i.e. the amounts to add to the sim joints) plus diagnostics.
    """
    config = config or CalibrationConfig()
    qpos_addrs = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in JOINT_NAMES
    }
    ids = joint_ids(model)
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    wrist_camera_matrix, undistort_map = load_intrinsics(
        wrist_intrinsics_path, WRIST_WIDTH, WRIST_HEIGHT, cv2
    )
    detector = make_cube_detector()
    # Same collision model the scripted run trusts, so look-at poses that fold the
    # wrist into the upper arm (or the table) at short radius are never commanded.
    collision_free = make_carry_collision_checker(model, *build_geom_sets(model))
    # Fast single-frame overhead locator, used to show the cube live in the viewer
    # while the operator places or moves it.
    track_live = _make_overhead_tracker(model, data, camera_name, overhead_cap)
    rng = np.random.default_rng()

    samples: list[JointZeroSample] = []
    visited: set[tuple[int, int]] = set()
    result: FitResult | None = None
    positions = 0

    while positions < config.max_positions:
        cube = _look_for_cube(
            follower, overhead_cap, camera_name, model, data, qpos_addrs, viewer, config, rng
        )
        if cube is None or not is_cube_pickup_allowed(cube.x, cube.y):
            print("No cube localized in the pickup zone.")
            if _prompt_with_live_cube(
                viewer,
                "Place the cube in the pickup zone, then Enter (q to stop): ",
                desired=None, track=track_live, prompt=prompt,
            ):
                break
            continue

        bin_ = _bin_of(kinematics, cube)
        visited.add(bin_)
        cube_world = np.array([cube.x, cube.y, cube.z], dtype=float)
        poses = lookat_poses(
            kinematics, cube, config,
            model=model, data=data, qpos_addrs=qpos_addrs,
            wrist_cam_id=wrist_cam_id, wrist_camera_matrix=wrist_camera_matrix,
            collision_free=collision_free,
        )
        print(f"Position {positions + 1} at bin {bin_}: {len(poses)} look-at poses.")

        detected = 0
        for pose in poses:
            if viewer is not None and not viewer.is_running():
                break
            _move_arm_to(follower, pose, config.gripper_rad, model, data, qpos_addrs, viewer, config)
            time.sleep(config.settle_s)
            sample = _measure_pose(
                follower, wrist_cap, detector, wrist_camera_matrix, undistort_map,
                model, data, qpos_addrs, ids, wrist_cam_id, cube_world, str(bin_), config,
            )
            if sample is not None:
                samples.append(sample)
                detected += 1
        positions += 1
        print(f"  detected the cube in {detected}/{len(poses)} poses.")

        if len(samples) < config.min_samples:
            print("  too few samples so far; need another cube position.")
        else:
            result = fit_robust(samples)
            _print_fit(result, len(samples))
            if positions >= config.min_positions and _std_ok(result, config):
                print("Fit converged.")
                break

        # Move the cube to a fresh bin: the robot does it itself when it can, and
        # only asks the operator (with a 3D cue) when it can't.
        desired_bin = _suggest_bin(visited, kinematics)
        if desired_bin is None:
            print("No further reachable cube positions to add; finishing with the current data.")
            break
        target = _bin_center_pose(kinematics, desired_bin)
        if relocate_cube is not None:
            print(f"Relocating the cube to bin {desired_bin} ({_describe_bin(desired_bin)})...")
            if relocate_cube(cube, target):
                continue
            print("  autonomous relocation failed; asking the operator.")

        live = getattr(viewer, "user_scn", None) is not None
        where = "the green box in the viewer" if live else _describe_bin(desired_bin)
        stop = _prompt_with_live_cube(
            viewer,
            f"Move the cube to {where}, then Enter (q to stop): ",
            desired=target, track=track_live, prompt=prompt,
        )
        if stop:
            break

    if result is None:
        if len(samples) < len(FIT_JOINTS):
            raise RuntimeError("not enough cube detections to fit joint zeros")
        result = fit_robust(samples)
        _print_fit(result, len(samples))

    return CalibrationResult(
        offsets_deg=params_to_offsets_deg(result.params),
        fit=result,
        samples=samples,
        positions=positions,
    )
