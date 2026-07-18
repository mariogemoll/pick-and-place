#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Session-start joint-zero calibration driver.

Sets up the real rig (overhead + wrist cameras, follower, sim model with solved
overhead extrinsics) and runs the report-only calibration routine in
``pick_and_place.session_calibration``: it measures the four arm joint zeros
"du jour" by driving the wrist camera through look-at orbits around the cube at
several operator-placed positions, then persists them to
``config/joint_zeros.json``.

The fitted values are the amounts to *add* to the sim joints (exporter sign),
directly comparable to the offline per-day values from ``fit_joint_zeros.py``.
This stage only reports them; wiring them into the scripted pipeline as a
feed-forward correction is a separate step. Run it, compare the printed zeros to
the known offline day values, and sanity-check the sign with a one-joint sweep
before trusting the correction.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from pick_and_place.episodes import _build_model
from pick_and_place.follower import (
    action_to_joints,
    make_so101_follower,
    real_frame_to_sim,
)
from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
from pick_and_place.kinematics import derive_kinematics
from pick_and_place.cam_align_solve import parse_index_or_path
from pick_and_place.overhead_detection import MockViewer
from pick_and_place.session_calibration import (
    CalibrationConfig,
    run_session_calibration,
)
from pick_and_place.trajectory import REST_ARM_JOINTS, REST_GRIPPER
from pick_and_place.workspace_overlays import PAN_AXIS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "config" / "joint_zeros.json"


def _open_camera(spec: str, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    cap = cv2.VideoCapture(parse_index_or_path(spec), backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # noqa: BLE001 - backend may not support it; flush covers us
        pass
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Could not open camera {spec!r}.")
    return cap


class _WristCamera:
    """Holds the wrist capture so it can be handed to the executor and reopened.

    The calibration routine reads frames through ``.read()``; autonomous
    relocation releases the device (so ``execute_episode`` can open it for the
    descent servo) and reopens it afterwards, transparently to the routine.
    """

    def __init__(self, spec: str, width: int, height: int) -> None:
        self._spec, self._w, self._h = spec, width, height
        self.cap = _open_camera(spec, width, height)

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reopen(self) -> None:
        self.cap = _open_camera(self._spec, self._w, self._h)


def _persist(output: Path, day: str, result, config: CalibrationConfig) -> None:
    residual_mm = float(np.sqrt((result.fit.residual**2).sum(axis=1).mean()) * 1000.0)
    entry = {
        "day": day,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "offsets_deg": {k: round(v, 3) for k, v in result.offsets_deg.items()},
        "std_deg": {k: round(v, 3) for k, v in result.fit.std_deg.items()},
        "positions": result.positions,
        "samples": len(result.samples),
        "residual_mm": round(residual_mm, 2),
    }
    store = {"latest": None, "history": []}
    if output.exists():
        store = json.loads(output.read_text())
        if store.get("latest") is not None:
            store["history"].insert(0, store["latest"])
    store["latest"] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(store, indent=1) + "\n")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="overhead camera index or path")
    parser.add_argument("--camera-name", default="overhead_camera", help="overhead camera name")
    parser.add_argument("--wrist-camera", default="1", help="wrist camera index or path")
    parser.add_argument("--overhead-intrinsics", type=Path, default=None)
    parser.add_argument(
        "--wrist-intrinsics",
        type=Path,
        default=None,
        help="wrist intrinsics JSON (default: local sidecar)",
    )
    parser.add_argument("--follower-port", required=True, help="follower serial port")
    parser.add_argument("--follower-id", default="folly", help="follower calibration id")
    parser.add_argument(
        "--viewer", action="store_true", help="show the 3D MuJoCo viewer (default: headless)"
    )
    parser.add_argument(
        "--no-recalibrate",
        action="store_true",
        help="use the saved overhead extrinsics instead of solving live at startup",
    )
    parser.add_argument(
        "--no-auto-relocate",
        action="store_true",
        help="always ask the operator to move the cube instead of the robot relocating it",
    )
    parser.add_argument("--day", default=datetime.date.today().strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from pick_and_place.camera_extrinsics import (
        apply_camera_extrinsics_to_model,
        load_local_camera_extrinsics,
    )
    from pick_and_place.camera_intrinsics import LOCAL_CAMERA_INTRINSICS_DIR

    wrist_intrinsics = args.wrist_intrinsics or LOCAL_CAMERA_INTRINSICS_DIR / "wrist_camera.json"
    if not wrist_intrinsics.exists():
        raise SystemExit(f"Missing wrist intrinsics at {wrist_intrinsics}.")

    print("Building scene...")
    dummy = CubePose(x=PAN_AXIS[0] + 0.24, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    model, data = _build_model(dummy, include_environment=True)
    # The autonomous-relocation fallback runs the hardware executor, whose 30 Hz
    # control loop requires a timestep that divides evenly into it. The stock
    # 500 Hz model timestep does not, so match the hardware runner's rate.
    from pick_and_place.executor import HARDWARE_SIMULATION_HZ

    model.opt.timestep = 1.0 / HARDWARE_SIMULATION_HZ
    apply_camera_extrinsics_to_model(model, load_local_camera_extrinsics())
    mujoco.mj_forward(model, data)
    kinematics = derive_kinematics(model)

    print("Opening overhead camera...")
    overhead_cap = _open_camera(args.camera, 1920, 1080)
    print("Opening wrist camera...")
    wrist = _WristCamera(args.wrist_camera, 1280, 720)

    print("Connecting to follower...")
    follower = make_so101_follower(
        args.follower_port, args.follower_id, disable_torque_on_disconnect=False
    )
    follower.connect()

    rng = np.random.default_rng()

    def relocate_cube(source, target) -> bool:
        """Pick the cube from ``source`` and place it at ``target`` (production
        pick-place + descent servo). Frees the wrist device for the executor and
        reopens it afterwards. Returns True only on a completed placement."""
        from pick_and_place.episodes import EpisodeSamplingError, prepare_episode
        from pick_and_place.executor import execute_episode

        current = action_to_joints(follower.get_observation(), np.zeros(6))
        start_joints, start_gripper = real_frame_to_sim(current)
        try:
            episode = prepare_episode(
                rng, source, target,
                start_joints=start_joints, start_gripper=start_gripper,
                model=model, data=data, include_environment=True, verbose=True,
            )
        except EpisodeSamplingError as exc:
            print(f"  could not plan a relocation: {exc}")
            return False
        wrist.release()
        try:
            status = execute_episode(
                episode,
                follower=follower,
                viewer=viewer,
                wrist_camera=args.wrist_camera,
                wrist_intrinsics=str(wrist_intrinsics),
            )
        except Exception as exc:  # noqa: BLE001 - a failed relocation just falls back to the operator
            print(f"  relocation aborted: {exc}")
            status = "restart"
        finally:
            wrist.reopen()
        return status == "success"

    viewer_ctx = MockViewer() if not args.viewer else mujoco.viewer.launch_passive(model, data)
    try:
        with viewer_ctx as viewer:
            if not args.no_recalibrate:
                from pick_and_place.cam_align_solve import (
                    ExtrinsicsSolveError,
                    apply_solve_result,
                    check_solve_plausible,
                    solve_overhead_extrinsics,
                )

                print("Solving overhead extrinsics from the workspace-frame tags...")
                result = solve_overhead_extrinsics(
                    model,
                    data,
                    overhead_cap,
                    camera_name=args.camera_name,
                    intrinsics_path=args.overhead_intrinsics,
                    cv2_module=cv2,
                )
                if result is None:
                    raise SystemExit(
                        "Overhead calibration failed: never saw all four workspace-frame tags."
                    )
                try:
                    check_solve_plausible(result)
                except ExtrinsicsSolveError as exc:
                    raise SystemExit(f"Overhead calibration rejected: {exc}") from exc
                apply_solve_result(model, data, args.camera_name, result)
                print(
                    f"Overhead extrinsics solved: {result.reprojection_error_px:.2f}px, "
                    f"{result.nominal_delta.translation_m * 1000.0:.1f}mm from nominal."
                )

            calibration = run_session_calibration(
                follower=follower,
                overhead_cap=overhead_cap,
                wrist_cap=wrist,
                viewer=viewer,
                model=model,
                data=data,
                kinematics=kinematics,
                camera_name=args.camera_name,
                wrist_intrinsics_path=wrist_intrinsics,
                relocate_cube=None if args.no_auto_relocate else relocate_cube,
            )

            print("\nFitted joint zero offsets (add to sim joints, exporter sign):")
            for name, value in calibration.offsets_deg.items():
                print(f"  {name}={value:+.2f}deg  (1-sigma {calibration.fit.std_deg[name]:.2f}deg)")
            _persist(args.output, args.day, calibration, CalibrationConfig())

            print("\nParking the arm to REST...")
            from pick_and_place.session_calibration import _move_arm_to

            qpos_addrs = {
                n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
            }
            _move_arm_to(
                follower, REST_ARM_JOINTS, REST_GRIPPER, model, data, qpos_addrs, viewer,
                CalibrationConfig(),
            )
    finally:
        overhead_cap.release()
        wrist.release()
        follower.disconnect()


if __name__ == "__main__":
    main()
