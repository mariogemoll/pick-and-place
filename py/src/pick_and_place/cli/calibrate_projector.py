# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Solve where the projector's pixels land on the workspace floor.

Projects a Gray code sequence, watches it with the overhead camera, and fits the
projector-pixel to workspace-metre homography. The projector may stand anywhere;
its pose is solved for, never measured.

**Exposure is locked for the whole sequence.** Every bit is read by comparing a
pattern against its inverse, so a camera that re-exposes between the two would
turn its own gain change into decoded bits. Auto-exposure is therefore switched
off and a fixed value used, which is also why the sequence starts with white and
black references -- they establish what "lit" means at that one exposure.

**The corner plates are read off the black reference.** With the projector dark
the four printed tags sit in plain ambient light, which is the condition their
detector was tuned for; under the projected pattern they would be striped over.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from pick_and_place.calibration.gray_code import (
    DEFAULT_STRIPE_PX,
    GrayCodePlan,
    correspondences,
    decode,
    frames,
)
from pick_and_place.calibration.projector_solve import (
    camera_to_workspace,
    restrict_to_workspace,
    solve_projector_to_workspace,
    workspace_plate_centers,
)
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.camera_calibration import (
    LOCAL_CAMERA_INTRINSICS_DIR,
    load_camera_intrinsics,
)
from pick_and_place.core.paths import REPO_ROOT
from pick_and_place.hardware.projector import blank, read_framebuffer_info, show
from pick_and_place.perception.image_rectify import build_undistort_map

DEFAULT_OUTPUT = REPO_ROOT / "config" / "projector" / "overhead_projector.json"


def build_parser() -> SuggestingArgumentParser:
    """Flags for ``pap calibrate-projector``."""
    parser = SuggestingArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--camera-index", type=int, default=2, help="V4L2 index of the overhead camera"
    )
    parser.add_argument(
        "--stripe-px",
        type=int,
        default=DEFAULT_STRIPE_PX,
        help="projector-pixel width of the finest coded stripe",
    )
    parser.add_argument(
        "--exposure", type=int, default=156, help="fixed V4L2 exposure for the sequence"
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.35,
        help="seconds to wait after projecting each pattern before capturing",
    )
    parser.add_argument("--fb-index", type=int, default=0, help="framebuffer index (/dev/fb<N>)")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="where to write the fit"
    )
    parser.add_argument(
        "--half-extent",
        type=float,
        default=0.230,
        help="half-width in metres of the board region to fit over",
    )
    parser.add_argument("--dry-run", action="store_true", help="solve but do not write the file")
    return parser


def _capture_sequence(args: argparse.Namespace, plan: GrayCodePlan, info) -> list[np.ndarray]:
    """Project each pattern in turn and grab one settled frame of it."""
    import cv2

    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, info.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, info.height)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual on V4L2
    camera.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    if not camera.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index}")

    captures: list[np.ndarray] = []
    try:
        for index, frame in enumerate(frames(plan)):
            show(frame, info)
            time.sleep(args.settle)
            # Drain the driver's queue so the frame kept is the projected one
            # rather than one buffered before the pattern changed.
            grabbed = None
            for _ in range(6):
                ok, grabbed = camera.read()
                if not ok:
                    raise SystemExit(f"camera read failed on pattern {index}")
            captures.append(cv2.cvtColor(grabbed, cv2.COLOR_BGR2GRAY))
            print(f"\r  captured {index + 1}/{plan.frame_count}", end="", flush=True)
        print()
    finally:
        camera.release()
        blank(info)
    return captures


def _detect_plates(gray: np.ndarray) -> dict[int, np.ndarray]:
    """Find the four workspace corner plates in an ambient-lit frame."""
    from pupil_apriltags import Detector

    detector = Detector(families="tagStandard41h12", nthreads=4, refine_edges=True)
    wanted = set(workspace_plate_centers())
    return {
        detection.tag_id: np.asarray(detection.center, dtype=float)
        for detection in detector.detect(gray)
        if detection.tag_id in wanted
    }


def run(args: argparse.Namespace) -> None:
    """Capture the sequence, solve the homography, and report how well it fits."""
    import cv2

    info = read_framebuffer_info(args.fb_index)
    plan = GrayCodePlan(info.width, info.height, stripe_px=args.stripe_px)
    print(
        f"{info.device} {info.width}x{info.height}: {plan.x_cells}x{plan.y_cells} cells "
        f"at {plan.stripe_px}px, {plan.frame_count} exposures"
    )

    captures = _capture_sequence(args, plan, info)

    intrinsics = load_camera_intrinsics(LOCAL_CAMERA_INTRINSICS_DIR / "overhead_camera.json")
    height, width = captures[0].shape
    undistort = build_undistort_map(intrinsics, width, height, cv2)
    rectified = [cv2.remap(c, undistort[0], undistort[1], cv2.INTER_LINEAR) for c in captures]

    plates = _detect_plates(rectified[1])
    print(f"corner plates detected: {sorted(plates)}")
    cam_to_ws = camera_to_workspace(plates)

    decoded = decode(rectified, plan)
    matched = correspondences(decoded, plan)
    print(f"decoded {decoded.coverage * 100:.1f}% of the frame into {len(matched)} cells")

    on_board_projector, on_board_camera = restrict_to_workspace(
        matched.projector_xy, matched.camera_xy, cam_to_ws, half_extent_m=args.half_extent
    )
    print(
        f"{len(on_board_projector)} of {len(matched)} cells landed on the board "
        f"(the rest spilled onto surfaces at other heights)"
    )

    fit = solve_projector_to_workspace(
        on_board_projector, on_board_camera, cam_to_ws, cv2_module=cv2
    )
    print(
        f"fit over {fit.point_count} inliers: "
        f"rms {fit.rms_mm:.2f} mm, worst {fit.max_mm:.2f} mm"
    )

    if args.dry_run:
        print("dry run; nothing written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "solved": date.today().isoformat(),
                "frame": "projector pixels to workspace-frame xy, metres",
                "projector_size": [info.width, info.height],
                "stripe_px": plan.stripe_px,
                "homography": fit.matrix.tolist(),
                "method": "Gray code structured light (pick_and_place.calibrate_projector)",
                "reference_tags": sorted(plates),
                "inlier_count": fit.point_count,
                "rms_mm": round(fit.rms_mm, 4),
                "max_mm": round(fit.max_mm, 4),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.output}")
