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

**The corner plates are read off both reference frames.** Under the projected
pattern they would be striped over, so only the flat white and flat black
exposures are usable -- and which of those works depends on the room, so both
are searched and the stronger read of each plate wins.
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
        "--plate-exposures",
        type=lambda value: [int(part) for part in value.split(",")],
        default=[156, 400, 900],
        help="comma-separated exposures to hunt the corner plates across",
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


def _open_camera(args: argparse.Namespace, info, cv2):
    """Open the overhead camera at the framebuffer's resolution."""
    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, info.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, info.height)
    if not camera.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index}")
    return camera


def _set_exposure(camera, value: int, cv2) -> None:
    """Pin the camera to one exposure, with auto-exposure off."""
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual on V4L2
    camera.set(cv2.CAP_PROP_EXPOSURE, value)


def _grab(camera, cv2) -> np.ndarray:
    """Return one settled grayscale frame, discarding what the driver had queued."""
    grabbed = None
    for _ in range(6):
        ok, grabbed = camera.read()
        if not ok:
            raise SystemExit("camera read failed")
    return cv2.cvtColor(grabbed, cv2.COLOR_BGR2GRAY)


def _plate_frames(camera, args: argparse.Namespace, info, cv2) -> list[np.ndarray]:
    """Bracket the plate hunt over projector state and exposure.

    Detecting the corner plates is a *separate* measurement from decoding the
    Gray code, and it wants a different exposure. The sequence needs one locked
    exposure so a pattern and its inverse are comparable; the plates just need
    to be legible, and on this rig one of them is only legible in a narrow band.
    Plate 14 sits where the room is dim and the projector's throw is weak: at
    exposure 156 it is too dark to decode and by 900 the frame around it has
    washed out, but at 400 it reads with a margin of 45. A single fixed exposure
    steps over that window and the solve fails with all four plates in plain
    sight.

    So the bracket runs before the exposure is locked, over both a dark and a
    lit projector, and :func:`_detect_plates` keeps whichever read of each plate
    came back strongest.
    """
    collected: list[np.ndarray] = []
    for level in (0, 255):
        show(np.full((info.height, info.width, 3), level, dtype=np.uint8), info)
        for exposure in args.plate_exposures:
            _set_exposure(camera, exposure, cv2)
            time.sleep(args.settle * 2)  # exposure changes settle slower than pattern swaps
            collected.append(_grab(camera, cv2))
    return collected


def _capture_sequence(camera, args: argparse.Namespace, plan: GrayCodePlan, info, cv2):
    """Project each pattern in turn and grab one settled frame of it."""
    _set_exposure(camera, args.exposure, cv2)
    time.sleep(args.settle * 2)

    captures: list[np.ndarray] = []
    for index, frame in enumerate(frames(plan)):
        show(frame, info)
        time.sleep(args.settle)
        captures.append(_grab(camera, cv2))
        print(f"\r  captured {index + 1}/{plan.frame_count}", end="", flush=True)
    print()
    return captures


def _detect_plates(*frames_gray: np.ndarray) -> dict[int, np.ndarray]:
    """Find the corner plates across several exposures, keeping the best read of each.

    Both reference frames are searched rather than just one, because which of
    them works depends on the room. Lit by the projector's white field the plates
    read with a far stronger decision margin -- 55 against 12 in a dark room --
    but the throw does not cover every plate equally, so a plate outside the
    bright part is found only on the dark reference. Taking the better read of
    each plate gets all four in conditions where neither frame alone would.
    """
    from pick_and_place.perception.cube_detection import make_tag_detector

    detector = make_tag_detector()
    wanted = set(workspace_plate_centers())
    best: dict[int, tuple[float, np.ndarray]] = {}
    for gray in frames_gray:
        for detection in detector.detect(gray):
            if detection.tag_id not in wanted:
                continue
            margin = float(detection.decision_margin)
            if detection.tag_id not in best or margin > best[detection.tag_id][0]:
                best[detection.tag_id] = (margin, np.asarray(detection.center, dtype=float))
    return {tag_id: center for tag_id, (_, center) in best.items()}


def run(args: argparse.Namespace) -> None:
    """Capture the sequence, solve the homography, and report how well it fits."""
    import cv2

    info = read_framebuffer_info(args.fb_index)
    plan = GrayCodePlan(info.width, info.height, stripe_px=args.stripe_px)
    print(
        f"{info.device} {info.width}x{info.height}: {plan.x_cells}x{plan.y_cells} cells "
        f"at {plan.stripe_px}px, {plan.frame_count} exposures"
    )

    camera = _open_camera(args, info, cv2)
    try:
        plate_captures = _plate_frames(camera, args, info, cv2)
        captures = _capture_sequence(camera, args, plan, info, cv2)
    finally:
        camera.release()
        blank(info)

    intrinsics = load_camera_intrinsics(LOCAL_CAMERA_INTRINSICS_DIR / "overhead_camera.json")
    height, width = captures[0].shape
    undistort = build_undistort_map(intrinsics, width, height, cv2)

    def rectify(frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, undistort[0], undistort[1], cv2.INTER_LINEAR)

    rectified = [rectify(c) for c in captures]

    plates = _detect_plates(*(rectify(c) for c in plate_captures))
    print(
        f"corner plates detected: {sorted(plates)} "
        f"(from {len(plate_captures)} bracketed frames)"
    )
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
