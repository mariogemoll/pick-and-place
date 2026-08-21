# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Show every attached camera's live feed, each in its own window."""

from __future__ import annotations

import argparse

import cv2

from pick_and_place.cli.suggest import SuggestingArgumentParser

MAX_CAMERAS = 10

# Uncompressed YUYV is the default on Linux, and two 640x480 streams do not fit
# in one USB controller's bandwidth: the second camera opens but every read()
# fails. MJPG compresses on the camera, so all the feeds stream together.
MJPG = cv2.VideoWriter_fourcc(*"MJPG")


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the all-cameras viewer."""
    parser = SuggestingArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=MAX_CAMERAS,
        help=f"highest OpenCV index to probe (default: {MAX_CAMERAS})",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Open every camera that answers, and show them until q is pressed."""
    cameras = []

    # Probing past the last camera is expected, so do not let the backend log a
    # multi-line error for every index that is not there.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

    for cam_id in range(args.max_cameras):
        cap = cv2.VideoCapture(cam_id)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, MJPG)
            ret, frame = cap.read()
            if ret:
                window_name = f"Camera {cam_id}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cameras.append((cam_id, cap, window_name))
                print(f"Opened camera {cam_id}")
            else:
                cap.release()
        else:
            cap.release()

    if not cameras:
        print("No cameras found")
        raise SystemExit(1)

    print("Press q in any window to quit")

    while True:
        for cam_id, cap, window_name in cameras:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for _, cap, _ in cameras:
        cap.release()

    cv2.destroyAllWindows()

def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
