# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Show one camera's live feed in a window."""

from __future__ import annotations

import argparse

import cv2

from pick_and_place.cli.suggest import SuggestingArgumentParser


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the single-camera viewer."""
    parser = SuggestingArgumentParser(description=__doc__)
    parser.add_argument(
        "camera",
        type=int,
        nargs="?",
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    """Show the feed until q is pressed."""
    camera_id = args.camera
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera {camera_id}")

    print(f"Showing camera {camera_id}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        cv2.imshow(f"Camera {camera_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()

def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
