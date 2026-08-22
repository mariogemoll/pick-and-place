# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate the standard vector A4 ChArUco PDF used by camera calibration.

Print the resulting PDF at 100% scale, without "fit to page", and verify one
square with a ruler before calibrating.

Example:

    cd py
    pap generate-charuco-board
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pick_and_place.calibration.charuco_board import make_board
from pick_and_place.cli.calibration import add_charuco_board_arguments
from pick_and_place.cli.common import add_output_argument
from pick_and_place.cli.suggest import SuggestingArgumentParser
from pick_and_place.core.paths import outputs_root


def write_pdf(path: Path, board, squares_x: int, squares_y: int, square_mm: float) -> None:
    """Write the OpenCV ChArUco board as native vector rectangles in an A4 PDF."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen.canvas import Canvas
    except ImportError as exc:
        raise SystemExit("board generation requires reportlab") from exc

    width = squares_x * square_mm
    height = squares_y * square_mm
    page_width, page_height = A4
    origin_x = (page_width - width * mm) / 2.0
    origin_y = (page_height - height * mm) / 2.0
    dictionary = board.getDictionary()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path), pagesize=A4, pageCompression=1)

    def rect(x: float, y: float, rect_width: float, rect_height: float, *, white: bool) -> None:
        canvas.setFillColorRGB(1.0, 1.0, 1.0) if white else canvas.setFillColorRGB(0.0, 0.0, 0.0)
        # ReportLab's origin is bottom-left; OpenCV's board coordinates are top-left.
        canvas.rect(
            origin_x + x * mm,
            origin_y + (height - y - rect_height) * mm,
            rect_width * mm,
            rect_height * mm,
            stroke=0,
            fill=1,
        )

    rect(0.0, 0.0, width, height, white=True)
    for row in range(squares_y):
        for column in range(squares_x):
            if (row + column) % 2 == 0:
                rect(column * square_mm, row * square_mm, square_mm, square_mm, white=False)

    # OpenCV supplies the exact marker locations and identifiers used by the
    # ChArUco board. A marker is a black border with white cells for bit value 1.
    for marker_id, corners_m in zip(board.getIds().ravel(), board.getObjPoints()):
        x0, y0 = (float(value) * 1000.0 for value in corners_m[0][:2])
        x1, _ = (float(value) * 1000.0 for value in corners_m[2][:2])
        marker_size = x1 - x0
        bits = dictionary.getBitsFromByteList(
            dictionary.bytesList[int(marker_id) : int(marker_id) + 1], 4
        )
        cell_size = marker_size / (bits.shape[0] + 2)
        rect(x0, y0, marker_size, marker_size, white=False)
        white_cells = canvas.beginPath()
        for row, values in enumerate(bits):
            for column, bit in enumerate(values):
                if bit:
                    white_cells.rect(
                        origin_x + (x0 + (column + 1) * cell_size) * mm,
                        origin_y + (height - y0 - (row + 2) * cell_size) * mm,
                        cell_size * mm,
                        cell_size * mm,
                    )
        canvas.setFillColorRGB(1.0, 1.0, 1.0)
        canvas.drawPath(white_cells, stroke=0, fill=1)
    canvas.showPage()
    canvas.save()


def build_parser() -> SuggestingArgumentParser:
    """Return the parser for the board renderer."""
    parser = SuggestingArgumentParser(description=__doc__)
    add_output_argument(parser, help="destination PDF for the printable board")
    add_charuco_board_arguments(parser)
    return parser


def validate(parser: SuggestingArgumentParser, args: argparse.Namespace) -> None:
    """Reject a board the parser cannot express, before anything is rendered.

    Only what is decidable from the arguments themselves. Whether OpenCV is
    installed and whether it has ChArUco support are facts about the machine,
    not about the command line, so they stay in :func:`run` as the runtime
    failures they are.
    """
    if args.squares_x < 3 or args.squares_y < 3:
        parser.error("the board needs at least 3 squares in each direction")
    if args.output is not None and args.output.suffix.lower() != ".pdf":
        parser.error("--output must be a .pdf file")
    if args.marker_mm >= args.square_mm:
        parser.error("--marker-mm must be smaller than --square-mm")


def run(args: argparse.Namespace) -> None:
    """Render the board at the requested geometry."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("board generation requires opencv-python") from exc
    if not hasattr(cv2, "aruco"):
        raise SystemExit("this OpenCV build has no ChArUco support; install opencv-python >= 4.7")

    output = args.output or outputs_root() / "charuco_a4_6x8_30mm.pdf"
    board = make_board(cv2, args.squares_x, args.squares_y, args.square_mm, args.marker_mm)

    write_pdf(output, board, args.squares_x, args.squares_y, args.square_mm)
    print(f"Wrote {output} (vector A4 PDF)")
    print(
        f"Board: {args.squares_x}x{args.squares_y} squares, {args.square_mm:g} mm squares, "
        f"{args.marker_mm:g} mm markers, DICT_4X4_50"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    run(args)


if __name__ == "__main__":
    main()
