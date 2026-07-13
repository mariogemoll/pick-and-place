#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Generate the standard vector A4 ChArUco PDF used by camera calibration.

Print the resulting PDF at 100% scale, without "fit to page", and verify one
square with a ruler before calibrating.

Example:

    cd py
    python scripts/generate_charuco_board.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


# A4 is 210 x 297 mm. This 180 x 240 mm board leaves a real printable margin
# on every side while still providing 35 ChArUco corners.
DEFAULT_SQUARES_X = 6
DEFAULT_SQUARES_Y = 8
DEFAULT_SQUARE_MM = 30.0
DEFAULT_MARKER_MM = 22.0


def make_board(cv2_module, squares_x: int, squares_y: int, square_mm: float, marker_mm: float):
    """Return the OpenCV ChArUco board shared with the calibration command."""
    if marker_mm >= square_mm:
        raise ValueError("marker size must be smaller than square size")
    dictionary = cv2_module.aruco.getPredefinedDictionary(cv2_module.aruco.DICT_4X4_50)
    return cv2_module.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm / 1000.0, marker_mm / 1000.0, dictionary
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/charuco_a4_6x8_30mm.pdf"))
    parser.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)
    parser.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("board generation requires opencv-python") from exc
    if not hasattr(cv2, "aruco"):
        raise SystemExit("this OpenCV build has no ChArUco support; install opencv-python >= 4.7")
    if args.squares_x < 3 or args.squares_y < 3:
        parser.error("the board needs at least 3 squares in each direction")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must be a .pdf file")
    try:
        board = make_board(cv2, args.squares_x, args.squares_y, args.square_mm, args.marker_mm)
    except ValueError as exc:
        parser.error(str(exc))

    write_pdf(args.output, board, args.squares_x, args.squares_y, args.square_mm)
    print(f"Wrote {args.output} (vector A4 PDF)")
    print(
        f"Board: {args.squares_x}x{args.squares_y} squares, {args.square_mm:g} mm squares, "
        f"{args.marker_mm:g} mm markers, DICT_4X4_50"
    )


if __name__ == "__main__":
    main()
