# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Small helpers for building SVG figures as plain strings."""

import base64
import mimetypes
from html import escape
from pathlib import Path

FONT = "Helvetica, Arial, sans-serif"


class Figure:
    def __init__(
        self,
        width: float,
        height: float,
        *,
        drawing_width: float | None = None,
        drawing_height: float | None = None,
    ):
        if (drawing_width is None) != (drawing_height is None):
            raise ValueError("drawing_width and drawing_height must be specified together")
        self.width = width
        self.height = height
        self.drawing_width = drawing_width or width
        self.drawing_height = drawing_height or height
        self.parts: list[str] = []

    def add(self, *parts: str) -> None:
        self.parts.extend(parts)

    def svg(self) -> str:
        head = (
            "<!-- SPDX-FileCopyrightText: 2026 Mario Gemoll -->\n"
            "<!-- SPDX-License-Identifier: 0BSD -->\n"
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            "<defs>\n"
            '<marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
            '<path d="M 0 0 L 7 3 L 0 6 z" fill="black"/>'
            "</marker>\n"
            "</defs>\n"
            f'<rect width="{self.width}" height="{self.height}" fill="white"/>\n'
        )
        scale_x = self.width / self.drawing_width
        scale_y = self.height / self.drawing_height
        drawing = "\n".join(self.parts)
        if scale_x != 1 or scale_y != 1:
            drawing = f'<g transform="scale({scale_x} {scale_y})">\n{drawing}\n</g>'
        return head + drawing + "\n</svg>\n"

    def save(self, path: Path) -> None:
        path.write_text(self.svg(), encoding="utf-8")
        print(f"wrote {path}")


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str = "white",
    stroke: str = "black",
    stroke_width: float = 1.2,
    rx: float = 0,
    dash: str | None = None,
) -> str:
    rounded_corner = f' rx="{rx}"' if rx else ""
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{rounded_corner}{dashed}/>'
    )


def text(
    x: float,
    y: float,
    content: str,
    size: float = 12,
    anchor: str = "middle",
    fill: str = "black",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(content)}</text>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, stroke_width: float = 1.4, dash: str | None = None) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="black" stroke-width="{stroke_width}"{dashed} marker-end="url(#arrowhead)"/>'
    )


def elbow_arrow(points: list[tuple[float, float]], radius: float = 12, stroke_width: float = 1.4) -> str:
    """Draw an arrow along an orthogonal polyline with rounded corners."""

    def sign(value: float) -> int:
        return (value > 0) - (value < 0)

    x0, y0 = points[0]
    path = [f"M {x0} {y0}"]
    for (previous_x, previous_y), (corner_x, corner_y), (next_x, next_y) in zip(
        points, points[1:-1], points[2:]
    ):
        path.append(f"L {corner_x - sign(corner_x - previous_x) * radius} {corner_y - sign(corner_y - previous_y) * radius}")
        path.append(f"Q {corner_x} {corner_y} {corner_x + sign(next_x - corner_x) * radius} {corner_y + sign(next_y - corner_y) * radius}")
    path.append(f"L {points[-1][0]} {points[-1][1]}")
    return f'<path d="{" ".join(path)}" fill="none" stroke="black" stroke-width="{stroke_width}" marker-end="url(#arrowhead)"/>'


def image(path: Path, x: float, y: float, width: float, height: float, stroke: str = "black", stroke_width: float = 1.2) -> str:
    """Embed a bitmap as a data URI, with a thin frame."""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        raise ValueError(f"could not determine MIME type for {path}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
        f'preserveAspectRatio="xMidYMid slice" href="data:{mime_type};base64,{data}"/>'
        + rect(x, y, width, height, fill="none", stroke=stroke, stroke_width=stroke_width)
    )
