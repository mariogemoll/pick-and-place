#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Find the rig's two cameras, work out which is overhead, and stream both.

Discovery walks ``/dev/v4l/by-path`` rather than probing OpenCV indices
blind: each USB camera exposes a capture node (``-video-index0``) and a
metadata node (``-video-index1``), and only the first is usable. Both rig
cameras report the same USB serial, so ``by-id`` collides and cannot tell
them apart; ``by-path`` keys on the physical port and does.

Identification is by what the camera can see. The workspace frame carries
four AprilTag corner plates (ids 12-15, ``tagStandard41h12``); the overhead
camera looks straight down at all four, the wrist camera does not. Most
corner tags visible wins.

    python3 camstream.py                 # identify, then serve on :8080
    python3 camstream.py --identify-only # just print the mapping and exit

Stop it (Ctrl-C) before running a policy, so the cameras are free.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from pick_and_place.cli.rig import add_capture_size_arguments

BOUNDARY = "frameboundary"
BY_PATH = Path("/dev/v4l/by-path")
TAG_FAMILY = "tagStandard41h12"
# Corner plates bolted to the workspace frame, from spec/workspace.py.
WORKSPACE_FRAME_TAG_IDS = frozenset({12, 13, 14, 15})


# --------------------------------------------------------------------------
# discovery


def device_name(index: int) -> str:
    """The kernel's name for a video node, e.g. 'Innomaker-U20CAM-1080p-S1'."""
    try:
        return Path(f"/sys/class/video4linux/video{index}/name").read_text().strip()
    except OSError:
        return ""


def by_path_map() -> dict[int, tuple[str, str]]:
    """video index -> (usb port, node kind), read from /dev/v4l/by-path."""
    mapping: dict[int, tuple[str, str]] = {}
    if not BY_PATH.is_dir():
        return mapping
    for link in sorted(BY_PATH.iterdir()):
        digits = re.search(r"(\d+)$", link.resolve().name)
        if digits is None:
            continue
        port = link.name.split("-usb-")[-1]
        kind = "capture" if port.endswith("-video-index0") else "metadata"
        port = re.sub(r"-video-index\d+$", "", port)
        mapping[int(digits.group(1))] = (port, kind)
    return mapping


def holders() -> dict[int, list[tuple[int, str]]]:
    """video index -> processes holding it open, from /proc/*/fd.

    A node that is merely *busy* looks identical to a broken one through
    OpenCV, which reports only that it could not open. Naming the holder is
    the difference between "unplug and replug" and "stop the policy run".
    """
    found: dict[int, list[tuple[int, str]]] = {}
    for fd_dir in Path("/proc").glob("*/fd"):
        try:
            pid = int(fd_dir.parent.name)
        except ValueError:
            continue
        try:
            entries = list(fd_dir.iterdir())
        except OSError:
            continue  # not ours, or gone
        for entry in entries:
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            match = re.fullmatch(r"/dev/video(\d+)", target)
            if match is None:
                continue
            try:
                comm = (fd_dir.parent / "comm").read_text().strip()
            except OSError:
                comm = "?"
            index = int(match.group(1))
            if pid not in [p for p, _ in found.get(index, [])]:
                found.setdefault(index, []).append((pid, comm))
    return found


class Node:
    """One /dev/videoN, whatever it turns out to be."""

    def __init__(self, index: int, name: str, port: str, kind: str) -> None:
        self.index = index
        self.name = name
        self.port = port
        self.kind = kind
        self.usable = False
        self.status = "not probed"
        self.resolution = ""
        self.role = ""
        self.tags = ""


def inventory(match: str) -> list[Node]:
    """Every video node on the box, probed to see which really capture.

    Listing the dead nodes matters as much as the live ones: each camera
    exposes a metadata sibling that opens cleanly under some backends but
    never yields a frame, and picking one by accident looks like a broken
    camera rather than a wrong index.
    """
    paths = by_path_map()
    busy = holders()
    nodes: list[Node] = []
    for device in sorted(Path("/dev").glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name))):
        index = int(re.sub(r"\D", "", device.name))
        name = device_name(index)
        if match and match.lower() not in name.lower():
            continue
        port, kind = paths.get(index, ("?", "?"))
        nodes.append(Node(index, name, port, kind))

    for node in nodes:
        if node.index in busy:
            who = ", ".join(f"pid {pid} {comm}" for pid, comm in busy[node.index])
            node.status = f"busy ({who})"
            continue
        cap = cv2.VideoCapture(node.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            node.status = "cannot open (V4L2)"
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            node.status = "opens, no frames"
            continue
        node.usable = True
        node.resolution = f"{frame.shape[1]}x{frame.shape[0]}"
        node.status = "capture"
    return nodes


# --------------------------------------------------------------------------
# identification


def grab(index: int, width: int, height: int, frames: int) -> list[np.ndarray]:
    """Take a few frames, discarding the first while exposure settles."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return []
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    grabbed: list[np.ndarray] = []
    try:
        for n in range(frames + 5):
            ok, frame = cap.read()
            if ok and frame is not None and n >= 5:
                grabbed.append(frame)
    finally:
        cap.release()
    return grabbed


def count_corner_tags(frames: list[np.ndarray]) -> tuple[int, set[int]]:
    """Best count of distinct workspace-frame tags seen in any single frame.

    Per frame rather than pooled across frames: a camera that catches one
    corner now and a different one later is not looking at the workspace.
    """
    from pupil_apriltags import Detector

    detector = Detector(families=TAG_FAMILY, nthreads=4, refine_edges=True)
    best: set[int] = set()
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        seen = {d.tag_id for d in detector.detect(gray)} & WORKSPACE_FRAME_TAG_IDS
        if len(seen) > len(best):
            best = seen
    return len(best), best


def identify(nodes: list[Node], width: int, height: int, frames: int) -> None:
    """Label the usable cameras 'overhead'/'wrist' by workspace-tag visibility."""
    usable = [node for node in nodes if node.usable]
    scores: dict[int, int] = {}
    for node in usable:
        shots = grab(node.index, width, height, frames)
        if not shots:
            node.role, node.tags = "unidentified", "no frames"
            continue
        count, ids = count_corner_tags(shots)
        scores[node.index] = count
        node.tags = ", ".join(str(i) for i in sorted(ids)) if ids else "none"
        print(f"  index {node.index} (usb {node.port}): {count}/4 corner tags [{node.tags}]")

    if not scores:
        return
    best = max(scores.values())
    if best == 0:
        print("\n  !! No workspace tags visible on any camera — cannot identify.")
        print("     Is the board in view and lit? Leaving both unlabelled.")
        for node in usable:
            node.role = "unidentified"
        return
    # Two cameras tying above zero means neither view is decisive.
    if list(scores.values()).count(best) > 1:
        print("\n  !! Tie on corner-tag count — labels are a guess, confirm by eye.")
    for node in usable:
        node.role = "overhead" if scores.get(node.index) == best else "wrist"


# --------------------------------------------------------------------------
# streaming


class Camera:
    """One capture device, grabbed continuously into a latest-frame slot."""

    def __init__(self, node: Node, width: int, height: int) -> None:
        self.index = node.index
        self.role = node.role
        self.port = node.port
        self.tags = node.tags
        self.width = width
        self.height = height
        self.frame: bytes | None = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # A one-frame buffer keeps the page showing now, not a second ago.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.frame = buf.tobytes()
        finally:
            cap.release()

    def latest(self) -> bytes | None:
        with self.lock:
            return self.frame

    def stop(self) -> None:
        self.running = False


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Rig cameras</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:1.5rem; background:#111; color:#eee;
         font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }}
  h1 {{ font-size:1.1rem; font-weight:600; margin:0 0 .25rem; }}
  p.sub {{ margin:0 0 1.25rem; color:#9aa; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:1.5rem; }}
  figure {{ margin:0; }}
  img {{ display:block; width:min(46vw,640px); height:auto;
         background:#000; border:1px solid #333; border-radius:6px; }}
  figcaption {{ margin-top:.5rem; font-weight:600; }}
  .role {{ color:#7ec8ff; }}
  .hint {{ color:#9aa; font-weight:400; }}
  code {{ background:#1c1c1c; padding:.1rem .35rem; border-radius:3px; }}
  h2 {{ font-size:.95rem; font-weight:600; margin:2rem 0 .5rem; }}
  .wrap {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:.35rem .8rem .35rem 0;
            border-bottom:1px solid #262626; white-space:nowrap; }}
  th {{ color:#9aa; font-weight:600; }}
  .ok {{ color:#7ee2a8; }}
  .dead {{ color:#e2a87e; }}
</style></head><body>
<h1>Rig cameras</h1>
<p class="sub">Identified by workspace-frame AprilTags (ids 12-15).
Run flags: <code>{flags}</code></p>
<div class="grid">{cards}</div>
<h2>All video nodes</h2>
<div class="wrap"><table>
<tr><th>node</th><th>status</th><th>kind</th><th>usb</th><th>role</th>
<th>corner tags</th><th>size</th><th>device</th></tr>
{rows}
</table></div>
</body></html>
"""

ROW = """<tr><td>/dev/video{index}</td><td class="{cls}">{status}</td><td>{kind}</td>
<td>{port}</td><td>{role}</td><td>{tags}</td><td>{resolution}</td><td>{name}</td></tr>"""

CARD = """<figure>
  <img src="/stream/{index}" alt="camera {index}">
  <figcaption><span class="role">{role}</span>
    <span class="hint">— index {index}, usb {port}, {tags}</span></figcaption>
</figure>"""


def build_handler(cameras: dict[int, Camera], nodes: list[Node], flags: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args) -> None:  # keep the console readable
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                cards = "".join(
                    CARD.format(
                        index=i,
                        role=(cameras[i].role or "unidentified").upper(),
                        port=cameras[i].port,
                        tags=f"tags {cameras[i].tags}" if cameras[i].tags else "",
                    )
                    for i in sorted(cameras)
                )
                rows = "".join(
                    ROW.format(
                        index=n.index,
                        cls="ok" if n.usable else "dead",
                        status=n.status,
                        kind=n.kind,
                        port=n.port,
                        role=n.role or "—",
                        tags=n.tags or "—",
                        resolution=n.resolution or "—",
                        name=n.name or "—",
                    )
                    for n in nodes
                )
                body = PAGE.format(cards=cards, rows=rows, flags=flags).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path.startswith("/stream/"):
                try:
                    index = int(self.path.rsplit("/", 1)[1])
                except ValueError:
                    self.send_error(404)
                    return
                camera = cameras.get(index)
                if camera is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
                )
                self.end_headers()
                try:
                    while True:
                        frame = camera.latest()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    return  # viewer closed the tab
                return

            self.send_error(404)

    return Handler


def lan_address() -> str:
    """Best-guess routable address of this box, for the URL to open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--match", default="", help="only nodes whose device name contains this (default: all)"
    )
    add_capture_size_arguments(parser, width=1280, height=720)
    parser.add_argument("--frames", type=int, default=8, help="frames to inspect per camera")
    parser.add_argument("--identify-only", action="store_true")
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()

    print("Video nodes on this box:")
    nodes = inventory(args.match)
    if not nodes:
        raise SystemExit("No /dev/video* nodes found.")
    for node in nodes:
        mark = "OK " if node.usable else "-- "
        size = f" {node.resolution}" if node.resolution else ""
        print(
            f"  {mark}/dev/video{node.index}  {node.status:<20} {node.kind:<9}"
            f" usb {node.port}{size}  {node.name}"
        )

    usable = [node for node in nodes if node.usable]
    if not usable:
        raise SystemExit("No node delivered a frame.")

    print("\nIdentifying by workspace-frame AprilTags ...")
    identify(nodes, args.width, args.height, args.frames)

    overhead = next((n.index for n in usable if n.role == "overhead"), None)
    wrist = next((n.index for n in usable if n.role == "wrist"), None)

    print("\nResult:")
    for node in usable:
        print(f"  index {node.index} -> {node.role or 'unidentified'}")
    flags = ""
    if overhead is not None and wrist is not None:
        flags = f"--camera {overhead} --wrist-camera {wrist}"
        print(f"\nRun the policy with:  {flags}")

    if args.identify_only:
        return

    cameras = {node.index: Camera(node, args.width, args.height) for node in usable}
    for camera in cameras.values():
        camera.start()

    server = ThreadingHTTPServer(
        (args.bind, args.port), build_handler(cameras, nodes, flags or "n/a")
    )
    server.daemon_threads = True
    print(f"\nStreaming. Open:  http://{lan_address()}:{args.port}/")
    print("Ctrl-C to stop (frees the cameras for the policy run).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        for camera in cameras.values():
            camera.stop()
        server.server_close()


if __name__ == "__main__":
    main()
