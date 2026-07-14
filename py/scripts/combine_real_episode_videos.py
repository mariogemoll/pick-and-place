#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Combine the live camera views recorded by ``pick_and_place/real.py``.

The inputs are one or more ``episodes/<timestamp>`` run directories or individual
``episode_####`` directories. Their episodes are played in input order (and run
episodes in recorded order), with each episode's native-rate
``*_live.mp4`` camera views tiled into a single video.  ``--cameras`` sets the
row-major tile order and may include ``3d``: a sim replay rendered from the
workspace camera's own pose (solved by PnP from its recorded AprilTags) that
mirrors the real workspace view.  The default places it beneath that view
(``workspace,3d,overhead,wrist``); move the ``3d`` token to relocate the tile,
or pass ``--no-view3d`` to drop it.  When the 3D tile is shown each episode is
trimmed to the span that has sim data.  Tiles are 4:3 (16:9 views are
center-cropped) and silent.  The run must have been captured with
``--live-videos``.  Pass ``--poster`` to also write a lossless ``poster.png`` of the
grid's first frame beside the output.

Example:
    python scripts/combine_real_episode_videos.py \\
        episodes/20260712_191914/episode_0001 \\
        episodes/20260713_090000/episode_0003 --out website/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np


# Synthetic tile that replays the recorded simulation state in a 3D overview,
# rendered from the workspace camera's own pose (solved by PnP from its recorded
# AprilTag detections), so it mirrors the real workspace view. It is placed in
# the tile grid wherever its token appears in ``--cameras``.
VIEW3D_TILE = "view3d"
VIEW3D_CAMERA = "view3d_camera"
VIEW3D_TARGET_DOT = "view3d_target_dot"
# User-facing tokens in ``--cameras`` that select the 3D overview tile.
VIEW3D_TOKENS = frozenset({"3d", "view3d"})

# Default tile order: the 3D overview sits directly beneath the workspace view it
# mirrors, with the two camera task views (overhead, wrist) filling the rest.
DEFAULT_TILES = ("workspace", "3d", "overhead", "wrist")


@dataclass(frozen=True)
class Episode:
    """The files and timing needed to render one recorded episode."""

    directory: Path
    cameras: dict[str, Path]
    fps: float
    duration: float
    visual_overlays: tuple[dict, ...]
    tag_overlays: dict[str, tuple[dict, ...]]
    wrist_size: tuple[int, int]
    camera_sizes: dict[str, tuple[int, int]]
    live_origin_wall_t: float | None


def live_duration(paths: dict[str, Path]) -> float:
    """Return the shared-clock duration of a set of live camera files."""
    durations: list[float] = []
    for path in paths.values():
        with av.open(path) as container:
            if container.duration is None:
                raise ValueError(f"{path}: could not determine live video duration")
            durations.append(float(container.duration / av.time_base))
    # Live videos use the same monotonic PTS origin.  Taking the smallest
    # container duration tolerates minor muxing-rounding differences.
    return min(durations)


def load_visual_overlays(directory: Path) -> tuple[dict, ...]:
    """Load the saved AprilTag and visual-servo drawing primitives."""
    path = directory / "visual_servo_overlays.jsonl"
    if not path.is_file():
        return ()
    return tuple(json.loads(line) for line in path.read_text().splitlines() if line)


def load_tag_overlays(directory: Path, live_origin_wall_t: float | None) -> dict[str, tuple[dict, ...]]:
    """Load every camera's stored AprilTag corners in the live-video timebase."""
    path = directory / "tag_locations.jsonl"
    if not path.is_file() or live_origin_wall_t is None:
        return {}
    by_camera: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("timebase") == "timeline":
            continue
        tags = [tag["corners"] for tag in entry["tags"]]
        # Keep empty detections as explicit clear states. Otherwise, a tag
        # outline remains visible until a later frame happens to detect a tag.
        by_camera.setdefault(entry["camera"], []).append(
            {"t": float(entry["t"]) - live_origin_wall_t, "tags": tags}
        )
    overlays: dict[str, tuple[dict, ...]] = {}
    for camera, entries in by_camera.items():
        # Older recordings did not label their timebase. Their file interleaves
        # native frames with delayed control-tick snapshots; retain the first
        # sample in each 25 ms window, which is the native camera sample.
        entries.sort(key=lambda entry: entry["t"])
        unique = []
        for entry in entries:
            if not unique or entry["t"] - unique[-1]["t"] >= 0.025:
                unique.append(entry)
        overlays[camera] = tuple(unique)
    return overlays


def ass_timestamp(seconds: float) -> str:
    """Format a timestamp for an ASS dialogue line."""
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    return f"{hours}:{minutes:02d}:{remainder / 100:05.2f}"


def write_overlay_subtitles(
    path: Path,
    overlays: tuple[dict, ...],
    source_size: tuple[int, int],
    cell_size: tuple[int, int],
    duration: float,
    *,
    stroke_scale: float = 1.0,
) -> None:
    """Draw saved geometry in the fill-cropped, resized tile without text."""
    source_width, source_height = source_size
    cell_width, cell_height = cell_size
    # Match the tile's fill-and-center-crop: scale to cover the cell, then the
    # (possibly negative) offset re-centers, so geometry stays on the image.
    scale = max(cell_width / source_width, cell_height / source_height)
    offset_x = (cell_width - source_width * scale) / 2
    offset_y = (cell_height - source_height * scale) / 2

    def point(raw):
        return offset_x + raw[0] * scale, offset_y + raw[1] * scale

    # \p4 gives the vertices one-eighth-pixel precision. A filled polygon
    # supplies the stroke, avoiding ASS's border renderer and its outline.
    def coordinate(value: float) -> int:
        return round(value * 8)

    def line(start, end, color, width=2):
        """Return a filled, subpixel-width ASS polygon for one line segment."""
        x1, y1 = point(start)
        x2, y2 = point(end)
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return ""
        half_width = width * stroke_scale / 2
        normal_x = (y1 - y2) * half_width / length
        normal_y = (x2 - x1) * half_width / length

        vertices = (
            (x1 + normal_x, y1 + normal_y),
            (x2 + normal_x, y2 + normal_y),
            (x2 - normal_x, y2 - normal_y),
            (x1 - normal_x, y1 - normal_y),
        )
        commands = [
            f"m {coordinate(vertices[0][0])} {coordinate(vertices[0][1])}",
            *(f"l {coordinate(x)} {coordinate(y)}" for x, y in vertices[1:]),
            "c",
        ]
        return rf"{{\an7\pos(0,0)\p4\1c{color}\bord0\shad0}}{' '.join(commands)}{{\p0}}"

    def offset_polygon(vertices, distance):
        """Offset a convex polygon's edges leftward, preserving joined corners."""
        shifted = []
        for index, vertex in enumerate(vertices):
            previous = vertices[index - 1]
            following = vertices[(index + 1) % len(vertices)]
            before_x, before_y = vertex[0] - previous[0], vertex[1] - previous[1]
            after_x, after_y = following[0] - vertex[0], following[1] - vertex[1]
            before_length = math.hypot(before_x, before_y)
            after_length = math.hypot(after_x, after_y)
            if before_length == 0 or after_length == 0:
                shifted.append(vertex)
                continue
            before_normal = (-before_y / before_length, before_x / before_length)
            after_normal = (-after_y / after_length, after_x / after_length)
            first = (vertex[0] + before_normal[0] * distance, vertex[1] + before_normal[1] * distance)
            second = (vertex[0] + after_normal[0] * distance, vertex[1] + after_normal[1] * distance)
            cross = before_x * after_y - before_y * after_x
            if abs(cross) < 1e-6:
                shifted.append(((first[0] + second[0]) / 2, (first[1] + second[1]) / 2))
                continue
            delta_x, delta_y = second[0] - first[0], second[1] - first[1]
            factor = (delta_x * after_y - delta_y * after_x) / cross
            shifted.append((first[0] + factor * before_x, first[1] + factor * before_y))
        return shifted

    def outline(corners, color, width=2):
        """Return one closed polygon ring for a continuous AprilTag border."""
        vertices = [point(corner) for corner in corners]
        if len(vertices) < 3:
            return ""
        area = sum(
            vertex[0] * following[1] - following[0] * vertex[1]
            for vertex, following in zip(vertices, vertices[1:] + vertices[:1])
        )
        half_width = width * stroke_scale / 2
        outer = offset_polygon(vertices, -half_width if area > 0 else half_width)
        inner = offset_polygon(vertices, half_width if area > 0 else -half_width)

        def path(points):
            return [
                f"m {coordinate(points[0][0])} {coordinate(points[0][1])}",
                *(f"l {coordinate(x)} {coordinate(y)}" for x, y in points[1:]),
                "c",
            ]

        return (
            rf"{{\an7\pos(0,0)\p4\1c{color}\bord0\shad0}}"
            f"{' '.join(path(outer) + path(list(reversed(inner))))}{{\\p0}}"
        )

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {cell_width}",
        f"PlayResY: {cell_height}",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: Servo,Arial,28,&H0000FFFF,&H0000FFFF,&H00000000,&H96000000,1,0,0,0,100,100,0,0,"
        "1,2,1,7,20,20,20,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    for index, overlay in enumerate(overlays):
        start = max(0.0, float(overlay["t"]))
        end = min(
            duration,
            float(overlays[index + 1]["t"]) if index + 1 < len(overlays) else start + 1 / 30,
        )
        drawings = []
        for corners in overlay.get("tags", []):
            drawings.append(outline(corners, "&H00FF00&"))
        drawings.extend(line(start, end, "&H00A5FF&") for start, end in overlay.get("cube_edges", []))
        if end <= start or not drawings:
            continue
        for drawing in drawings:
            lines.append(
                f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},Servo,,0,0,0,,{drawing}"
            )
    path.write_text("\n".join(lines) + "\n")


def escape_filter_path(path: Path) -> str:
    """Escape a filesystem path used as an ffmpeg filter option."""
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


class _RecordedTag:
    """A saved AprilTag detection, shaped like a live ``solve_camera_pose`` input."""

    __slots__ = ("tag_id", "corners", "center")

    def __init__(self, tag_id: int, corners) -> None:
        self.tag_id = tag_id
        self.corners = np.asarray(corners, dtype=float)
        self.center = self.corners.mean(axis=0)


def build_view3d_model(render_size: tuple[int, int]):
    """Compile the standard scene with a free ``view3d_camera`` on the world body.

    The cube's free joint makes the model's ``qpos`` line up with the recorded
    ``sim_qpos``; the camera is left at the origin for ``solve_view3d_pose`` to
    place from the workspace-frame tags.
    """
    import mujoco

    from pick_and_place import build_scene
    from pick_and_place.episodes import cube_quat_from_pose
    from pick_and_place.geometry import CUBE_HALF_SIZE, CubePose
    from pick_and_place.workspace_overlays import PAN_AXIS

    width, height = render_size
    source = CubePose(x=PAN_AXIS[0] + 0.1, y=PAN_AXIS[1], z=CUBE_HALF_SIZE)
    # A finite tabletop lets the rendered view retain the physical setup's
    # visible north edge instead of extending the floor indefinitely.
    spec = build_scene(include_environment=True, tabletop=True)
    target_dot = spec.worldbody.add_body(name=VIEW3D_TARGET_DOT)
    target_dot.add_geom(
        name=f"{VIEW3D_TARGET_DOT}_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(0.0, 0.0, 0.004),
        size=(0.008, 0.0, 0.0),
        rgba=(0.1, 0.7, 1.0, 0.0),
        contype=0,
        conaffinity=0,
    )
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, height)
    spec.worldbody.add_camera(name=VIEW3D_CAMERA)
    cube = spec.body("pick_cube")
    cube.pos = (source.x, source.y, source.z)
    cube.quat = cube_quat_from_pose(source)
    cube.add_freejoint()
    model = spec.compile()
    return model, mujoco.MjData(model)


def place_view3d_target_dot(model, data, episode_metadata: dict) -> None:
    """Show the recorded drop-target centre as a dot when metadata provides it."""
    import mujoco

    metadata = episode_metadata.get("episode_metadata", {})
    target_x = metadata.get("target_x")
    target_y = metadata.get("target_y")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{VIEW3D_TARGET_DOT}_geom")
    if geom_id < 0:
        return
    if target_x is None or target_y is None:
        model.geom_rgba[geom_id, 3] = 0.0
        return
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, VIEW3D_TARGET_DOT)
    model.body_pos[body_id] = (float(target_x), float(target_y), 0.0)
    model.geom_rgba[geom_id, 3] = 1.0
    mujoco.mj_forward(model, data)


def solve_view3d_pose(model, data, directory: Path, matrix: np.ndarray) -> bool:
    """Place ``view3d_camera`` at the workspace camera's PnP-solved pose.

    Averages a solve over every recorded workspace frame that shows at least one
    workspace-frame tag. Returns ``False`` (leaving the camera unmoved) when the
    recording has no usable workspace tag detections.
    """
    import mujoco

    from pick_and_place.cam_align_solve import (
        TAG_GEOMS,
        average_results,
        solve_camera_pose,
    )
    import cv2

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, VIEW3D_CAMERA)
    nominal_pos = model.cam_pos[camera_id].copy()
    nominal_quat = model.cam_quat[camera_id].copy()
    dist = np.zeros(5)
    results = []
    for line in (directory / "tag_locations.jsonl").read_text().splitlines():
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("camera") != "workspace":
            continue
        detections = [
            _RecordedTag(tag["id"], tag["corners"])
            for tag in entry["tags"]
            if tag["id"] in TAG_GEOMS
        ]
        if not detections:
            continue
        result = solve_camera_pose(
            frame_rgb=None,
            model=model,
            data=data,
            camera_name=VIEW3D_CAMERA,
            matrix=matrix,
            dist=dist,
            detector=None,
            detections=detections,
            min_workspace_tags=1,
            cv2_module=cv2,
            nominal_pos=nominal_pos,
            nominal_quat=nominal_quat,
        )
        if result is not None:
            results.append(result)
    if not results:
        model.cam_pos[camera_id] = nominal_pos
        model.cam_quat[camera_id] = nominal_quat
        mujoco.mj_forward(model, data)
        return False
    average = average_results(results, nominal_pos=nominal_pos, nominal_quat=nominal_quat)
    model.cam_pos[camera_id] = np.array(average.pos, dtype=float)
    model.cam_quat[camera_id] = np.array(average.quat, dtype=float)
    mujoco.mj_forward(model, data)
    return True


def render_view3d(
    model,
    data,
    sim_qpos: np.ndarray,
    times: np.ndarray,
    duration: float,
    out_path: Path,
    render_size: tuple[int, int],
) -> None:
    """Render the sim replay to an MP4 sharing the live videos' timebase.

    ``times`` are the sim frames' offsets from the live-capture origin. The first
    pose is held from t=0 to cover the gap before the first recorded control tick
    (the live cameras are already rolling then), and the final pose is held to
    ``duration`` so the tile spans the whole grid.
    """
    import av
    import mujoco

    width, height = render_size
    renderer = mujoco.Renderer(model, height=height, width=width)
    time_base = Fraction(1, 1_000_000)
    container = av.open(str(out_path), "w")
    stream = container.add_stream("libx264", rate=30, options={"preset": "ultrafast"})
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = time_base
    stream.codec_context.time_base = time_base

    def encode(image: np.ndarray, pts: int) -> None:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        for packet in stream.encode(frame):
            container.mux(packet)

    def render(qpos: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=VIEW3D_CAMERA)
        return renderer.render().copy()

    last_pts = -1
    last_image: np.ndarray | None = None
    if len(times) and float(times[0]) > 1e-4:
        encode(render(sim_qpos[0]), 0)
        last_pts = 0
    for qpos, offset in zip(sim_qpos, times):
        last_image = render(qpos)
        last_pts = max(last_pts + 1, round(float(offset) * 1_000_000))
        encode(last_image, last_pts)
    end_pts = round(duration * 1_000_000)
    if last_image is not None and end_pts > last_pts:
        encode(last_image, end_pts)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    renderer.close()


def render_view3d_videos(
    episodes: list["Episode"], cell_size: tuple[int, int], temp_dir: Path
) -> tuple[dict[Path, Path], dict[Path, tuple[float, float]]]:
    """Render a workspace-pose sim overview per episode; empty if unavailable.

    Returns the rendered tile per episode and the ``[first_tick, last_tick]``
    window (in the live-video timebase) that the whole grid is trimmed to, so
    the combined video only spans the phase with sim data. Every episode must
    carry workspace intrinsics, tag detections and a sim timeline; if any lacks
    them the 3D tile is skipped for the whole run so the grid stays uniform.
    """
    metadata = {}
    for episode in episodes:
        camera = json.loads((episode.directory / "episode.json").read_text()).get(
            "cameras", {}
        ).get("workspace")
        timeline = episode.directory / "timeline.npz"
        tags = episode.directory / "tag_locations.jsonl"
        if camera is None or "camera_matrix" not in camera or not timeline.is_file() or not tags.is_file():
            print("3D view skipped: a workspace camera, its tags, and a sim timeline are required.")
            return {}, {}
        metadata[episode.directory] = camera

    # Render at the workspace aspect ratio so the solved intrinsics stay correct;
    # ffmpeg pads the tile into the requested cell afterwards.
    first = metadata[episodes[0].directory]
    aspect = int(first["width"]) / int(first["height"])
    render_height = cell_size[1]
    render_width = max(2, round(render_height * aspect / 2) * 2)
    render_size = (render_width, render_height)

    model, data = build_view3d_model(render_size)
    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, VIEW3D_CAMERA)
    videos: dict[Path, Path] = {}
    windows: dict[Path, tuple[float, float]] = {}
    for index, episode in enumerate(episodes):
        episode_metadata = json.loads((episode.directory / "episode.json").read_text())
        camera = metadata[episode.directory]
        matrix = np.array(camera["camera_matrix"], dtype=float)
        model.cam_fovy[camera_id] = math.degrees(
            2.0 * math.atan((int(camera["height"]) / 2.0) / matrix[1, 1])
        )
        if not solve_view3d_pose(model, data, episode.directory, matrix):
            print(f"3D view skipped: no workspace tags solved in {episode.directory.name}.")
            return {}, {}
        place_view3d_target_dot(model, data, episode_metadata)
        timeline = np.load(episode.directory / "timeline.npz")
        sim_qpos = np.asarray(timeline["sim_qpos"], dtype=float)
        if sim_qpos.ndim != 2 or sim_qpos.shape[1] != model.nq:
            print(f"3D view skipped: {episode.directory.name} sim_qpos does not match the scene.")
            return {}, {}
        origin = episode.live_origin_wall_t
        wall_t = np.asarray(timeline["wall_t"], dtype=float)
        times = wall_t - (origin if origin is not None else (wall_t[0] if len(wall_t) else 0.0))
        out_path = temp_dir / f"episode_{index:04d}_view3d.mp4"
        render_view3d(model, data, sim_qpos, times, episode.duration, out_path, render_size)
        videos[episode.directory] = out_path
        start = max(0.0, float(times[0]))
        end = min(episode.duration, float(times[-1]) + 1.0 / episode.fps)
        windows[episode.directory] = (start, max(end, start))
    return videos, windows


def load_episode(directory: Path, camera_names: tuple[str, ...]) -> Episode:
    """Load one complete recorded episode."""
    metadata_path = directory / "episode.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{directory}: missing episode.json")
    metadata = json.loads(metadata_path.read_text())
    cameras = metadata.get("cameras", {})
    missing = [name for name in camera_names if name not in cameras]
    if missing:
        raise ValueError(f"{directory}: missing requested camera(s): {', '.join(missing)}")
    live_videos = {name: camera.get("live_video") for name, camera in cameras.items()}
    unavailable = [name for name in camera_names if not live_videos.get(name)]
    if unavailable:
        raise ValueError(
            f"{directory}: no live video for {', '.join(unavailable)}; "
            "record with real.py --live-videos"
        )
    paths = {name: directory / live_videos[name] for name in camera_names}
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{directory}: missing video file(s): {', '.join(absent)}")
    fps = float(metadata["fps"])
    if fps <= 0:
        raise ValueError(f"{directory}: fps must be positive")
    wrist_camera = cameras.get("wrist", {})
    return Episode(
        directory=directory,
        cameras=paths,
        fps=fps,
        duration=live_duration(paths),
        visual_overlays=load_visual_overlays(directory),
        tag_overlays=load_tag_overlays(directory, metadata.get("live_capture_origin_wall_t")),
        wrist_size=(int(wrist_camera["width"]), int(wrist_camera["height"])),
        camera_sizes={name: (int(camera["width"]), int(camera["height"])) for name, camera in cameras.items()},
        live_origin_wall_t=metadata.get("live_capture_origin_wall_t"),
    )


def load_episodes(root: Path, camera_names: tuple[str, ...]) -> list[Episode]:
    """Load complete episodes in a run's recorded order."""
    directories = [directory for directory in sorted(root.glob("episode_*")) if (directory / "episode.json").is_file()]
    if not directories:
        raise FileNotFoundError(f"No complete episode_#### directories found in {root}")
    return [load_episode(directory, camera_names) for directory in directories]


def missing_replay_data(episode: Episode, tiles: tuple[str, ...]) -> list[str]:
    """Return the recorded inputs absent from an episode's requested replay tiles."""
    missing: list[str] = []
    if "wrist" in tiles and not (episode.directory / "visual_servo_overlays.jsonl").is_file():
        missing.append("visual-servo overlays")
    if VIEW3D_TILE not in tiles:
        return missing

    metadata = json.loads((episode.directory / "episode.json").read_text())
    workspace = metadata.get("cameras", {}).get("workspace", {})
    if "camera_matrix" not in workspace:
        missing.append("workspace camera calibration")
    if not (episode.directory / "timeline.npz").is_file():
        missing.append("simulation timeline")
    tags_path = episode.directory / "tag_locations.jsonl"
    if not tags_path.is_file():
        missing.append("workspace AprilTag detections")
    elif not any(
        json.loads(line).get("camera") == "workspace" and json.loads(line).get("tags")
        for line in tags_path.read_text().splitlines()
        if line
    ):
        missing.append("workspace AprilTag detections")
    return missing


def parse_size(value: str) -> tuple[int, int]:
    """Parse an even ``WIDTHxHEIGHT`` cell size suitable for yuv420p."""
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("width and height must be positive even numbers")
    return width, height


def tile_source(episode: Episode, tile: str, view3d_videos: dict[Path, Path]) -> Path:
    """Return the input video for one tile of one episode."""
    return view3d_videos[episode.directory] if tile == VIEW3D_TILE else episode.cameras[tile]


def tile_filter(
    input_index: int,
    label: str,
    cell_size: tuple[int, int],
    fps: float,
    overlay: str,
    window: tuple[float, float],
) -> str:
    """Scale/crop one tile to the 4:3 cell, burn overlays, and trim to ``window``.

    The first ``setpts`` lands the clip on the shared live-video origin so the
    ``[start, end]`` trim (measured from that origin) drops the pre/post-roll;
    the second re-zeros the trimmed clip for concatenation.
    """
    cell_width, cell_height = cell_size
    start, end = window
    return (
        f"[{input_index}:v]fps={fps:g},"
        f"scale={cell_width}:{cell_height}:force_original_aspect_ratio=increase,"
        f"crop={cell_width}:{cell_height},"
        f"setsar=1{overlay},setpts=PTS-STARTPTS,"
        f"trim=start={start:.9f}:end={end:.9f},setpts=PTS-STARTPTS[{label}]"
    )


def episode_overlay(
    episode: Episode, tile: str, overlay_subtitles: dict[tuple[Path, str], list[Path]]
) -> str:
    """Concatenated ``ass`` filter clauses for one tile of one episode."""
    overlay = ""
    for subtitle_path in overlay_subtitles.get((episode.directory, tile), []):
        overlay += f",ass=filename='{escape_filter_path(subtitle_path)}'"
    return overlay


def build_command(
    episodes: list[Episode],
    tiles: tuple[str, ...],
    cell_size: tuple[int, int],
    fps: float,
    output: Path,
    overlay_subtitles: dict[tuple[Path, str], list[Path]],
    view3d_videos: dict[Path, Path],
    trim_windows: dict[Path, tuple[float, float]],
    crf: int,
    preset: str,
    *,
    still: bool = False,
) -> list[str]:
    """Build the ffmpeg invocation that tiles then joins every episode.

    ``tiles`` is the row-major tile order and may include ``VIEW3D_TILE``. With
    ``still`` the output is a single poster frame instead of the full video.
    """
    cell_width, cell_height = cell_size
    columns = min(2, len(tiles))
    layout = "|".join(
        f"{(index % columns) * cell_width}_{(index // columns) * cell_height}"
        for index in range(len(tiles))
    )
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y"]
    filters: list[str] = []
    grid_labels: list[str] = []
    input_index = 0

    for episode_index, episode in enumerate(episodes):
        window = trim_windows.get(episode.directory, (0.0, episode.duration))
        view_labels: list[str] = []
        for camera_index, camera in enumerate(tiles):
            command.extend(("-i", str(tile_source(episode, camera, view3d_videos))))
            label = f"e{episode_index}c{camera_index}"
            overlay = episode_overlay(episode, camera, overlay_subtitles)
            filters.append(tile_filter(input_index, label, cell_size, fps, overlay, window))
            view_labels.append(f"[{label}]")
            input_index += 1
        grid_label = f"e{episode_index}v"
        filters.append(
            "".join(view_labels)
            + f"xstack=inputs={len(view_labels)}:layout={layout}:fill=black[{grid_label}]"
        )
        grid_labels.append(f"[{grid_label}]")
        if still:
            break

    filters.append("".join(grid_labels) + f"concat=n={len(grid_labels)}:v=1:a=0[video]")
    command.extend(("-filter_complex", ";".join(filters), "-map", "[video]"))
    if still:
        command.extend(("-frames:v", "1", "-c:v", "png", "-update", "1", str(output)))
    else:
        command.extend(
            (
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                "-profile:v", "high", "-level:v", "4.1", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output),
            )
        )
    return command


def prepare_render_inputs(
    episodes: list[Episode],
    camera_names: tuple[str, ...],
    tiles: tuple[str, ...],
    cell_size: tuple[int, int],
    temp_dir: Path,
    *,
    overlay_stroke_scale: float = 1.0,
) -> tuple[dict[tuple[Path, str], list[Path]], dict[Path, Path], dict[Path, tuple[float, float]]]:
    """Create cell-size-specific overlays and optional 3D replay tiles."""
    overlay_subtitles: dict[tuple[Path, str], list[Path]] = {}
    for index, episode in enumerate(episodes):
        for camera in camera_names:
            tags = episode.tag_overlays.get(camera, ())
            if tags:
                subtitle_path = temp_dir / f"episode_{index:04d}_{camera}_tags.ass"
                write_overlay_subtitles(
                    subtitle_path,
                    tags,
                    episode.camera_sizes[camera],
                    cell_size,
                    episode.duration,
                    stroke_scale=overlay_stroke_scale,
                )
                overlay_subtitles[episode.directory, camera] = [subtitle_path]
        if "wrist" in camera_names and episode.visual_overlays:
            subtitle_path = temp_dir / f"episode_{index:04d}_servo.ass"
            write_overlay_subtitles(
                subtitle_path,
                episode.visual_overlays,
                episode.wrist_size,
                cell_size,
                episode.duration,
                stroke_scale=overlay_stroke_scale,
            )
            overlay_subtitles.setdefault((episode.directory, "wrist"), []).append(subtitle_path)
    view3d_videos, trim_windows = (
        render_view3d_videos(episodes, cell_size, temp_dir)
        if VIEW3D_TILE in tiles
        else ({}, {})
    )
    return overlay_subtitles, view3d_videos, trim_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="run directory or individual episode_#### directory; sources are joined in this order",
    )
    parser.add_argument("--out", type=Path, default=None, help="output MP4 (default: RUN/combined.mp4)")
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_TILES),
        help="comma-separated tiles in row-major order, filling a 2-column grid; "
        "use '3d' for the sim overview placed anywhere among the camera views "
        f"(default: {','.join(DEFAULT_TILES)})",
    )
    parser.add_argument(
        "--cell",
        type=parse_size,
        default=(720, 540),
        help="tile size; 16:9 camera views are center-cropped to fill it (default: 720x540, 4:3)",
    )
    parser.add_argument("--fps", type=float, default=None, help="output frame rate (default: recording rate)")
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 constant-quality value, 0-51; lower is higher quality and larger (default: 23)",
    )
    parser.add_argument(
        "--preset",
        default="slow",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"),
        help="H.264 encoder speed/compression tradeoff (default: slow)",
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help="also write lossless poster.png (the tiled grid's first frame) beside the output",
    )
    parser.add_argument(
        "--poster-cell",
        type=parse_size,
        default=None,
        help="tile size for --poster (default: --cell)",
    )
    parser.add_argument(
        "--no-view3d",
        dest="view3d",
        action="store_false",
        help="drop the '3d' sim-overview tile from the layout",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="include episodes missing recorded overlay or 3D-replay inputs",
    )
    args = parser.parse_args()

    tiles = tuple(
        VIEW3D_TILE if name.strip() in VIEW3D_TOKENS else name.strip()
        for name in args.cameras.split(",")
        if name.strip()
    )
    if len(set(tiles)) != len(tiles):
        parser.error("--cameras must not repeat a tile")
    if not args.view3d:
        tiles = tuple(tile for tile in tiles if tile != VIEW3D_TILE)
    camera_names = tuple(tile for tile in tiles if tile != VIEW3D_TILE)
    if not camera_names:
        parser.error("--cameras must name at least one camera view")
    episodes: list[Episode] = []
    for source in args.sources:
        if (source / "episode.json").is_file():
            episode = load_episode(source, camera_names)
            episodes.append(episode)
        else:
            episodes.extend(load_episodes(source, camera_names))
    if not args.include_incomplete:
        complete_episodes = []
        for episode in episodes:
            missing = missing_replay_data(episode, tiles)
            if missing:
                print(f"Skipping {episode.directory.name}: missing {', '.join(missing)}.")
            else:
                complete_episodes.append(episode)
        episodes = complete_episodes
        if not episodes:
            parser.error("No episodes contain all requested replay inputs")
    fps = args.fps if args.fps is not None else episodes[0].fps
    if fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if args.out is None and len(args.sources) > 1:
        parser.error("--out is required when combining multiple sources")
    output = args.out if args.out is not None else args.sources[0] / "combined.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="combine-real-episode-videos-") as temp_dir:
        render_temp_dir = Path(temp_dir) / "video"
        render_temp_dir.mkdir()
        overlay_subtitles, view3d_videos, trim_windows = prepare_render_inputs(
            episodes, camera_names, tiles, args.cell, render_temp_dir
        )
        # Drop the 3D tile if its render was unavailable so the grid stays whole.
        if VIEW3D_TILE in tiles and not view3d_videos:
            tiles = tuple(tile for tile in tiles if tile != VIEW3D_TILE)

        subprocess.run(
            build_command(
                episodes, tiles, args.cell, fps, output,
                overlay_subtitles, view3d_videos, trim_windows, args.crf, args.preset,
            ),
            check=True,
        )
        print(f"Wrote {output} ({len(episodes)} episode(s), {len(tiles)} tile(s))")
        if args.poster:
            poster_path = output.with_name("poster.png")
            poster_cell = args.poster_cell if args.poster_cell is not None else args.cell
            poster_temp_dir = Path(temp_dir) / "poster"
            poster_temp_dir.mkdir()
            poster_stroke_scale = min(
                poster_cell[0] / args.cell[0], poster_cell[1] / args.cell[1]
            )
            poster_overlays, poster_view3d, poster_windows = prepare_render_inputs(
                episodes,
                camera_names,
                tiles,
                poster_cell,
                poster_temp_dir,
                overlay_stroke_scale=poster_stroke_scale,
            )
            poster_tiles = tiles if VIEW3D_TILE not in tiles or poster_view3d else tuple(
                tile for tile in tiles if tile != VIEW3D_TILE
            )
            subprocess.run(
                build_command(
                    episodes, poster_tiles, poster_cell, fps, poster_path,
                    poster_overlays, poster_view3d, poster_windows, args.crf, args.preset,
                    still=True,
                ),
                check=True,
            )
            print(f"Wrote {poster_path}")


if __name__ == "__main__":
    main()
