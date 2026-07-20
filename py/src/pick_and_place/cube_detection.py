# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Fuse the pick cube's visible AprilTag faces into a single cube pose.

The pick cube (:mod:`pick_and_place.scene`) carries one ``tagStandard41h12``
sticker on each of its six faces, ids :data:`CUBE_TAG_IDS`. Every visible tag
contributes its four decoded corners, at their known positions on the cube, to
**one** rigid PnP solve. Pooling corners this way fuses all visible faces in a
single least-squares pose and, for the common overhead case where only one face
is visible, lets us resolve the planar two-fold ambiguity (the orientation
"flip") by keeping the solution whose face actually points back at the camera.

The pose returned here lives in the **camera** frame (OpenCV convention: x
right, y down, z forward), oriented to match MuJoCo's cube *body* frame. Mapping
it into the world frame with the overhead camera extrinsics is a separate step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from pick_and_place.scene import PICK_CUBE_APRILTAG_IDS, PICK_CUBE_HALF_SIZE

# The cube tag ids, in MuJoCo cube-texture order (right, left, up, down, front,
# back). Imported from the scene so the detector and the rendered cube can never
# drift apart.
CUBE_TAG_IDS: tuple[int, ...] = PICK_CUBE_APRILTAG_IDS

# tagStandard41h12 geometry: total 9 cells wide, black border at 5 cells. The
# quad the detector returns spans the black border, so its metric edge is 5/9 of
# the printed graphic edge, not the full graphic.
TAG_BORDER_FRACTION = 5.0 / 9.0

# Printed AprilTag graphic edge on each 30 mm cube face (the cube texture spec in
# ``render_apriltag_textures.py`` renders a 20 mm tag on a 30 mm sticker).
CUBE_TAG_GRAPHIC_M = 0.020

# Tag-local corner offsets (z = 0 tag plane, in units of the black-border
# half-edge) in the order pupil-apriltags returns ``det.corners``. Scaled by the
# half-edge and pushed through each face transform, these give four metric 3-D
# points per visible face in the shared cube-centre frame for one joint PnP.
_TAG_CORNERS_LOCAL = np.array(
    [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=float
)

# Per-face rigid transform, tag frame -> shared cube-centre frame F, as
# (rotation rows, translation in units of the cube half-edge). The translation
# is the face centre, ``half * outward_normal``. These are fixed by the cube
# geometry and the texture layout; ``test_cube_detection`` plants the rendered
# cube at a known pose and checks the solve recovers it to well under a
# millimetre, which is what pins these values down.
_FACE_EXTRINSICS: dict[int, tuple[list[list[int]], list[int]]] = {
    0: ([[0, 0, -1], [0, 1, 0], [1, 0, 0]], [1, 0, 0]),
    1: ([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [-1, 0, 0]),
    2: ([[1, 0, 0], [0, 0, 1], [0, -1, 0]], [0, -1, 0]),
    3: ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], [0, 1, 0]),
    4: ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, -1]),
    5: ([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0, 1]),
}

# The cube-centre frame F above is fixed to a reference face and lands 180 deg
# about x off MuJoCo's cube-body frame (how the cubemap texture maps onto the
# box). Centres coincide, so only the orientation needs this constant relabel.
_CUBE_F_TO_BODY = np.diag([1.0, -1.0, -1.0])

# OpenCV camera frame (x right, y down, z forward) -> MuJoCo camera frame
# (x right, y up, z back): flip y and z.
_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def _face_transforms() -> dict[int, NDArray]:
    """Return ``T_F_tag``: each face's tag frame in the cube-centre frame F."""
    transforms: dict[int, NDArray] = {}
    for tag_id, (rows, offset) in _FACE_EXTRINSICS.items():
        transform = np.eye(4)
        transform[:3, :3] = np.array(rows, dtype=float)
        transform[:3, 3] = np.array(offset, dtype=float) * PICK_CUBE_HALF_SIZE
        transforms[tag_id] = transform
    return transforms


_FACE_T = _face_transforms()


@dataclass(frozen=True)
class CubePoseEstimate:
    """A pick-cube pose in the OpenCV camera frame, plus solve diagnostics.

    ``rotation`` and ``position`` give the cube body frame in camera coordinates
    (the cube centre is ``position``). ``reproj_px`` is the RMS corner
    reprojection error, and ``num_candidates`` is how many PnP solutions were
    weighed -- 2 means a single-face flip had to be disambiguated.
    """

    rotation: NDArray
    position: NDArray
    face_ids: tuple[int, ...]
    num_faces_total: int
    reproj_px: float
    num_candidates: int

    @property
    def num_faces_used(self) -> int:
        return len(self.face_ids)

    def matrix(self) -> NDArray:
        """Return the 4x4 camera-from-cube transform."""
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.position
        return transform


def make_cube_detector(quad_decimate: float = 1.0, nthreads: int = 4):
    """Create a pupil-apriltags detector tuned for the cube tags.

    ``quad_decimate`` downsamples the image for the quad-detection pass only;
    ``refine_edges`` still locates the final corners at full resolution. Values
    above 1.0 (e.g. 1.5-2.0) cut detection latency markedly for a small accuracy
    cost, which is the main lever for a snappier live feed.

    ``nthreads`` should stay at its default for the single-process live control
    loop, where lower per-call latency matters. Callers that run many detectors
    concurrently across OS processes (e.g. sharded sim recording) should pass
    ``nthreads=1`` to avoid oversubscribing the machine; it costs latency, not
    accuracy, since the algorithm itself is unaffected by thread count.

    Note that ``nthreads=1`` is *not* a fix for the rare ``libapriltag``
    segfault — that crash reproduces under strictly single-threaded detection,
    so it is not a data race however tempting that reading is. Crash
    containment is :mod:`pick_and_place.detector_process`, not this parameter.
    """
    from pupil_apriltags import Detector

    return Detector(
        families="tagStandard41h12",
        nthreads=nthreads,
        quad_decimate=float(quad_decimate),
        refine_edges=True,
    )


def detect_tags(frame_rgb: NDArray, detector) -> list:
    """Detect every AprilTag in an RGB frame (corners only, no per-tag pose)."""
    import cv2

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return detector.detect(gray)


def detect_cube_faces(frame_rgb: NDArray, detector) -> list:
    """Detect just the cube tags in an RGB frame."""
    return [det for det in detect_tags(frame_rgb, detector) if det.tag_id in _FACE_T]


def _tag_correspondences(detections) -> tuple[NDArray, NDArray, list[int]] | None:
    """Pool visible cube tags into object (cube-centre frame) / image corners."""
    half_edge = CUBE_TAG_GRAPHIC_M * TAG_BORDER_FRACTION / 2.0
    object_points: list[NDArray] = []
    image_points: list[NDArray] = []
    ids: list[int] = []
    for det in detections:
        transform = _FACE_T.get(det.tag_id)
        if transform is None:
            continue
        corners = (_TAG_CORNERS_LOCAL * half_edge) @ transform[:3, :3].T + transform[:3, 3]
        object_points.append(corners)
        image_points.append(np.asarray(det.corners, dtype=float))
        ids.append(det.tag_id)
    if not object_points:
        return None
    return np.concatenate(object_points), np.concatenate(image_points), sorted(ids)


def _rotation_angle_deg(a: NDArray, b: NDArray) -> float:
    """Shortest rotation angle (degrees) between two rotation matrices."""
    from scipy.spatial.transform import Rotation

    r_a = Rotation.from_matrix(a)
    r_b = Rotation.from_matrix(b)
    return float(np.degrees((r_a.inv() * r_b).magnitude()))


def _outside_face_score(rotation: NDArray, translation: NDArray, ids: list[int]) -> float:
    """Positive when the camera sees the *outside* of every visible face.

    For each face we take its centre and outward normal in camera coordinates;
    the dot of (camera -> face centre, reversed) with the outward normal is
    positive exactly when that face points back toward the camera. The minimum
    over visible faces flags a flipped (physically impossible) PnP solution.
    """
    scores = []
    for tag_id in ids:
        transform = _FACE_T[tag_id]
        face_centre = rotation @ transform[:3, 3] + translation
        outward = rotation @ (-transform[:3, 2])
        scores.append(float(np.dot(-face_centre, outward)))
    return min(scores) if scores else 0.0


def _solve_rigid_pose(
    object_points: NDArray,
    image_points: NDArray,
    ids: list[int],
    camera_matrix: NDArray,
    dist: NDArray,
    prior_rotation: NDArray | None = None,
):
    """Solve one rigid pose from pooled tag corners, resolving the planar flip.

    A single coplanar face is solved with IPPE, which returns both ambiguous
    solutions; several faces span depth and are solved with SQPNP. Among the
    physically valid candidates (face pointing at the camera) we keep, when a
    ``prior_rotation`` (camera-frame, body convention) is given, the one closest
    to it -- the correct single-face solution is stable frame to frame while the
    flip is a large jump, so temporal consistency rejects it. Without a prior we
    fall back to the lowest reprojection error. Returns
    ``(rotation, translation, num_candidates)`` or ``None``.
    """
    import cv2

    flags = cv2.SOLVEPNP_IPPE if len(ids) == 1 else cv2.SOLVEPNP_SQPNP
    retval, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, dist, flags=flags
    )
    if not retval:
        return None

    candidates = []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=float).ravel()
        error = float(np.asarray(errors).reshape(-1)[index]) if errors is not None else float("inf")
        invalid = not _outside_face_score(rotation, translation, ids) > 0.0
        if prior_rotation is None:
            key = (invalid, error)
        else:
            key = (invalid, _rotation_angle_deg(rotation @ _CUBE_F_TO_BODY, prior_rotation))
        candidates.append((key, rotation, translation))

    _, rotation, translation = min(candidates, key=lambda candidate: candidate[0])
    return rotation, translation, len(candidates)


def fuse_cube_faces(
    detections,
    camera_matrix: NDArray,
    dist: NDArray | None = None,
    prior_rotation: NDArray | None = None,
) -> CubePoseEstimate | None:
    """Solve one cube pose from all visible faces. ``None`` if none are usable.

    ``prior_rotation`` is the previous frame's ``CubePoseEstimate.rotation``; pass
    it to resolve the single-face flip by temporal consistency.
    """
    import cv2

    correspondences = _tag_correspondences(detections)
    if correspondences is None:
        return None
    object_points, image_points, ids = correspondences
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=float)

    solved = _solve_rigid_pose(
        object_points, image_points, ids, camera_matrix, dist, prior_rotation
    )
    if solved is None:
        return None
    rotation, translation, num_candidates = solved

    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(object_points, rvec, translation, camera_matrix, dist)
    reproj_px = float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean())

    return CubePoseEstimate(
        rotation=rotation @ _CUBE_F_TO_BODY,
        position=translation,
        face_ids=tuple(ids),
        num_faces_total=len(detections),
        reproj_px=reproj_px,
        num_candidates=num_candidates,
    )


def estimate_cube_pose(
    frame_rgb: NDArray,
    detector,
    camera_matrix: NDArray,
    dist: NDArray | None = None,
    prior_rotation: NDArray | None = None,
) -> CubePoseEstimate | None:
    """Detect the cube tags in one frame and solve a single cube pose."""
    return fuse_cube_faces(
        detect_cube_faces(frame_rgb, detector), camera_matrix, dist, prior_rotation
    )


def _average_rotation(rotations: list[NDArray], weights: NDArray) -> NDArray:
    """Weighted chordal mean of rotations, projected back onto SO(3)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()


class PoseEMA:
    """Low-pass a stream of ``(rotation, position)`` to damp per-frame jitter.

    ``alpha`` is the weight on the running estimate: 0 disables smoothing (pass
    the new pose straight through), higher is steadier but laggier. Position is
    blended linearly; rotation uses a chordal mean of the running and new
    rotations.

    Each ``update`` takes a ``confidence`` in ``[0, 1]`` that scales how hard the
    new sample pulls the estimate: the new-sample weight is
    ``(1 - alpha) * confidence``. A low-confidence frame (e.g. a lone, depth-blind
    single-face solve) only nudges the estimate, while a high-confidence frame
    (several faces) moves it at the full configured responsiveness.
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._rotation: NDArray | None = None
        self._position: NDArray | None = None

    def update(
        self, rotation: NDArray, position: NDArray, confidence: float = 1.0
    ) -> tuple[NDArray, NDArray]:
        rotation = np.asarray(rotation, dtype=float)
        position = np.asarray(position, dtype=float)
        if self._rotation is None or self.alpha <= 0.0:
            self._rotation, self._position = rotation, position
        else:
            new_weight = float(np.clip((1.0 - self.alpha) * confidence, 0.0, 1.0))
            weights = np.array([1.0 - new_weight, new_weight])
            self._rotation = _average_rotation([self._rotation, rotation], weights)
            self._position = (1.0 - new_weight) * self._position + new_weight * position
        return self._rotation, self._position


class OrientationStabilizer:
    """Reject transient orientation flips using a short history of accepted poses.

    A single-face tag pose has a two-fold flip ambiguity; even with the best
    per-frame pick, a noisy frame can occasionally jump to the mirror solution.
    This keeps a short window of recent rotations, offers their consensus as the
    next solve's prior (so a lone flipped frame can't capture the branch the way a
    single previous frame can), and holds back any frame that disagrees with that
    consensus by more than ``flip_deg`` -- unless the disagreement persists for
    ``confirm`` frames, which is genuine motion and re-anchors the window. Exposes
    the running rate of held (likely-flip) frames.

    A frame flagged ``authoritative`` (a multi-face solve, which is geometrically
    unambiguous -- no flip) is always accepted and re-anchors the window to it.
    This is the only way out of a wrong-flip lock: once the consensus seeds onto
    the mirror solution, single-face frames keep reinforcing it, but the first
    multi-face frame snaps the consensus back to the true orientation.
    """

    def __init__(self, window: int = 8, flip_deg: float = 20.0, confirm: int = 3):
        self._history: deque[NDArray] = deque(maxlen=max(1, window))
        self.flip_deg = float(flip_deg)
        self.confirm = int(confirm)
        self._pending = 0
        self._held: NDArray | None = None
        self._frames = 0
        self._rejected = 0

    def reference(self) -> NDArray | None:
        """Consensus rotation of the recent history, for use as the solve prior."""
        if not self._history:
            return None
        return _average_rotation(list(self._history), np.ones(len(self._history)))

    def update(self, rotation: NDArray, authoritative: bool = False) -> tuple[NDArray, bool]:
        """Accept ``rotation`` or hold the last good one. Returns ``(rotation, held)``.

        ``authoritative`` marks an unambiguous (multi-face) solve: it bypasses the
        flip check and re-anchors the consensus window to ``rotation``.
        """
        self._frames += 1
        if authoritative:
            self._history.clear()
            self._history.append(rotation)
            self._pending = 0
            self._held = rotation
            return rotation, False
        reference = self.reference()
        if reference is None or _rotation_angle_deg(rotation, reference) <= self.flip_deg:
            self._pending = 0
            self._history.append(rotation)
            self._held = rotation
            return rotation, False
        self._pending += 1
        if self._pending >= self.confirm:  # sustained disagreement -> real motion
            self._history.clear()
            self._history.append(rotation)
            self._pending = 0
            self._held = rotation
            return rotation, False
        self._rejected += 1
        return self._held, True

    @property
    def flip_rate(self) -> float:
        """Fraction of frames held back as likely flips."""
        return self._rejected / self._frames if self._frames else 0.0


def cube_pose_to_world(
    estimate: CubePoseEstimate,
    camera_position: NDArray,
    camera_rotation: NDArray,
) -> tuple[NDArray, NDArray]:
    """Map a camera-frame cube estimate into world coordinates.

    ``camera_position`` / ``camera_rotation`` are the MuJoCo camera frame in world
    (``data.cam_xpos`` and ``data.cam_xmat`` after ``mj_forward``). Returns the
    cube body ``(rotation, position)`` in the world frame, ready to drop into the
    matching MuJoCo body.
    """
    camera_position = np.asarray(camera_position, dtype=float)
    camera_rotation = np.asarray(camera_rotation, dtype=float)
    rotation = camera_rotation @ _CV_TO_MJ @ estimate.rotation
    position = camera_position + camera_rotation @ (_CV_TO_MJ @ estimate.position)
    return rotation, position


@dataclass(frozen=True)
class CubePose:
    """A fused, flip-resolved, stabilized cube pose in the **world** frame.

    ``held`` is true when this frame was rejected as a likely flip, in which case
    the pose is the last trusted one rather than the raw measurement. ``flip_rate``
    is the running fraction of held frames -- a live read on how ambiguous the
    current view is.
    """

    rotation: NDArray
    position: NDArray
    face_ids: tuple[int, ...]
    reproj_px: float
    flip_rate: float
    held: bool

    @property
    def num_faces(self) -> int:
        return len(self.face_ids)

    def matrix(self) -> NDArray:
        """Return the 4x4 world-from-cube transform."""
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.position
        return transform


class CubeTracker:
    """Track the pick cube from a calibrated camera into a world-frame pose.

    Owns the whole estimation pipeline so the viewer and the real-world controller
    share identical behaviour: detect -> fuse all visible faces with flip
    resolution -> reject transient single-face flips against a short history ->
    optionally smooth -> map to world. Pass ``smooth=0`` for the raw (lowest-lag)
    estimate, or a small EMA factor to steady a live view.

    Smoothing is confidence-weighted: a single visible face barely observes depth,
    so its frames pull the smoother at only ``single_face_weight`` of the usual
    authority, while two-or-more-face frames (which span depth) update fully. This
    keeps the pose from jumping each time a grazing side face flickers in and out.

    Flip rejection runs in the camera frame, which is exact for a fixed camera
    such as the overhead; a moving (wrist) camera would stabilize in world.
    """

    def __init__(
        self,
        *,
        smooth: float = 0.0,
        history: int = 8,
        single_face_weight: float = 0.25,
        quad_decimate: float = 1.0,
        detector=None,
    ):
        self.detector = detector if detector is not None else make_cube_detector(quad_decimate)
        self._ema = PoseEMA(smooth)
        self._stabilizer = OrientationStabilizer(window=history) if history > 0 else None
        self.single_face_weight = float(single_face_weight)
        self._last: CubePose | None = None

    def update(
        self,
        detections,
        camera_matrix: NDArray,
        camera_position: NDArray,
        camera_rotation: NDArray,
        *,
        dist: NDArray | None = None,
    ) -> CubePose | None:
        """Update from already-detected cube tags. ``None`` if none are usable."""
        prior = self._stabilizer.reference() if self._stabilizer is not None else None
        estimate = fuse_cube_faces(detections, camera_matrix, dist, prior_rotation=prior)
        if estimate is None:
            return None

        held = False
        if self._stabilizer is not None:
            # A multi-face solve is unambiguous, so let it override (and re-anchor)
            # the flip consensus instead of being held against a wrong lock.
            rotation_cam, held = self._stabilizer.update(
                estimate.rotation, authoritative=estimate.num_faces_used >= 2
            )
            estimate = replace(estimate, rotation=rotation_cam)
        flip_rate = self._stabilizer.flip_rate if self._stabilizer is not None else 0.0

        if held and self._last is not None:
            # Likely flip: keep the last trusted pose, just refresh diagnostics.
            self._last = replace(
                self._last, held=True, flip_rate=flip_rate,
                face_ids=estimate.face_ids, reproj_px=estimate.reproj_px,
            )
            return self._last

        # A lone face barely constrains depth, so it pulls the smoother gently;
        # two or more faces span depth and update at full responsiveness.
        confidence = 1.0 if estimate.num_faces_used >= 2 else self.single_face_weight
        rotation, position = cube_pose_to_world(estimate, camera_position, camera_rotation)
        rotation, position = self._ema.update(rotation, position, confidence)
        self._last = CubePose(
            rotation=rotation,
            position=position,
            face_ids=estimate.face_ids,
            reproj_px=estimate.reproj_px,
            flip_rate=flip_rate,
            held=held,
        )
        return self._last

    def update_frame(
        self,
        frame_rgb: NDArray,
        camera_matrix: NDArray,
        camera_position: NDArray,
        camera_rotation: NDArray,
        *,
        dist: NDArray | None = None,
    ) -> CubePose | None:
        """Detect the cube tags in ``frame_rgb`` and update in one call."""
        detections = detect_cube_faces(frame_rgb, self.detector)
        return self.update(detections, camera_matrix, camera_position, camera_rotation, dist=dist)
