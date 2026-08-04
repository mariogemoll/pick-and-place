# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Shared least-squares core for fitting arm joint zero offsets.

Perturbing a revolute joint (world axis ``a``, anchor ``p``) by ``dtheta``
changes the measured cube delta ``real - sim`` by ``-dtheta * (a x (q - p))`` in
world coordinates (``q`` = cube position): moving the sim joint toward the real
joint's true zero removes the offset the measurement sees. Stacking the four arm
joints (shoulder_pan / shoulder_lift / elbow_flex / wrist_flex) gives a linear
least-squares problem whose solution is the amount to *add* to each sim joint
(exporter sign) to null the offset.

Both paths that fit these offsets share this module:

- offline (``scripts/fit_joint_zeros.py``): ``sim`` is the overhead-localized
  cube rendered through the posed sim arm, ``real`` is the wrist detection;
- live (``session_calibration``): ``sim`` is the overhead-localized cube,
  ``real`` is the wrist detection lifted to world through the model-predicted
  wrist-camera pose.

Both produce the same ``delta = real - sim`` and the same design columns, so the
fit is identical and the live path needs no sim rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

FIT_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
TRIM_MAD_FACTOR = 4.0


@dataclass
class JointZeroSample:
    """One measured cube delta and its per-joint design columns."""

    delta: np.ndarray  # world, meters, real - sim
    columns: np.ndarray  # 3 x len(FIT_JOINTS): -a_j x (q - p_j)
    group: str = ""  # optional grouping label (e.g. recording day)


@dataclass
class FitResult:
    params: np.ndarray  # radians to add to each FIT_JOINTS sim joint
    residual: np.ndarray  # n_kept x 3, world meters
    keep: np.ndarray  # bool mask over the input samples
    std_deg: dict[str, float]  # 1-sigma parameter uncertainty, degrees
    design: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, len(FIT_JOINTS))))


def joint_ids(model: mujoco.MjModel) -> dict[str, int]:
    """MuJoCo joint ids for ``FIT_JOINTS`` (for ``data.xaxis`` / ``data.xanchor``)."""
    return {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in FIT_JOINTS
    }


def build_columns(
    data: mujoco.MjData, ids: dict[str, int], cube_world: np.ndarray
) -> np.ndarray:
    """Design columns ``-a_j x (q - p_j)`` for a model already posed and forwarded.

    ``data`` must have had ``mj_forward`` run at the sim joint state; ``ids`` comes
    from :func:`joint_ids`; ``cube_world`` is the cube position ``q`` in meters.
    """
    return np.stack(
        [
            -np.cross(data.xaxis[ids[name]], cube_world - data.xanchor[ids[name]])
            for name in FIT_JOINTS
        ],
        axis=1,
    )


def _fit(samples: list[JointZeroSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    design = np.concatenate([s.columns for s in samples], axis=0)
    target = np.concatenate([s.delta for s in samples])
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = (target - design @ params).reshape(-1, 3)
    return params, residual, np.linalg.norm(residual, axis=1), design


def _parameter_std_deg(design: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    """1-sigma parameter uncertainty from ``sigma^2 * diag((A^T A)^-1)``."""
    n_rows, n_params = design.shape
    dof = max(1, n_rows - n_params)
    sigma2 = float((residual**2).sum() / dof)
    try:
        cov_diag = np.diag(np.linalg.inv(design.T @ design)) * sigma2
    except np.linalg.LinAlgError:
        cov_diag = np.full(n_params, np.inf)
    std_deg = np.degrees(np.sqrt(np.clip(cov_diag, 0.0, None)))
    return {name: float(v) for name, v in zip(FIT_JOINTS, std_deg)}


def fit_robust(samples: list[JointZeroSample]) -> FitResult:
    """Least-squares fit with one MAD-trim pass dropping gross residual outliers."""
    _, _, norms, _ = _fit(samples)
    median = np.median(norms)
    mad = np.median(np.abs(norms - median))
    keep = norms < median + TRIM_MAD_FACTOR * mad
    kept = [s for s, k in zip(samples, keep) if k]
    params, residual, _, design = _fit(kept)
    return FitResult(
        params=params,
        residual=residual,
        keep=keep,
        std_deg=_parameter_std_deg(design, residual.reshape(-1)),
        design=design,
    )


def params_to_offsets_deg(params: np.ndarray) -> dict[str, float]:
    """Fitted radian params as the exporter's ``--joint-offsets-deg`` dict."""
    return {name: float(np.degrees(v)) for name, v in zip(FIT_JOINTS, params)}
