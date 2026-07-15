# SPDX-License-Identifier: MIT
#
# Milestone M4 — subject-specific plantar foot geometry from a static C3D, feeding the
# distributed contact models (M5/M7). This replaces the analytic ``sample_flat_sole`` /
# ``sample_ellipsoid_sole`` beds with a tapered plantar footprint sized to the subject's
# real foot and placed in the model's ``calcn`` body frame, so the gold-standard motion
# drives contact on the subject's own foot.
#
# Two ingredients are combined:
#   * subject foot *dimensions* (heel width, forefoot width, foot length, ball offset,
#     toe length) measured from the static-trial plantar markers (HEE/HEE2/HEE3, MTH1,
#     MTH5, HLX, TOE) — a port of the foot-calibration logic in
#     ``data/scripts/calibrate_lower_body_elipsoid_from_static_c3d.py``; these are
#     rigid-frame-invariant scalars, so no lab->model registration is needed.
#   * anatomical *anchors* from the fitted model — the scaled local offsets of the
#     ``RCAL`` (heel), ``RMT5`` (5th metatarsal) and ``RTOE`` (toe) markers, which all
#     live on ``calcn_{r,l}``. These define the foot frame's origin/orientation/scale in
#     the exact place the fitted+scaled skeleton puts the foot, keeping the sole
#     consistent with the exported ``.motion`` (whose ``calcn`` body pose drives contact).
#
# The result is a :class:`biomech.contact.elastic_foundation.FootSole` (points/normals/
# areas + a per-patch compliance ``modulus`` map for a soft heel pad, stiffer forefoot,
# and a relieved medial arch) expressed in the OpenSim ``calcn`` body frame — ready for
# ``hydroelastic.evaluate_contact`` / ``kinematics.evaluate_foot_contact_from_motion``.
#
# Frame note: the sole lives in the raw OpenSim ``calcn`` body frame (x forward, y up,
# z lateral for the right foot); ``export.motion`` already bakes the Y-up->Z-up rotation
# into the exported body quaternion, so body-frame points rotate correctly into Z-up
# world when driven by the motion clip. The plantar normal is -y in the body frame.

"""Subject-specific plantar foot geometry from a static C3D (M4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from biomech.contact.elastic_foundation import FootSole


# ---------------------------------------------------------------------------
# Subject foot dimensions (from static plantar markers)
# ---------------------------------------------------------------------------


@dataclass
class FootDimensions:
    """Subject foot measurements (meters), side ``"R"`` or ``"L"``."""

    side: str
    heel_width: float
    forefoot_width: float
    foot_length: float
    heel_to_ball: float
    toe_length: float


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalize a ~zero vector")
    return v / n


def foot_dimensions_from_markers(
    markers: Mapping[str, np.ndarray], side: str
) -> FootDimensions:
    """Measure foot dimensions from averaged static plantar markers (port of
    ``foot_calibration``). ``markers`` maps label -> ``(3,)`` mean position (any frame;
    the measurements are rigid-invariant)."""
    heel = np.asarray(markers[f"{side}HEE"], dtype=np.float64)
    heel_2 = np.asarray(markers[f"{side}HEE2"], dtype=np.float64)
    heel_3 = np.asarray(markers[f"{side}HEE3"], dtype=np.float64)
    mth1 = np.asarray(markers[f"{side}MTH1"], dtype=np.float64)
    mth5 = np.asarray(markers[f"{side}MTH5"], dtype=np.float64)
    hlx = np.asarray(markers[f"{side}HLX"], dtype=np.float64)

    heel_center = np.mean(np.stack([heel, heel_2, heel_3]), axis=0)
    ball_center = 0.5 * (mth1 + mth5)
    toe_tip = hlx  # HLX (hallux) is the most anterior plantar landmark

    forward = toe_tip - heel_center
    forward[2] = 0.0
    forward = _unit(forward)

    heel_width = float(np.linalg.norm(heel_3 - heel_2))
    forefoot_width = float(np.linalg.norm(mth5 - mth1))
    foot_length = float((toe_tip - heel_center) @ forward)
    heel_to_ball = float((ball_center - heel_center) @ forward)
    toe_length = max(0.03, float((toe_tip - ball_center) @ forward))

    return FootDimensions(
        side=side,
        heel_width=heel_width,
        forefoot_width=forefoot_width,
        foot_length=foot_length,
        heel_to_ball=heel_to_ball,
        toe_length=toe_length,
    )


def average_static_markers(
    labels: Sequence[str],
    points: np.ndarray,
    frame_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """NaN-aware per-marker mean over a (static) frame window. ``points`` is
    ``(F, n_markers, 3)`` (lab frame)."""
    points = np.asarray(points, dtype=np.float64)
    if frame_range is not None:
        lo, hi = frame_range
        points = points[lo:hi]
    out: Dict[str, np.ndarray] = {}
    for i, lbl in enumerate(labels):
        col = points[:, i, :]
        finite = np.isfinite(col).all(axis=1)
        if finite.any():
            out[lbl] = col[finite].mean(axis=0)
    return out


def foot_dimensions_from_session(
    session, side: str, frame_range: Optional[Tuple[int, int]] = None
) -> FootDimensions:
    """Measure foot dimensions from a (static) ``CaptureSession``."""
    means = average_static_markers(
        session.marker_labels, session.markers, frame_range
    )
    return foot_dimensions_from_markers(means, side)


# ---------------------------------------------------------------------------
# Model anchors (scaled marker offsets on the calcn body)
# ---------------------------------------------------------------------------


@dataclass
class FootAnchors:
    """Anatomical anchors in the ``calcn`` body frame (meters)."""

    heel: np.ndarray  # RCAL / LCAL offset
    mt5: np.ndarray  # RMT5 / LMT5 offset
    toe: np.ndarray  # RTOE / LTOE offset
    side: str


def _group_index(spec, body_name: str) -> int:
    for gi, group in enumerate(spec.scale_groups):
        if body_name in group:
            return gi
    raise KeyError(f"body {body_name} not in any scale group")


def calcn_anchors_from_spec(
    spec, side: str, group_scales: Optional[np.ndarray] = None
) -> FootAnchors:
    """Scaled ``calcn`` marker offsets (heel/mt5/toe) for a given side."""
    body = f"calcn_{side.lower()}"
    if group_scales is None:
        s = np.ones(3, dtype=np.float64)
    else:
        gi = _group_index(spec, body)
        s = np.asarray(group_scales, dtype=np.float64)[3 * gi : 3 * gi + 3]
    heel = spec.marker(f"{side}CAL").offset * s
    mt5 = spec.marker(f"{side}MT5").offset * s
    toe = spec.marker(f"{side}TOE").offset * s
    return FootAnchors(heel=heel, mt5=mt5, toe=toe, side=side)


# ---------------------------------------------------------------------------
# Sole construction (tapered footprint in the calcn body frame)
# ---------------------------------------------------------------------------


@dataclass
class SolePads:
    """Per-region relative compliance (stiffness) of the plantar bed.

    Values are relative stiffness multipliers (``FootSole.modulus``): the soft heel
    fat pad and the relieved medial arch carry less stiffness than the forefoot.
    """

    heel: float = 0.6
    arch: float = 0.15
    forefoot: float = 1.0
    toe: float = 0.8


def _foot_axes(anchors: FootAnchors) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (forward, lateral, up) orthonormal axes in the calcn frame from anchors."""
    forward = _unit(anchors.toe - anchors.heel)
    lat0 = anchors.mt5 - anchors.heel
    lat0 = lat0 - (lat0 @ forward) * forward
    lateral = _unit(lat0)  # points toward the 5th metatarsal (lateral)
    up = _unit(np.cross(forward, lateral))
    # up should point toward the ankle (+y in the OpenSim calcn frame)
    if up[1] < 0.0:
        up = -up
        lateral = -lateral
    lateral = _unit(np.cross(up, forward))
    return forward, lateral, up


def build_subject_sole(
    dims: FootDimensions,
    anchors: FootAnchors,
    nx: int = 16,
    ny: int = 6,
    plantar_drop: float = 0.02,
    toe_taper: float = 0.6,
    pads: Optional[SolePads] = None,
) -> FootSole:
    """Build a tapered subject plantar sole in the ``calcn`` body frame.

    Args:
        dims: subject foot dimensions (widths/length/ball) from the static C3D.
        anchors: scaled ``calcn`` marker anchors (heel/mt5/toe) defining placement.
        nx, ny: sample resolution along the foot / across its width.
        plantar_drop: distance below the anchor plane to the plantar surface (m).
        toe_taper: forefoot-width fraction at the toe tip (outline narrowing).
        pads: per-region compliance map (soft heel pad / relieved arch / forefoot / toe).

    Returns:
        a :class:`FootSole` (points/normals/areas + ``modulus`` compliance map) in the
        calcn body frame, plantar normal ``-up``.
    """
    pads = pads or SolePads()
    forward, lateral, up = _foot_axes(anchors)
    origin = anchors.heel.astype(np.float64)

    L = dims.foot_length
    s_ball = float(np.clip(dims.heel_to_ball / max(L, 1e-6), 0.4, 0.8))
    s_heel = min(0.2, 0.5 * s_ball)

    # width(s) control profile: heel pad -> ball (forefoot) -> tapered toe
    s_nodes = np.array([0.0, s_heel, s_ball, 1.0])
    w_nodes = np.array([
        dims.heel_width,
        dims.heel_width,
        dims.forefoot_width,
        dims.forefoot_width * toe_taper,
    ])

    ss = (np.arange(nx) + 0.5) / nx  # patch centers along length
    ds = L / nx

    pts = []
    nrm = []
    areas = []
    mods = []
    for s in ss:
        w = float(np.interp(s, s_nodes, w_nodes))
        dw = w / ny
        ts = (np.arange(ny) + 0.5) / ny - 0.5  # in (-0.5, 0.5)
        along = s * L
        for t in ts:
            lat = t * w
            p = origin + along * forward + lat * lateral - plantar_drop * up
            pts.append(p)
            nrm.append(-up)
            areas.append(ds * dw)
            # compliance region: t<0 is medial (opposite the lateral axis)
            if s < s_heel:
                m = pads.heel
            elif s > s_ball:
                m = pads.toe if s > 0.85 else pads.forefoot
            elif t < -0.1:
                m = pads.arch  # medial midfoot (arch) relief
            else:
                m = pads.forefoot
            mods.append(m)

    return FootSole(
        points=np.array(pts, dtype=np.float64),
        normals=np.array(nrm, dtype=np.float64),
        areas=np.array(areas, dtype=np.float64),
        modulus=np.array(mods, dtype=np.float64),
    )


def subject_sole_from_session(
    session,
    spec,
    side: str,
    group_scales: Optional[np.ndarray] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    **sole_kwargs,
) -> FootSole:
    """Convenience: measure dims from a static session + place with model anchors."""
    dims = foot_dimensions_from_session(session, side, frame_range)
    anchors = calcn_anchors_from_spec(spec, side, group_scales)
    return build_subject_sole(dims, anchors, **sole_kwargs)
