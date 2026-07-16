# SPDX-License-Identifier: MIT
#
# OpenSim-style *marker placement* for the foot, from a static (calibration) trial.
#
# The stock Rajagopal2015 marker set is sparse on the foot: only ``RCAL`` (heel),
# ``RMT5`` (5th-met head) and ``RTOE`` (placed at the toe tip) on ``calcn``, and *nothing*
# on ``toes``. But the S001 Plug-in-Gait capture carries a much richer foot set (verified
# in the C3D, 100% present in both the static and dynamic trials):
#
#   * ``HEE``/``HEE2``/``HEE3`` — a triangular calcaneus cluster (fully constrains the
#     hindfoot orientation, which the single ``RCAL`` cannot),
#   * ``MTH1``/``MTH5`` — 1st/5th metatarsal heads,
#   * ``TOE`` — on the **met-2 head** (i.e. on ``calcn``, at the MTP line — NOT the toe tip
#     the stock model places it at, a ~2.7 cm forward mismatch that the fit was absorbing
#     by pitching the foot into a spurious plantarflexion offset),
#   * ``HLX`` — on the **hallux**, ~5 cm distal to the MTP line, i.e. on the ``toes``
#     segment. This is the only marker distal to the MTP joint, so mapping it is what makes
#     ``mtp_angle`` observable at all.
#
# This module runs the OpenSim "marker placement" step: fit the *static* trial with the
# existing (sparse) markers to get segment scales + a foot-flat pose, then express each rich
# foot marker's measured position in its owning body frame. Those become the model marker
# offsets. Re-seating ``RCAL``/``RTOE``/``RMT5`` and adding the cluster/met-1/hallux markers
# both (a) over-constrains the calcaneus so the ankle stops absorbing foot placement error,
# and (b) puts a marker on ``toes`` so the MTP angle is recovered.
#
# Frames: markers come in the lab (Z-up) frame; the skeleton FK is OpenSim (Y-up). We reuse
# ``marker_map.R_PM2OS`` for the rotation, exactly as ``build_observations`` does.

"""OpenSim-style foot marker placement from a static trial (enriches the model marker set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from biomech.fitting.marker_map import R_PM2OS
from biomech.osim.spec import MarkerSpec, SkeletonSpec


# New model markers to add: model name -> (owning body, capture label), per side.
# The body assignment is the anatomically-correct segment (calcn for hindfoot/forefoot,
# toes distal to the MTP joint).
_FOOT_ADDITIONS = {
    "R": {
        "RCAL2": ("calcn_r", "RHEE2"),
        "RCAL3": ("calcn_r", "RHEE3"),
        "RMT1": ("calcn_r", "RMTH1"),
        "RTOE_TIP": ("toes_r", "RHLX"),
    },
    "L": {
        "LCAL2": ("calcn_l", "LHEE2"),
        "LCAL3": ("calcn_l", "LHEE3"),
        "LMT1": ("calcn_l", "LMTH1"),
        "LTOE_TIP": ("toes_l", "LHLX"),
    },
}

# Existing foot markers whose model offset we re-seat from the static trial (marker
# placement). ``RTOE`` is the important one: the stock model puts it at the toe tip but the
# capture marker is on the met-2 head.
_FOOT_RESEAT = {
    "RCAL": ("calcn_r", "RHEE"),
    "RTOE": ("calcn_r", "RTOE"),
    "RMT5": ("calcn_r", "RMTH5"),
    "LCAL": ("calcn_l", "LHEE"),
    "LTOE": ("calcn_l", "LTOE"),
    "LMT5": ("calcn_l", "LMTH5"),
}


@dataclass
class MarkerPlacement:
    """Result of the static foot marker placement."""

    added: List[str]                 # new model marker names added to the spec
    reseated: List[str]              # existing model markers whose offset changed
    unlocked: List[str]              # coordinate names unlocked (e.g. mtp_angle_{r,l})
    offsets: Dict[str, np.ndarray]   # model name -> body-frame offset (unscaled, meters)
    residual_mm: Dict[str, float]    # model name -> static placement residual (mm)
    group_scales: np.ndarray         # scales used for placement (from the static fit)
    poses: np.ndarray                # (Fw, ndof) fitted static poses used for placement
    window: Tuple[int, int]          # frame window used for the static fit


def unlock_mtp(spec: SkeletonSpec, sides: Sequence[str] = ("r", "l")) -> List[str]:
    """Unlock the ``mtp_angle_{r,l}`` coordinates so the MTP DOF can be fit.

    The stock Rajagopal2015 model ships the metatarsophalangeal joint ``locked`` (frozen
    at 0), which is the standard choice when there is no marker distal to the MTP. Once a
    hallux/toes marker is added (see :func:`place_foot_markers`) the MTP becomes
    observable, so we free it. Returns the list of coordinate names unlocked.
    """
    unlocked: List[str] = []
    for side in sides:
        try:
            joint = spec.joint(f"mtp_{side}")
        except KeyError:
            continue
        for coord in joint.coordinates:
            if coord.locked:
                coord.locked = False
                unlocked.append(coord.name)
    return unlocked


def _body_group_map(spec: SkeletonSpec) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for gi, group in enumerate(spec.scale_groups):
        for name in group:
            out[name] = gi
    return out


def _capture_positions_opensim(
    session, label: str, frame_range: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Per-frame world positions of a capture label, rotated lab Z-up -> OpenSim Y-up.

    Returns ``(F, 3)`` with NaN where the marker is missing.
    """
    labels = list(session.marker_labels)
    if label not in labels:
        raise KeyError(f"capture label {label!r} not in session")
    pts = np.asarray(session.markers, dtype=np.float64)[:, labels.index(label), :]
    if frame_range is not None:
        lo, hi = frame_range
        pts = pts[lo:hi]
    return np.einsum("ij,fj->fi", R_PM2OS, pts)


def place_foot_markers(
    spec: SkeletonSpec,
    static_session,
    *,
    mapping=None,
    marker_config=None,
    device: str = "cpu",
    frame_range: Optional[Tuple[int, int]] = None,
    sides: Sequence[str] = ("R", "L"),
    reseat: bool = True,
    unlock_mtp_joint: bool = True,
) -> MarkerPlacement:
    """Add the rich foot markers to ``spec`` (in place) via static marker placement.

    Fits the static trial with the current (sparse) marker set to obtain segment scales +
    a foot-flat pose, then expresses each rich foot marker's measured position in its owning
    body frame. New markers are appended to ``spec.markers``; existing foot markers are
    re-seated (unless ``reseat=False``). Returns a :class:`MarkerPlacement` describing the
    change.

    The offset stored on a :class:`MarkerSpec` is *unscaled*: FK applies
    ``world = T_body @ (group_scale ⊙ offset)``, so we divide the measured body-frame
    position by the fitted group scale before storing.
    """
    # Local import to avoid a heavy import at module load (mirrors pipeline.reconstruct).
    from biomech.contact.pipeline import reconstruct_window
    from biomech.skeleton.skeleton import WarpSkeleton

    F = np.asarray(static_session.markers).shape[0]
    window = frame_range if frame_range is not None else (0, F)

    # 1) Fit the static trial with the *current* (pre-augmentation) marker set.
    result, _obs, _anat = reconstruct_window(
        static_session, spec, window, mapping=mapping,
        marker_config=marker_config, device=device,
    )
    group_scales = np.asarray(result.group_scales, dtype=np.float64)
    poses = np.asarray(result.poses, dtype=np.float64)  # (Fw, ndof)

    # 2) FK the fitted static poses -> per-frame body transforms (OpenSim frame).
    skel = WarpSkeleton(spec, device=device)
    world, _ = skel.forward(poses, group_scales)  # (Fw, B, 4, 4)
    body_index = {b.name: i for i, b in enumerate(spec.bodies)}
    body_group = _body_group_map(spec)
    scales3 = group_scales.reshape(-1, 3)

    lo, hi = window

    def _place(model_name: str, body: str, label: str) -> Tuple[np.ndarray, float]:
        # measured positions over the fit window, OpenSim frame
        p = _capture_positions_opensim(static_session, label)[lo:hi]  # (Fw, 3)
        T = world[:, body_index[body], :, :]  # (Fw, 4, 4)
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        # local = R^T (p - t): position in the (scaled) body frame
        local = np.einsum("fji,fj->fi", R, p - t)  # R^T @ (p - t)
        mask = np.isfinite(local).all(axis=1)
        if not mask.any():
            raise ValueError(f"no finite frames to place marker {model_name!r} ({label})")
        local_mean = local[mask].mean(axis=0)
        s = scales3[body_group[body]]
        offset = local_mean / s  # unscaled offset stored on the MarkerSpec
        # placement residual: spread of the per-frame local positions (rigid-body noise)
        resid_mm = float(np.linalg.norm(local[mask] - local_mean, axis=1).mean() * 1e3)
        return offset, resid_mm

    offsets: Dict[str, np.ndarray] = {}
    residual_mm: Dict[str, float] = {}
    added: List[str] = []
    reseated: List[str] = []

    # 3) Re-seat existing foot markers (marker placement on the stock set).
    if reseat:
        for model_name, (body, label) in _FOOT_RESEAT.items():
            side = model_name[0]
            if side not in sides:
                continue
            try:
                m = spec.marker(model_name)
            except KeyError:
                continue
            offset, resid = _place(model_name, body, label)
            m.offset = offset
            offsets[model_name] = offset
            residual_mm[model_name] = resid
            reseated.append(model_name)

    # 4) Add the new rich foot markers.
    existing = {m.name for m in spec.markers}
    for side in sides:
        for model_name, (body, label) in _FOOT_ADDITIONS[side].items():
            if model_name in existing:
                continue
            offset, resid = _place(model_name, body, label)
            spec.markers.append(
                MarkerSpec(name=model_name, body=body, offset=offset, fixed=True)
            )
            offsets[model_name] = offset
            residual_mm[model_name] = resid
            added.append(model_name)

    # 5) Unlock the MTP joint now that a toes marker constrains it.
    unlocked: List[str] = []
    if unlock_mtp_joint:
        unlocked = unlock_mtp(spec, sides=tuple(s.lower() for s in sides))

    return MarkerPlacement(
        added=added,
        reseated=reseated,
        unlocked=unlocked,
        offsets=offsets,
        residual_mm=residual_mm,
        group_scales=group_scales,
        poses=poses,
        window=(lo, hi),
    )
