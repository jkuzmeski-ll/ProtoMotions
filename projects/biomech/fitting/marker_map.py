# SPDX-License-Identifier: MIT
#
# Real-data bridge — map a captured marker set (Vicon Plug-in-Gait, as in the S001
# instrumented-treadmill capture) into the Rajagopal2015 model's marker order so the
# marker fitters (``fitting/ik.py``, ``fitting/ik_initializer.py``,
# ``fitting/marker_fitter.py``) can consume real observations.
#
# All fitters take observations in ``skel.marker_names()`` order with NaN marking a
# missing/absent marker; this module produces exactly that ``(F, M, 3)`` array from a
# ``CaptureSession`` (or raw labels + points). Model marker names with no measured
# counterpart (virtual joint centres like ``*JC``, medial-elbow, extra shoulder/upper-arm
# cluster markers) are left as NaN.
#
# Frames: the capture is in the lab frame (Z-up, meters, ``biomech`` lab==world), while
# the Warp/DART skeleton FK works in OpenSim's native Y-up frame. Observations are
# rotated lab Z-up -> OpenSim Y-up (the inverse of ``export.motion.R_OS2PM``) so the
# fitted ``q`` is in the canonical OpenSim frame (``export.motion`` converts it back to
# Z-up on the way out).

"""Map captured (Plug-in-Gait) markers into Rajagopal2015 marker order (real-data bridge)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# lab (Z-up) -> OpenSim (Y-up): inverse of export.motion.R_OS2PM.
#   os_x = lab_x, os_y = lab_z, os_z = -lab_y
R_PM2OS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


# Rajagopal2015 marker name -> S001 Plug-in-Gait label. Only physically-measured
# landmarks are mapped; virtual joint centres (``*JC``, ``*_tibial_plateau``) and model
# markers with no PiG counterpart are intentionally omitted (-> NaN).
S001_TO_RAJAGOPAL: Dict[str, str] = {
    # pelvis
    "RASI": "RASI", "LASI": "LASI", "RPSI": "RPSI", "LPSI": "LPSI",
    # thigh cluster (tracking)
    "RTH1": "RTHI", "RTH2": "RTH2", "RTH3": "RTH3",
    "LTH1": "LTHI", "LTH2": "LTH2", "LTH3": "LTH3",
    # knee (lateral/medial femoral condyle)
    "RLFC": "RKNE", "RMFC": "RMKNE", "LLFC": "LKNE", "LMFC": "LMKNE",
    # tibia cluster (tracking)
    "RTB1": "RTIB", "RTB2": "RTIB2", "RTB3": "RTIB3",
    "LTB1": "LTIB", "LTB2": "LTIB2", "LTB3": "LTIB3",
    # ankle (lateral/medial malleolus)
    "RLMAL": "RANK", "RMMAL": "RMANK", "LLMAL": "LANK", "LMMAL": "LMANK",
    # foot — calcaneus cluster (HEE/HEE2/HEE3), met heads (MTH1/MTH5), met-2 head (TOE)
    "RCAL": "RHEE", "RCAL2": "RHEE2", "RCAL3": "RHEE3",
    "RMT1": "RMTH1", "RMT5": "RMTH5", "RTOE": "RTOE",
    "LCAL": "LHEE", "LCAL2": "LHEE2", "LCAL3": "LHEE3",
    "LMT1": "LMTH1", "LMT5": "LMTH5", "LTOE": "LTOE",
    # hallux — the only marker distal to the MTP joint (on the toes segment); makes
    # mtp_angle observable. These model markers are added by fitting.marker_placement.
    "RTOE_TIP": "RHLX", "LTOE_TIP": "LHLX",
    # torso / shoulders
    "C7": "C7", "CLAV": "CLAV", "RACR": "RSHO", "LACR": "LSHO",
    # arms (approximate; upper-body is secondary to the lower-body focus)
    "RLEL": "RELB", "LLEL": "LELB",
    "RUA1": "RUPA", "LUA1": "LUPA",
    "RFAsuperior": "RFRM", "RFAradius": "RWRA", "RFAulna": "RWRB",
    "LFAsuperior": "LFRM", "LFAradius": "LWRA", "LFAulna": "LWRB",
}

# Dynamic captures may still contain medial knee / medial ankle calibration markers.
# These are useful for static scaling / joint-center calibration, but they are often
# removed for dynamic Plug-in-Gait IK because they can overconstrain gait angles when
# placement/soft-tissue error differs from the model. Keep them named separately so S001
# benchmarks can test excluding them from dynamic IK without deleting the measurements.
S001_STATIC_CALIBRATION_MARKERS: Set[str] = {
    "RMFC", "LMFC", "RMMAL", "LMMAL",
}

# Bony landmarks trustworthy for scaling / joint-centre estimation (vs soft-tissue
# cluster plates, which are tracking-only).
S001_ANATOMICAL: Set[str] = {
    "RASI", "LASI", "RPSI", "LPSI",
    "RLFC", "RMFC", "LLFC", "LMFC",
    "RLMAL", "RMMAL", "LLMAL", "LMMAL",
    "RCAL", "RCAL2", "RCAL3", "RMT1", "RTOE", "RMT5", "RTOE_TIP",
    "LCAL", "LCAL2", "LCAL3", "LMT1", "LTOE", "LMT5", "LTOE_TIP",
    "C7", "CLAV", "RACR", "LACR", "RLEL", "LLEL",
}

# Model markers belonging to the lower body (pelvis + legs + feet).
LOWER_BODY_MARKERS: Set[str] = {
    "RASI", "LASI", "RPSI", "LPSI", "RHJC", "LHJC",
    "RTH1", "RTH2", "RTH3", "LTH1", "LTH2", "LTH3",
    "RLFC", "RMFC", "LLFC", "LMFC", "RKJC", "LKJC",
    "R_tibial_plateau", "L_tibial_plateau",
    "RTB1", "RTB2", "RTB3", "LTB1", "LTB2", "LTB3",
    "RLMAL", "RMMAL", "LLMAL", "LMMAL", "RAJC", "LAJC",
    "RCAL", "RCAL2", "RCAL3", "RMT1", "RTOE", "RMT5", "RTOE_TIP",
    "LCAL", "LCAL2", "LCAL3", "LMT1", "LTOE", "LMT5", "LTOE_TIP",
}


@dataclass
class MarkerMap:
    """A mapping from model marker names to captured labels + anatomical flags."""

    model_to_capture: Dict[str, str]
    anatomical: Set[str] = field(default_factory=set)

    def restrict(self, keep_model_markers: Set[str]) -> "MarkerMap":
        """Return a submap keeping only the given model marker names."""
        return MarkerMap(
            model_to_capture={
                k: v for k, v in self.model_to_capture.items() if k in keep_model_markers
            },
            anatomical={a for a in self.anatomical if a in keep_model_markers},
        )


def s001_marker_map(lower_body_only: bool = False) -> MarkerMap:
    """The S001 (Plug-in-Gait) -> Rajagopal2015 marker map."""
    mm = MarkerMap(
        model_to_capture=dict(S001_TO_RAJAGOPAL),
        anatomical=set(S001_ANATOMICAL),
    )
    if lower_body_only:
        mm = mm.restrict(LOWER_BODY_MARKERS)
    return mm


def build_observations(
    capture_labels: Sequence[str],
    capture_points: np.ndarray,
    model_marker_names: Sequence[str],
    mapping: MarkerMap,
    to_opensim: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder captured markers into the model's marker order.

    Args:
        capture_labels: captured marker labels, e.g. ``session.marker_labels``.
        capture_points: ``(F, n_captured, 3)`` marker positions (lab Z-up, meters, NaN
            gaps), e.g. ``session.markers``.
        model_marker_names: ``skel.marker_names()`` (the target order).
        mapping: model->capture name map.
        to_opensim: rotate lab Z-up -> OpenSim Y-up (recommended; the FK is in Y-up).

    Returns:
        ``(obs, present)`` where ``obs`` is ``(F, M, 3)`` in model-marker order (NaN for
        unmapped/absent markers) and ``present`` is a ``(M,)`` bool of which model
        markers were mapped to a measured label at all.
    """
    capture_points = np.asarray(capture_points, dtype=np.float64)
    if capture_points.ndim != 3 or capture_points.shape[2] != 3:
        raise ValueError("capture_points must be (F, n_captured, 3)")
    F = capture_points.shape[0]
    label_index = {lbl: i for i, lbl in enumerate(capture_labels)}

    M = len(model_marker_names)
    obs = np.full((F, M, 3), np.nan, dtype=np.float64)
    present = np.zeros(M, dtype=bool)

    for mi, mname in enumerate(model_marker_names):
        cap = mapping.model_to_capture.get(mname)
        if cap is None:
            continue
        if isinstance(cap, (list, tuple)):
            # centroid marker: per-frame NaN-aware mean of several capture labels
            cols = [label_index[c] for c in cap if c in label_index]
            if not cols:
                continue
            pts = capture_points[:, cols, :]  # (F, k, 3)
            with np.errstate(invalid="ignore"):
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", category=RuntimeWarning)
                    obs[:, mi, :] = np.nanmean(pts, axis=1)  # NaN where all members missing
            present[mi] = True
            continue
        ci = label_index.get(cap)
        if ci is None:
            continue
        obs[:, mi, :] = capture_points[:, ci, :]
        present[mi] = True

    if to_opensim:
        # rotate every finite marker into the OpenSim Y-up frame
        obs = np.einsum("ij,fmj->fmi", R_PM2OS, obs)

    return obs, present


def observations_from_session(
    session,
    model_marker_names: Sequence[str],
    mapping: Optional[MarkerMap] = None,
    lower_body_only: bool = False,
    to_opensim: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: build model-ordered observations from a ``CaptureSession``."""
    mm = mapping or s001_marker_map(lower_body_only=lower_body_only)
    return build_observations(
        session.marker_labels,
        session.markers,
        model_marker_names,
        mm,
        to_opensim=to_opensim,
    )


def anatomical_mask(
    model_marker_names: Sequence[str],
    mapping: MarkerMap,
) -> np.ndarray:
    """Build a ``(M,)`` bool of which model markers are anatomical landmarks (for scaling).

    Aligned to ``model_marker_names`` order; only markers that are both mapped and flagged
    anatomical in ``mapping`` are ``True``.
    """
    return np.array(
        [
            (m in mapping.anatomical) and (m in mapping.model_to_capture)
            for m in model_marker_names
        ],
        dtype=bool,
    )


def mapping_coverage(
    model_marker_names: Sequence[str],
    capture_labels: Sequence[str],
    mapping: MarkerMap,
) -> Dict[str, List[str]]:
    """Report which model markers are mapped/measured, and which capture labels are unused."""
    cap_set = set(capture_labels)
    mapped = []
    missing_in_capture = []
    unmapped_model = []
    used_caps: Set[str] = set()
    for mname in model_marker_names:
        cap = mapping.model_to_capture.get(mname)
        if cap is None:
            unmapped_model.append(mname)
            continue
        caps = cap if isinstance(cap, (list, tuple)) else (cap,)
        present_caps = [c for c in caps if c in cap_set]
        if present_caps:
            mapped.append(mname)
            used_caps.update(present_caps)
        else:
            missing_in_capture.append(mname)
    unused_capture = [c for c in capture_labels if c not in used_caps]
    return {
        "mapped": mapped,
        "unmapped_model": unmapped_model,
        "missing_in_capture": missing_in_capture,
        "unused_capture": unused_capture,
    }
