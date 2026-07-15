# SPDX-License-Identifier: MIT
#
# Windows-native port of Nimble's marker-fit error reporting
# (``dart/biomechanics/IKErrorReport.{hpp,cpp}``). Nimble computes, for a fitted
# skeleton + marker offsets + per-frame poses, the reprojection error between the model
# markers (FK) and the observed markers: overall RMS / max, per-frame RMS / max, and
# per-marker RMS / max / mean over the visible frames. It is the standard "how good is
# this fit" report used everywhere in AddBiomechanics.
#
# This port keeps the same statistics but computes the model marker positions with the
# ported Warp skeleton FK (``WarpSkeleton.forward``) instead of DART. NaN observations
# are treated as "missing" (not counted), exactly like Nimble's visibility masking.

"""Marker-fit error report (port of Nimble ``IKErrorReport``, M2 reporting)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biomech.skeleton.skeleton import WarpSkeleton


@dataclass
class MarkerErrorReport:
    """Reprojection-error summary for a fitted marker set (all distances in meters)."""

    rms: float  # overall RMS marker error over all visible (frame, marker) pairs
    max: float  # overall worst single visible marker error
    mean: float  # overall mean marker error
    per_frame_rms: np.ndarray  # (F,) per-frame RMS over that frame's visible markers
    per_frame_max: np.ndarray  # (F,) per-frame worst marker error
    per_marker_rms: np.ndarray  # (M,) per-marker RMS over its visible frames (NaN if never seen)
    per_marker_max: np.ndarray  # (M,) per-marker worst error (NaN if never seen)
    per_marker_mean: np.ndarray  # (M,) per-marker mean error (NaN if never seen)
    marker_names: list[str] = field(default_factory=list)
    num_visible: int = 0  # total number of visible (frame, marker) pairs

    # -- convenience ---------------------------------------------------
    def worst_markers(self, k: int | None = None) -> list[tuple[str, float]]:
        """``(name, rms)`` sorted worst-first over markers that were ever visible."""
        order = np.argsort(np.where(np.isfinite(self.per_marker_rms), -self.per_marker_rms, np.inf))
        out = [
            (self.marker_names[i], float(self.per_marker_rms[i]))
            for i in order
            if np.isfinite(self.per_marker_rms[i])
        ]
        return out if k is None else out[:k]

    def format(self, k: int = 10) -> str:
        """Human-readable multi-line report (mirrors Nimble's ``printReport``)."""
        lines = [
            "Marker error report",
            f"  frames={self.per_frame_rms.size}  markers={self.per_marker_rms.size}"
            f"  visible_pairs={self.num_visible}",
            f"  overall RMS = {self.rms * 1e3:8.3f} mm"
            f"   max = {self.max * 1e3:8.3f} mm"
            f"   mean = {self.mean * 1e3:8.3f} mm",
            f"  worst {k} markers (RMS):",
        ]
        for name, rms in self.worst_markers(k):
            lines.append(f"    {name:<10s} {rms * 1e3:8.3f} mm")
        return "\n".join(lines)


def marker_errors(
    skel: WarpSkeleton,
    observations: np.ndarray,
    poses: np.ndarray,
    group_scales: np.ndarray | None = None,
    marker_offsets: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> MarkerErrorReport:
    """Build a :class:`MarkerErrorReport` for a fitted skeleton.

    Parameters
    ----------
    skel : WarpSkeleton
    observations : (F, M, 3)
        Observed marker world positions in the model frame, aligned to
        ``skel.marker_names()``; NaN where missing.
    poses : (F, ndof)
        Per-frame generalized coordinates (e.g. ``MarkerFitResult.poses``).
    group_scales : (3G,), optional
        Anisotropic per-group scale (default unit scale).
    marker_offsets : (M, 3), optional
        If given, the skeleton's marker offsets are set to these before FK (restored
        afterwards) so the report reflects the fitted offsets.
    weights : (M,), optional
        Per-marker nonnegative weights. Errors are plain Euclidean distances; weights
        only affect the *weighted* overall RMS reported via ``num_visible`` folding.
        By default all markers count equally.

    Notes
    -----
    Errors are Euclidean marker distances (meters). This matches Nimble's
    ``IKErrorReport`` which reports geometric distance, not the weighted least-squares
    objective the solver minimizes.
    """
    obs = np.asarray(observations, dtype=np.float64)
    assert obs.ndim == 3 and obs.shape[2] == 3, obs.shape
    F, M, _ = obs.shape
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None]
    assert poses.shape[0] == F, (poses.shape, F)

    prev_off = None
    if marker_offsets is not None:
        prev_off = skel.marker_offsets().copy()
        skel.set_marker_offsets(np.asarray(marker_offsets, dtype=np.float64))
    try:
        _, model_mk = skel.forward(poses, group_scales)  # (F, M, 3)
    finally:
        if prev_off is not None:
            skel.set_marker_offsets(prev_off)

    visible = np.isfinite(obs).all(axis=2)  # (F, M)
    dist = np.linalg.norm(np.where(visible[..., None], model_mk - obs, 0.0), axis=2)  # (F, M)

    if weights is None:
        w = np.ones(M)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
    wvis = (w[None, :] * visible).astype(np.float64)  # (F, M)

    def _rms(d2_sum, n):
        return float(np.sqrt(d2_sum / n)) if n > 0 else float("nan")

    # per-frame
    d2 = dist**2
    nf = visible.sum(axis=1)
    per_frame_rms = np.array(
        [_rms(d2[t, visible[t]].sum(), nf[t]) for t in range(F)]
    )
    per_frame_max = np.array(
        [float(dist[t, visible[t]].max()) if nf[t] else float("nan") for t in range(F)]
    )

    # per-marker
    nm = visible.sum(axis=0)
    per_marker_rms = np.full(M, np.nan)
    per_marker_max = np.full(M, np.nan)
    per_marker_mean = np.full(M, np.nan)
    for m in range(M):
        if nm[m]:
            dm = dist[visible[:, m], m]
            per_marker_rms[m] = float(np.sqrt((dm**2).mean()))
            per_marker_max[m] = float(dm.max())
            per_marker_mean[m] = float(dm.mean())

    # overall (weighted by marker weight * visibility)
    total_w = wvis.sum()
    rms = float(np.sqrt((wvis * d2).sum() / total_w)) if total_w > 0 else float("nan")
    mean = float((wvis * dist).sum() / total_w) if total_w > 0 else float("nan")
    mx = float(dist[visible].max()) if visible.any() else float("nan")

    return MarkerErrorReport(
        rms=rms,
        max=mx,
        mean=mean,
        per_frame_rms=per_frame_rms,
        per_frame_max=per_frame_max,
        per_marker_rms=per_marker_rms,
        per_marker_max=per_marker_max,
        per_marker_mean=per_marker_mean,
        marker_names=skel.marker_names(),
        num_visible=int(visible.sum()),
    )
