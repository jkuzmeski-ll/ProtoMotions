# SPDX-License-Identifier: MIT
#
# Windows-native port of Nimble's marker-fit error reporting
# (``dart/biomechanics/IKErrorReport.{hpp,cpp}``). Nimble computes, for a fitted
# skeleton + marker offsets + per-frame poses, the reprojection error between the model
# markers (FK) and the observed markers: overall RMS / max, per-frame RMS / max, and
# per-marker RMS / max / mean over the visible frames. It is the standard "how good is
# this fit" report used everywhere in AddBiomechanics.
#
# This port keeps the same statistics but computes the model marker positions AND all of
# the per-frame / per-marker reductions on the Warp GPU skeleton (``WarpSkeleton``): the
# FK runs on the device, a distance kernel produces the ``(F, M)`` visible marker error
# matrix, and per-frame / per-marker reduction kernels replace the old Python loops.
# NaN observations are treated as "missing" (not counted), exactly like Nimble's
# visibility masking.

"""Marker-fit error report (port of Nimble ``IKErrorReport``, M2 reporting)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from biomech.skeleton.skeleton import WarpSkeleton


# ---------------------------------------------------------------------------
# Device kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _dist_kernel(
    markers: wp.array2d(dtype=wp.vec3d),  # (F, M) model FK markers
    obs: wp.array2d(dtype=wp.vec3d),  # (F, M) observed (NaNs pre-zeroed)
    vis: wp.array2d(dtype=wp.float64),  # (F, M) 1 = visible, 0 = missing
    dist: wp.array2d(dtype=wp.float64),  # (F, M) output euclidean error (0 if missing)
):
    f, m = wp.tid()
    if vis[f, m] > wp.float64(0.0):
        d = markers[f, m] - obs[f, m]
        dist[f, m] = wp.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    else:
        dist[f, m] = wp.float64(0.0)


@wp.kernel
def _per_frame_kernel(
    dist: wp.array2d(dtype=wp.float64),  # (F, M)
    vis: wp.array2d(dtype=wp.float64),  # (F, M)
    M: wp.int32,
    sumsq: wp.array(dtype=wp.float64),  # (F,)
    cnt: wp.array(dtype=wp.float64),  # (F,)
    mx: wp.array(dtype=wp.float64),  # (F,)
):
    f = wp.tid()
    s = wp.float64(0.0)
    c = wp.float64(0.0)
    mm = wp.float64(0.0)
    for m in range(M):
        if vis[f, m] > wp.float64(0.0):
            d = dist[f, m]
            s += d * d
            c += wp.float64(1.0)
            if d > mm:
                mm = d
    sumsq[f] = s
    cnt[f] = c
    mx[f] = mm


@wp.kernel
def _per_marker_kernel(
    dist: wp.array2d(dtype=wp.float64),  # (F, M)
    vis: wp.array2d(dtype=wp.float64),  # (F, M)
    Fr: wp.int32,
    sumsq: wp.array(dtype=wp.float64),  # (M,)
    ssum: wp.array(dtype=wp.float64),  # (M,)
    cnt: wp.array(dtype=wp.float64),  # (M,)
    mx: wp.array(dtype=wp.float64),  # (M,)
):
    m = wp.tid()
    s = wp.float64(0.0)
    sm = wp.float64(0.0)
    c = wp.float64(0.0)
    mm = wp.float64(0.0)
    for f in range(Fr):
        if vis[f, m] > wp.float64(0.0):
            d = dist[f, m]
            s += d * d
            sm += d
            c += wp.float64(1.0)
            if d > mm:
                mm = d
    sumsq[m] = s
    ssum[m] = sm
    cnt[m] = c
    mx[m] = mm


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
    objective the solver minimizes. All heavy work (FK + per-frame/per-marker
    reductions) runs on the Warp device.
    """
    obs = np.asarray(observations, dtype=np.float64)
    assert obs.ndim == 3 and obs.shape[2] == 3, obs.shape
    F, M, _ = obs.shape
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None]
    assert poses.shape[0] == F, (poses.shape, F)

    dev = skel.device
    G = skel.topo.num_groups
    gs = np.ones(3 * G, dtype=np.float64) if group_scales is None else (
        np.asarray(group_scales, dtype=np.float64).ravel()
    )

    visible = np.isfinite(obs).all(axis=2)  # (F, M)
    obs_z = np.where(visible[..., None], obs, 0.0)

    prev_off = None
    if marker_offsets is not None:
        prev_off = skel.marker_offsets().copy()
        skel.set_marker_offsets(np.asarray(marker_offsets, dtype=np.float64))
    try:
        d_poses = wp.array(poses, dtype=wp.float64, device=dev)
        d_scales = wp.array(gs, dtype=wp.float64, device=dev)
        _, d_markers = skel._run_wp(d_poses, d_scales)  # (F, M) vec3d on device

        d_obs = wp.array(obs_z.reshape(F, M, 3), dtype=wp.vec3d, device=dev)
        d_vis = wp.array(visible.astype(np.float64), dtype=wp.float64, device=dev)
        d_dist = wp.zeros((F, M), dtype=wp.float64, device=dev)
        wp.launch(
            _dist_kernel, dim=(F, M), inputs=[d_markers, d_obs, d_vis],
            outputs=[d_dist], device=dev,
        )

        d_pf_sumsq = wp.zeros(F, dtype=wp.float64, device=dev)
        d_pf_cnt = wp.zeros(F, dtype=wp.float64, device=dev)
        d_pf_max = wp.zeros(F, dtype=wp.float64, device=dev)
        wp.launch(
            _per_frame_kernel, dim=F, inputs=[d_dist, d_vis, M],
            outputs=[d_pf_sumsq, d_pf_cnt, d_pf_max], device=dev,
        )
        d_pm_sumsq = wp.zeros(M, dtype=wp.float64, device=dev)
        d_pm_sum = wp.zeros(M, dtype=wp.float64, device=dev)
        d_pm_cnt = wp.zeros(M, dtype=wp.float64, device=dev)
        d_pm_max = wp.zeros(M, dtype=wp.float64, device=dev)
        wp.launch(
            _per_marker_kernel, dim=M, inputs=[d_dist, d_vis, F],
            outputs=[d_pm_sumsq, d_pm_sum, d_pm_cnt, d_pm_max], device=dev,
        )
        dist = d_dist.numpy()
        pf_sumsq = d_pf_sumsq.numpy()
        pf_cnt = d_pf_cnt.numpy()
        pf_max = d_pf_max.numpy()
        pm_sumsq = d_pm_sumsq.numpy()
        pm_sum = d_pm_sum.numpy()
        pm_cnt = d_pm_cnt.numpy()
        pm_max = d_pm_max.numpy()
    finally:
        if prev_off is not None:
            skel.set_marker_offsets(prev_off)

    # per-frame (NaN where the frame had no visible marker)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_frame_rms = np.where(pf_cnt > 0, np.sqrt(pf_sumsq / pf_cnt), np.nan)
    per_frame_max = np.where(pf_cnt > 0, pf_max, np.nan)

    # per-marker (NaN where the marker was never visible)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_marker_rms = np.where(pm_cnt > 0, np.sqrt(pm_sumsq / pm_cnt), np.nan)
        per_marker_mean = np.where(pm_cnt > 0, pm_sum / pm_cnt, np.nan)
    per_marker_max = np.where(pm_cnt > 0, pm_max, np.nan)

    # overall (weighted by marker weight * visibility)
    if weights is None:
        w = np.ones(M)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
    wvis = w[None, :] * visible  # (F, M)
    total_w = wvis.sum()
    d2 = dist**2
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
