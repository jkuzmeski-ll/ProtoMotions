# SPDX-License-Identifier: MIT
#
# Windows-native port of the *spirit* of Nimble's marker cleanup
# (``dart/biomechanics/MarkerFixer.{hpp,cpp}``). Nimble's MarkerFixer runs before
# initialization to (a) drop gross outliers / mislabeled swaps, and (b) gap-fill short
# dropouts, using RANSAC rigid-body fits over marker clusters. Raw optical mocap has
# dropouts (occlusion) and spikes (ghost markers / label swaps); feeding those straight
# into IK poisons the fit, so cleaning first is standard practice.
#
# This port keeps the two decisions that actually matter for a downstream least-squares
# IK, computed with plain NumPy (no RANSAC solver dependency):
#
#   1. rigid-body consistency: markers on the *same* skeleton body keep (almost) constant
#      pairwise distances. We build a robust per-body template of pairwise distances
#      (median over frames), then per frame flag a marker whose distances to its
#      same-body neighbours deviate from the template by more than a robust threshold as
#      an outlier and blank it (NaN). This catches swaps and ghost markers exactly like
#      Nimble's rigid checks, without needing a fitted skeleton.
#   2. spike + gap handling: a per-marker robust velocity gate blanks single-frame jumps
#      (median + k*MAD of frame-to-frame displacement), then short NaN gaps
#      (<= ``max_gap`` frames, including the just-blanked spikes) are linearly filled.
#      Long gaps are left NaN so the IK visibility mask handles them.
#
# All checks are robust (median / MAD), so a few bad frames do not corrupt the template.

"""Robust marker cleanup: outlier rejection + gap fill (port of Nimble MarkerFixer)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MarkerFixConfig:
    """Settings for :func:`fix_markers`."""

    # Rigid-body pairwise-distance check (needs a per-marker body grouping).
    rigid_tol_m: float = 0.03  # absolute distance slack before the robust gate kicks in
    rigid_mad_k: float = 6.0  # reject if |dist - template| > tol + k * MAD(dist)
    min_body_markers: int = 3  # only bodies with this many markers get the rigid check
    # Per-marker spike gate on frame-to-frame displacement.
    spike_mad_k: float = 8.0
    spike_min_disp_m: float = 0.05  # never flag jumps smaller than this (noise floor)
    # Gap filling.
    max_gap: int = 10  # linearly fill NaN runs no longer than this many frames
    fill_gaps: bool = True


@dataclass
class MarkerFixReport:
    n_rigid_rejected: int = 0
    n_spikes_rejected: int = 0
    n_filled: int = 0
    per_marker_rejected: np.ndarray | None = None  # (M,) count blanked per marker
    per_marker_filled: np.ndarray | None = None  # (M,) count filled per marker
    marker_names: list[str] = field(default_factory=list)

    def format(self) -> str:
        return (
            "MarkerFixer report\n"
            f"  rigid-body outliers blanked : {self.n_rigid_rejected}\n"
            f"  velocity spikes blanked     : {self.n_spikes_rejected}\n"
            f"  samples gap-filled          : {self.n_filled}"
        )


def _robust_scale(x: np.ndarray) -> float:
    """MAD-based robust standard-deviation estimate (0 if degenerate)."""
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _fill_short_gaps(track: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Linearly interpolate NaN runs of length <= ``max_gap`` in a (F, 3) trajectory."""
    out = track.copy()
    F = out.shape[0]
    valid = np.isfinite(out).all(axis=1)
    if valid.sum() < 2:
        return out, 0
    idx = np.where(valid)[0]
    filled = 0
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (b - a)
                out[a + k] = (1.0 - t) * out[a] + t * out[b]
            filled += gap
    return out, filled


def fix_markers(
    observations: np.ndarray,
    body_of_marker: np.ndarray | None = None,
    marker_names: list[str] | None = None,
    config: MarkerFixConfig | None = None,
) -> tuple[np.ndarray, MarkerFixReport]:
    """Clean a raw ``(F, M, 3)`` marker array (NaN = missing) before IK.

    Parameters
    ----------
    observations : (F, M, 3)
        Raw observed marker world positions; NaN where missing.
    body_of_marker : (M,) int, optional
        Body/segment index each marker is attached to (e.g. ``skel.topo.m_body``). When
        given, markers on the same body get the rigid-body pairwise-distance check.
    marker_names : list[str], optional
        For the report only.
    config : MarkerFixConfig, optional

    Returns
    -------
    (cleaned, report) : (np.ndarray, MarkerFixReport)
        ``cleaned`` is a fresh ``(F, M, 3)`` array with outliers blanked to NaN and short
        gaps filled. The input is not modified.
    """
    cfg = config or MarkerFixConfig()
    obs = np.array(observations, dtype=np.float64)  # copy
    assert obs.ndim == 3 and obs.shape[2] == 3, obs.shape
    F, M, _ = obs.shape

    per_marker_rejected = np.zeros(M, dtype=int)
    per_marker_filled = np.zeros(M, dtype=int)

    # 1. rigid-body pairwise-distance outlier rejection ---------------------------
    n_rigid = 0
    if body_of_marker is not None:
        bom = np.asarray(body_of_marker, dtype=int).ravel()
        assert bom.size == M, (bom.size, M)
        for body in np.unique(bom):
            members = np.where(bom == body)[0]
            if members.size < cfg.min_body_markers:
                continue
            # template pairwise distances (median over frames where both visible)
            for a_i in range(members.size):
                ma = members[a_i]
                # distance from ma to each other member across frames
                for b_i in range(a_i + 1, members.size):
                    mb = members[b_i]
                    d = np.linalg.norm(obs[:, ma] - obs[:, mb], axis=1)  # (F,)
                    good = np.isfinite(d)
                    if good.sum() < 3:
                        continue
                    templ = np.median(d[good])
                    scale = _robust_scale(d[good])
                    thr = cfg.rigid_tol_m + cfg.rigid_mad_k * scale
                    bad = good & (np.abs(d - templ) > thr)
                    # a distance is bad because ma or mb is off; count the violation but
                    # do not blank yet (accumulate a per-marker vote below)
                    if bad.any():
                        # blank the marker with the larger local deviation from its own
                        # median position in the offending frames
                        for t in np.where(bad)[0]:
                            # decide which endpoint is the outlier: the one whose distance
                            # to *other* same-body markers is also inconsistent
                            va = _consistency(obs, t, ma, members, templ_cache=None)
                            vb = _consistency(obs, t, mb, members, templ_cache=None)
                            victim = ma if va >= vb else mb
                            if np.isfinite(obs[t, victim]).all():
                                obs[t, victim] = np.nan
                                per_marker_rejected[victim] += 1
                                n_rigid += 1

    # 2. per-marker spike gate ---------------------------------------------------
    n_spike = 0
    for m in range(M):
        track = obs[:, m]
        vis = np.isfinite(track).all(axis=1)
        if vis.sum() < 3:
            continue
        disp = np.full(F, np.nan)
        prev = None
        for t in range(F):
            if vis[t]:
                if prev is not None:
                    disp[t] = np.linalg.norm(track[t] - track[prev])
                prev = t
        d = disp[np.isfinite(disp)]
        if d.size < 3:
            continue
        med = np.median(d)
        scale = _robust_scale(d)
        thr = max(cfg.spike_min_disp_m, med + cfg.spike_mad_k * scale)
        for t in range(F):
            if np.isfinite(disp[t]) and disp[t] > thr:
                # blank the isolated spike sample
                obs[t, m] = np.nan
                per_marker_rejected[m] += 1
                n_spike += 1

    # 3. gap fill ----------------------------------------------------------------
    n_filled = 0
    if cfg.fill_gaps:
        for m in range(M):
            filled_track, k = _fill_short_gaps(obs[:, m], cfg.max_gap)
            if k:
                obs[:, m] = filled_track
                per_marker_filled[m] += k
                n_filled += k

    return obs, MarkerFixReport(
        n_rigid_rejected=n_rigid,
        n_spikes_rejected=n_spike,
        n_filled=n_filled,
        per_marker_rejected=per_marker_rejected,
        per_marker_filled=per_marker_filled,
        marker_names=list(marker_names) if marker_names is not None else [],
    )


def _consistency(obs: np.ndarray, t: int, m: int, members: np.ndarray, templ_cache) -> float:
    """How inconsistent marker ``m`` is with its same-body neighbours at frame ``t``.

    Returns the median absolute deviation of ``m``'s current pairwise distances from
    their per-pair medians over time. A larger value means ``m`` is more likely the
    outlier of a violating pair.
    """
    devs = []
    pm = obs[t, m]
    if not np.isfinite(pm).all():
        return np.inf
    for other in members:
        if other == m:
            continue
        d_now = np.linalg.norm(pm - obs[t, other])
        if not np.isfinite(d_now):
            continue
        d_all = np.linalg.norm(obs[:, m] - obs[:, other], axis=1)
        d_all = d_all[np.isfinite(d_all)]
        if d_all.size < 3:
            continue
        devs.append(abs(d_now - np.median(d_all)))
    return float(np.median(devs)) if devs else 0.0
