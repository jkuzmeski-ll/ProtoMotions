# SPDX-License-Identifier: MIT

"""Tests for the marker-fit error report (biomech.fitting.report, IKErrorReport port)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.ik import position_limits
from biomech.fitting.report import marker_errors
from biomech.osim import parse_osim
from biomech.skeleton.skeleton import WarpSkeleton

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_skel_cache: WarpSkeleton | None = None


def _skel() -> WarpSkeleton:
    global _skel_cache
    if _skel_cache is None:
        _skel_cache = WarpSkeleton(parse_osim(str(_OSIM)), device="cpu")
    return _skel_cache


def _feasible_poses(skel, F, scale, seed):
    rng = np.random.default_rng(seed)
    lo, hi = position_limits(skel.spec)
    ndof = skel.topo.num_dofs
    q = rng.uniform(-scale, scale, size=(F, ndof))
    q[:, 3:6] += rng.uniform(-0.3, 0.3, size=(F, 3))
    q = np.clip(q, lo, hi)
    ranged = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    q[:, ranged] = np.clip(q[:, ranged], lo[ranged] + 1e-3, hi[ranged] - 1e-3)
    return q


def test_zero_error_when_poses_match():
    skel = _skel()
    q = _feasible_poses(skel, 6, 0.2, 0)
    _, obs = skel.forward(q)
    rep = marker_errors(skel, obs, q)
    assert rep.rms < 1e-9, rep.rms
    assert rep.max < 1e-9, rep.max
    assert rep.num_visible == obs.shape[0] * obs.shape[1]


def test_per_marker_error_matches_injected_offset():
    skel = _skel()
    q = _feasible_poses(skel, 8, 0.2, 1)
    _, obs = skel.forward(q)
    obs = obs.copy()
    m = 3
    obs[:, m] += np.array([0.005, 0.0, 0.0])  # 5 mm shift on one marker, all frames
    rep = marker_errors(skel, obs, q)
    assert abs(rep.per_marker_rms[m] - 0.005) < 1e-6, rep.per_marker_rms[m]
    assert abs(rep.per_marker_max[m] - 0.005) < 1e-6
    worst = rep.worst_markers(1)
    assert worst[0][0] == skel.marker_names()[m]


def test_nan_observations_are_not_counted():
    skel = _skel()
    q = _feasible_poses(skel, 5, 0.2, 2)
    _, obs = skel.forward(q)
    obs = obs.copy()
    obs[:, 0] = np.nan  # marker 0 never visible
    obs[0, 1] = np.nan  # marker 1 missing on frame 0 only
    rep = marker_errors(skel, obs, q)
    assert not np.isfinite(rep.per_marker_rms[0])
    assert np.isfinite(rep.per_marker_rms[1])
    M = obs.shape[1]
    F = obs.shape[0]
    assert rep.num_visible == F * M - F - 1
    # report still near zero on the visible markers
    assert rep.rms < 1e-9


def test_format_runs():
    skel = _skel()
    q = _feasible_poses(skel, 4, 0.2, 3)
    _, obs = skel.forward(q)
    rep = marker_errors(skel, obs, q)
    text = rep.format(k=5)
    assert "Marker error report" in text
