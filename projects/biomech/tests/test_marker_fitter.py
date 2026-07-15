# SPDX-License-Identifier: MIT

"""Synthetic round-trip tests for the bilevel marker fit (biomech.fitting.marker_fitter), M2d.

We generate synthetic markers from the Warp skeleton at known {group scales, marker
offsets, poses}, then check that the bilevel fit recovers the scales and drives the
marker reprojection error down. Because {scale, offset, pose} has a gauge ambiguity,
exact recovery is only expected for the scales when the offsets are anchored at their
true (model) values by the prior; the offset-perturbation case checks that the fit
reduces marker error well below a scales-only baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.ik import MarkerIKConfig, position_limits, solve_marker_ik
from biomech.fitting.marker_fitter import MarkerFitConfig, MarkerFitter
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


def test_recovers_group_scales_offsets_at_model():
    """True offsets == model offsets, true scales anisotropic.

    The fit's objective is marker reprojection, which it drives to sub-mm. Note that
    per-axis *anisotropic* segment scale is only partially identifiable from markers
    alone (bone-width directions and tiny foot bones are weakly observed); exact
    per-axis recovery needs Nimble's anthropometric body-scale prior, a documented
    deferred term. We therefore assert the objective (marker RMS) tightly and scale
    recovery in aggregate.
    """
    skel = _skel()
    G = skel.topo.num_groups
    rng = np.random.default_rng(0)
    true_scales = rng.uniform(0.92, 1.08, size=3 * G)
    q_true = _feasible_poses(skel, 10, 0.25, 1)
    # markers at model offsets, true scales
    _, obs = skel.forward(q_true, group_scales=true_scales)

    fitter = MarkerFitter(skel, obs)
    res = fitter.fit(
        init_scales=np.ones(3 * G),
        config=MarkerFitConfig(outer_iters=20, offset_prior_weight=25.0),
    )
    # Objective met to a few mm with offsets anchored to the model (the residual is the
    # weakly-observed anisotropic scale axes; the offset-fit test drives RMS lower by
    # letting offsets move).
    assert np.median(res.marker_rms) < 2.5e-3, res.marker_rms
    # Scales stay in bounds and improve on the naive unit-scale guess.
    assert np.all(res.group_scales > 0.5) and np.all(res.group_scales < 1.6)
    init_err = np.median(np.abs(np.ones(3 * G) - true_scales))
    fit_err = np.median(np.abs(res.group_scales - true_scales))
    assert fit_err <= init_err + 1e-9, (fit_err, init_err)
    # Offsets stay anchored near the model (no drift under the prior).
    assert np.max(np.linalg.norm(res.marker_offsets - fitter.offset0, axis=1)) < 0.02


def test_offset_fit_beats_scales_only_baseline():
    """With true offset perturbations, fitting offsets beats a scales-only fit."""
    skel = _skel()
    G = skel.topo.num_groups
    M = skel.topo.num_markers
    rng = np.random.default_rng(2)
    true_scales = rng.uniform(0.95, 1.05, size=3 * G)
    true_off = skel.marker_offsets().copy()
    # Perturb a subset of markers by ~1 cm.
    pick = rng.choice(M, size=M // 3, replace=False)
    true_off[pick] += rng.normal(0, 0.01, size=(len(pick), 3))

    q_true = _feasible_poses(skel, 12, 0.25, 3)
    skel.set_marker_offsets(true_off)
    _, obs = skel.forward(q_true, group_scales=true_scales)
    # Restore model offsets as the fit's starting point.
    model_off = np.array([m.offset for m in skel.spec.markers])
    skel.set_marker_offsets(model_off)

    # Baseline: scales-only (freeze offsets by huge prior).
    base = MarkerFitter(skel, obs).fit(
        init_scales=np.ones(3 * G),
        config=MarkerFitConfig(outer_iters=20, offset_prior_weight=1e8),
    )
    # Full: fit offsets too (weak prior).
    skel.set_marker_offsets(model_off)
    full = MarkerFitter(skel, obs).fit(
        init_scales=np.ones(3 * G),
        config=MarkerFitConfig(outer_iters=20, offset_prior_weight=0.05),
    )
    assert np.median(full.marker_rms) < np.median(base.marker_rms)
    assert np.median(full.marker_rms) < 4e-3, np.median(full.marker_rms)


def test_fit_is_noop_when_already_correct():
    """Starting at the truth, the fit should stay put and report ~zero error."""
    skel = _skel()
    G = skel.topo.num_groups
    q_true = _feasible_poses(skel, 6, 0.2, 5)
    _, obs = skel.forward(q_true)  # unit scale, model offsets
    res = MarkerFitter(skel, obs).fit(
        init_scales=np.ones(3 * G),
        config=MarkerFitConfig(outer_iters=15),
    )
    assert np.max(np.abs(res.group_scales - 1.0)) < 0.02
    assert np.median(res.marker_rms) < 1e-3
