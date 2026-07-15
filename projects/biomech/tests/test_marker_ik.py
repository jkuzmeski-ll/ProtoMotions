# SPDX-License-Identifier: MIT

"""Round-trip tests for the batched marker IK (biomech.fitting.ik), M2c pose solver.

There is no direct Nimble golden for the per-frame LM step (Nimble's ``refineIK`` is an
internal descent whose *path* is implementation-specific; the recovered pose is set by
the least-squares minimum, not the schedule). We therefore validate by round-trip: take
the Warp skeleton, forward-kinematic a batch of random poses to synthetic markers, then
recover the poses with :func:`solve_marker_ik` and check that the reprojected marker
error is driven to ~zero and the poses are recovered. Missing-marker masking, per-marker
weighting, and anisotropic group scaling are exercised too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.ik import (
    MarkerIKConfig,
    position_limits,
    solve_marker_ik,
)
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


def _random_poses(skel: WarpSkeleton, F: int, scale: float, seed: int) -> np.ndarray:
    """Random *feasible* poses: small joint angles + a real root pose.

    Poses are made feasible w.r.t. the model's joint limits so the clamping solver
    can recover them exactly: locked DOFs collapse to their default, and ranged
    DOFs are pulled strictly interior.
    """
    rng = np.random.default_rng(seed)
    lo, hi = position_limits(skel.spec)
    ndof = skel.topo.num_dofs
    q = rng.uniform(-scale, scale, size=(F, ndof))
    # Give the free root a nontrivial pose so IK must recover global placement.
    q[:, 0:3] += rng.uniform(-0.3, 0.3, size=(F, 3))  # root orientation (euler)
    q[:, 3:6] += rng.uniform(-0.5, 0.5, size=(F, 3))  # root translation
    # Clamp to limits (locked DOFs -> default), then pull ranged DOFs interior.
    q = np.clip(q, lo, hi)
    ranged = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    q[:, ranged] = np.clip(q[:, ranged], lo[ranged] + 1e-3, hi[ranged] - 1e-3)
    return q


def test_position_limits_shapes_and_lock():
    skel = _skel()
    lo, hi = position_limits(skel.spec)
    assert lo.shape == (skel.topo.num_dofs,)
    assert hi.shape == (skel.topo.num_dofs,)
    assert np.all(lo <= hi)


def test_recovers_single_pose_to_zero_error():
    skel = _skel()
    q_true = _random_poses(skel, 1, scale=0.2, seed=0)
    _, markers = skel.forward(q_true)  # (1, M, 3)
    q_init = q_true + 0.05  # small perturbation
    res = solve_marker_ik(
        skel, markers, q_init, config=MarkerIKConfig(max_iters=200)
    )
    assert res.marker_rms[0] < 1e-6, res.marker_rms
    # Reprojection recovers the true markers to sub-micron.
    _, mk = skel.forward(res.q)
    assert np.max(np.abs(mk - markers)) < 1e-5


def test_recovers_batch_of_frames():
    skel = _skel()
    F = 8
    q_true = _random_poses(skel, F, scale=0.25, seed=1)
    _, markers = skel.forward(q_true)
    q_init = q_true + np.random.default_rng(2).uniform(-0.05, 0.05, q_true.shape)
    res = solve_marker_ik(
        skel, markers, q_init, config=MarkerIKConfig(max_iters=200)
    )
    assert res.q.shape == (F, skel.topo.num_dofs)
    assert np.all(res.marker_rms < 1e-6), res.marker_rms


def test_recovers_under_anisotropic_scaling():
    skel = _skel()
    G = skel.topo.num_groups
    rng = np.random.default_rng(3)
    scales = rng.uniform(0.85, 1.15, size=3 * G)
    q_true = _random_poses(skel, 4, scale=0.2, seed=4)
    _, markers = skel.forward(q_true, group_scales=scales)
    q_init = q_true + 0.03
    res = solve_marker_ik(
        skel,
        markers,
        q_init,
        group_scales=scales,
        config=MarkerIKConfig(max_iters=200),
    )
    assert np.all(res.marker_rms < 1e-6), res.marker_rms


def test_missing_markers_are_masked():
    skel = _skel()
    F = 4
    M = skel.topo.num_markers
    q_true = _random_poses(skel, F, scale=0.2, seed=5)
    _, markers = skel.forward(q_true)
    # Drop 10 markers on some frames by inserting NaN (auto-detected).
    rng = np.random.default_rng(6)
    obs = markers.copy()
    for f in range(F):
        drop = rng.choice(M, size=10, replace=False)
        obs[f, drop, :] = np.nan
    q_init = q_true + 0.04
    res = solve_marker_ik(
        skel, obs, q_init, config=MarkerIKConfig(max_iters=250)
    )
    # Still recovers pose from the remaining markers (well over-determined).
    _, mk = skel.forward(res.q)
    visible = np.isfinite(obs).all(axis=2)
    err = np.linalg.norm(mk - np.nan_to_num(obs), axis=2)
    assert np.max(err[visible]) < 1e-4


def test_zero_weight_marker_is_ignored():
    skel = _skel()
    q_true = _random_poses(skel, 1, scale=0.2, seed=7)
    _, markers = skel.forward(q_true)
    obs = markers.copy()
    weights = np.ones(skel.topo.num_markers)
    # Corrupt one marker but give it zero weight; solution must ignore it.
    obs[0, 0, :] += 0.5
    weights[0] = 0.0
    res = solve_marker_ik(
        skel, obs, q_true + 0.03, weights=weights,
        config=MarkerIKConfig(max_iters=200),
    )
    _, mk = skel.forward(res.q)
    # All other markers reprojected essentially exactly.
    err = np.linalg.norm(mk[0, 1:] - markers[0, 1:], axis=1)
    assert np.max(err) < 1e-5
