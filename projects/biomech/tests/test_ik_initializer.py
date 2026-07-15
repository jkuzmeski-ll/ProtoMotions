# SPDX-License-Identifier: MIT

"""Synthetic round-trip tests for the closed-form IK initializer (biomech.fitting.ik_initializer), M2c.

No direct Nimble golden exists yet for the initializer (that needs the S001 marker map
+ a WSL reference run). We validate the closed-form math by round-trip on the Warp
skeleton: forward-kinematic known poses (and known scales) to synthetic markers, then
check that

  * the MDS joint-center solver recovers the true joint centers (unit scale, where the
    model's neutral joint<->marker distances match the data exactly), and
  * the closed-form group-scale estimate returns unit scale, and
  * the full pipeline drives the marker reprojection error to ~zero.

An anisotropic-scale case checks the pipeline runs and returns sane, bounded scales
(exact scale recovery under scaling needs the deferred prescale / iterative polishing
passes, so its tolerance is loose).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.ik import MarkerIKConfig, position_limits
from biomech.fitting.ik_initializer import IKInitializer
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


def _feasible_poses(skel: WarpSkeleton, F: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo, hi = position_limits(skel.spec)
    ndof = skel.topo.num_dofs
    q = rng.uniform(-scale, scale, size=(F, ndof))
    q[:, 0:3] += rng.uniform(-0.2, 0.2, size=(F, 3))
    q[:, 3:6] += rng.uniform(-0.4, 0.4, size=(F, 3))
    q = np.clip(q, lo, hi)
    ranged = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    q[:, ranged] = np.clip(q[:, ranged], lo[ranged] + 1e-3, hi[ranged] - 1e-3)
    return q


def _fk_joint_centers(skel: WarpSkeleton, q: np.ndarray, scales=None) -> np.ndarray:
    """Ground-truth joint-center world positions from FK (F, J, 3)."""
    world, _ = skel.forward(q, scales)  # (F, B, 4, 4)
    F = world.shape[0]
    J = skel.topo.num_joints
    out = np.zeros((F, J, 3))
    for j in range(J):
        pb = int(skel.topo.j_parent_body[j])
        Tp = skel.topo.T_parent[j]
        for t in range(F):
            jw = (world[t, pb] @ Tp) if pb >= 0 else Tp
            out[t, j] = jw[:3, 3]
    return out


def test_topology_selects_multi_marker_joints():
    skel = _skel()
    q = _feasible_poses(skel, 4, 0.2, 0)
    _, mk = skel.forward(q)
    init = IKInitializer(skel, mk)
    # At least the big lower-limb joints should qualify (>=3 adjacent markers).
    names = [skel.spec.joints[j].name for j in init.active_joints]
    assert len(init.active_joints) >= 5, names
    assert any("hip" in n or "walker_knee" in n or "knee" in n for n in names), names


def test_mds_recovers_joint_centers_unit_scale():
    skel = _skel()
    q = _feasible_poses(skel, 6, 0.25, 1)
    _, mk = skel.forward(q)  # unit scale
    gt = _fk_joint_centers(skel, q)  # (F, J, 3)

    init = IKInitializer(skel, mk)
    solved = init.closed_form_mds_joint_centers()

    # The free root joint (ground_pelvis) has no rigid pivot center, and the
    # SimmSpline-coupled knee (walker_knee) translates with flexion, so neither is a
    # fixed-distance point -- exclude them (Nimble treats the root specially too). The
    # remaining revolute/euler joints are true pivots and recover to sub-mm at unit
    # scale (where the model's neutral joint<->marker distances match the data).
    exclude = {"ground_pelvis", "walker_knee_l", "walker_knee_r"}
    max_err = 0.0
    checked = 0
    for t in range(len(solved)):
        for j, center in solved[t].items():
            if skel.spec.joints[j].name in exclude:
                continue
            err = float(np.linalg.norm(center - gt[t, j]))
            max_err = max(max_err, err)
            checked += 1
    assert checked > 0
    assert max_err < 1e-3, max_err


def test_group_scales_recovered_unit_scale():
    skel = _skel()
    q = _feasible_poses(skel, 10, 0.3, 2)
    _, mk = skel.forward(q)
    init = IKInitializer(skel, mk)
    scales = init.estimate_group_scales()
    assert scales.shape == (skel.topo.num_groups * 3,)
    # Groups that got real observations should come back ~1; unobserved default to 1.
    assert np.all(scales > 0.5) and np.all(scales < 1.5)
    assert np.median(np.abs(scales - 1.0)) < 0.02


def test_full_run_reduces_marker_error_unit_scale():
    skel = _skel()
    q = _feasible_poses(skel, 6, 0.25, 3)
    _, mk = skel.forward(q)
    init = IKInitializer(skel, mk)
    res = init.run(config=MarkerIKConfig(max_iters=250))
    assert res.poses.shape == (6, skel.topo.num_dofs)
    # Closed-form scales ~1 and IK drives marker error down.
    assert np.median(np.abs(res.group_scales - 1.0)) < 0.05
    assert np.median(res.marker_rms) < 5e-3, res.marker_rms


def test_anisotropic_scaling_runs_and_is_bounded():
    skel = _skel()
    G = skel.topo.num_groups
    rng = np.random.default_rng(4)
    scales = rng.uniform(0.9, 1.1, size=3 * G)
    q = _feasible_poses(skel, 8, 0.25, 5)
    _, mk = skel.forward(q, group_scales=scales)
    init = IKInitializer(skel, mk)
    est = init.estimate_group_scales()
    # Bounded and finite; the initializer must not blow up under scaling.
    assert np.all(np.isfinite(est))
    assert np.all(est > 0.6) and np.all(est < 1.4)


def test_prescale_recovers_uniform_scale():
    """A uniformly scaled subject: the prescale estimate recovers the scalar."""
    skel = _skel()
    G = skel.topo.num_groups
    s = 1.12
    q = _feasible_poses(skel, 6, 0.2, 6)
    _, mk = skel.forward(q, group_scales=np.full(3 * G, s))
    init = IKInitializer(skel, mk)
    pre = init.estimate_prescale()
    assert abs(pre - s) < 0.02, pre


def test_prescale_is_unity_at_model_scale():
    skel = _skel()
    q = _feasible_poses(skel, 6, 0.2, 7)
    _, mk = skel.forward(q)  # unit scale
    init = IKInitializer(skel, mk)
    assert abs(init.estimate_prescale() - 1.0) < 0.01


def test_prescale_fills_unobserved_groups():
    """Groups with no usable marker pairs fall back to the prescale, not 1.0."""
    skel = _skel()
    G = skel.topo.num_groups
    s = 1.15
    q = _feasible_poses(skel, 6, 0.2, 8)
    _, mk = skel.forward(q, group_scales=np.full(3 * G, s))
    init = IKInitializer(skel, mk)
    est_pre = init.estimate_group_scales(prescale=True)
    # unobserved axes/groups now sit near the prescale, not pinned to 1.0
    init2 = IKInitializer(skel, mk)
    est_no = init2.estimate_group_scales(default_scale=1.0, prescale=False)
    # the prescale version is at least as close to the true uniform scale in aggregate
    assert np.median(np.abs(est_pre - s)) <= np.median(np.abs(est_no - s)) + 1e-9
