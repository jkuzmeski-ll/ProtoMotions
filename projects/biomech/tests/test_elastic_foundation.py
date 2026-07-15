# SPDX-License-Identifier: MIT

"""Tests for the distributed elastic-foundation foot contact (biomech.contact, M5).

Validates the Winkler-foundation contact law under prescribed kinematics:

- a flat sole pressed to a known depth produces the analytic vertical load and a
  centered COP,
- a foot above the ground carries no load,
- a sliding foot develops Coulomb friction that opposes its slip velocity and is
  bounded by ``mu * fn``,
- the reduction of per-patch forces into net GRF / COP / free-moment is consistent,
- the sole samplers produce sane areas and unit normals, and
- the Warp GPU kernel matches the NumPy reference (skipped when no CUDA device).

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

import numpy as np

from biomech.contact.elastic_foundation import (
    ElasticFoundationParams,
    FootSole,
    evaluate_contact,
    point_forces_numpy,
    point_forces_warp,
    reduce_wrench,
    sample_ellipsoid_sole,
    sample_flat_sole,
)
from biomech.tests import SkipTest

# identity xyzw quaternion
_QID = np.array([0.0, 0.0, 0.0, 1.0])


def _require_warp_cuda():
    try:
        import warp as wp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"warp not available: {exc}")
    if not wp.is_cuda_available():
        raise SkipTest("no CUDA device available for warp")
    return wp


# ---------------------------------------------------------------------------
# Sole geometry
# ---------------------------------------------------------------------------


def test_flat_sole_geometry():
    length, width, nx, ny = 0.25, 0.10, 10, 5
    sole = sample_flat_sole(length, width, nx, ny)
    assert sole.n == nx * ny
    # total tributary area == footprint
    assert abs(sole.total_area - length * width) < 1e-12
    # all patches share the footprint area equally
    assert np.allclose(sole.areas, (length * width) / (nx * ny))
    # normals are unit and point straight down (sole faces the ground)
    assert np.allclose(np.linalg.norm(sole.normals, axis=1), 1.0)
    assert np.allclose(sole.normals, np.array([0.0, 0.0, -1.0]))
    # samples span the footprint and lie in the z=0 plane
    assert np.allclose(sole.points[:, 2], 0.0)
    assert sole.points[:, 0].min() > -length / 2 - 1e-9
    assert sole.points[:, 0].max() < length / 2 + 1e-9


def test_ellipsoid_sole_geometry():
    sole = sample_ellipsoid_sole((0.12, 0.04, 0.03), n_theta=16, n_phi=8)
    assert sole.n == 16 * 8
    # areas are strictly positive
    assert np.all(sole.areas > 0.0)
    # normals are unit
    assert np.allclose(np.linalg.norm(sole.normals, axis=1), 1.0)
    # lower-cap normals point downward
    assert np.all(sole.normals[:, 2] < 0.0)
    # lower-cap points sit at or below the ellipsoid center
    assert np.all(sole.points[:, 2] <= 1e-9)


def test_sole_scaled_area_and_normals():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    scaled = sole.scaled(2.0, 3.0, 1.0)
    # tributary area scales with the ground-plane factors
    assert abs(scaled.total_area - sole.total_area * 2.0 * 3.0) < 1e-12
    # normals stay unit
    assert np.allclose(np.linalg.norm(scaled.normals, axis=1), 1.0)


# ---------------------------------------------------------------------------
# Contact law (NumPy reference)
# ---------------------------------------------------------------------------


def test_static_flat_foot_normal_load():
    length, width, nx, ny = 0.2, 0.1, 8, 4
    sole = sample_flat_sole(length, width, nx, ny)
    params = ElasticFoundationParams(k_bed=5e6, c_bed=5e3, mu=0.9)

    depth = 0.002  # 2 mm penetration
    ground_z = 0.0
    # place the (z=0) sole at world z = -depth so penetration == depth everywhere
    body_pos = np.array([[0.3, 0.1, ground_z - depth]])
    body_quat = _QID[None, :]
    linvel = np.zeros((1, 3))
    angvel = np.zeros((1, 3))

    pf, pw = point_forces_numpy(
        sole, params, body_pos, body_quat, linvel, angvel, ground_z=ground_z
    )
    pred = reduce_wrench(pf, pw, ground_z=ground_z)

    expected_fz = params.k_bed * depth * sole.total_area
    assert abs(pred.total_normal[0] - expected_fz) < 1e-6 * expected_fz
    # no horizontal load at rest
    assert np.allclose(pred.grf[0, :2], 0.0, atol=1e-9)
    assert abs(pred.grf[0, 2] - expected_fz) < 1e-6 * expected_fz
    # COP sits at the (translated) footprint centroid, on the ground plane
    assert abs(pred.cop[0, 0] - body_pos[0, 0]) < 1e-9
    assert abs(pred.cop[0, 1] - body_pos[0, 1]) < 1e-9
    assert abs(pred.cop[0, 2] - ground_z) < 1e-12
    # a symmetric static bed has no vertical free moment
    assert abs(pred.free_moment_z[0]) < 1e-6


def test_above_ground_no_load():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    params = ElasticFoundationParams()
    body_pos = np.array([[0.0, 0.0, 0.05]])  # 5 cm above ground
    pred = evaluate_contact(
        sole,
        params,
        body_pos,
        _QID[None, :],
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        ground_z=0.0,
    )
    assert pred.total_normal[0] == 0.0
    assert np.allclose(pred.grf[0], 0.0)
    # unloaded frame -> NaN COP (swing convention, matches ForcePlate)
    assert np.all(np.isnan(pred.cop[0]))


def test_sliding_friction_opposes_velocity_and_is_bounded():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    params = ElasticFoundationParams(k_bed=5e6, c_bed=0.0, mu=0.8, v_eps=1e-3)

    depth = 0.003
    vx = 0.5  # sliding in +x, well above v_eps
    body_pos = np.array([[0.0, 0.0, -depth]])
    linvel = np.array([[vx, 0.0, 0.0]])
    pf, pw = point_forces_numpy(
        sole, params, body_pos, _QID[None, :], linvel, np.zeros((1, 3))
    )
    pred = reduce_wrench(pf, pw)

    fn = pred.total_normal[0]
    # friction points opposite to the slip velocity (-x)
    assert pred.grf[0, 0] < 0.0
    assert abs(pred.grf[0, 1]) < 1e-9
    # magnitude bounded by Coulomb cone, and near mu*fn for fast sliding
    ft_mag = abs(pred.grf[0, 0])
    assert ft_mag < params.mu * fn + 1e-6
    ratio = ft_mag / (params.mu * fn)
    assert ratio > 0.99  # vx >> v_eps -> saturated


def test_damping_adds_load_on_penetration_and_relieves_on_lift():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    params = ElasticFoundationParams(k_bed=5e6, c_bed=1e5, mu=0.0)
    depth = 0.002
    body_pos = np.array([[0.0, 0.0, -depth], [0.0, 0.0, -depth]])
    quat = np.tile(_QID, (2, 1))
    # frame 0 pressing down (vz<0), frame 1 lifting off (vz>0)
    linvel = np.array([[0.0, 0.0, -0.1], [0.0, 0.0, +0.1]])
    pf, pw = point_forces_numpy(sole, params, body_pos, quat, linvel, np.zeros((2, 3)))
    pred = reduce_wrench(pf, pw)
    static = params.k_bed * depth * sole.total_area
    assert pred.total_normal[0] > static  # damping adds load when compressing
    assert pred.total_normal[1] < static  # damping relieves load when separating
    assert pred.total_normal[1] >= 0.0  # never adhesive


# ---------------------------------------------------------------------------
# Reduction consistency
# ---------------------------------------------------------------------------


def test_reduce_wrench_grf_is_patch_sum():
    rng = np.random.default_rng(0)
    F, N = 4, 20
    pf = rng.normal(size=(F, N, 3))
    pf[:, :, 2] = np.abs(pf[:, :, 2]) + 1.0  # ensure loaded
    pw = rng.normal(size=(F, N, 3))
    pred = reduce_wrench(pf, pw)
    assert np.allclose(pred.grf, pf.sum(axis=1))
    assert np.allclose(pred.total_normal, pf[:, :, 2].sum(axis=1))
    # COP is the normal-weighted centroid
    w = pf[:, :, 2] / pf[:, :, 2].sum(axis=1, keepdims=True)
    cx = np.sum(w * pw[:, :, 0], axis=1)
    assert np.allclose(pred.cop[:, 0], cx)


# ---------------------------------------------------------------------------
# Warp GPU parity
# ---------------------------------------------------------------------------


def test_warp_matches_numpy():
    _require_warp_cuda()
    sole = sample_ellipsoid_sole((0.12, 0.045, 0.03), n_theta=20, n_phi=10)
    params = ElasticFoundationParams(k_bed=4e6, c_bed=8e3, mu=0.7, v_eps=1e-3)

    rng = np.random.default_rng(3)
    F = 8
    body_pos = rng.normal(scale=0.05, size=(F, 3))
    body_pos[:, 2] -= 0.02  # drive some penetration
    # random small rotations (near identity), normalized xyzw
    q = rng.normal(scale=0.1, size=(F, 4))
    q[:, 3] += 1.0
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    linvel = rng.normal(scale=0.3, size=(F, 3))
    angvel = rng.normal(scale=0.3, size=(F, 3))

    pf_np, pw_np = point_forces_numpy(sole, params, body_pos, q, linvel, angvel)
    pf_wp, pw_wp = point_forces_warp(sole, params, body_pos, q, linvel, angvel)

    # float32 warp vs float64 numpy; scale tolerance to force magnitude
    fscale = max(1.0, np.abs(pf_np).max())
    assert np.abs(pw_np - pw_wp).max() < 1e-4
    assert np.abs(pf_np - pf_wp).max() < 1e-3 * fscale


def test_evaluate_contact_warp_backend_matches_numpy():
    _require_warp_cuda()
    sole = sample_flat_sole(0.2, 0.1, 10, 5)
    params = ElasticFoundationParams()
    rng = np.random.default_rng(7)
    F = 5
    body_pos = rng.normal(scale=0.03, size=(F, 3))
    body_pos[:, 2] -= 0.01
    quat = np.tile(_QID, (F, 1))
    linvel = rng.normal(scale=0.2, size=(F, 3))
    angvel = np.zeros((F, 3))

    pred_np = evaluate_contact(sole, params, body_pos, quat, linvel, angvel, backend="numpy")
    pred_wp = evaluate_contact(sole, params, body_pos, quat, linvel, angvel, backend="warp")

    fscale = max(1.0, np.abs(pred_np.grf).max())
    assert np.abs(pred_np.grf - pred_wp.grf).max() < 1e-3 * fscale
    # COP where both are loaded
    loaded = (pred_np.total_normal > 1e-6) & (pred_wp.total_normal > 1e-6)
    if np.any(loaded):
        assert np.nanmax(np.abs(pred_np.cop[loaded] - pred_wp.cop[loaded])) < 1e-3
