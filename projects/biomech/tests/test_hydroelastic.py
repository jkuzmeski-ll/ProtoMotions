# SPDX-License-Identifier: MIT

"""Tests for the hydroelastic pressure-field foot contact (biomech.contact, M7).

Validates the pressure-field law generalizes the Winkler bed:

- in the linear limit (b=0, no Hunt-Crossley, uniform modulus, mu_s=mu_d) it reproduces
  ``biomech.contact.elastic_foundation`` exactly,
- hyperelastic stiffening (b>0) increases force super-linearly with depth,
- Hunt-Crossley dissipation adds load on penetration but never pulls (non-adhesive) and
  scales with the elastic pressure,
- a spatially-varying modulus shifts the centre of pressure toward the stiffer region,
- Stribeck friction peaks near zero sliding speed and relaxes to mu_d when fast,
- the Warp kernel matches the NumPy reference (skipped without CUDA).

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

import numpy as np

from biomech.contact.elastic_foundation import (
    ElasticFoundationParams,
    FootSole,
    sample_ellipsoid_sole,
    sample_flat_sole,
)
from biomech.contact.elastic_foundation import (
    point_forces_numpy as ef_point_forces,
)
from biomech.contact.hydroelastic import (
    HydroelasticParams,
    evaluate_contact,
    point_forces_numpy,
    point_forces_warp,
)
from biomech.tests import SkipTest

_QID = np.array([0.0, 0.0, 0.0, 1.0])


def _require_warp_cuda():
    try:
        import warp as wp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"warp not available: {exc}")
    if not wp.is_cuda_available():
        raise SkipTest("no CUDA device available for warp")


def test_reduces_to_winkler_in_linear_limit():
    sole = sample_flat_sole(0.22, 0.09, 8, 4)
    rng = np.random.default_rng(0)
    F = 6
    body_pos = rng.normal(scale=0.03, size=(F, 3))
    body_pos[:, 2] -= 0.01  # some penetration
    quat = np.tile(_QID, (F, 1))
    linvel = rng.normal(scale=0.2, size=(F, 3))
    angvel = np.zeros((F, 3))

    # Winkler: fn = area*(k*d + c*vn); set c=0.
    ef = ElasticFoundationParams(k_bed=5e6, c_bed=0.0, mu=0.7, v_eps=1e-3)
    # Hydroelastic linear limit: b=0, hc_alpha=0, uniform modulus, mu_s=mu_d.
    he = HydroelasticParams(
        k_bed=5e6, stiffen_b=0.0, hc_alpha=0.0, mu_d=0.7, mu_s=0.7,
        v_stribeck=0.05, v_eps=1e-3,
    )
    pf_ef, _ = ef_point_forces(sole, ef, body_pos, quat, linvel, angvel)
    pf_he, _ = point_forces_numpy(sole, he, body_pos, quat, linvel, angvel)
    assert np.abs(pf_ef - pf_he).max() < 1e-9


def test_hyperelastic_stiffening_superlinear():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    params = HydroelasticParams(k_bed=5e6, stiffen_b=50.0, hc_alpha=0.0, mu_d=0.0, mu_s=0.0)
    depths = np.array([0.001, 0.002, 0.004])
    pos = np.zeros((3, 3))
    pos[:, 2] = -depths
    quat = np.tile(_QID, (3, 1))
    z = np.zeros((3, 3))
    pf, pw = point_forces_numpy(sole, params, pos, quat, z, z)
    fz = pf[:, :, 2].sum(axis=1)
    # linear model would double force when depth doubles; stiffening -> more than double
    assert fz[1] / fz[0] > 2.0
    assert fz[2] / fz[1] > 2.0


def test_hunt_crossley_dissipation_non_adhesive():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    params = HydroelasticParams(k_bed=5e6, stiffen_b=0.0, hc_alpha=5.0, mu_d=0.0, mu_s=0.0)
    depth = 0.002
    pos = np.array([[0, 0, -depth], [0, 0, -depth], [0, 0, -depth]], dtype=float)
    quat = np.tile(_QID, (3, 1))
    # frame 0: compressing (vz<0), 1: static, 2: separating fast (vz>0)
    linvel = np.array([[0, 0, -0.05], [0, 0, 0.0], [0, 0, +10.0]], dtype=float)
    angvel = np.zeros((3, 3))
    pf, _ = point_forces_numpy(sole, params, pos, quat, linvel, angvel)
    fz = pf[:, :, 2].sum(axis=1)
    static = 5e6 * depth * sole.total_area
    assert fz[0] > static  # dissipation adds load while compressing
    assert abs(fz[1] - static) < 1e-6 * static  # no rate -> pure elastic
    assert fz[2] >= 0.0  # never adhesive even for large separation velocity
    assert fz[2] < 1e-6 * static  # clamped to ~0


def test_spatial_modulus_shifts_cop():
    # stiffer forefoot (+x) than heel (-x) should pull COP toward +x
    length, width, nx, ny = 0.24, 0.09, 10, 4
    sole = sample_flat_sole(length, width, nx, ny)
    # modulus grows with x
    xs = sole.points[:, 0]
    mod = 1.0 + 4.0 * (xs - xs.min()) / (xs.max() - xs.min())
    stiff = FootSole(sole.points, sole.normals, sole.areas, modulus=mod)

    params = HydroelasticParams(k_bed=5e6, stiffen_b=0.0, hc_alpha=0.0, mu_d=0.0, mu_s=0.0)
    depth = 0.002
    pos = np.array([[0.0, 0.0, -depth]])
    quat = _QID[None, :]
    z = np.zeros((1, 3))

    pf_u, pw = point_forces_numpy(sole, params, pos, quat, z, z)
    from biomech.contact.elastic_foundation import reduce_wrench

    cop_uniform = reduce_wrench(pf_u, pw).cop[0, 0]
    pf_s, pw2 = point_forces_numpy(stiff, params, pos, quat, z, z)
    cop_stiff = reduce_wrench(pf_s, pw2).cop[0, 0]
    assert cop_stiff > cop_uniform  # COP shifted toward the stiffer forefoot


def test_stribeck_friction_peaks_at_low_speed():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    params = HydroelasticParams(
        k_bed=5e6, stiffen_b=0.0, hc_alpha=0.0, mu_d=0.4, mu_s=0.9,
        v_stribeck=0.05, v_eps=1e-4,
    )
    depth = 0.002
    pos = np.array([[0, 0, -depth], [0, 0, -depth]], dtype=float)
    quat = np.tile(_QID, (2, 1))
    # slow slide (near static peak) vs fast slide (dynamic mu)
    linvel = np.array([[0.01, 0, 0], [2.0, 0, 0]], dtype=float)
    angvel = np.zeros((2, 3))
    pf, _ = point_forces_numpy(sole, params, pos, quat, linvel, angvel)
    fn = pf[:, :, 2].sum(axis=1)
    ftx = np.abs(pf[:, :, 0].sum(axis=1))
    mu_slow = ftx[0] / fn[0]
    mu_fast = ftx[1] / fn[1]
    assert mu_slow > mu_fast  # static > dynamic
    assert abs(mu_fast - 0.4) < 0.05  # fast -> dynamic mu
    assert mu_slow > 0.7  # slow -> near static mu


def test_warp_matches_numpy():
    _require_warp_cuda()
    sole = sample_ellipsoid_sole((0.12, 0.045, 0.03), n_theta=20, n_phi=10)
    # give it a spatial modulus too
    xs = sole.points[:, 0]
    sole = FootSole(sole.points, sole.normals, sole.areas,
                    modulus=1.0 + (xs - xs.min()))
    params = HydroelasticParams(
        k_bed=4e6, stiffen_b=30.0, hc_alpha=2.0, mu_d=0.5, mu_s=0.8,
        v_stribeck=0.05, v_eps=1e-3,
    )
    rng = np.random.default_rng(5)
    F = 8
    body_pos = rng.normal(scale=0.05, size=(F, 3))
    body_pos[:, 2] -= 0.02
    q = rng.normal(scale=0.1, size=(F, 4))
    q[:, 3] += 1.0
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    linvel = rng.normal(scale=0.3, size=(F, 3))
    angvel = rng.normal(scale=0.3, size=(F, 3))

    pf_np, pw_np = point_forces_numpy(sole, params, body_pos, q, linvel, angvel)
    pf_wp, pw_wp = point_forces_warp(sole, params, body_pos, q, linvel, angvel)
    fscale = max(1.0, np.abs(pf_np).max())
    assert np.abs(pw_np - pw_wp).max() < 1e-4
    assert np.abs(pf_np - pf_wp).max() < 2e-3 * fscale


def test_evaluate_contact_smoke():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    params = HydroelasticParams()
    pos = np.array([[0.0, 0.0, -0.002], [0.0, 0.0, 0.05]])  # loaded, then airborne
    quat = np.tile(_QID, (2, 1))
    z = np.zeros((2, 3))
    pred = evaluate_contact(sole, params, pos, quat, z, z, backend="numpy")
    assert pred.total_normal[0] > 0.0
    assert pred.total_normal[1] == 0.0
    assert np.all(np.isfinite(pred.cop[0]))
    assert np.all(np.isnan(pred.cop[1]))
