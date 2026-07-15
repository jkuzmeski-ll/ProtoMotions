# SPDX-License-Identifier: MIT

"""Tests for the closed-form IKInitializer kernels (biomech.fitting.closed_form), M2c.

These kernels are mathematically exact on ideal data, so we validate them against
synthetic analytic ground truth (points with known pairwise distances, markers
rotating about a known center/axis, a known rigid transform, known segment scales)
rather than against Nimble goldens. A one-time Nimble golden run for the full S001
pipeline is a later parity check (needs the S001 marker->model map).
"""

from __future__ import annotations

import numpy as np

from biomech.fitting import closed_form as cf
from biomech.tests import SkipTest


def _rot(axis, angle):
    """Rodrigues rotation matrix (no external deps)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rand_rot(rng):
    v = rng.normal(size=3)
    ang = rng.uniform(0, 2 * np.pi)
    return _rot(v, ang)


def test_mds_reconstructs_pairwise_distances():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(3, 7))
    D = np.array([[np.sum((P[:, i] - P[:, j]) ** 2) for j in range(7)] for i in range(7)])
    X = cf.point_cloud_from_distance_matrix(D)
    D2 = np.array([[np.sum((X[:, i] - X[:, j]) ** 2) for j in range(7)] for i in range(7)])
    assert np.abs(D - D2).max() < 1e-9


def test_kabsch_recovers_rigid_transform():
    rng = np.random.default_rng(1)
    R = _rand_rot(rng)
    t = rng.normal(size=3)
    L = rng.normal(size=(8, 3))
    W = (R @ L.T).T + t
    T = cf.point_cloud_to_point_cloud_transform(L, W)
    assert np.abs(T[:3, :3] - R).max() < 1e-9
    assert np.abs(T[:3, 3] - t).max() < 1e-9
    # applying T to local reproduces world
    Wp = (T[:3, :3] @ L.T).T + T[:3, 3]
    assert np.abs(Wp - W).max() < 1e-9


def test_gamage_lasenby_recovers_rotation_axis():
    rng = np.random.default_rng(2)
    axis = np.array([0.2, 0.7, -0.5])
    axis /= np.linalg.norm(axis)
    traces = []
    for _ in range(4):
        base = rng.normal(size=3)
        tr = [(_rot(axis, th) @ base) for th in np.linspace(0, 2 * np.pi, 60)]
        traces.append(np.array(tr))
    est, cond = cf.gamage_lasenby_2002_axis_fit(traces)
    err = min(np.abs(est - axis).max(), np.abs(-est - axis).max())
    assert err < 1e-6, err
    assert cond > 100  # pure rotation -> degenerate along the axis


def test_least_squares_sphere_fit_zero_noise():
    rng = np.random.default_rng(3)
    center = np.array([0.5, -0.3, 0.2])
    traces = []
    for m in range(3):
        base = rng.normal(size=3)
        base = base / np.linalg.norm(base) * (0.3 + 0.1 * m)
        tr = [center + _rand_rot(rng) @ base for _ in range(40)]
        traces.append(np.array(tr))
    est = cf.least_squares_concentric_sphere_fit(traces)
    assert np.abs(est - center).max() < 1e-9


def test_chang_pollard_recovers_center_with_noise():
    try:
        import scipy.linalg  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"scipy required for generalized eig: {exc}")
    rng = np.random.default_rng(4)
    center = np.array([0.5, -0.3, 0.2])
    traces = []
    for m in range(3):
        base = rng.normal(size=3)
        base = base / np.linalg.norm(base) * (0.3 + 0.1 * m)
        tr = [
            center + _rand_rot(rng) @ base + rng.normal(scale=1e-3, size=3)
            for _ in range(80)
        ]
        traces.append(np.array(tr))
    est = cf.chang_pollard_2006_joint_center(traces)
    assert np.abs(est - center).max() < 5e-3, np.abs(est - center).max()


def test_get_local_scale_recovers_anisotropic_scale():
    rng = np.random.default_rng(5)
    lp = [rng.normal(size=3) for _ in range(4)]
    true_scale = np.array([1.1, 0.9, 1.05])
    pairs = []
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        d = np.linalg.norm(true_scale * (lp[i] - lp[j]))
        pairs.append((i, j, d, 1.0))
    s = cf.get_local_scale(lp, pairs, 1.0)
    assert np.abs(s - true_scale).max() < 1e-6, s


def test_get_local_scale_defaults_without_observations():
    s = cf.get_local_scale([np.zeros(3)], [], default_axis_scale=1.2)
    assert np.allclose(s, 1.2)


def test_find_cubic_real_roots():
    # (x-1)(x-2)(x+3) = x^3 - 7x + 6  -> roots {1, 2, -3}
    roots = sorted(cf.find_cubic_real_roots(1.0, 0.0, -7.0, 6.0))
    assert np.allclose(roots, [-3.0, 1.0, 2.0], atol=1e-9)


def test_center_point_on_axis_slides_to_optimum():
    center = np.zeros(3)
    axis = np.array([1.0, 0.0, 0.0])
    # target center at x=0.7 keeps all three points at their stated radii
    pr = [
        (np.array([0.7, 0.5, 0.0]), 0.5),
        (np.array([0.7, -0.5, 0.0]), 0.5),
        (np.array([0.7, 0.0, 0.3]), 0.3),
    ]
    res = cf.center_point_on_axis(center, axis, pr)
    assert np.abs(res - np.array([0.7, 0.0, 0.0])).max() < 1e-6, res
