# SPDX-License-Identifier: MIT
#
# Windows-native port of the reusable closed-form numerical kernels in Nimble's
# ``dart/biomechanics/IKInitializer.cpp`` (and the Kabsch solve in
# ``dart/math/Geometry.cpp``). These are the robust, gradient-free building blocks of
# the AddBiomechanics initialization pass that give a good seed for the nonlinear
# marker fit:
#
#   * point_cloud_from_distance_matrix       classical MDS (IKInitializer.cpp:3456)
#   * least_squares_concentric_sphere_fit    sphere fit    (IKInitializer.cpp:3872)
#   * chang_pollard_2006_joint_center        center of rotation, Chang & Pollard 2006
#                                            (IKInitializer.cpp:3598)
#   * gamage_lasenby_2002_axis_fit           axis of rotation, Gamage & Lasenby 2002
#                                            (IKInitializer.cpp:3946)
#   * get_local_scale                        anisotropic segment scale from distances
#                                            (IKInitializer.cpp:3487)
#   * find_cubic_real_roots / center_point_on_axis   (IKInitializer.cpp:3972 / 4010)
#   * point_cloud_to_point_cloud_transform   weighted Kabsch (Geometry.cpp:4651)
#
# All the small dense linear algebra (SVD, generalized eigendecomposition,
# eigendecomposition) is host-side NumPy; there is no Warp equivalent for arbitrary
# small-matrix factorizations. The batched-over-frames work that sits *on top* of
# these kernels (building marker distance matrices, per-frame Kabsch pose recovery)
# is the natural Warp target and is accelerated in the initializer/pose stages.

"""Closed-form joint-center / scale / axis kernels (port of Nimble IKInitializer)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

Vec3 = np.ndarray
Trace = Sequence[np.ndarray]  # a marker trace: (T, 3) array or list of (3,) points


def _as_trace(trace) -> np.ndarray:
    return np.asarray(trace, dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Classical MDS
# ---------------------------------------------------------------------------


def point_cloud_from_distance_matrix(squared_distances: np.ndarray) -> np.ndarray:
    """Classical MDS: recover a 3xN point cloud from a squared-distance matrix.

    Port of ``IKInitializer::getPointCloudFromDistanceMatrix``. ``squared_distances``
    is the ``n x n`` matrix of squared pairwise distances. Returns a ``3 x n`` array
    whose columns are points reproducing those distances (up to a rigid transform).
    """
    D = np.asarray(squared_distances, dtype=np.float64)
    n = D.shape[0]
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D @ J
    # Symmetric eigendecomposition (ascending eigenvalues, like Eigen SelfAdjoint).
    evals, evecs = np.linalg.eigh(B)
    k = 3
    k_evecs = evecs[:, -k:]  # rightCols(3): the 3 largest
    k_evals = np.sqrt(np.maximum(evals[-k:], 1e-16))
    return np.diag(k_evals) @ k_evecs.T


# ---------------------------------------------------------------------------
# Sphere / center-of-rotation fits
# ---------------------------------------------------------------------------


def least_squares_concentric_sphere_fit(
    traces: Sequence[Trace], max_samples: int = 500
) -> Vec3:
    """Least-squares concentric-sphere center (port of ``leastSquaresConcentricSphereFit``).

    Each trace shares the same center but may have its own radius. Robust on
    zero-noise (synthetic) data, biases radius toward zero under ambiguity.
    """
    traces = [_as_trace(t) for t in traces]
    if not traces:
        return np.zeros(3)
    min_len = min(t.shape[0] for t in traces)
    idx = np.arange(min_len)
    if min_len > max_samples:
        idx = _evenly_spaced(min_len, max_samples)

    dim = len(traces) * len(idx)
    f = np.zeros(dim)
    A = np.zeros((dim, 3 + len(traces)))
    row = 0
    for ci, t in enumerate(traces):
        for i in idx:
            p = t[i]
            f[row] = p @ p
            A[row, 0] = 2 * p[0]
            A[row, 1] = 2 * p[1]
            A[row, 2] = 2 * p[2]
            A[row, 3 + ci] = 1.0
            row += 1
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    return c[:3]


def chang_pollard_2006_joint_center(traces: Sequence[Trace]) -> Vec3:
    """Center of rotation via Chang & Pollard 2006 (port of the multi-marker method).

    ``traces`` are marker positions in a frame where the joint center is fixed. Solves
    the constrained generalized eigenproblem (S, C) and picks the min-cost solution;
    falls back to the concentric-sphere fit when there is no valid (noiseless) solution.
    """
    traces = [_as_trace(t) for t in traces]
    SCALE = 50.0
    num = len(traces)
    u_dim = 4 + num

    Ds = []
    for t in traces:
        sp = SCALE * t
        D = np.zeros((sp.shape[0], 4))
        D[:, 0] = np.sum(sp * sp, axis=1)
        D[:, 1:4] = sp
        Ds.append(D)

    total = sum(d.shape[0] for d in Ds)
    Dfull = np.zeros((total, u_dim))
    r = 0
    for i, d in enumerate(Ds):
        Dfull[r : r + d.shape[0], 0:4] = d
        Dfull[r : r + d.shape[0], 4 + i] = 1.0
        r += d.shape[0]

    S = Dfull.T @ Dfull
    C = np.zeros((u_dim, u_dim))
    C[1, 1] = C[2, 2] = C[3, 3] = num
    for i in range(4, u_dim):
        C[i, 0] = -2.0
        C[0, i] = -2.0

    evals, evecs = _generalized_eig(S, C)

    best_cost = np.inf
    center = np.zeros(3)
    for i in range(len(evals)):
        lam = evals[i]
        if abs(lam.imag) > 0 or lam.real <= 1e-12:
            continue
        v = np.real(evecs[:, i])
        constraint = v @ (C @ v)
        if constraint <= 0:
            continue
        scale_by = np.sqrt(1.0 / constraint)
        if not np.isfinite(scale_by):
            continue
        u = v * scale_by
        if np.isnan(u).any():
            continue
        a = u[0]
        if abs(a) < 1e-8:
            continue
        cost = (u @ (S @ u)) / (a * a)
        cand = u[1:4] / (-2.0 * a)
        if cost < best_cost:
            best_cost = cost
            center = cand

    if not np.isfinite(best_cost):
        return least_squares_concentric_sphere_fit(traces)
    return center / SCALE


# ---------------------------------------------------------------------------
# Axis fit
# ---------------------------------------------------------------------------


def gamage_lasenby_2002_axis_fit(traces: Sequence[Trace]) -> tuple[Vec3, float]:
    """Axis of rotation via Gamage & Lasenby 2002 (port of ``gamageLasenby2002AxisFit``).

    Returns ``(axis, condition_number)`` where ``axis`` is the smallest-variance
    direction (the rotation axis) and the condition number flags degeneracy.
    """
    A = np.zeros((3, 3))
    for t in traces:
        t = _as_trace(t)
        mean = t.mean(axis=0)
        mean_outer = (t[:, :, None] * t[:, None, :]).mean(axis=0)
        A += mean_outer - np.outer(mean, mean)
    U, s, Vt = np.linalg.svd(A)
    axis = Vt[2, :]  # V.col(2)
    cond = s[0] / s[2] if s[2] != 0 else np.inf
    return axis, float(cond)


# ---------------------------------------------------------------------------
# Local (segment) scale from pairwise distances
# ---------------------------------------------------------------------------


def get_local_scale(
    local_points: Sequence[Vec3],
    pair_distances_with_weights: Sequence[tuple[int, int, float, float]],
    default_axis_scale: float,
) -> Vec3:
    """Anisotropic per-axis scale for a segment (port of ``getLocalScale``).

    ``pair_distances_with_weights`` are ``(i, j, target_distance, weight)`` between
    ``local_points[i]`` and ``local_points[j]``. Solves the per-axis squared-scale
    least-squares and defaults unreliable axes to ``default_axis_scale``.
    """
    local_points = [np.asarray(p, dtype=np.float64) for p in local_points]
    m = len(pair_distances_with_weights)
    if m == 0:
        return np.full(3, default_axis_scale)
    if m == 1:
        i, j, d, _w = pair_distances_with_weights[0]
        a, b = local_points[i], local_points[j]
        ratio = d / np.linalg.norm(a - b)
        if (
            np.isnan(ratio)
            or ratio < 0.75 * default_axis_scale
            or ratio > 1.25 * default_axis_scale
        ):
            ratio = default_axis_scale
        return np.full(3, ratio)

    A = np.zeros((m, 3))
    dist = np.zeros(m)
    for k, (i, j, d, w) in enumerate(pair_distances_with_weights):
        a, b = local_points[i], local_points[j]
        A[k, 0] = w * (a[0] - b[0]) ** 2
        A[k, 1] = w * (a[1] - b[1]) ** 2
        A[k, 2] = w * (a[2] - b[2]) ** 2
        dist[k] = w * d * d

    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    # svd.solve(distances) via the thin SVD pseudo-inverse
    scale_squared = Vt.T @ (np.divide(U.T @ dist, sv, out=np.zeros_like(sv), where=sv != 0))
    scale = np.sqrt(np.abs(scale_squared))

    output_sensitivity = Vt.T @ sv  # matrixV() * singularValues()
    for i in range(3):
        s_i = output_sensitivity[i]
        if abs(s_i) < 0.002 or abs(s_i) > 100:
            scale[i] = default_axis_scale
        elif (
            np.isnan(scale[i])
            or scale[i] < 0.75 * default_axis_scale
            or scale[i] > 1.25 * default_axis_scale
        ):
            scale[i] = default_axis_scale
    return scale


# ---------------------------------------------------------------------------
# Cubic roots + point-on-axis refinement
# ---------------------------------------------------------------------------


def find_cubic_real_roots(a: float, b: float, c: float, d: float) -> list[float]:
    """Real roots of ``a x^3 + b x^2 + c x + d`` (port of Nimble's ``SolveP3``).

    Faithful Cardano solver (``IKInitializer.cpp:221``); unlike a companion-matrix
    eigensolver it handles the degenerate triple-root case exactly, matching Nimble.
    """
    if a == 0.0:
        return [float(r.real) for r in np.roots([b, c, d]) if abs(r.imag) < 1e-9]
    return _solve_p3(b / a, c / a, d / a)


def _root3(x: float) -> float:
    return np.cbrt(x)


def _solve_p3(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``x^3 + a x^2 + b x + c = 0`` (Cardano; port of ``SolveP3``)."""
    eps = 1e-14
    two_pi = 2.0 * np.pi
    a2 = a * a
    q = (a2 - 3.0 * b) / 9.0
    r = (a * (2.0 * a2 - 9.0 * b) + 27.0 * c) / 54.0
    if abs(q) < eps:
        if abs(r) < eps:
            return [-a / 3.0]  # three identical roots
        x0 = _root3(-r / 2.0)
        return [x0]
    r2 = r * r
    q3 = q * q * q
    if r2 <= (q3 + eps):
        t = r / np.sqrt(q3)
        t = max(-1.0, min(1.0, t))
        t = np.arccos(t)
        a_over_3 = a / 3.0
        qq = -2.0 * np.sqrt(q)
        return [
            qq * np.cos(t / 3.0) - a_over_3,
            qq * np.cos((t + two_pi) / 3.0) - a_over_3,
            qq * np.cos((t - two_pi) / 3.0) - a_over_3,
        ]
    A = -_root3(abs(r) + np.sqrt(r2 - q3))
    if r < 0:
        A = -A
    B = 0.0 if A == 0.0 else q / A
    a_over_3 = a / 3.0
    x0 = (A + B) - a_over_3
    x2_imag = 0.5 * np.sqrt(3.0) * (A - B)
    if abs(x2_imag) < eps:
        # one real root plus a repeated real root
        return [x0, -0.5 * (A + B) - a_over_3]
    return [x0]


def center_point_on_axis(
    center: Vec3,
    axis: Vec3,
    points_and_radii: Sequence[tuple[Vec3, float]],
    weights: Sequence[float] | None = None,
) -> Vec3:
    """Slide ``center`` along ``axis`` to best fit the (point, radius) constraints.

    Port of ``centerPointOnAxis``: minimizes
    ``sum_i w_i (||f(x) - p_i||^2 - r_i^2)^2`` with ``f(x) = center + x*axis`` by
    solving the resulting cubic and picking the lowest-loss real root.
    """
    center = np.asarray(center, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if weights is None:
        weights = [1.0] * len(points_and_radii)

    e = axis @ axis
    a = b = c = d = 0.0
    for i, (p, radius) in enumerate(points_and_radii):
        w = weights[i] if i < len(weights) else 1.0
        p = np.asarray(p, dtype=np.float64)
        f = (center - p) @ axis
        g = (center - p) @ (center - p) - radius * radius
        a += w * 4.0 * e * e
        b += w * 12.0 * e * f
        c += w * (4.0 * e * g + 8.0 * f * f)
        d += w * 4.0 * f * g

    roots = find_cubic_real_roots(a, b, c, d)
    if not roots:
        return center

    best_root = roots[0]
    best_loss = np.inf
    for root in roots:
        resulting = center + root * axis
        loss = 0.0
        for i, (p, radius) in enumerate(points_and_radii):
            w = weights[i] if i < len(weights) else 1.0
            p = np.asarray(p, dtype=np.float64)
            err = (p - resulting) @ (p - resulting) - radius * radius
            loss += w * err * err
        if loss < best_loss:
            best_loss = loss
            best_root = root
    return center + best_root * axis


# ---------------------------------------------------------------------------
# Weighted Kabsch (rigid point-cloud alignment)
# ---------------------------------------------------------------------------


def point_cloud_to_point_cloud_transform(
    local_points: Sequence[Vec3],
    world_points: Sequence[Vec3],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted rigid transform ``T`` with ``T * local ≈ world`` (port of the Kabsch
    solve in ``math::getPointCloudToPointCloudTransform``). Returns a ``4x4`` matrix.
    """
    L = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
    W = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    n = L.shape[0]
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
    sw = w.sum()

    local_c = (L * w[:, None]).sum(axis=0) / sw
    world_c = (W * w[:, None]).sum(axis=0) / sw
    Lc = L - local_c
    Wc = W - world_c

    cov = (w[:, None] * Wc).T @ Lc  # sum w * world * local^T
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        s = np.eye(3)
        s[2, 2] = -1.0
        R = U @ s @ Vt
    t = world_c - R @ local_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Point-cloud alignment used by the MDS joint-center solver
# ---------------------------------------------------------------------------


def map_point_cloud_to_data(
    point_cloud: np.ndarray, first_n_points: Sequence[Vec3]
) -> np.ndarray:
    """Rigidly map an MDS point cloud onto observed data (port of ``mapPointCloudToData``).

    ``point_cloud`` is ``3 x N`` (from :func:`point_cloud_from_distance_matrix`) whose
    first ``len(first_n_points)`` columns correspond to the observed points
    ``first_n_points``. Finds the rigid transform aligning those leading columns to the
    observations and applies it to the *whole* cloud (so the trailing columns -- e.g. an
    unknown joint center -- are placed consistently). Returns the transformed ``3 x N``.

    NOTE: unlike a standard Kabsch solve, this deliberately does **not** enforce a
    proper (det=+1) rotation. Classical MDS can reconstruct a *reflected* copy of the
    data, and forcing a right-handed rotation here would corrupt the fit; Nimble skips
    the determinant fix for exactly this reason. Any leftover chirality ambiguity in
    the trailing (joint-center) column is resolved separately by
    :func:`ensure_on_same_side_of_plane`.
    """
    cloud = np.asarray(point_cloud, dtype=np.float64)
    target = np.asarray(first_n_points, dtype=np.float64).reshape(-1, 3).T  # 3 x k
    k = target.shape[1]
    source = cloud[:, :k]

    src_c = source.mean(axis=1, keepdims=True)
    tgt_c = target.mean(axis=1, keepdims=True)
    src0 = source - src_c
    tgt0 = target - tgt_c

    cov = tgt0 @ src0.T  # 3x3
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt  # no determinant fix (reflections allowed), matching Nimble
    return R @ (cloud - src_c) + tgt_c


def is_coplanar(points: Sequence[Vec3], threshold: float = 1e-3) -> bool:
    """True if all points lie (near) a common plane (port of ``isCoplanar``).

    Fewer than 4 points are always considered coplanar. Uses the plane through the
    first three points and flags non-coplanarity if any other point is farther than
    ``threshold`` from it.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < 4:
        return True
    normal = np.cross(P[1] - P[0], P[2] - P[0])
    nn = np.linalg.norm(normal)
    if nn < 1e-15:
        return True
    normal = normal / nn
    dots = (P[3:] - P[0]) @ normal
    return bool(np.all(np.abs(dots) <= threshold))


def ensure_on_same_side_of_plane(
    neutral_points: Sequence[Vec3],
    neutral_goal: Vec3,
    actual_points: Sequence[Vec3],
    ambiguous_reconstruction: Vec3,
) -> Vec3:
    """Reflect a reconstructed point to the correct side of a support plane.

    Port of ``ensureOnSameSideOfPlane``. When the support markers are (near) coplanar,
    MDS leaves the joint center ambiguous between the two sides of that plane. This
    picks the side matching the neutral (rest-pose) model geometry, reflecting the
    reconstruction across the plane of the ``actual_points`` if needed.
    """
    Np = np.asarray(neutral_points, dtype=np.float64).reshape(-1, 3)
    Ap = np.asarray(actual_points, dtype=np.float64).reshape(-1, 3)
    rec = np.asarray(ambiguous_reconstruction, dtype=np.float64)
    if Np.shape[0] < 3 or Ap.shape[0] < 3:
        return rec
    neutral_normal = np.cross(Np[1] - Np[0], Np[2] - Np[0])
    neutral_normal = neutral_normal / (np.linalg.norm(neutral_normal) + 1e-300)
    neutral_dist = float((np.asarray(neutral_goal) - Np[0]) @ neutral_normal)

    actual_normal = np.cross(Ap[1] - Ap[0], Ap[2] - Ap[0])
    actual_normal = actual_normal / (np.linalg.norm(actual_normal) + 1e-300)
    rec_dist = float((rec - Ap[0]) @ actual_normal)

    if neutral_dist * rec_dist < 0:
        return rec - 2.0 * rec_dist * actual_normal
    return rec


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------



def _evenly_spaced(n: int, k: int) -> np.ndarray:
    """Port of ``math::evenlySpacedTimesteps``: ``k`` indices spanning ``[0, n)``."""
    if k >= n:
        return np.arange(n)
    return np.floor(np.linspace(0, n - 1, k)).astype(int)


def _generalized_eig(S: np.ndarray, C: np.ndarray):
    """Generalized eigenproblem ``S v = lambda C v`` (C symmetric but indefinite).

    Mirrors Eigen's ``GeneralizedEigenSolver`` result (possibly complex). We use
    ``scipy.linalg.eig`` when available, else reduce via a pseudo-inverse of C.
    """
    try:
        from scipy.linalg import eig as _scipy_eig

        evals, evecs = _scipy_eig(S, C)
        return evals, evecs
    except Exception:
        # Fallback: solve C^{-1} S v = lambda v. C is indefinite but generally
        # invertible for this construction.
        M = np.linalg.solve(C, S)
        evals, evecs = np.linalg.eig(M)
        return evals, evecs
