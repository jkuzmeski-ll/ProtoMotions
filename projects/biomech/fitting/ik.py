# SPDX-License-Identifier: MIT
#
# Batched, Warp-driven per-frame marker IK. This is the per-frame pose solver used by
# both the closed-form initializer (``estimatePosesClosedForm``) and the bilevel marker
# fit (``MarkerFitter``). It is a Windows-native re-expression of Nimble's
# ``Skeleton::fitMarkersToWorldPositions`` + ``math::solveIK`` / ``refineIK``
# (``dart/dynamics/Skeleton.cpp:7890`` and ``dart/math/IKSolver.cpp``).
#
# Design (why this shape):
#   * The expensive quantities -- marker forward kinematics and the marker Jacobian
#     ``d(marker world pos)/dq`` -- are computed by the Warp skeleton
#     (``biomech.skeleton.WarpSkeleton``). The Warp autodiff Jacobian costs ``3*M``
#     backward passes *regardless of the number of frames*, so IK over an entire
#     trial's frames is batched essentially for free: one FK + one Jacobian sweep per
#     iteration covers all frames at once.
#   * Nimble's damped-least-squares step solves ``delta = J^T (J J^T + lambda I)^-1 r``
#     because for a single frame ``J`` has more rows (3*M) than columns (ndof). By the
#     push-through identity ``J^T (J J^T + lambda I)^-1 = (J^T J + lambda I)^-1 J^T``,
#     this is *identical* to the cheaper ``ndof x ndof`` normal-equation form, which is
#     what we solve here (batched with ``np.linalg.solve`` over frames).
#   * We use a standard Levenberg-Marquardt trust-region schedule (per-frame adaptive
#     damping with accept/reject + line-search revert) rather than Nimble's exact
#     lr/transpose bookkeeping. The pose IK is a well-posed weighted least squares once
#     scales+offsets are fixed, so the recovered ``q`` is set by the minimum, not the
#     descent path; we validate by round-trip recovery to machine precision.
#
# Weighting: each marker gets a nonnegative weight; a per-(frame, marker) visibility
# mask handles missing/occluded markers. Rows of both the residual and the Jacobian are
# scaled by ``sqrt(weight) * mask`` (proper weighted least squares; reduces to plain
# masking when all weights are 1). Poses are clamped to the model's joint limits.

"""Batched marker IK on the Warp skeleton (port of Nimble ``fitMarkersToWorldPositions``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biomech.osim.spec import SkeletonSpec
from biomech.skeleton.skeleton import WarpSkeleton


def _as_2d_q(q: np.ndarray, F: int, ndof: int) -> np.ndarray:
    q2 = np.atleast_2d(np.asarray(q, dtype=np.float64)).copy()
    if q2.shape[0] == 1 and F > 1:
        q2 = np.repeat(q2, F, axis=0)
    assert q2.shape == (F, ndof), f"q_init must be ({F},{ndof}), got {q2.shape}"
    return q2




@dataclass
class MarkerIKConfig:
    """Solver settings for :func:`solve_marker_ik` (mirrors Nimble ``IKConfig``)."""

    max_iters: int = 100
    convergence_threshold: float = 1e-10  # stop when squared-error change < this
    damping_init: float = 1e-3  # initial LM damping lambda (per frame)
    damping_min: float = 1e-9
    damping_max: float = 1e9
    damping_down: float = 0.5  # multiply lambda by this on an accepted step
    damping_up: float = 4.0  # multiply lambda by this on a rejected step
    clamp_to_limits: bool = True
    jacobian: str = "fd"  # "fd" = fast GPU central difference, "autodiff" = exact Warp AD


def position_limits(spec: SkeletonSpec) -> tuple[np.ndarray, np.ndarray]:
    """Per-DOF ``(lower, upper)`` position limits in DOF order.

    Locked coordinates collapse to their default value; unclamped coordinates get
    +/-inf so they are effectively free. Matches DART's position limit semantics
    (``CoordinateSpec.limit_lo/hi`` already collapses locked DOFs).
    """
    lo: list[float] = []
    hi: list[float] = []
    for joint in spec.joints:
        for c in joint.coordinates:
            if c.locked:
                lo.append(c.default_value)
                hi.append(c.default_value)
            elif c.clamped:
                lo.append(c.range_lo)
                hi.append(c.range_hi)
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _row_weights(
    weights: np.ndarray | None, mask: np.ndarray | None, F: int, M: int
) -> np.ndarray:
    """Build per-(frame, marker) row weights ``sqrt(weight) * mask``, shape ``(F, M)``."""
    if weights is None:
        w = np.ones(M, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        assert w.shape == (M,), f"weights must be ({M},), got {w.shape}"
    w = np.sqrt(np.maximum(w, 0.0))
    rw = np.broadcast_to(w, (F, M)).astype(np.float64).copy()
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        if m.ndim == 1:
            m = np.broadcast_to(m, (F, M))
        assert m.shape == (F, M), f"mask must be ({F},{M}) or ({M},), got {m.shape}"
        rw = rw * m
    return rw


@dataclass
class MarkerIKResult:
    q: np.ndarray  # (F, ndof) recovered poses
    marker_rms: np.ndarray  # (F,) per-frame weighted RMS marker error (meters)
    iters: int  # LM iterations actually run
    final_loss: np.ndarray  # (F,) final weighted squared marker error


def solve_marker_ik(
    skel: WarpSkeleton,
    observed_markers: np.ndarray,
    q_init: np.ndarray,
    group_scales: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    config: MarkerIKConfig | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> MarkerIKResult:
    """Batched Levenberg-Marquardt marker IK over frames.

    Parameters
    ----------
    skel : WarpSkeleton
        The differentiable OpenSim skeleton (fixed ``group_scales`` here).
    observed_markers : (F, M, 3) or (M, 3)
        Target marker world positions in the model's native frame (OpenSim Y-up
        meters, same frame as :meth:`WarpSkeleton.forward`). NaNs are treated as
        missing (their rows are masked out automatically).
    q_init : (F, ndof) or (ndof,)
        Initial poses (e.g. from the closed-form initializer).
    group_scales : (3*G,), optional
        Fixed segment scales; defaults to unit scale.
    weights : (M,), optional
        Per-marker nonnegative weights (default all 1).
    mask : (F, M) or (M,), optional
        Per-(frame, marker) visibility (1 = observed, 0 = missing). Combined with
        automatic NaN detection.
    config : MarkerIKConfig, optional
    lower, upper : (ndof,), optional
        Joint position limits; default derived from the parsed model via
        :func:`position_limits`.

    Returns
    -------
    MarkerIKResult
    """
    cfg = config or MarkerIKConfig()
    obs = np.asarray(observed_markers, dtype=np.float64)
    if obs.ndim == 2:
        obs = obs[None]
    F, M, _ = obs.shape
    ndof = skel.topo.num_dofs
    assert M == skel.topo.num_markers, (
        f"observed markers M={M} != model markers {skel.topo.num_markers}"
    )

    q = np.atleast_2d(np.asarray(q_init, dtype=np.float64)).copy()
    if q.shape[0] == 1 and F > 1:
        q = np.repeat(q, F, axis=0)
    assert q.shape == (F, ndof), f"q_init must be ({F},{ndof}), got {q.shape}"

    if lower is None or upper is None:
        lo_d, hi_d = position_limits(skel.spec)
        lower = lo_d if lower is None else lower
        upper = hi_d if upper is None else upper
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    # Missing markers: explicit mask AND-ed with finite-observation detection.
    finite = np.asarray(np.isfinite(obs).all(axis=2), dtype=bool)  # (F, M)
    rw = _row_weights(weights, mask, F, M)  # (F, M)
    rw = rw * finite
    # Replace NaNs so they never poison FK-independent arithmetic (rows are zeroed
    # by rw anyway).
    obs = np.where(finite[..., None], obs, 0.0)

    if cfg.clamp_to_limits:
        q = np.clip(q, lower, upper)

    lam = np.full(F, cfg.damping_init, dtype=np.float64)
    eye = np.eye(ndof, dtype=np.float64)

    def _loss(qc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(residual (F,M,3), per-frame squared loss (F,))`` at poses ``qc``."""
        _, mk = skel.forward(qc, group_scales)  # (F, M, 3)
        res = (mk - obs) * rw[..., None]
        loss = (res**2).reshape(F, -1).sum(axis=1)
        return res, loss

    res, loss = _loss(q)
    iters = 0
    for iters in range(1, cfg.max_iters + 1):
        # Jacobian of marker positions wrt q (F, M, 3, ndof), weighted row-wise.
        # The finite-difference path is a two-kernel GPU implementation that avoids the
        # legacy 3*M sequential Warp-autodiff backward passes. It is the default because
        # it agrees with autodiff to numerical precision for this IK use case and is the
        # difference between seconds and minutes on real windows.
        if cfg.jacobian == "fd":
            jac = skel.marker_jacobian_wrt_q_fd(q, group_scales)
        elif cfg.jacobian == "autodiff":
            jac = skel.marker_jacobian_wrt_q(q, group_scales)
        else:
            raise ValueError(f"unknown IK Jacobian backend: {cfg.jacobian!r}")
        if jac.ndim == 3:  # single frame -> add batch dim
            jac = jac[None]
        jac = jac * rw[:, :, None, None]  # scale rows by sqrt(weight)*mask

        J = jac.reshape(F, M * 3, ndof)
        r = res.reshape(F, M * 3)
        # Normal equations: H = J^T J, g = J^T r.  (identical to Nimble's DLS form)
        H = np.einsum("fri,frj->fij", J, J)
        g = np.einsum("fri,fr->fi", J, r)

        # Per-frame LM step with accept/reject on the damping.
        # Vectorized solve, then evaluate the trial poses in one batched FK.
        A = H + lam[:, None, None] * eye[None]
        try:
            delta = np.linalg.solve(A, g[..., None])[..., 0]  # (F, ndof)
        except np.linalg.LinAlgError:
            delta = np.stack(
                [np.linalg.lstsq(A[f], g[f], rcond=None)[0] for f in range(F)]
            )
        q_try = q - delta
        if cfg.clamp_to_limits:
            q_try = np.clip(q_try, lower, upper)
        res_try, loss_try = _loss(q_try)

        accept = loss_try < loss
        # Accepted frames: take the step, decrease damping.
        q[accept] = q_try[accept]
        res[accept] = res_try[accept]
        loss_change = loss - loss_try
        loss[accept] = loss_try[accept]
        lam[accept] = np.maximum(lam[accept] * cfg.damping_down, cfg.damping_min)
        # Rejected frames: keep pose, increase damping (smaller, safer step next).
        lam[~accept] = np.minimum(lam[~accept] * cfg.damping_up, cfg.damping_max)

        # Convergence: every frame either converged (tiny accepted improvement) or is
        # stuck at max damping (cannot make progress).
        converged = (accept & (loss_change < cfg.convergence_threshold)) | (
            ~accept & (lam >= cfg.damping_max)
        )
        if converged.all():
            break

    # Per-frame weighted RMS over observed coordinates (meters).
    n_obs = np.maximum(rw.astype(bool).sum(axis=1), 1)  # markers per frame
    marker_rms = np.sqrt(loss / (3.0 * n_obs))
    return MarkerIKResult(
        q=q, marker_rms=marker_rms, iters=iters, final_loss=loss
    )


