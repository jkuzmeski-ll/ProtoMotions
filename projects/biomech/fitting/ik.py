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
#     (``biomech.skeleton.WarpSkeleton``). The finite-difference Jacobian costs two
#     GPU kernel launches *regardless of the number of frames*, so IK over an entire
#     trial's frames is batched essentially for free.
#   * The ENTIRE Levenberg-Marquardt loop now runs on the device: the marker residual,
#     per-frame loss, the normal equations ``H = J^T J`` / ``g = J^T r``, the per-frame
#     damped SPD solve (an in-kernel float64 Cholesky), the trial-pose evaluation, and
#     the accept/reject + damping schedule are all Warp kernels. The giant
#     ``(F, M, 3, ndof)`` Jacobian and the ``(F, ndof, ndof)`` normal equations never
#     leave the GPU; only a single per-iteration "all frames converged" flag (F ints)
#     is copied to the host to decide when to stop.
#   * Nimble's damped-least-squares step solves ``delta = J^T (J J^T + lambda I)^-1 r``.
#     By the push-through identity this equals the cheaper ``ndof x ndof`` normal-equation
#     form ``(J^T J + lambda I)^-1 J^T r`` which is what the device kernels build and
#     solve. We use a standard LM trust-region schedule (per-frame adaptive damping with
#     accept/reject); the pose IK is a well-posed weighted least squares once scales +
#     offsets are fixed, so the recovered ``q`` is set by the minimum, not the path.
#
# Weighting: each marker gets a nonnegative weight; a per-(frame, marker) visibility
# mask handles missing/occluded markers. Rows of both the residual and the Jacobian are
# scaled by ``sqrt(weight) * mask`` (proper weighted least squares; reduces to plain
# masking when all weights are 1). Poses are clamped to the model's joint limits.

"""Batched marker IK on the Warp skeleton (port of Nimble ``fitMarkersToWorldPositions``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from biomech.osim.spec import SkeletonSpec
from biomech.skeleton.skeleton import WarpSkeleton


# ---------------------------------------------------------------------------
# Device kernels for the Levenberg-Marquardt loop (all float64, per-frame batched)
# ---------------------------------------------------------------------------


@wp.kernel
def _residual_kernel(
    markers: wp.array2d(dtype=wp.vec3d),  # (F, M) model marker world positions
    obs: wp.array2d(dtype=wp.vec3d),  # (F, M) observed (NaNs pre-zeroed)
    rw: wp.array2d(dtype=wp.float64),  # (F, M) row weight = sqrt(weight) * mask
    res: wp.array2d(dtype=wp.vec3d),  # (F, M) output weighted residual
):
    f, m = wp.tid()
    w = rw[f, m]
    d = markers[f, m] - obs[f, m]
    res[f, m] = wp.vec3d(d[0] * w, d[1] * w, d[2] * w)


@wp.kernel
def _loss_kernel(
    res: wp.array2d(dtype=wp.vec3d),  # (F, M) weighted residual
    M: wp.int32,
    loss: wp.array(dtype=wp.float64),  # (F,) per-frame squared error
):
    f = wp.tid()
    acc = wp.float64(0.0)
    for m in range(M):
        r = res[f, m]
        acc += r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
    loss[f] = acc


@wp.kernel
def _assemble_H_kernel(
    jac: wp.array4d(dtype=wp.float64),  # (F, M, 3, ndof) unweighted marker Jacobian
    rw: wp.array2d(dtype=wp.float64),  # (F, M) row weight
    M: wp.int32,
    H: wp.array3d(dtype=wp.float64),  # (F, ndof, ndof) output H = Jw^T Jw
):
    f, i, j = wp.tid()
    if j >= i:
        acc = wp.float64(0.0)
        for m in range(M):
            w2 = rw[f, m] * rw[f, m]
            acc += w2 * (
                jac[f, m, 0, i] * jac[f, m, 0, j]
                + jac[f, m, 1, i] * jac[f, m, 1, j]
                + jac[f, m, 2, i] * jac[f, m, 2, j]
            )
        H[f, i, j] = acc
        H[f, j, i] = acc


@wp.kernel
def _assemble_g_kernel(
    jac: wp.array4d(dtype=wp.float64),  # (F, M, 3, ndof)
    rw: wp.array2d(dtype=wp.float64),  # (F, M)
    res: wp.array2d(dtype=wp.vec3d),  # (F, M) weighted residual (already * rw)
    M: wp.int32,
    g: wp.array2d(dtype=wp.float64),  # (F, ndof) output g = Jw^T r
):
    f, i = wp.tid()
    acc = wp.float64(0.0)
    for m in range(M):
        w = rw[f, m]
        r = res[f, m]
        acc += w * (
            jac[f, m, 0, i] * r[0]
            + jac[f, m, 1, i] * r[1]
            + jac[f, m, 2, i] * r[2]
        )
    g[f, i] = acc


@wp.kernel
def _lm_solve_kernel(
    H: wp.array3d(dtype=wp.float64),  # (F, ndof, ndof)
    g: wp.array2d(dtype=wp.float64),  # (F, ndof)
    lam: wp.array(dtype=wp.float64),  # (F,) per-frame damping
    ndof: wp.int32,
    A: wp.array3d(dtype=wp.float64),  # (F, ndof, ndof) scratch (Cholesky factor)
    delta: wp.array2d(dtype=wp.float64),  # (F, ndof) output step
):
    """Per-frame solve of ``(H + lam I) delta = g`` via in-place float64 Cholesky."""
    f = wp.tid()
    lm = lam[f]
    # A = H + lam I
    for i in range(ndof):
        for j in range(ndof):
            A[f, i, j] = H[f, i, j]
        A[f, i, i] = A[f, i, i] + lm
    # Cholesky A = L L^T (lower triangle, in place). A is SPD because lam > 0.
    for i in range(ndof):
        for j in range(i + 1):
            s = A[f, i, j]
            for k in range(j):
                s -= A[f, i, k] * A[f, j, k]
            if i == j:
                if s <= wp.float64(0.0):
                    s = wp.float64(1e-300)
                A[f, i, j] = wp.sqrt(s)
            else:
                A[f, i, j] = s / A[f, j, j]
    # forward solve L y = g  (store y in delta)
    for i in range(ndof):
        s = g[f, i]
        for k in range(i):
            s -= A[f, i, k] * delta[f, k]
        delta[f, i] = s / A[f, i, i]
    # back solve L^T x = y
    for ii in range(ndof):
        i = ndof - 1 - ii
        s = delta[f, i]
        for k in range(i + 1, ndof):
            s -= A[f, k, i] * delta[f, k]
        delta[f, i] = s / A[f, i, i]


@wp.kernel
def _trial_pose_kernel(
    q: wp.array2d(dtype=wp.float64),  # (F, ndof)
    delta: wp.array2d(dtype=wp.float64),  # (F, ndof)
    lo: wp.array(dtype=wp.float64),  # (ndof,)
    hi: wp.array(dtype=wp.float64),  # (ndof,)
    clamp: wp.int32,
    q_try: wp.array2d(dtype=wp.float64),  # (F, ndof) output
):
    f, i = wp.tid()
    v = q[f, i] - delta[f, i]
    if clamp == 1:
        if v < lo[i]:
            v = lo[i]
        if v > hi[i]:
            v = hi[i]
    q_try[f, i] = v


@wp.kernel
def _accept_kernel(
    loss: wp.array(dtype=wp.float64),  # (F,) current loss (updated in place)
    loss_try: wp.array(dtype=wp.float64),  # (F,) trial loss
    q_try: wp.array2d(dtype=wp.float64),  # (F, ndof)
    res_try: wp.array2d(dtype=wp.vec3d),  # (F, M)
    ndof: wp.int32,
    M: wp.int32,
    conv_thresh: wp.float64,
    dmin: wp.float64,
    dmax: wp.float64,
    ddown: wp.float64,
    dup: wp.float64,
    q: wp.array2d(dtype=wp.float64),  # (F, ndof) updated in place
    res: wp.array2d(dtype=wp.vec3d),  # (F, M) updated in place
    lam: wp.array(dtype=wp.float64),  # (F,) updated in place
    converged: wp.array(dtype=wp.int32),  # (F,) output flag
):
    f = wp.tid()
    lcur = loss[f]
    lt = loss_try[f]
    if lt < lcur:
        for i in range(ndof):
            q[f, i] = q_try[f, i]
        for m in range(M):
            res[f, m] = res_try[f, m]
        change = lcur - lt
        loss[f] = lt
        nl = lam[f] * ddown
        if nl < dmin:
            nl = dmin
        lam[f] = nl
        if change < conv_thresh:
            converged[f] = 1
        else:
            converged[f] = 0
    else:
        nl = lam[f] * dup
        if nl > dmax:
            nl = dmax
        lam[f] = nl
        if lam[f] >= dmax:
            converged[f] = 1
        else:
            converged[f] = 0


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
    """Batched Levenberg-Marquardt marker IK over frames (fully device-resident).

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

    q = _as_2d_q(q_init, F, ndof)

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

    dev = skel.device
    G = skel.topo.num_groups
    if group_scales is None:
        gs = np.ones(3 * G, dtype=np.float64)
    else:
        gs = np.asarray(group_scales, dtype=np.float64).ravel()

    # ---- upload everything once; the loop stays on the device ----
    d_q = wp.array(q, dtype=wp.float64, device=dev)
    d_scales = wp.array(gs, dtype=wp.float64, device=dev)
    d_obs = wp.array(obs.reshape(F, M, 3), dtype=wp.vec3d, device=dev)
    d_rw = wp.array(rw, dtype=wp.float64, device=dev)
    d_lo = wp.array(lower, dtype=wp.float64, device=dev)
    d_hi = wp.array(upper, dtype=wp.float64, device=dev)
    d_lam = wp.array(
        np.full(F, cfg.damping_init, dtype=np.float64), dtype=wp.float64, device=dev
    )

    d_res = wp.zeros((F, M), dtype=wp.vec3d, device=dev)
    d_res_try = wp.zeros((F, M), dtype=wp.vec3d, device=dev)
    d_loss = wp.zeros(F, dtype=wp.float64, device=dev)
    d_loss_try = wp.zeros(F, dtype=wp.float64, device=dev)
    d_H = wp.zeros((F, ndof, ndof), dtype=wp.float64, device=dev)
    d_g = wp.zeros((F, ndof), dtype=wp.float64, device=dev)
    d_A = wp.zeros((F, ndof, ndof), dtype=wp.float64, device=dev)
    d_delta = wp.zeros((F, ndof), dtype=wp.float64, device=dev)
    d_q_try = wp.zeros((F, ndof), dtype=wp.float64, device=dev)
    d_converged = wp.zeros(F, dtype=wp.int32, device=dev)

    clamp_flag = 1 if cfg.clamp_to_limits else 0

    def _eval(d_pose, d_res_out, d_loss_out):
        """FK -> weighted residual -> per-frame loss, all on device."""
        _, d_markers = skel._run_wp(d_pose, d_scales)
        wp.launch(
            _residual_kernel,
            dim=(F, M),
            inputs=[d_markers, d_obs, d_rw],
            outputs=[d_res_out],
            device=dev,
        )
        wp.launch(
            _loss_kernel, dim=F, inputs=[d_res_out, M], outputs=[d_loss_out], device=dev
        )

    def _jacobian(d_pose):
        if cfg.jacobian == "fd":
            return skel.marker_jacobian_wrt_q_fd_wp(d_pose, d_scales)
        elif cfg.jacobian == "autodiff":
            jac_host = skel.marker_jacobian_wrt_q(d_pose.numpy(), gs)
            if jac_host.ndim == 3:
                jac_host = jac_host[None]
            return wp.array(jac_host, dtype=wp.float64, device=dev)
        raise ValueError(f"unknown IK Jacobian backend: {cfg.jacobian!r}")

    _eval(d_q, d_res, d_loss)

    iters = 0
    for iters in range(1, cfg.max_iters + 1):
        d_jac = _jacobian(d_q)  # (F, M, 3, ndof) device, unweighted
        wp.launch(
            _assemble_H_kernel,
            dim=(F, ndof, ndof),
            inputs=[d_jac, d_rw, M],
            outputs=[d_H],
            device=dev,
        )
        wp.launch(
            _assemble_g_kernel,
            dim=(F, ndof),
            inputs=[d_jac, d_rw, d_res, M],
            outputs=[d_g],
            device=dev,
        )
        wp.launch(
            _lm_solve_kernel,
            dim=F,
            inputs=[d_H, d_g, d_lam, ndof],
            outputs=[d_A, d_delta],
            device=dev,
        )
        wp.launch(
            _trial_pose_kernel,
            dim=(F, ndof),
            inputs=[d_q, d_delta, d_lo, d_hi, clamp_flag],
            outputs=[d_q_try],
            device=dev,
        )
        _eval(d_q_try, d_res_try, d_loss_try)
        wp.launch(
            _accept_kernel,
            dim=F,
            inputs=[
                d_loss,
                d_loss_try,
                d_q_try,
                d_res_try,
                ndof,
                M,
                wp.float64(cfg.convergence_threshold),
                wp.float64(cfg.damping_min),
                wp.float64(cfg.damping_max),
                wp.float64(cfg.damping_down),
                wp.float64(cfg.damping_up),
            ],
            outputs=[d_q, d_res, d_lam, d_converged],
            device=dev,
        )
        # Only a single tiny (F,) flag is read back to decide the stopping condition.
        if bool(d_converged.numpy().all()):
            break

    q_out = d_q.numpy()
    loss = d_loss.numpy()

    # Per-frame weighted RMS over observed coordinates (meters).
    n_obs = np.maximum((rw > 0).sum(axis=1), 1)  # markers per frame
    marker_rms = np.sqrt(loss / (3.0 * n_obs))
    return MarkerIKResult(q=q_out, marker_rms=marker_rms, iters=iters, final_loss=loss)
