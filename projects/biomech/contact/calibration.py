# SPDX-License-Identifier: MIT
#
# Milestone M6 (contact rung 2) — calibrate the elastic-foundation contact parameters
# (k_bed, c_bed, mu) against measured 6-axis GRF/COP under prescribed (gold-standard)
# kinematics.
#
# Physical identifiability: for a given foot trajectory + sole geometry the *net* ground
# reaction is
#     Fz = Σ area·(k·d + c·ṅ)              (linear in k, c),
#     F_xy ≈ -mu · Σ fn · v̂_t            (linear in mu given fn),
# so the three parameters are well determined by the measured *force*. The centre of
# pressure is fixed mainly by the kinematics + plantar geometry (the normal-force
# weighting is nearly scale-invariant in k), so COP is treated here as a *diagnostic*
# of the reconstruction, not a calibration target. That keeps the fit well-posed.
#
# "Use Newton as much as possible": every residual/Jacobian evaluation runs the whole
# trajectory through the Warp contact kernel (``evaluate_contact(..., backend="warp")``),
# so a Levenberg–Marquardt loop with a 3-column finite-difference Jacobian costs four
# batched GPU launches per iteration. Parameters are optimized in log-space so they stay
# strictly positive without constraints.
#
# World Z-up; inputs are the same per-frame foot pose + spatial velocity used by
# ``biomech.contact.elastic_foundation`` / ``biomech.contact.kinematics`` and the
# measured signals are the ``grf``/``cop_world`` fields of
# ``biomech.io.force_plate.ForcePlate``.

"""Calibrate elastic-foundation contact params against measured GRF/COP (M6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from biomech.contact.elastic_foundation import (
    ContactPrediction,
    ElasticFoundationParams,
    FootSole,
    evaluate_contact,
)


# ---------------------------------------------------------------------------
# Targets / config / result
# ---------------------------------------------------------------------------


@dataclass
class CalibrationTarget:
    """Measured ground reaction to fit (world frame, SI).

    ``grf`` is the force ON the subject ``(F, 3)`` (``ForcePlate.grf``); ``cop`` is the
    measured centre of pressure ``(F, 3)`` (``ForcePlate.cop_world``, NaN in swing) and
    is optional (used only for diagnostics).
    """

    grf: np.ndarray  # (F, 3)
    cop: Optional[np.ndarray] = None  # (F, 3), NaN in swing


@dataclass
class CalibrationConfig:
    max_iters: int = 40
    fz_threshold: float = 20.0  # N; frames above this count as "loaded" (measured)
    horizontal_weight: float = 1.0  # relative weight of Fx/Fy vs Fz residuals
    # Objective: "perframe" matches instantaneous GRF every frame (sensitive to
    # per-frame kinematic noise / ground registration); "aggregate" matches
    # per-stance-phase mean + peak GRF (robust to unbiased per-frame noise), which is
    # the recommended mode for dynamic (walk/run) windows.
    objective: str = "perframe"
    agg_bins: int = 3  # sub-bins per stance segment for the "aggregate" objective
    fd_eps: float = 1e-4  # finite-difference step in log-parameter space
    lm_lambda0: float = 1e-3
    lm_lambda_up: float = 3.0
    lm_lambda_down: float = 0.5
    tol_cost: float = 1e-9  # stop when relative cost improvement < tol
    k_floor: float = 1.0e3
    c_floor: float = 0.0
    mu_floor: float = 1.0e-3
    verbose: bool = False


@dataclass
class CalibrationResult:
    params: object  # ElasticFoundationParams or HydroelasticParams (calibrated)
    cost_history: List[float] = field(default_factory=list)
    force_rms: float = 0.0  # N, over all frames
    vertical_rms: float = 0.0  # N, Fz over all frames
    cop_rms: Optional[float] = None  # m, over commonly-loaded frames (diagnostic)
    n_iters: int = 0
    prediction: Optional[ContactPrediction] = None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _force_scale(target: CalibrationTarget, cfg: CalibrationConfig) -> float:
    """A robust force normalizer (~body weight) so residuals are O(1)."""
    fz = np.abs(target.grf[:, 2])
    peak = float(np.nanmax(fz)) if fz.size else 1.0
    return max(peak, cfg.fz_threshold, 1.0)


def _predict(
    sole: FootSole,
    theta: np.ndarray,
    v_eps: float,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    ground_z: float,
    backend: str,
    device: str,
    keep_points: bool = False,
) -> ContactPrediction:
    params = ElasticFoundationParams(
        k_bed=float(theta[0]), c_bed=float(theta[1]), mu=float(theta[2]), v_eps=v_eps
    )
    return evaluate_contact(
        sole, params, body_pos, body_quat, body_linvel, body_angvel,
        ground_z=ground_z, backend=backend, device=device, keep_points=keep_points,
    )


def _residual(pred: ContactPrediction, target: CalibrationTarget,
              fscale: float, w_h: float) -> np.ndarray:
    """Weighted force residual over all frames: (3F,)."""
    diff = (pred.grf - target.grf) / fscale
    diff[:, 0] *= w_h
    diff[:, 1] *= w_h
    return diff.ravel()


def _residual_aggregate(pred: ContactPrediction, target: CalibrationTarget,
                        fscale: float, w_h: float,
                        segs: Sequence[tuple], n_bins: int = 3) -> np.ndarray:
    """Sub-bin-averaged per-stance-phase GRF residual (robust to per-frame noise).

    Each contiguous stance segment is split into up to ``n_bins`` equal time bins; we
    match the *mean* force in each bin (vertical, plus horizontal when weighted). A bin
    mean averages out unbiased per-frame kinematic noise (so it is far less sensitive to
    reconstruction jitter than a per-frame fit), while several bins per segment still
    capture the loading/unloading shape needed to separate stiffness from dissipation.
    Unlike a peak/max feature, a mean is not biased upward by noise.
    """
    res: List[float] = []
    for a, b in segs:
        n = b - a
        if n <= 0:
            continue
        nb = int(min(max(1, n_bins), n))
        edges = np.linspace(a, b, nb + 1).astype(int)
        for i in range(nb):
            s, e = int(edges[i]), int(edges[i + 1])
            if e <= s:
                continue
            gp = pred.grf[s:e]
            gm = target.grf[s:e]
            res.append((gp[:, 2].mean() - gm[:, 2].mean()) / fscale)
            if w_h > 0.0:
                res.append(w_h * (gp[:, 0].mean() - gm[:, 0].mean()) / fscale)
                res.append(w_h * (gp[:, 1].mean() - gm[:, 1].mean()) / fscale)
    return np.asarray(res, dtype=np.float64)


def _make_residual_fn(cfg: CalibrationConfig, target: CalibrationTarget, fscale: float):
    """Build the residual functional ``fn(pred) -> vec`` for the configured objective."""
    w_h = float(cfg.horizontal_weight)
    if cfg.objective == "perframe":
        return lambda pred: _residual(pred, target, fscale, w_h)
    if cfg.objective == "aggregate":
        from biomech.contact.stance import segment_contacts
        segs = segment_contacts(target.grf[:, 2], cfg.fz_threshold)
        if not segs:
            segs = [(0, target.grf.shape[0])]
        nb = cfg.agg_bins
        return lambda pred: _residual_aggregate(pred, target, fscale, w_h, segs, nb)
    raise ValueError(f"unknown calibration objective {cfg.objective!r}")


def calibrate_elastic_foundation(
    sole: FootSole,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    target: CalibrationTarget,
    init_params: Optional[ElasticFoundationParams] = None,
    ground_z: float = 0.0,
    backend: str = "numpy",
    device: str = "cuda",
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    """Fit ``(k_bed, c_bed, mu)`` to a measured GRF trajectory (Levenberg–Marquardt).

    Args:
        sole: plantar bed (foot frame).
        body_pos/quat/linvel/angvel: ``(F, ...)`` foot pose + spatial velocity (world,
            xyzw) — e.g. from ``biomech.contact.kinematics.foot_trajectory_from_motion``.
        target: measured GRF (and optional COP) to fit.
        init_params: starting parameters (defaults to the ``ElasticFoundationParams``
            defaults). ``v_eps`` is held fixed (a numerical regularizer, not calibrated).
        backend: ``"numpy"`` or ``"warp"`` for the forward evaluations.

    Returns:
        :class:`CalibrationResult` with the fitted params, cost history, and RMS
        force/COP diagnostics.
    """
    cfg = config or CalibrationConfig()
    init = init_params or ElasticFoundationParams()
    v_eps = init.v_eps
    floors = np.array([cfg.k_floor, cfg.c_floor, cfg.mu_floor], dtype=np.float64)

    target = CalibrationTarget(
        grf=np.asarray(target.grf, dtype=np.float64),
        cop=None if target.cop is None else np.asarray(target.cop, dtype=np.float64),
    )
    fscale = _force_scale(target, cfg)
    resfn = _make_residual_fn(cfg, target, fscale)

    # log-space params (strictly positive); c may be exactly 0 -> shift by floor+eps
    theta = np.array([init.k_bed, init.c_bed, init.mu], dtype=np.float64)
    theta = np.maximum(theta, floors + 1e-9)
    # For log-space we optimize p = log(theta - floor) so the floor is a hard lower bound.
    shift = floors
    p = np.log(theta - shift)

    def theta_of(p_):
        # clamp the exponent so rejected (over-aggressive) LM trial steps don't overflow
        return shift + np.exp(np.clip(p_, -50.0, 40.0))

    def eval_res(p_):
        th = theta_of(p_)
        pred = _predict(
            sole, th, v_eps, body_pos, body_quat, body_linvel, body_angvel,
            ground_z, backend, device,
        )
        return resfn(pred), pred

    r, pred = eval_res(p)
    cost = 0.5 * float(r @ r)
    lam = cfg.lm_lambda0
    history = [cost]

    n_iters = 0
    for it in range(cfg.max_iters):
        n_iters = it + 1
        # forward finite-difference Jacobian in log space (3 extra evaluations)
        M = r.shape[0]
        J = np.zeros((M, 3), dtype=np.float64)
        for j in range(3):
            pp = p.copy()
            pp[j] += cfg.fd_eps
            rj, _ = eval_res(pp)
            J[:, j] = (rj - r) / cfg.fd_eps

        JtJ = J.T @ J
        Jtr = J.T @ r
        diag = np.diag(np.clip(np.diag(JtJ), 1e-12, None))

        # inner LM: try steps, growing lambda on rejection
        improved = False
        for _ in range(20):
            try:
                dp = np.linalg.solve(JtJ + lam * diag, -Jtr)
            except np.linalg.LinAlgError:
                lam *= cfg.lm_lambda_up
                continue
            p_new = p + dp
            r_new, pred_new = eval_res(p_new)
            cost_new = 0.5 * float(r_new @ r_new)
            if cost_new < cost:
                p, r, pred, cost = p_new, r_new, pred_new, cost_new
                lam *= cfg.lm_lambda_down
                improved = True
                break
            lam *= cfg.lm_lambda_up
        history.append(cost)
        if cfg.verbose:
            th = theta_of(p)
            print(f"[cal] it={it} cost={cost:.6e} k={th[0]:.3e} "
                  f"c={th[1]:.3e} mu={th[2]:.3f} lam={lam:.2e}")
        if not improved:
            break
        if len(history) >= 2 and abs(history[-2] - history[-1]) <= cfg.tol_cost * (
            history[-2] + 1e-30
        ):
            break

    theta = theta_of(p)
    final = ElasticFoundationParams(
        k_bed=float(theta[0]), c_bed=float(theta[1]), mu=float(theta[2]), v_eps=v_eps
    )
    # final prediction (keep points off; recompute cheaply)
    pred = _predict(
        sole, theta, v_eps, body_pos, body_quat, body_linvel, body_angvel,
        ground_z, backend, device,
    )

    df = pred.grf - target.grf
    force_rms = float(np.sqrt(np.mean(np.sum(df ** 2, axis=1))))
    vertical_rms = float(np.sqrt(np.mean(df[:, 2] ** 2)))

    cop_rms = None
    if target.cop is not None:
        both = (
            np.isfinite(pred.cop).all(axis=1)
            & np.isfinite(target.cop).all(axis=1)
        )
        if np.any(both):
            dc = pred.cop[both, :2] - target.cop[both, :2]
            cop_rms = float(np.sqrt(np.mean(np.sum(dc ** 2, axis=1))))

    return CalibrationResult(
        params=final,
        cost_history=history,
        force_rms=force_rms,
        vertical_rms=vertical_rms,
        cop_rms=cop_rms,
        n_iters=n_iters,
        prediction=pred,
    )


# ---------------------------------------------------------------------------
# Generic LM core + hydroelastic calibration (M7 params)
# ---------------------------------------------------------------------------

# per-parameter hard lower bounds (floors) for log-space optimization
_PARAM_FLOORS = {
    "k_bed": 1.0e3,
    "stiffen_b": 0.0,
    "hc_alpha": 0.0,
    "mu_d": 1.0e-3,
    "mu_s": 1.0e-3,
    "v_stribeck": 1.0e-3,
}


def _lm_optimize(p0, eval_res, cfg: CalibrationConfig):
    """Levenberg–Marquardt in a free-parameter log-space.

    ``eval_res(p) -> (residual_vec, prediction)``. Returns
    ``(p, r, pred, history, n_iters)``.
    """
    p = np.asarray(p0, dtype=np.float64).copy()
    npar = p.shape[0]
    r, pred = eval_res(p)
    cost = 0.5 * float(r @ r)
    lam = cfg.lm_lambda0
    history = [cost]
    n_iters = 0
    for it in range(cfg.max_iters):
        n_iters = it + 1
        M = r.shape[0]
        J = np.zeros((M, npar), dtype=np.float64)
        for j in range(npar):
            pp = p.copy()
            pp[j] += cfg.fd_eps
            rj, _ = eval_res(pp)
            J[:, j] = (rj - r) / cfg.fd_eps
        JtJ = J.T @ J
        Jtr = J.T @ r
        diag = np.diag(np.clip(np.diag(JtJ), 1e-12, None))
        improved = False
        for _ in range(20):
            try:
                dp = np.linalg.solve(JtJ + lam * diag, -Jtr)
            except np.linalg.LinAlgError:
                lam *= cfg.lm_lambda_up
                continue
            r_new, pred_new = eval_res(p + dp)
            cost_new = 0.5 * float(r_new @ r_new)
            if cost_new < cost:
                p, r, pred, cost = p + dp, r_new, pred_new, cost_new
                lam *= cfg.lm_lambda_down
                improved = True
                break
            lam *= cfg.lm_lambda_up
        history.append(cost)
        if not improved:
            break
        if abs(history[-2] - history[-1]) <= cfg.tol_cost * (history[-2] + 1e-30):
            break
    return p, r, pred, history, n_iters


def calibrate_hydroelastic(
    sole: FootSole,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    target: CalibrationTarget,
    init_params=None,
    free_params: Sequence[str] = ("k_bed", "hc_alpha", "mu_d"),
    ground_z: float = 0.0,
    backend: str = "numpy",
    device: str = "cuda",
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    """Fit a subset of :class:`HydroelasticParams` to a measured GRF trajectory.

    Same LM machinery as :func:`calibrate_elastic_foundation`, but for the M7
    pressure-field law and a configurable ``free_params`` set (the rest of the
    :class:`HydroelasticParams` are held at their ``init_params`` values). Optimized in
    per-parameter log-space so each stays above its physical floor. Recommended free set
    for GRF-only data: ``("k_bed", "hc_alpha", "mu_d")`` — stiffening ``b`` and the
    Stribeck shape are weakly identified from net force alone.
    """
    from biomech.contact.hydroelastic import HydroelasticParams
    from biomech.contact.hydroelastic import evaluate_contact as he_eval

    cfg = config or CalibrationConfig()
    init = init_params or HydroelasticParams()
    for name in free_params:
        if name not in _PARAM_FLOORS:
            raise ValueError(f"unknown/again non-calibratable param {name!r}")

    target = CalibrationTarget(
        grf=np.asarray(target.grf, dtype=np.float64),
        cop=None if target.cop is None else np.asarray(target.cop, dtype=np.float64),
    )
    fscale = _force_scale(target, cfg)
    resfn = _make_residual_fn(cfg, target, fscale)

    free = list(free_params)
    floors = np.array([_PARAM_FLOORS[n] for n in free], dtype=np.float64)
    init_vals = np.array([float(getattr(init, n)) for n in free], dtype=np.float64)
    init_vals = np.maximum(init_vals, floors + 1e-9)
    p0 = np.log(init_vals - floors)

    def params_of(p_):
        vals = floors + np.exp(np.clip(p_, -50.0, 40.0))
        kw = {n: float(v) for n, v in zip(free, vals)}
        return HydroelasticParams(
            k_bed=kw.get("k_bed", init.k_bed),
            stiffen_b=kw.get("stiffen_b", init.stiffen_b),
            hc_alpha=kw.get("hc_alpha", init.hc_alpha),
            mu_d=kw.get("mu_d", init.mu_d),
            mu_s=kw.get("mu_s", init.mu_s),
            v_stribeck=kw.get("v_stribeck", init.v_stribeck),
            v_eps=init.v_eps,
        )

    def eval_res(p_):
        params = params_of(p_)
        pred = he_eval(
            sole, params, body_pos, body_quat, body_linvel, body_angvel,
            ground_z=ground_z, backend=backend, device=device,
        )
        return resfn(pred), pred

    p, r, pred, history, n_iters = _lm_optimize(p0, eval_res, cfg)
    final = params_of(p)

    df = pred.grf - target.grf
    force_rms = float(np.sqrt(np.mean(np.sum(df ** 2, axis=1))))
    vertical_rms = float(np.sqrt(np.mean(df[:, 2] ** 2)))
    cop_rms = None
    if target.cop is not None:
        both = np.isfinite(pred.cop).all(axis=1) & np.isfinite(target.cop).all(axis=1)
        if np.any(both):
            dc = pred.cop[both, :2] - target.cop[both, :2]
            cop_rms = float(np.sqrt(np.mean(np.sum(dc ** 2, axis=1))))

    return CalibrationResult(
        params=final,
        cost_history=history,
        force_rms=force_rms,
        vertical_rms=vertical_rms,
        cop_rms=cop_rms,
        n_iters=n_iters,
        prediction=pred,
    )
