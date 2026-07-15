# SPDX-License-Identifier: MIT
#
# Windows-native port of the *structure* of Nimble's bilevel marker fit
# (``dart/biomechanics/MarkerFitter.cpp``). Nimble minimizes marker reprojection error
# over {group scales, marker offsets, per-frame poses} with analytic Jacobians and
# IPOPT. IPOPT is unavailable on Windows, so we keep Nimble's problem formulation and
# solve it by block coordinate descent, where every block sub-step is exact or a proper
# descent step (so the whole thing converges without optimizer babysitting):
#
#   poses    : per-frame Warp Levenberg-Marquardt marker IK
#              (``biomech.fitting.ik.solve_marker_ik``) -- exact given scales + offsets.
#   offsets  : closed-form per-marker least squares. With the poses and scales fixed,
#              each marker's world position is linear in its local offset
#              (marker = R_body (scale (.) offset) + p_body), so the offset that best
#              reprojects it across frames -- regularized toward the model offset by the
#              prior -- has a 3x3 closed form (diagonal, since R is orthonormal).
#   scales   : Gauss-Newton / Levenberg-Marquardt step on the group scales, using a
#              finite-difference scale Jacobian of the marker residual (3G columns).
#
# Poses are held fixed while updating scales/offsets, which is valid at the inner-IK
# optimum by the envelope theorem (dL/dq = 0). The offset prior (a quadratic pull toward
# the model .osim offsets, stronger for anatomical landmarks) resolves the
# scale/offset/pose gauge ambiguity. The anthropometric body-scale prior
# (``Anthropometrics``) is deferred; joint limits are enforced by the inner IK clamp.
#
# All heavy work (FK, marker Jacobian, pose LM) runs on the Warp skeleton in float64.

"""Bilevel marker fit: {scales, offsets, poses} (port of Nimble MarkerFitter, M2d)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik
from biomech.skeleton.skeleton import WarpSkeleton


@dataclass
class MarkerFitConfig:
    """Settings for :meth:`MarkerFitter.fit`."""

    outer_iters: int = 25
    # Quadratic offset prior weight (relative to unit marker weight). Keeps offsets near
    # the model; anatomical markers are anchored ``anatomical_prior_factor`` x harder.
    offset_prior_weight: float = 1.0
    anatomical_prior_factor: float = 25.0
    # Levenberg-Marquardt damping (relative to the diagonal of J^T J) for the scale step.
    scale_lm_damping: float = 1e-3
    scale_fd_eps: float = 1e-5
    # Weak Tikhonov pull of the group scales toward neutral (1.0), plus an optional
    # vector anthropometric prior. If ``scale_prior_target``/``weights`` are supplied,
    # they add 0.5 * sum_i w_i * (scale_i - target_i)^2 to the scale step. This is the
    # lightweight Windows-native equivalent of OpenSim/Nimble-style segment scaling:
    # keep weakly observed axes anatomically plausible instead of letting marker offsets
    # or soft-tissue clusters drag them to degenerate values.
    scale_prior_weight: float = 0.05
    scale_prior_target: np.ndarray | None = None
    scale_prior_weights: np.ndarray | None = None
    scale_bounds: tuple[float, float] = (0.5, 1.6)
    optimize_scales: bool = True
    # Max |offset - model offset| (meters). Use a smaller value with anthropometric
    # fixed scales to stop tracking clusters from absorbing soft-tissue artifacts.
    offset_max_delta: float = 0.05
    convergence_rel: float = 1e-7  # stop when relative loss change < this
    inner: MarkerIKConfig = field(default_factory=lambda: MarkerIKConfig(max_iters=80))
    inner_first: MarkerIKConfig = field(
        default_factory=lambda: MarkerIKConfig(max_iters=300)
    )
    final_inner: MarkerIKConfig = field(
        default_factory=lambda: MarkerIKConfig(max_iters=80)
    )


@dataclass
class MarkerFitResult:
    group_scales: np.ndarray  # (3G,)
    marker_offsets: np.ndarray  # (M, 3)
    poses: np.ndarray  # (F, ndof)
    marker_rms: np.ndarray  # (F,) per-frame weighted RMS (m)
    loss_history: list[float] = field(default_factory=list)
    outer_iters: int = 0


class MarkerFitter:
    """Bilevel {group scales, marker offsets, poses} marker fit for a :class:`WarpSkeleton`.

    Parameters
    ----------
    skel : WarpSkeleton
    observations : (F, M, 3)
        Observed marker world positions in the model frame, aligned to
        ``skel.marker_names()``; NaN where missing.
    weights : (M,), optional
        Per-marker nonnegative weights (default all 1).
    anatomical : (M,) bool, optional
        Anatomical-landmark flags (default from the model's ``fixed`` markers); used to
        anchor those offsets harder.
    """

    def __init__(
        self,
        skel: WarpSkeleton,
        observations: np.ndarray,
        weights: np.ndarray | None = None,
        anatomical: np.ndarray | None = None,
    ):
        self.skel = skel
        self.obs = np.asarray(observations, dtype=np.float64)
        assert self.obs.ndim == 3 and self.obs.shape[2] == 3, self.obs.shape
        self.F, self.M, _ = self.obs.shape
        assert self.M == skel.topo.num_markers

        self.visible = np.asarray(
            np.isfinite(self.obs).all(axis=2), dtype=bool
        )  # (F, M)
        self.obs_z = np.where(self.visible[..., None], self.obs, 0.0)
        if weights is None:
            weights = np.ones(self.M)
        self.weights = np.asarray(weights, dtype=np.float64).ravel()
        if anatomical is None:
            anatomical = np.array(
                [m.anatomical for m in skel.spec.markers], dtype=bool
            )
        self.anatomical = np.asarray(anatomical, dtype=bool)

        # per-(frame, marker) linear loss weight (weight * visibility)
        self.w_lin = (self.weights[None, :] * self.visible).astype(np.float64)

        # marker -> body / group index
        self.m_body = np.asarray(skel.topo.m_body, dtype=int)
        bidx = {b.name: i for i, b in enumerate(skel.spec.bodies)}
        body_group = np.zeros(skel.topo.num_bodies, dtype=int)
        for gi, group in enumerate(skel.spec.scale_groups):
            for name in group:
                body_group[bidx[name]] = gi
        self.m_group = body_group[self.m_body]

        # model marker offsets = prior mean
        self.offset0 = skel.marker_offsets().copy()

    # ------------------------------------------------------------------
    def _residual(self, poses, scales, offsets):
        """Weighted marker residual (F*M*3,) at fixed poses (sqrt-weighted rows)."""
        self.skel.set_marker_offsets(offsets)
        _, mk = self.skel.forward(poses, scales)
        r = (mk - self.obs_z) * np.sqrt(self.w_lin)[..., None]
        return r.reshape(-1)

    def _offset_update(self, world, scales, prior_w):
        """Closed-form per-marker offset least squares (regularized to the model)."""
        offsets = self.offset0.copy()
        R = world[:, :, :3, :3]  # (F, B, 3, 3)
        p = world[:, :, :3, 3]  # (F, B, 3)
        for m in range(self.M):
            b = self.m_body[m]
            g = self.m_group[m]
            s = scales[3 * g : 3 * g + 3]
            vis = self.visible[:, m]
            pw = prior_w[m]
            if vis.any():
                Rb = R[vis, b]  # (k, 3, 3)
                pb = p[vis, b]  # (k, 3)
                wm = self.w_lin[vis, m]  # (k,)
                bt = self.obs[vis, m] - pb  # (k, 3)
                # R^T bt per frame, weight-summed
                acc = np.einsum("k,kji,kj->i", wm, Rb, bt)  # note Rb^T via index swap
                sw = wm.sum()
            else:
                acc = np.zeros(3)
                sw = 0.0
            diag = sw * s * s + pw  # (3,)
            rhs = s * acc + pw * self.offset0[m]
            o = rhs / diag
            # clamp offset delta magnitude
            d = o - self.offset0[m]
            nrm = np.linalg.norm(d)
            if nrm > self._offset_max:
                o = self.offset0[m] + d * (self._offset_max / nrm)
            offsets[m] = o
        return offsets

    def _scale_step(self, poses, scales, offsets, damping):
        """One Levenberg-Marquardt step on the group scales (FD Jacobian).

        Includes a weak Tikhonov pull toward neutral scale (``scale_prior``) so the
        weakly-observed anisotropic axes do not drift to the bounds.
        """
        G3 = scales.size
        r0 = self._residual(poses, scales, offsets)
        J = np.zeros((r0.size, G3), dtype=np.float64)
        eps = self._fd_eps
        for i in range(G3):
            s2 = scales.copy()
            s2[i] += eps
            J[:, i] = (self._residual(poses, s2, offsets) - r0) / eps
        H = J.T @ J
        g = J.T @ r0
        sp_w = self._scale_prior_weights
        sp_t = self._scale_prior_target
        A = H + damping * np.diag(np.maximum(np.diag(H), 1e-12)) + np.diag(sp_w)
        rhs = g + sp_w * (scales - sp_t)  # gradient of 1/2 * ||scale-target||_W^2
        try:
            delta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, rhs, rcond=None)[0]
        new = np.clip(scales - delta, self._scale_lo, self._scale_hi)
        return new, float(r0 @ r0)

    # ------------------------------------------------------------------
    def fit(
        self,
        init_scales: np.ndarray | None = None,
        init_offsets: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        config: MarkerFitConfig | None = None,
    ) -> MarkerFitResult:
        cfg = config or MarkerFitConfig()
        G = self.skel.topo.num_groups
        ndof = self.skel.topo.num_dofs
        self._offset_max = cfg.offset_max_delta
        self._fd_eps = cfg.scale_fd_eps
        self._scale_lo, self._scale_hi = cfg.scale_bounds
        base_target = np.ones(3 * G, dtype=np.float64)
        base_weights = np.full(3 * G, float(cfg.scale_prior_weight), dtype=np.float64)
        if cfg.scale_prior_target is not None:
            tgt = np.asarray(cfg.scale_prior_target, dtype=np.float64).ravel()
            assert tgt.shape == (3 * G,), tgt.shape
            base_target = tgt
        if cfg.scale_prior_weights is not None:
            w = np.asarray(cfg.scale_prior_weights, dtype=np.float64).ravel()
            assert w.shape == (3 * G,), w.shape
            base_weights = base_weights + np.maximum(w, 0.0)
        self._scale_prior_target = base_target
        self._scale_prior_weights = base_weights

        scales = (
            np.ones(3 * G) if init_scales is None
            else np.asarray(init_scales, dtype=np.float64).ravel().copy()
        )
        offsets = (
            self.offset0.copy() if init_offsets is None
            else np.asarray(init_offsets, dtype=np.float64).reshape(self.M, 3).copy()
        )
        prior_w = np.where(
            self.anatomical,
            cfg.offset_prior_weight * cfg.anatomical_prior_factor,
            cfg.offset_prior_weight,
        )

        poses = None if q_init is None else np.asarray(q_init, dtype=np.float64).copy()

        def _seed():
            seed = np.zeros((self.F, ndof))
            for t in range(self.F):
                if self.visible[t].any():
                    seed[t, 3:6] = self.obs[t, self.visible[t]].mean(axis=0)
            return seed

        def _solve_ik(obs, q0, scales_arg, ik_cfg):
            return solve_marker_ik(
                self.skel, obs, q_init=q0, group_scales=scales_arg,
                weights=self.weights, config=ik_cfg,
            )

        history: list[float] = []
        last = np.inf
        it = 0
        for it in range(1, cfg.outer_iters + 1):
            # 1. poses (inner IK)
            self.skel.set_marker_offsets(offsets)
            ik = _solve_ik(
                self.obs,
                (_seed() if poses is None else poses),
                scales,
                (cfg.inner_first if poses is None else cfg.inner),
            )
            poses = ik.q

            # 2. offsets (closed form; body FK does not depend on offsets)
            world, _ = self.skel.forward(poses, scales)
            offsets = self._offset_update(world, scales, prior_w)

            # 3. scales (LM step) + track loss. If an external anthropometric model has
            # already set the scales, skip the expensive finite-difference scale step and
            # just track the current marker loss.
            if cfg.optimize_scales:
                scales, loss = self._scale_step(poses, scales, offsets, cfg.scale_lm_damping)
            else:
                r = self._residual(poses, scales, offsets)
                loss = float(r @ r)
            history.append(loss)
            if abs(last - loss) <= cfg.convergence_rel * max(last, 1e-30):
                break
            last = loss

        # final poses at the converged parameters
        self.skel.set_marker_offsets(offsets)
        ik = _solve_ik(
            self.obs,
            (_seed() if poses is None else poses),
            scales,
            cfg.final_inner,
        )
        return MarkerFitResult(
            group_scales=scales,
            marker_offsets=offsets,
            poses=ik.q,
            marker_rms=ik.marker_rms,
            loss_history=history,
            outer_iters=it,
        )
