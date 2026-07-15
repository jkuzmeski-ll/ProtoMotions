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
import warp as wp

from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik
from biomech.fitting.ik import _residual_kernel as _ik_residual_kernel
from biomech.skeleton.skeleton import WarpSkeleton


@wp.kernel
def _offset_update_kernel(
    world: wp.array2d(dtype=wp.mat44d),  # (F, B) body transforms
    obs: wp.array2d(dtype=wp.vec3d),  # (F, M) observed markers (NaNs pre-zeroed)
    wlin: wp.array2d(dtype=wp.float64),  # (F, M) linear weight * visibility
    m_body: wp.array(dtype=wp.int32),  # (M,)
    m_group: wp.array(dtype=wp.int32),  # (M,)
    scales: wp.array(dtype=wp.float64),  # (3G,)
    offset0: wp.array(dtype=wp.vec3d),  # (M,) model (prior-mean) offsets
    prior_w: wp.array(dtype=wp.float64),  # (M,) per-marker prior weight
    offset_max: wp.float64,
    Fr: wp.int32,
    offsets_out: wp.array(dtype=wp.vec3d),  # (M,) output offsets
):
    """Closed-form per-marker offset least squares (regularized toward the model).

    Each marker is independent (one thread per marker); with poses and scales fixed the
    marker world position is linear in its local offset, so the best offset has a 3x3
    diagonal closed form (R is orthonormal). Mirrors the host reference exactly.
    """
    m = wp.tid()
    b = m_body[m]
    g = m_group[m]
    sx = scales[3 * g]
    sy = scales[3 * g + 1]
    sz = scales[3 * g + 2]
    pw = prior_w[m]

    ax = wp.float64(0.0)
    ay = wp.float64(0.0)
    az = wp.float64(0.0)
    sw = wp.float64(0.0)
    for f in range(Fr):
        w = wlin[f, m]
        if w > wp.float64(0.0):
            T = world[f, b]
            o = obs[f, m]
            btx = o[0] - T[0, 3]
            bty = o[1] - T[1, 3]
            btz = o[2] - T[2, 3]
            # R^T @ bt : component i = R[0,i]*btx + R[1,i]*bty + R[2,i]*btz
            ax += w * (T[0, 0] * btx + T[1, 0] * bty + T[2, 0] * btz)
            ay += w * (T[0, 1] * btx + T[1, 1] * bty + T[2, 1] * btz)
            az += w * (T[0, 2] * btx + T[1, 2] * bty + T[2, 2] * btz)
            sw += w

    o0 = offset0[m]
    ox = (sx * ax + pw * o0[0]) / (sw * sx * sx + pw)
    oy = (sy * ay + pw * o0[1]) / (sw * sy * sy + pw)
    oz = (sz * az + pw * o0[2]) / (sw * sz * sz + pw)

    dx = ox - o0[0]
    dy = oy - o0[1]
    dz = oz - o0[2]
    nrm = wp.sqrt(dx * dx + dy * dy + dz * dz)
    if nrm > offset_max:
        sc = offset_max / nrm
        ox = o0[0] + dx * sc
        oy = o0[1] + dy * sc
        oz = o0[2] + dz * sc
    offsets_out[m] = wp.vec3d(ox, oy, oz)


@wp.kernel
def _scale_H_kernel(
    mk_base: wp.array2d(dtype=wp.vec3d),  # (F, M) markers at current scales
    mk_pert: wp.array3d(dtype=wp.vec3d),  # (F, 3G, M) markers with scale p bumped +eps
    wlin: wp.array2d(dtype=wp.float64),  # (F, M) linear weight * visibility
    eps: wp.float64,
    Fr: wp.int32,
    M: wp.int32,
    H: wp.array2d(dtype=wp.float64),  # (3G, 3G) Gauss-Newton normal matrix
):
    """H[p,q] = sum_{f,m} wlin * (mk_pert[p]-mk_base) . (mk_pert[q]-mk_base) / eps^2."""
    p, q = wp.tid()
    if q < p:
        return
    acc = wp.float64(0.0)
    for f in range(Fr):
        for m in range(M):
            w = wlin[f, m]
            if w > wp.float64(0.0):
                base = mk_base[f, m]
                dp = mk_pert[f, p, m] - base
                dq = mk_pert[f, q, m] - base
                acc += w * (dp[0] * dq[0] + dp[1] * dq[1] + dp[2] * dq[2])
    val = acc / (eps * eps)
    H[p, q] = val
    H[q, p] = val


@wp.kernel
def _scale_g_kernel(
    mk_base: wp.array2d(dtype=wp.vec3d),  # (F, M)
    mk_pert: wp.array3d(dtype=wp.vec3d),  # (F, 3G, M)
    obs: wp.array2d(dtype=wp.vec3d),  # (F, M) observed (NaNs pre-zeroed)
    wlin: wp.array2d(dtype=wp.float64),  # (F, M)
    eps: wp.float64,
    Fr: wp.int32,
    M: wp.int32,
    g: wp.array(dtype=wp.float64),  # (3G,) gradient J^T r0
):
    """g[p] = sum_{f,m} wlin * (mk_pert[p]-mk_base) . (mk_base-obs) / eps."""
    p = wp.tid()
    acc = wp.float64(0.0)
    for f in range(Fr):
        for m in range(M):
            w = wlin[f, m]
            if w > wp.float64(0.0):
                base = mk_base[f, m]
                dp = mk_pert[f, p, m] - base
                e = base - obs[f, m]
                acc += w * (dp[0] * e[0] + dp[1] * e[1] + dp[2] * e[2])
    g[p] = acc / eps


@wp.kernel
def _scale_loss_kernel(
    mk_base: wp.array2d(dtype=wp.vec3d),  # (F, M)
    obs: wp.array2d(dtype=wp.vec3d),  # (F, M)
    wlin: wp.array2d(dtype=wp.float64),  # (F, M)
    M: wp.int32,
    loss: wp.array(dtype=wp.float64),  # (F,) per-frame weighted squared error
):
    f = wp.tid()
    acc = wp.float64(0.0)
    for m in range(M):
        w = wlin[f, m]
        d = mk_base[f, m] - obs[f, m]
        acc += w * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    loss[f] = acc


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

        # device-resident arrays reused across the fit (offset update on GPU)
        dev = skel.device
        self._dev = dev
        self.d_m_body = wp.array(self.m_body.astype(np.int32), dtype=wp.int32, device=dev)
        self.d_m_group = wp.array(self.m_group.astype(np.int32), dtype=wp.int32, device=dev)
        self.d_obs_z = wp.array(
            self.obs_z.reshape(self.F, self.M, 3), dtype=wp.vec3d, device=dev
        )
        self.d_wlin = wp.array(self.w_lin, dtype=wp.float64, device=dev)
        # sqrt-row-weight for the marker residual (matches sqrt(w_lin) rows)
        self.d_rw = wp.array(np.sqrt(self.w_lin), dtype=wp.float64, device=dev)
        self.d_offset0 = wp.array(self.offset0, dtype=wp.vec3d, device=dev)

    # ------------------------------------------------------------------
    def _residual(self, poses, scales, offsets):
        """Weighted marker residual (F*M*3,) at fixed poses, computed on the GPU.

        FK + marker positions + sqrt-weighted residual all run on the Warp device; only
        the flattened residual vector is returned to the host for the (tiny 3G x 3G)
        scale normal-equation assembly.
        """
        self.skel.set_marker_offsets(offsets)
        dev = self._dev
        d_poses = wp.array(np.asarray(poses, dtype=np.float64), dtype=wp.float64, device=dev)
        d_scales = wp.array(
            np.asarray(scales, dtype=np.float64).ravel(), dtype=wp.float64, device=dev
        )
        _, d_markers = self.skel._run_wp(d_poses, d_scales)
        d_res = wp.zeros((self.F, self.M), dtype=wp.vec3d, device=dev)
        wp.launch(
            _ik_residual_kernel,
            dim=(self.F, self.M),
            inputs=[d_markers, self.d_obs_z, self.d_rw],
            outputs=[d_res],
            device=dev,
        )
        return d_res.numpy().reshape(-1)

    def _offset_update(self, poses, scales, prior_w):
        """Closed-form per-marker offset least squares (regularized to the model), on GPU.

        One Warp kernel thread per marker; body FK does not depend on the offsets, so we
        run FK once on the device and let every marker solve its 3x3 diagonal system in
        parallel. Identical result to the per-marker host reference it replaces.
        """
        dev = self._dev
        d_poses = wp.array(np.asarray(poses, dtype=np.float64), dtype=wp.float64, device=dev)
        d_scales = wp.array(np.asarray(scales, dtype=np.float64).ravel(), dtype=wp.float64, device=dev)
        d_world, _ = self.skel._run_wp(d_poses, d_scales)
        d_prior_w = wp.array(np.asarray(prior_w, dtype=np.float64), dtype=wp.float64, device=dev)
        d_out = wp.zeros(self.M, dtype=wp.vec3d, device=dev)
        wp.launch(
            _offset_update_kernel,
            dim=self.M,
            inputs=[
                d_world, self.d_obs_z, self.d_wlin, self.d_m_body, self.d_m_group,
                d_scales, self.d_offset0, d_prior_w, wp.float64(self._offset_max), self.F,
            ],
            outputs=[d_out],
            device=dev,
        )
        return d_out.numpy().reshape(self.M, 3)

    def _scale_step(self, poses, scales, offsets, damping):
        """One Levenberg-Marquardt step on the group scales (device FD Jacobian).

        The 3G-column finite-difference scale Jacobian is never materialized on the host:
        two Warp kernels perturb every group-scale component in parallel and assemble the
        3G x 3G Gauss-Newton normal equations (``H``, ``g``) directly on the GPU. Only the
        compact 3G x 3G system and the per-frame loss are read back for the tiny host
        solve. Includes a weak Tikhonov pull toward neutral scale (``scale_prior``) so the
        weakly-observed anisotropic axes do not drift to the bounds. Numerically identical
        to the previous per-column host implementation (one-sided FD, sqrt-weighted).
        """
        self.skel.set_marker_offsets(offsets)
        dev = self._dev
        G3 = scales.size
        d_poses = wp.array(
            np.asarray(poses, dtype=np.float64), dtype=wp.float64, device=dev
        )
        d_scales = wp.array(
            np.asarray(scales, dtype=np.float64).ravel(), dtype=wp.float64, device=dev
        )
        mk_base, mk_pert = self.skel.markers_scale_perturbed_wp(
            d_poses, d_scales, eps=self._fd_eps
        )
        epsd = wp.float64(self._fd_eps)
        d_H = wp.zeros((G3, G3), dtype=wp.float64, device=dev)
        d_g = wp.zeros(G3, dtype=wp.float64, device=dev)
        d_loss = wp.zeros(self.F, dtype=wp.float64, device=dev)
        wp.launch(
            _scale_H_kernel,
            dim=(G3, G3),
            inputs=[mk_base, mk_pert, self.d_wlin, epsd, self.F, self.M],
            outputs=[d_H],
            device=dev,
        )
        wp.launch(
            _scale_g_kernel,
            dim=G3,
            inputs=[mk_base, mk_pert, self.d_obs_z, self.d_wlin, epsd, self.F, self.M],
            outputs=[d_g],
            device=dev,
        )
        wp.launch(
            _scale_loss_kernel,
            dim=self.F,
            inputs=[mk_base, self.d_obs_z, self.d_wlin, self.M],
            outputs=[d_loss],
            device=dev,
        )
        H = d_H.numpy()
        g = d_g.numpy()
        loss = float(d_loss.numpy().sum())
        sp_w = self._scale_prior_weights
        sp_t = self._scale_prior_target
        A = H + damping * np.diag(np.maximum(np.diag(H), 1e-12)) + np.diag(sp_w)
        rhs = g + sp_w * (scales - sp_t)  # gradient of 1/2 * ||scale-target||_W^2
        try:
            delta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, rhs, rcond=None)[0]
        new = np.clip(scales - delta, self._scale_lo, self._scale_hi)
        return new, loss

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

            # 2. offsets (closed form on GPU; body FK does not depend on offsets)
            offsets = self._offset_update(poses, scales, prior_w)

            # 3. scales (LM step) + track loss. If an external anthropometric model has
            # already set the scales, skip the expensive finite-difference scale step and
            # just track the current marker loss.
            if cfg.optimize_scales:
                scales, loss = self._scale_step(poses, scales, offsets, cfg.scale_lm_damping)
            else:
                r = self._residual(poses, scales, offsets)
                loss = float(r @ r)
            history.append(loss)
            # Relative-change convergence, but only once we have two losses to compare;
            # on the first iteration ``last`` is +inf so the test would spuriously fire
            # (inf <= inf), stopping after a single outer step and ignoring outer_iters.
            if it > 1 and abs(last - loss) <= cfg.convergence_rel * max(last, 1e-30):
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
