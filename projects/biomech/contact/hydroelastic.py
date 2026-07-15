# SPDX-License-Identifier: MIT
#
# Milestone M7 (contact rung 3) — hydroelastic / pressure-field foot contact as a Warp
# kernel, evaluated under prescribed (gold-standard) kinematics.
#
# Motivation (the project's research goal): point/sphere contact cannot represent the
# plantar *pressure distribution*, and a plain Winkler bed (rung 1) is linear and
# energetically inconsistent at separation. This rung implements a pressure-field model
# in the spirit of the hydroelastic ("pressure field") contact used in Drake
# (Elandt et al. 2019) specialized to a compliant foot on rigid ground, which is exactly
# "between FEA and point contact":
#
#   * Each plantar patch carries a scalar pressure ``p`` (Pa) from the foot's internal
#     pressure field evaluated at the local penetration depth. For a foot of
#     characteristic thickness ``H`` and hydroelastic modulus ``E`` the linear field is
#     ``p = (E/H)·d`` — i.e. a Winkler bed with ``k = E/H`` — but the field may vary
#     spatially (``FootSole.modulus`` maps soft heel pad vs stiff forefoot) and stiffen
#     with compression (soft tissue is hyperelastic): ``p = k·d·(1 + b·d)``.
#   * Dissipation is Hunt–Crossley (``p ← p·(1 + a·ṅ)`` clamped ≥ 0): it scales with the
#     elastic pressure, is energetically consistent, and — unlike additive linear damping
#     — produces **no adhesive pull** at lift-off.
#   * Friction is a pressure-dependent regularized Coulomb law with an optional Stribeck
#     transition from static to dynamic ``mu``.
#
# In the linear limit (``b=0``, ``a=0``, uniform modulus, ``mu_s=mu_d``) this reduces
# exactly to ``biomech.contact.elastic_foundation`` (verified in tests).
#
# "Use Newton as much as possible": the per-patch law is a Warp kernel (one thread per
# (frame, patch)); the net wrench / COP reduction and calibration reuse the rung-1/rung-2
# infrastructure (``reduce_wrench`` / ``ContactPrediction``). World Z-up; inputs are the
# same per-frame foot pose + spatial velocity (xyzw) used everywhere in ``biomech``.

"""Hydroelastic (pressure-field) foot contact, Warp-accelerated (M7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from biomech.contact.elastic_foundation import (
    ContactPrediction,
    FootSole,
    _quat_rotate_np,
    reduce_wrench,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class HydroelasticParams:
    """Pressure-field foot-contact parameters.

    ``k_bed``       : linear foundation stiffness per area per depth (N/m^3) == E/H.
    ``stiffen_b``   : hyperelastic stiffening coefficient (1/m); ``p = k·d·(1 + b·d)``.
    ``hc_alpha``    : Hunt–Crossley dissipation constant (s/m); ``p ← p·(1 + α·ṅ)`` ≥ 0.
    ``mu_d``        : dynamic (sliding) Coulomb friction coefficient.
    ``mu_s``        : static friction coefficient (``>= mu_d``) for the Stribeck peak.
    ``v_stribeck``  : Stribeck velocity scale (m/s) for the static→dynamic transition.
    ``v_eps``       : tangential-velocity regularization (m/s).
    """

    k_bed: float = 5.0e6
    stiffen_b: float = 0.0
    hc_alpha: float = 1.0
    mu_d: float = 0.8
    mu_s: float = 0.9
    v_stribeck: float = 0.05
    v_eps: float = 1.0e-3


# ---------------------------------------------------------------------------
# NumPy reference (float64, authoritative for tests)
# ---------------------------------------------------------------------------


def point_forces_numpy(
    sole: FootSole,
    params: HydroelasticParams,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    ground_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-patch world force ``(F, N, 3)`` and world position ``(F, N, 3)`` (reference)."""
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_quat = np.asarray(body_quat, dtype=np.float64)
    body_linvel = np.asarray(body_linvel, dtype=np.float64)
    body_angvel = np.asarray(body_angvel, dtype=np.float64)
    F = body_pos.shape[0]
    N = sole.n

    pl = sole.points  # (N,3)
    mod = sole.modulus_or_ones()  # (N,)
    pw = body_pos[:, None, :] + _quat_rotate_np(
        body_quat[:, None, :], np.broadcast_to(pl, (F, N, 3))
    )
    r = pw - body_pos[:, None, :]
    v = body_linvel[:, None, :] + np.cross(
        np.broadcast_to(body_angvel[:, None, :], (F, N, 3)), r
    )

    d = ground_z - pw[:, :, 2]  # penetration depth
    vn = -v[:, :, 2]  # penetration rate
    contact = d > 0.0

    # hydroelastic pressure: linear field * hyperelastic stiffening
    k_eff = params.k_bed * mod[None, :]  # (F?,N) via broadcast -> (1,N)
    p_elastic = k_eff * d * (1.0 + params.stiffen_b * d)
    # Hunt–Crossley dissipation (non-adhesive): scales with elastic pressure
    p = p_elastic * (1.0 + params.hc_alpha * vn)
    p = np.where(contact, np.clip(p, 0.0, None), 0.0)
    fn = sole.areas[None, :] * p  # (F,N)

    # pressure-dependent Stribeck friction
    vt = v.copy()
    vt[:, :, 2] = 0.0
    vt_mag = np.linalg.norm(vt, axis=2)  # (F,N)
    mu_eff = params.mu_d + (params.mu_s - params.mu_d) * np.exp(
        -((vt_mag / max(params.v_stribeck, 1e-12)) ** 2)
    )
    ft = -mu_eff[:, :, None] * fn[:, :, None] * vt / (vt_mag[:, :, None] + params.v_eps)

    force = np.zeros((F, N, 3), dtype=np.float64)
    force[:, :, :2] = ft[:, :, :2]
    force[:, :, 2] = fn
    return force, pw


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------


_HE_KERNEL = None


def _ensure_kernel(wp):
    global _HE_KERNEL
    if _HE_KERNEL is not None:
        return

    @wp.kernel
    def he_kernel(
        body_pos: wp.array(dtype=wp.vec3),
        body_quat: wp.array(dtype=wp.quat),
        body_linvel: wp.array(dtype=wp.vec3),
        body_angvel: wp.array(dtype=wp.vec3),
        pts_local: wp.array(dtype=wp.vec3),
        area: wp.array(dtype=wp.float32),
        modulus: wp.array(dtype=wp.float32),
        ground_z: wp.float32,
        k_bed: wp.float32,
        stiffen_b: wp.float32,
        hc_alpha: wp.float32,
        mu_d: wp.float32,
        mu_s: wp.float32,
        v_stribeck: wp.float32,
        v_eps: wp.float32,
        out_force: wp.array2d(dtype=wp.vec3),
        out_point: wp.array2d(dtype=wp.vec3),
    ):
        f, i = wp.tid()
        q = body_quat[f]
        p0 = body_pos[f]
        pw = p0 + wp.quat_rotate(q, pts_local[i])
        out_point[f, i] = pw
        d = ground_z - pw[2]
        if d <= 0.0:
            out_force[f, i] = wp.vec3(0.0, 0.0, 0.0)
            return
        r = pw - p0
        v = body_linvel[f] + wp.cross(body_angvel[f], r)
        vn = -v[2]
        k_eff = k_bed * modulus[i]
        p_elastic = k_eff * d * (1.0 + stiffen_b * d)
        p = p_elastic * (1.0 + hc_alpha * vn)
        if p < 0.0:
            p = 0.0
        fn = area[i] * p
        vt = wp.vec3(v[0], v[1], 0.0)
        vt_mag = wp.length(vt)
        stb = vt_mag / v_stribeck
        mu_eff = mu_d + (mu_s - mu_d) * wp.exp(-stb * stb)
        scale = -mu_eff * fn / (vt_mag + v_eps)
        out_force[f, i] = wp.vec3(scale * v[0], scale * v[1], fn)

    _HE_KERNEL = he_kernel


def point_forces_warp(
    sole: FootSole,
    params: HydroelasticParams,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    ground_z: float = 0.0,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp implementation of :func:`point_forces_numpy` (float32, batched over F*N)."""
    import warp as wp

    _ensure_kernel(wp)

    F = int(body_pos.shape[0])
    N = sole.n
    dv = device

    bp = wp.array(np.asarray(body_pos, np.float32), dtype=wp.vec3, device=dv)
    bq = wp.array(np.asarray(body_quat, np.float32), dtype=wp.quat, device=dv)
    bl = wp.array(np.asarray(body_linvel, np.float32), dtype=wp.vec3, device=dv)
    ba = wp.array(np.asarray(body_angvel, np.float32), dtype=wp.vec3, device=dv)
    pts = wp.array(np.asarray(sole.points, np.float32), dtype=wp.vec3, device=dv)
    area = wp.array(np.asarray(sole.areas, np.float32), dtype=wp.float32, device=dv)
    mod = wp.array(
        np.asarray(sole.modulus_or_ones(), np.float32), dtype=wp.float32, device=dv
    )

    out_f = wp.zeros((F, N), dtype=wp.vec3, device=dv)
    out_p = wp.zeros((F, N), dtype=wp.vec3, device=dv)

    wp.launch(
        _HE_KERNEL,
        dim=(F, N),
        inputs=[
            bp, bq, bl, ba, pts, area, mod,
            float(ground_z), float(params.k_bed), float(params.stiffen_b),
            float(params.hc_alpha), float(params.mu_d), float(params.mu_s),
            float(params.v_stribeck), float(params.v_eps),
        ],
        outputs=[out_f, out_p],
        device=dv,
    )
    return out_f.numpy().astype(np.float64), out_p.numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Top-level evaluate
# ---------------------------------------------------------------------------


def evaluate_contact(
    sole: FootSole,
    params: HydroelasticParams,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    ground_z: float = 0.0,
    backend: str = "numpy",
    device: str = "cuda",
    keep_points: bool = False,
) -> ContactPrediction:
    """Predict GRF/COP for a foot trajectory with the hydroelastic pressure-field law.

    Mirrors :func:`biomech.contact.elastic_foundation.evaluate_contact`; see the module
    docstring for the pressure/dissipation/friction law. Returns a
    :class:`ContactPrediction` (same GRF/COP/free-moment fields as ``ForcePlate``).
    """
    if backend == "warp":
        pf, pw = point_forces_warp(
            sole, params, body_pos, body_quat, body_linvel, body_angvel,
            ground_z=ground_z, device=device,
        )
    elif backend == "numpy":
        pf, pw = point_forces_numpy(
            sole, params, body_pos, body_quat, body_linvel, body_angvel,
            ground_z=ground_z,
        )
    else:
        raise ValueError("backend must be 'numpy' or 'warp'")

    pred = reduce_wrench(pf, pw, ground_z=ground_z)
    if not keep_points:
        pred.point_forces = None
        pred.point_world = None
    return pred
