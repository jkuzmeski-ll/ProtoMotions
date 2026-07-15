# SPDX-License-Identifier: MIT
#
# Milestone M5 (rung 1) — distributed elastic-foundation (Winkler) foot contact as a
# Warp kernel, evaluated under prescribed (gold-standard) kinematics.
#
# Motivation (the project's research goal): a single point contact, or even a few
# spheres, cannot capture the plantar contact *surface*. An elastic foundation models
# the sole as a dense bed of independent springs distributed over the actual foot
# geometry — "between FEA and point contact." Each sample patch of the sole carries a
# local normal pressure proportional to its penetration into the ground (Winkler
# foundation), plus damping and Coulomb friction, and the patch forces integrate to a
# net ground-reaction wrench + centre of pressure that can be compared directly against
# the instrumented-treadmill 6-axis GRF/COP (``biomech.io.force_plate``).
#
# "Use Newton as much as possible": the per-patch contact law is a **Warp kernel**
# (one thread per (frame, sample point)), so a whole trajectory over a dense sole
# evaluates on the GPU. Under *prescribed kinematics* we do not integrate dynamics — we
# drive the foot bodies with the fitted ``q(t)`` (foot pose + spatial velocity from the
# gold-standard Warp FK) and predict GRF/COP. This is rung 1; rung 2 (M6) batches the
# calibration of ``(k, c, mu)`` against measured GRF/COP, and rung 3 (M7) swaps the
# Winkler law for a hydroelastic / tactile-rich law on the subject plantar SDF (M4).
#
# Conventions: SI units, world Z-up (matches ``biomech`` lab==world). Foot pose is a
# world position + xyzw quaternion (COMMON convention, as exported by
# ``biomech.export.motion``). The ground is the plane ``z = ground_z`` with +Z normal.

"""Distributed elastic-foundation (Winkler) foot contact, Warp-accelerated (M5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Sole geometry (a bed of sample patches in the foot body frame)
# ---------------------------------------------------------------------------


@dataclass
class FootSole:
    """A discretized plantar surface: sample points + normals + patch areas.

    All quantities are in the **foot body frame** (meters). ``normal`` is the outward
    sole normal (informational for rung 1, which uses the world ground normal), and
    ``area`` is the tributary area of each sample patch (so ``sum(area)`` is the total
    modeled contact area).
    """

    points: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3)
    areas: np.ndarray  # (N,)
    modulus: Optional[np.ndarray] = None  # (N,) relative stiffness map (None = uniform 1.0)

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def total_area(self) -> float:
        return float(np.sum(self.areas))

    def modulus_or_ones(self) -> np.ndarray:
        """Per-patch relative stiffness, defaulting to a uniform field of ones."""
        if self.modulus is None:
            return np.ones(self.n, dtype=np.float64)
        return np.asarray(self.modulus, dtype=np.float64)

    def scaled(self, sx: float, sy: float, sz: float) -> "FootSole":
        """Anisotropically scale the sole (e.g. to subject foot length/width)."""
        s = np.array([sx, sy, sz], dtype=np.float64)
        pts = self.points * s
        # normals scale by the inverse-transpose, then renormalize
        nrm = self.normals / s
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        areas = self.areas * (sx * sy)  # tributary area scales with the ground plane
        return FootSole(
            points=pts,
            normals=nrm,
            areas=areas,
            modulus=None if self.modulus is None else self.modulus.copy(),
        )

    def translated(self, offset: np.ndarray) -> "FootSole":
        return FootSole(
            points=self.points + np.asarray(offset, dtype=np.float64),
            normals=self.normals.copy(),
            areas=self.areas.copy(),
            modulus=None if self.modulus is None else self.modulus.copy(),
        )


def sample_flat_sole(
    length: float,
    width: float,
    nx: int,
    ny: int,
    z: float = 0.0,
    x_center: float = 0.0,
) -> FootSole:
    """A flat rectangular plantar bed in the foot frame (normal ``-z``, sole faces down).

    ``length`` runs along foot +x (heel->toe), ``width`` along +y. The patch is centered
    at ``x_center`` and sits at height ``z``. Tributary areas are uniform.
    """
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    gx, gy = np.meshgrid(xs * length + x_center, ys * width, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    nrm = np.tile(np.array([0.0, 0.0, -1.0]), (pts.shape[0], 1))
    area = np.full(pts.shape[0], (length * width) / (nx * ny))
    return FootSole(points=pts, normals=nrm, areas=area)


def sample_ellipsoid_sole(
    radii: Tuple[float, float, float],
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_theta: int = 12,
    n_phi: int = 6,
    max_polar_deg: float = 80.0,
) -> FootSole:
    """The lower cap of an ellipsoid (Brown/McPhee-style plantar lobe) as a bed.

    Samples the surface for polar angle from the downward pole up to ``max_polar_deg``
    (the part that can touch the ground). Normals point outward (downward on the cap);
    tributary areas come from the local surface-element Jacobian.
    """
    a, b, c = radii
    cx, cy, cz = center
    # polar phi measured from -z (downward pole); azimuth theta about z
    phis = np.linspace(0.0, np.deg2rad(max_polar_deg), n_phi + 1)
    phis = 0.5 * (phis[:-1] + phis[1:])
    dphi = np.deg2rad(max_polar_deg) / n_phi
    thetas = (np.arange(n_theta) + 0.5) * (2.0 * np.pi / n_theta)
    dtheta = 2.0 * np.pi / n_theta

    pts = []
    nrm = []
    area = []
    for phi in phis:
        # unit sphere point on the lower cap: z = -cos(phi)
        sz = -np.cos(phi)
        sr = np.sin(phi)
        for th in thetas:
            ux, uy, uz = sr * np.cos(th), sr * np.sin(th), sz
            p = np.array([a * ux + cx, b * uy + cy, c * uz + cz])
            # outward normal of an ellipsoid ~ (x/a^2, y/b^2, z/c^2)
            gnrm = np.array([ux / a, uy / b, uz / c])
            gnrm = gnrm / (np.linalg.norm(gnrm) + 1e-12)
            # surface area element of the sphere ~ sin(phi) dphi dtheta, mapped by ellipsoid
            # use the local metric determinant approximation via the parameter Jacobian
            da = _ellipsoid_area_element(a, b, c, phi, th) * dphi * dtheta
            pts.append(p)
            nrm.append(gnrm)
            area.append(da)
    return FootSole(
        points=np.array(pts), normals=np.array(nrm), areas=np.array(area)
    )


def _ellipsoid_area_element(a, b, c, phi, theta) -> float:
    """|∂P/∂phi × ∂P/∂theta| for P(phi,theta) on the lower cap (phi from -z)."""
    sr, cr = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    # P = (a sr ct, b sr st, -c cr)
    dphi = np.array([a * cr * ct, b * cr * st, c * sr])
    dtheta = np.array([-a * sr * st, b * sr * ct, 0.0])
    return float(np.linalg.norm(np.cross(dphi, dtheta)))


# ---------------------------------------------------------------------------
# Elastic-foundation contact law
# ---------------------------------------------------------------------------


@dataclass
class ElasticFoundationParams:
    """Winkler foundation + Coulomb friction parameters.

    ``k_bed``  : foundation stiffness per unit area per unit depth (N/m^3).
    ``c_bed``  : foundation damping per unit area per unit penetration-rate (N*s/m^3).
    ``mu``     : Coulomb friction coefficient.
    ``v_eps``  : tangential-velocity regularization (m/s) for smooth friction.
    """

    k_bed: float = 5.0e6
    c_bed: float = 5.0e3
    mu: float = 0.9
    v_eps: float = 1.0e-3


@dataclass
class ContactPrediction:
    """Per-frame predicted ground reaction from the distributed contact model."""

    grf: np.ndarray  # (F, 3) net force ON the foot (world, N)
    cop: np.ndarray  # (F, 3) centre of pressure (world, m); NaN when unloaded
    free_moment_z: np.ndarray  # (F,) vertical free moment about the COP (N*m)
    total_normal: np.ndarray  # (F,) sum of patch normal forces (N)
    point_forces: Optional[np.ndarray] = None  # (F, N, 3) per-patch world force
    point_world: Optional[np.ndarray] = None  # (F, N, 3) per-patch world position


# ---------------------------------------------------------------------------
# NumPy reference (float64, authoritative for tests)
# ---------------------------------------------------------------------------


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` (..., 3) by xyzw quaternions ``q`` (..., 4)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # rotation via q * v * q^-1, vectorized
    t = 2.0 * np.cross(q[..., :3], v)
    return v + w[..., None] * t + np.cross(q[..., :3], t)


def point_forces_numpy(
    sole: FootSole,
    params: ElasticFoundationParams,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    ground_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-patch world force ``(F, N, 3)`` and world position ``(F, N, 3)`` (reference).

    Force ON the foot: normal along +Z (Winkler ``k*d`` + damping, no adhesion) and
    regularized Coulomb friction opposing the patch's tangential slip velocity.
    """
    body_pos = np.asarray(body_pos, dtype=np.float64)
    body_quat = np.asarray(body_quat, dtype=np.float64)
    body_linvel = np.asarray(body_linvel, dtype=np.float64)
    body_angvel = np.asarray(body_angvel, dtype=np.float64)
    F = body_pos.shape[0]
    N = sole.n

    # world positions: p = body_pos + R(q) * p_local
    pl = sole.points  # (N,3)
    pw = body_pos[:, None, :] + _quat_rotate_np(
        body_quat[:, None, :], np.broadcast_to(pl, (F, N, 3))
    )
    r = pw - body_pos[:, None, :]  # lever arm from body origin
    v = body_linvel[:, None, :] + np.cross(
        np.broadcast_to(body_angvel[:, None, :], (F, N, 3)), r
    )

    d = ground_z - pw[:, :, 2]  # penetration depth (>0 below ground)
    vn = -v[:, :, 2]  # penetration rate (>0 moving down)
    contact = d > 0.0

    fn = sole.areas[None, :] * (params.k_bed * d + params.c_bed * vn)
    fn = np.where(contact, np.clip(fn, 0.0, None), 0.0)

    vt = v.copy()
    vt[:, :, 2] = 0.0
    vt_mag = np.linalg.norm(vt, axis=2)
    ft = -params.mu * fn[:, :, None] * vt / (vt_mag[:, :, None] + params.v_eps)

    force = np.zeros((F, N, 3), dtype=np.float64)
    force[:, :, :2] = ft[:, :, :2]
    force[:, :, 2] = fn
    return force, pw


# ---------------------------------------------------------------------------
# Warp kernel (GPU; "use Newton as much as possible")
# ---------------------------------------------------------------------------


def point_forces_warp(
    sole: FootSole,
    params: ElasticFoundationParams,
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
    d = device

    bp = wp.array(np.asarray(body_pos, np.float32), dtype=wp.vec3, device=d)
    bq = wp.array(np.asarray(body_quat, np.float32), dtype=wp.quat, device=d)
    bl = wp.array(np.asarray(body_linvel, np.float32), dtype=wp.vec3, device=d)
    ba = wp.array(np.asarray(body_angvel, np.float32), dtype=wp.vec3, device=d)
    pts = wp.array(np.asarray(sole.points, np.float32), dtype=wp.vec3, device=d)
    area = wp.array(np.asarray(sole.areas, np.float32), dtype=wp.float32, device=d)

    out_f = wp.zeros((F, N), dtype=wp.vec3, device=d)
    out_p = wp.zeros((F, N), dtype=wp.vec3, device=d)

    wp.launch(
        _EF_KERNEL,
        dim=(F, N),
        inputs=[
            bp, bq, bl, ba, pts, area,
            float(ground_z), float(params.k_bed), float(params.c_bed),
            float(params.mu), float(params.v_eps),
        ],
        outputs=[out_f, out_p],
        device=d,
    )
    return out_f.numpy().astype(np.float64), out_p.numpy().astype(np.float64)


_EF_KERNEL = None


def _ensure_kernel(wp):
    global _EF_KERNEL
    if _EF_KERNEL is not None:
        return

    @wp.kernel
    def ef_kernel(
        body_pos: wp.array(dtype=wp.vec3),
        body_quat: wp.array(dtype=wp.quat),
        body_linvel: wp.array(dtype=wp.vec3),
        body_angvel: wp.array(dtype=wp.vec3),
        pts_local: wp.array(dtype=wp.vec3),
        area: wp.array(dtype=wp.float32),
        ground_z: wp.float32,
        k_bed: wp.float32,
        c_bed: wp.float32,
        mu: wp.float32,
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
        fn = area[i] * (k_bed * d + c_bed * vn)
        if fn < 0.0:
            fn = 0.0
        vt = wp.vec3(v[0], v[1], 0.0)
        vt_mag = wp.length(vt)
        scale = -mu * fn / (vt_mag + v_eps)
        out_force[f, i] = wp.vec3(scale * v[0], scale * v[1], fn)

    _EF_KERNEL = ef_kernel


# ---------------------------------------------------------------------------
# Reduction: per-patch forces -> net GRF / COP / free moment
# ---------------------------------------------------------------------------


def reduce_wrench(
    point_forces: np.ndarray,
    point_world: np.ndarray,
    ground_z: float = 0.0,
    fz_threshold: float = 1e-6,
) -> ContactPrediction:
    """Aggregate per-patch forces ``(F, N, 3)`` into net GRF, COP, and free moment.

    COP is the ground-plane point where the normal forces balance (the standard
    pressure centroid); ``free_moment_z`` is the residual vertical moment about it.
    Frames with total normal force below ``fz_threshold`` get a ``NaN`` COP (swing).
    """
    point_forces = np.asarray(point_forces, dtype=np.float64)
    point_world = np.asarray(point_world, dtype=np.float64)
    F = point_forces.shape[0]

    fn = point_forces[:, :, 2]  # (F, N)
    total_normal = np.sum(fn, axis=1)  # (F,)
    grf = np.sum(point_forces, axis=1)  # (F, 3)

    cop = np.full((F, 3), np.nan)
    free_mz = np.zeros(F, dtype=np.float64)
    loaded = total_normal > fz_threshold
    if np.any(loaded):
        w = fn[loaded] / total_normal[loaded][:, None]  # normal-weighted
        px = np.sum(w * point_world[loaded, :, 0], axis=1)
        py = np.sum(w * point_world[loaded, :, 1], axis=1)
        cop[loaded, 0] = px
        cop[loaded, 1] = py
        cop[loaded, 2] = ground_z
        # vertical free moment about the COP: sum (r x f)_z with r = p - cop
        rx = point_world[loaded, :, 0] - px[:, None]
        ry = point_world[loaded, :, 1] - py[:, None]
        fx = point_forces[loaded, :, 0]
        fy = point_forces[loaded, :, 1]
        free_mz[loaded] = np.sum(rx * fy - ry * fx, axis=1)

    return ContactPrediction(
        grf=grf,
        cop=cop,
        free_moment_z=free_mz,
        total_normal=total_normal,
        point_forces=point_forces,
        point_world=point_world,
    )


# ---------------------------------------------------------------------------
# Top-level evaluate
# ---------------------------------------------------------------------------


def evaluate_contact(
    sole: FootSole,
    params: ElasticFoundationParams,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_linvel: np.ndarray,
    body_angvel: np.ndarray,
    ground_z: float = 0.0,
    backend: str = "numpy",
    device: str = "cuda",
    keep_points: bool = False,
) -> ContactPrediction:
    """Predict GRF/COP for a foot trajectory under prescribed kinematics.

    Args:
        sole: discretized plantar bed (foot frame).
        params: elastic-foundation + friction parameters.
        body_pos/quat/linvel/angvel: ``(F, ...)`` foot pose + spatial velocity (world,
            xyzw quaternion) — e.g. from ``biomech.export.motion`` / the gold-standard FK.
        backend: ``"numpy"`` (float64 reference) or ``"warp"`` (GPU float32).
        keep_points: keep the per-patch arrays on the result (memory-heavy for dense soles).
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
