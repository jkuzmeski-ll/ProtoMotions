# SPDX-License-Identifier: MIT
#
# Milestone M2e — DynamicsFitter (residual reduction + mass identification from GRF).
#
# This is a Windows-native port of the *formulation* in Nimble's
# ``dart/biomechanics/DynamicsFitter.cpp`` (``ResidualForceHelper`` + the linear
# mass/inertia identification), computing the articulated-dynamics terms with the
# **MuJoCo solver** (``mujoco.mj_inverse`` / ``mj_applyFT``) on the M3-exported MJCF
# instead of porting DART's Featherstone. The exported MJCF is the same model that
# drives the Newton ``SolverMuJoCo`` for the contact-research side (M5+), so the
# dynamics used here are exactly the dynamics the sim will use.
#
# Nimble's core object (``ResidualForceHelper``, DynamicsFitter.cpp:57) computes, per
# frame, the inverse-dynamics generalized force
#
#     tau = M(q) q̈ + C(q, q̇) - Σ_i J_i(q)ᵀ W_i          (DynamicsFitter.cpp:69-111)
#
# where ``W_i`` is a measured external (ground-reaction) wrench on force-body ``i``,
# and calls the first 6 entries (the un-actuated floating-base root) the *residual*
# ("hand of god" force). A dynamically consistent trajectory + GRF drives that root
# residual to zero. Nimble then solves a **linear** least-squares over the per-body
# inertial parameters (mass / COM / inertia) to minimize the stacked root residual,
# regularized toward the anthropometric initial guess.
#
# MuJoCo mapping (all validated against ``mj_inverse`` to ~1e-13, see
# ``tests/test_dynamics_fitter.py``):
#   - ``M q̈ + C`` (with the coupled-knee ``<equality>`` constraint forces correctly
#     accounted for)         == ``mj_inverse`` -> ``d.qfrc_inverse``.
#   - ``Σ J_iᵀ W_i``          == ``mj_applyFT(force, torque, point, body)`` accumulated.
#   - residual                == ``qfrc_inverse[:6] - Σ Fs[:6]``.
#   - ``d(residual)/d(mass_b)`` is exactly linear -> a finite-difference regressor over
#     ``m.body_mass`` is exact, giving Nimble's linear mass identification directly on
#     the MuJoCo model.
#
# Scope (feet / lower-body first, per project direction): this delivers the residual
# machinery, the GRF/COP -> contact-wrench adapter (split-belt treadmill: right belt =
# right foot, left belt = left foot), and the linear **per-segment mass** identification.
# COM/full-inertia identification and Warp-batched multi-frame residuals are the next
# increment (see ``docs/21_nimble_source_map.md``).

"""DynamicsFitter (M2e): GRF residual + mass identification on the MuJoCo solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Contact wrench (measured ground reaction on a body), world frame, SI units
# ---------------------------------------------------------------------------


@dataclass
class Contact:
    """A measured external wrench applied to ``body`` at a world ``point``.

    ``force`` is the ground-reaction force **on the subject** (N, world frame),
    ``torque`` the free moment (N*m, world frame; for a 6-axis plate this is
    ``[0, 0, free_moment_z]``), and ``point`` the centre of pressure (m, world).
    A swing (unloaded) foot is represented by a zero-force contact or simply omitted.
    """

    body: str
    force: np.ndarray  # (3,)
    torque: np.ndarray  # (3,)
    point: np.ndarray  # (3,)


@dataclass
class ResidualReport:
    """Per-frame root-residual diagnostics over a trajectory."""

    force_residual: np.ndarray  # (F, 3) linear root residual (N)
    torque_residual: np.ndarray  # (F, 3) angular root residual (N*m)
    force_norm: np.ndarray  # (F,)
    torque_norm: np.ndarray  # (F,)

    @property
    def mean_force_norm(self) -> float:
        return float(np.mean(self.force_norm)) if self.force_norm.size else 0.0

    @property
    def mean_torque_norm(self) -> float:
        return float(np.mean(self.torque_norm)) if self.torque_norm.size else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "frames": int(self.force_norm.size),
            "mean_force_residual_N": self.mean_force_norm,
            "max_force_residual_N": (
                float(np.max(self.force_norm)) if self.force_norm.size else 0.0
            ),
            "mean_torque_residual_Nm": self.mean_torque_norm,
            "max_torque_residual_Nm": (
                float(np.max(self.torque_norm)) if self.torque_norm.size else 0.0
            ),
        }


@dataclass
class MassIdentificationResult:
    body_names: List[str]
    initial_mass: np.ndarray  # (nb,)
    fitted_mass: np.ndarray  # (nb,)
    residual_before: ResidualReport
    residual_after: ResidualReport

    def summary(self) -> Dict[str, object]:
        return {
            "total_mass_before_kg": float(np.sum(self.initial_mass)),
            "total_mass_after_kg": float(np.sum(self.fitted_mass)),
            "max_abs_mass_change_kg": float(
                np.max(np.abs(self.fitted_mass - self.initial_mass))
            ),
            "residual_before": self.residual_before.summary(),
            "residual_after": self.residual_after.summary(),
        }


# ---------------------------------------------------------------------------
# ResidualForceHelper (MuJoCo-backed)
# ---------------------------------------------------------------------------


class ResidualHelper:
    """Port of Nimble ``ResidualForceHelper`` backed by ``mujoco.mj_inverse``.

    Wraps a MuJoCo model built from the M3-exported MJCF. All dynamics quantities are
    evaluated on that model, so they match the model the Newton ``SolverMuJoCo`` will
    integrate for contact research.
    """

    def __init__(self, mjcf_xml: str):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self._body_id: Dict[str, int] = {}
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name:
                self._body_id[name] = b

    # -- ids ---------------------------------------------------------------
    def body_id(self, name: str) -> int:
        return self._body_id[name]

    def has_body(self, name: str) -> bool:
        return name in self._body_id

    # -- core --------------------------------------------------------------
    def inverse_dynamics(
        self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray
    ) -> np.ndarray:
        """``qfrc_inverse = M q̈ + C`` (equality-constraint forces accounted for)."""
        d = self.data
        d.qpos[:] = np.asarray(qpos, dtype=np.float64)
        d.qvel[:] = np.asarray(qvel, dtype=np.float64)
        d.qacc[:] = np.asarray(qacc, dtype=np.float64)
        self._mj.mj_inverse(self.model, d)
        return d.qfrc_inverse.copy()

    def contact_generalized_force(
        self, contacts: Sequence[Contact], qpos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """``Σ_i J_iᵀ W_i`` for measured wrenches (``mj_applyFT`` accumulated).

        Requires body kinematics for the current pose. Pass ``qpos`` to (re)evaluate
        kinematics; otherwise the pose from the last :meth:`inverse_dynamics` /
        :meth:`root_residual` call is reused (the common path in :meth:`root_residual`).
        """
        d = self.data
        if qpos is not None:
            d.qpos[:] = np.asarray(qpos, dtype=np.float64)
            self._mj.mj_kinematics(self.model, d)
            self._mj.mj_comPos(self.model, d)
        Fs = np.zeros(self.nv, dtype=np.float64)
        for c in contacts:
            self._mj.mj_applyFT(
                self.model,
                d,
                np.asarray(c.force, dtype=np.float64),
                np.asarray(c.torque, dtype=np.float64),
                np.asarray(c.point, dtype=np.float64),
                self._body_id[c.body],
                Fs,
            )
        return Fs

    def root_residual(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        contacts: Sequence[Contact],
    ) -> np.ndarray:
        """Floating-base root residual ``(6,)`` = ``qfrc_inverse[:6] - Fs[:6]``.

        First 3 = linear (force) residual, last 3 = angular (torque) residual, in the
        MuJoCo free-joint convention (translational DOFs world, rotational DOFs local).
        """
        qfrc = self.inverse_dynamics(qpos, qvel, qacc)
        # kinematics for the same pose are populated by mj_inverse; reuse them
        Fs = self.contact_generalized_force(contacts, qpos=None)
        return qfrc[:6] - Fs[:6]

    def residual_report(
        self,
        qpos_t: np.ndarray,
        qvel_t: np.ndarray,
        qacc_t: np.ndarray,
        contacts_t: Sequence[Sequence[Contact]],
    ) -> ResidualReport:
        """Root residual over a whole trajectory."""
        F = qpos_t.shape[0]
        res = np.zeros((F, 6), dtype=np.float64)
        for f in range(F):
            res[f] = self.root_residual(
                qpos_t[f], qvel_t[f], qacc_t[f], contacts_t[f]
            )
        return ResidualReport(
            force_residual=res[:, :3],
            torque_residual=res[:, 3:],
            force_norm=np.linalg.norm(res[:, :3], axis=1),
            torque_norm=np.linalg.norm(res[:, 3:], axis=1),
        )

    # -- mass parameter access --------------------------------------------
    def get_masses(self, body_names: Sequence[str]) -> np.ndarray:
        return np.array(
            [self.model.body_mass[self._body_id[n]] for n in body_names],
            dtype=np.float64,
        )

    def set_masses(self, body_names: Sequence[str], masses: np.ndarray) -> None:
        for n, mval in zip(body_names, np.asarray(masses, dtype=np.float64)):
            self.model.body_mass[self._body_id[n]] = float(mval)

    # -- full inertial parameters (10 per body: m, m·c, I about body origin) ----
    def get_inertial_params(self, body_names: Sequence[str]) -> np.ndarray:
        """Linear inertial parameters ``(nb, 10)`` = ``[m, m·cx, m·cy, m·cz, Ixx, Iyy,
        Izz, Ixy, Ixz, Iyz]`` with the inertia taken about the **body frame origin**
        (the parameterization in which the inverse-dynamics residual is linear)."""
        return np.stack(
            [_get_body_phi(self._mj, self.model, self._body_id[n]) for n in body_names]
        )

    def set_inertial_params(
        self, body_names: Sequence[str], phi: np.ndarray
    ) -> None:
        phi = np.asarray(phi, dtype=np.float64).reshape(len(body_names), 10)
        for n, p in zip(body_names, phi):
            _set_body_phi(self._mj, self.model, self._body_id[n], p)


# ---------------------------------------------------------------------------
# Linear inertial-parameter <-> MuJoCo body-inertial mapping
# ---------------------------------------------------------------------------


def _sym_from6(v: np.ndarray) -> np.ndarray:
    """``[xx, yy, zz, xy, xz, yz]`` -> symmetric 3x3."""
    return np.array(
        [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]],
        dtype=np.float64,
    )


def _six_from_sym(M: np.ndarray) -> np.ndarray:
    return np.array(
        [M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]], dtype=np.float64
    )


def _get_body_phi(mj, model, bid: int) -> np.ndarray:
    """MuJoCo body inertial (mass, ipos=COM, principal inertia+iquat) -> ``phi[10]``."""
    mass = float(model.body_mass[bid])
    c = np.array(model.body_ipos[bid], dtype=np.float64)
    diag = np.array(model.body_inertia[bid], dtype=np.float64)
    q = np.array(model.body_iquat[bid], dtype=np.float64)
    R = np.zeros(9, dtype=np.float64)
    mj.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    I_com = R @ np.diag(diag) @ R.T
    # parallel axis: I_origin = I_com + m (||c||^2 I - c c^T)
    I_origin = I_com + mass * (float(c @ c) * np.eye(3) - np.outer(c, c))
    return np.concatenate([[mass], mass * c, _six_from_sym(I_origin)])


def _set_body_phi(mj, model, bid: int, phi: np.ndarray) -> None:
    """``phi[10]`` -> MuJoCo body inertial (diagonalized principal frame)."""
    mass = float(phi[0])
    c = phi[1:4] / mass
    I_origin = _sym_from6(phi[4:10])
    I_com = I_origin - mass * (float(c @ c) * np.eye(3) - np.outer(c, c))
    w, V = np.linalg.eigh(I_com)  # ascending eigenvalues, orthonormal columns
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]  # ensure a proper rotation
    q = np.zeros(4, dtype=np.float64)
    mj.mju_mat2Quat(q, V.reshape(9))
    model.body_mass[bid] = mass
    model.body_ipos[bid] = c
    model.body_inertia[bid] = w
    model.body_iquat[bid] = q


def inertia_is_physical(phi: np.ndarray, min_mass: float = 1e-3) -> bool:
    """Positive mass and a positive-definite COM inertia obeying the triangle ineq."""
    mass = float(phi[0])
    if mass < min_mass:
        return False
    c = phi[1:4] / mass
    I_com = _sym_from6(phi[4:10]) - mass * (
        float(c @ c) * np.eye(3) - np.outer(c, c)
    )
    w = np.linalg.eigvalsh(I_com)
    if np.any(w <= 0):
        return False
    # principal moments must satisfy the triangle inequalities
    a, b, cc = np.sort(w)
    return (a + b) >= cc - 1e-9


def project_physical(
    phi: np.ndarray, min_mass: float = 1e-3, eps: float = 1e-8
) -> np.ndarray:
    """Project a 10-param block onto the physically valid inertia cone.

    Clamps mass to ``>= min_mass`` and the COM principal moments to positive values
    obeying the triangle inequality (keeping the COM and principal axes). Used by the
    identification line search so an aggressive step stays a valid rigid body.
    """
    phi = np.asarray(phi, dtype=np.float64).copy()
    m = max(float(phi[0]), min_mass)
    phi[0] = m
    c = phi[1:4] / m
    shift = m * (float(c @ c) * np.eye(3) - np.outer(c, c))
    I_com = _sym_from6(phi[4:10]) - shift
    w, V = np.linalg.eigh(I_com)
    w = np.clip(w, eps, None)
    order = np.argsort(w)
    ws = w[order]
    if ws[2] > ws[0] + ws[1]:  # enforce triangle inequality
        ws[2] = ws[0] + ws[1]
    w[order] = ws
    I_com_p = V @ np.diag(w) @ V.T
    phi[4:10] = _six_from_sym(I_com_p + shift)
    return phi


# ---------------------------------------------------------------------------
# GPU-batched residual over frames (mujoco_warp / Newton MuJoCo)
# ---------------------------------------------------------------------------


class BatchedResidualHelper:
    """Frame-batched ``ResidualForceHelper`` on ``mujoco_warp`` (the Newton MuJoCo GPU).

    Evaluates the inverse-dynamics residual for a whole ``F``-frame trajectory in one
    batched ``mujoco_warp.inverse`` launch (one warp world per frame) instead of a
    per-frame Python loop over ``mj_inverse``. Contact wrenches map to generalized
    forces via ``mujoco_warp.xfrc_accumulate`` (measured GRF at the COP is converted to
    the equivalent wrench at the body COM ``xipos``). Validated against the CPU
    :class:`ResidualHelper` to float32 precision (~1e-4).

    This is the ``use-Newton-as-much-as-possible`` accelerator for the mass
    identification: the linear mass regressor needs ``nbody + 1`` residual passes, each
    now a single GPU launch over all frames. Numerics are float32 (GPU); the CPU helper
    remains the authoritative float64 reference.
    """

    def __init__(self, mjcf_xml: str, nframes: int):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        self._mj = mujoco
        self._mjw = mjw
        self._wp = wp
        self.nframes = int(nframes)
        self.mjm = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.model = mjw.put_model(self.mjm)
        self.data = mjw.make_data(self.mjm, nworld=self.nframes)
        self.nv = int(self.mjm.nv)
        self.nq = int(self.mjm.nq)
        self.nbody = int(self.mjm.nbody)
        self._body_id: Dict[str, int] = {}
        for b in range(self.mjm.nbody):
            name = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, b)
            if name:
                self._body_id[name] = b
        self._qpos_set = False

    def has_body(self, name: str) -> bool:
        return name in self._body_id

    def set_kinematics(
        self, qpos_t: np.ndarray, qvel_t: np.ndarray, qacc_t: np.ndarray
    ) -> None:
        if qpos_t.shape[0] != self.nframes:
            raise ValueError(
                f"expected {self.nframes} frames, got {qpos_t.shape[0]}"
            )
        self.data.qpos.assign(np.ascontiguousarray(qpos_t, dtype=np.float32))
        self.data.qvel.assign(np.ascontiguousarray(qvel_t, dtype=np.float32))
        self.data.qacc.assign(np.ascontiguousarray(qacc_t, dtype=np.float32))
        self._qpos_set = True

    def inverse_dynamics(self) -> np.ndarray:
        """Batched ``qfrc_inverse`` ``(F, nv)`` for the currently-set kinematics."""
        self._mjw.inverse(self.model, self.data)
        return self.data.qfrc_inverse.numpy().copy()

    def contact_generalized_force(
        self, contacts_t: Sequence[Sequence[Contact]]
    ) -> np.ndarray:
        """Batched ``Σ J_iᵀ W_i`` ``(F, nv)`` (mass-independent; compute once).

        Requires kinematics to have been evaluated (call :meth:`inverse_dynamics`
        first) so body COM positions ``xipos`` are populated.
        """
        xipos = self.data.xipos.numpy()  # (F, nbody, 3)
        xfrc = np.zeros((self.nframes, self.nbody, 6), dtype=np.float32)
        for f, frame in enumerate(contacts_t):
            for c in frame:
                bid = self._body_id[c.body]
                force = np.asarray(c.force, dtype=np.float64)
                torque = np.asarray(c.torque, dtype=np.float64)
                point = np.asarray(c.point, dtype=np.float64)
                # equivalent wrench at the body COM (xfrc_applied acts at xipos)
                torque_com = torque + np.cross(point - xipos[f, bid], force)
                xfrc[f, bid, :3] += force.astype(np.float32)
                xfrc[f, bid, 3:] += torque_com.astype(np.float32)
        self.data.xfrc_applied.assign(xfrc)
        qfrc = self._wp.zeros((self.nframes, self.nv), dtype=self._wp.float32)
        self._mjw.xfrc_accumulate(self.model, self.data, qfrc)
        return qfrc.numpy().copy()

    def root_residual_batch(
        self,
        qpos_t: np.ndarray,
        qvel_t: np.ndarray,
        qacc_t: np.ndarray,
        contacts_t: Sequence[Sequence[Contact]],
    ) -> np.ndarray:
        """Root residual ``(F, 6)`` = ``qfrc_inverse[:, :6] - Fs[:, :6]``."""
        self.set_kinematics(qpos_t, qvel_t, qacc_t)
        qfrc = self.inverse_dynamics()
        Fs = self.contact_generalized_force(contacts_t)
        return qfrc[:, :6] - Fs[:, :6]

    def residual_report(
        self,
        qpos_t: np.ndarray,
        qvel_t: np.ndarray,
        qacc_t: np.ndarray,
        contacts_t: Sequence[Sequence[Contact]],
    ) -> ResidualReport:
        res = self.root_residual_batch(qpos_t, qvel_t, qacc_t, contacts_t)
        return ResidualReport(
            force_residual=res[:, :3],
            torque_residual=res[:, 3:],
            force_norm=np.linalg.norm(res[:, :3], axis=1),
            torque_norm=np.linalg.norm(res[:, 3:], axis=1),
        )

    # -- mass parameter access (model.body_mass is (1, nbody) on device) ----
    def get_masses(self, body_names: Sequence[str]) -> np.ndarray:
        bm = self.model.body_mass.numpy()
        return np.array(
            [bm[0, self._body_id[n]] for n in body_names], dtype=np.float64
        )

    def set_masses(self, body_names: Sequence[str], masses: np.ndarray) -> None:
        bm = self.model.body_mass.numpy()
        for n, mval in zip(body_names, np.asarray(masses, dtype=np.float64)):
            bm[0, self._body_id[n]] = float(mval)
        self.model.body_mass.assign(bm)


def identify_masses_batched(
    helper: "BatchedResidualHelper",
    qpos_t: np.ndarray,
    qvel_t: np.ndarray,
    qacc_t: np.ndarray,
    contacts_t: Sequence[Sequence[Contact]],
    body_names: Sequence[str],
    reg: float = 1e-2,
    min_mass: float = 1e-3,
    fd_step: float = 1.0,
) -> MassIdentificationResult:
    """GPU-batched linear per-segment mass identification (see :func:`identify_masses`).

    Identical formulation to :func:`identify_masses` but every residual pass is a single
    ``mujoco_warp`` launch over all frames, and the mass-independent contact force is
    computed once and reused across the ``nbody + 1`` regressor evaluations.
    """
    qpos_t = np.ascontiguousarray(qpos_t, dtype=np.float32)
    qvel_t = np.ascontiguousarray(qvel_t, dtype=np.float32)
    qacc_t = np.ascontiguousarray(qacc_t, dtype=np.float32)
    F = qpos_t.shape[0]
    nb = len(body_names)

    helper.set_kinematics(qpos_t, qvel_t, qacc_t)
    m0 = helper.get_masses(body_names)

    def id6() -> np.ndarray:
        return helper.inverse_dynamics()[:, :6]

    # contact force is mass-independent -> compute once (kinematics already set)
    helper.inverse_dynamics()  # populate xipos
    Fs6 = helper.contact_generalized_force(contacts_t)[:, :6]

    def stacked_residual() -> np.ndarray:
        return (id6() - Fs6).reshape(-1)

    r0 = stacked_residual()
    report_before = ResidualReport(
        force_residual=r0.reshape(F, 6)[:, :3],
        torque_residual=r0.reshape(F, 6)[:, 3:],
        force_norm=np.linalg.norm(r0.reshape(F, 6)[:, :3], axis=1),
        torque_norm=np.linalg.norm(r0.reshape(F, 6)[:, 3:], axis=1),
    )

    A = np.zeros((F * 6, nb), dtype=np.float64)
    for b in range(nb):
        helper.set_masses([body_names[b]], m0[b : b + 1] + fd_step)
        A[:, b] = (stacked_residual() - r0) / fd_step
        helper.set_masses([body_names[b]], m0[b : b + 1])

    aug_A = np.vstack([A, np.sqrt(reg) * np.eye(nb)])
    aug_b = np.concatenate([-r0, np.zeros(nb)])
    dm, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)

    e0 = float(r0 @ r0)
    m_fit = m0.copy()
    alpha = 1.0
    for _ in range(20):
        cand = m0 + alpha * dm
        if np.all(cand >= min_mass):
            helper.set_masses(body_names, cand)
            r = stacked_residual()
            if float(r @ r) <= e0 + 1e-6:
                m_fit = cand
                break
        alpha *= 0.5
    helper.set_masses(body_names, m_fit)

    rA = stacked_residual().reshape(F, 6)
    report_after = ResidualReport(
        force_residual=rA[:, :3],
        torque_residual=rA[:, 3:],
        force_norm=np.linalg.norm(rA[:, :3], axis=1),
        torque_norm=np.linalg.norm(rA[:, 3:], axis=1),
    )
    return MassIdentificationResult(
        body_names=list(body_names),
        initial_mass=m0,
        fitted_mass=m_fit,
        residual_before=report_before,
        residual_after=report_after,
    )


# ---------------------------------------------------------------------------
# Kinematics: velocities/accelerations from a pose trajectory (free-joint aware)
# ---------------------------------------------------------------------------


def velocities_from_positions(
    helper: ResidualHelper, qpos_t: np.ndarray, dt: float
) -> np.ndarray:
    """Central finite-difference velocities in the tangent space ``(F, nv)``.

    Uses ``mj_differentiatePos`` so the free-joint quaternion is differentiated on the
    manifold (not component-wise), matching how ``mj_inverse`` expects ``qvel``.
    """
    mj = helper._mj
    m = helper.model
    qpos_t = np.asarray(qpos_t, dtype=np.float64)
    F = qpos_t.shape[0]
    qvel = np.zeros((F, helper.nv), dtype=np.float64)
    if F < 2:
        return qvel
    buf = np.zeros(helper.nv, dtype=np.float64)
    for f in range(F):
        if f == 0:
            mj.mj_differentiatePos(m, buf, dt, qpos_t[0], qpos_t[1])
        elif f == F - 1:
            mj.mj_differentiatePos(m, buf, dt, qpos_t[F - 2], qpos_t[F - 1])
        else:
            mj.mj_differentiatePos(m, buf, 2.0 * dt, qpos_t[f - 1], qpos_t[f + 1])
        qvel[f] = buf
    return qvel


def accelerations_from_velocities(qvel_t: np.ndarray, dt: float) -> np.ndarray:
    """Central finite-difference accelerations ``(F, nv)`` (qvel is already tangent)."""
    qvel_t = np.asarray(qvel_t, dtype=np.float64)
    F = qvel_t.shape[0]
    a = np.zeros_like(qvel_t)
    if F < 2:
        return a
    a[1:-1] = (qvel_t[2:] - qvel_t[:-2]) / (2.0 * dt)
    a[0] = (qvel_t[1] - qvel_t[0]) / dt
    a[-1] = (qvel_t[-1] - qvel_t[-2]) / dt
    return a


# ---------------------------------------------------------------------------
# GRF/COP -> Contact adapter (split-belt instrumented treadmill)
# ---------------------------------------------------------------------------


def contacts_from_grf(
    force_world: np.ndarray,
    cop_world: np.ndarray,
    body: str,
    free_moment_z: Optional[np.ndarray] = None,
    fz_threshold: float = 1e-6,
) -> List[List[Contact]]:
    """Build per-frame single-foot contacts from resampled GRF/COP arrays.

    Args:
        force_world: ``(F, 3)`` ground-reaction force on the subject (N, world).
        cop_world: ``(F, 3)`` centre of pressure (m, world). ``NaN`` marks swing.
        body: MuJoCo body the wrench acts on (e.g. ``"calcn_r"``).
        free_moment_z: ``(F,)`` vertical free moment (N*m); zeros if omitted.
        fz_threshold: below this vertical load the frame is treated as swing (no
            contact) to avoid injecting a wrench at a ``NaN`` COP.

    Returns a length-``F`` list; each entry is a (possibly empty) list of contacts.
    """
    force_world = np.asarray(force_world, dtype=np.float64)
    cop_world = np.asarray(cop_world, dtype=np.float64)
    F = force_world.shape[0]
    fmz = (
        np.zeros(F, dtype=np.float64)
        if free_moment_z is None
        else np.asarray(free_moment_z, dtype=np.float64)
    )
    out: List[List[Contact]] = []
    for f in range(F):
        fz = force_world[f, 2]
        if not np.isfinite(cop_world[f]).all() or abs(fz) < fz_threshold:
            out.append([])
            continue
        out.append(
            [
                Contact(
                    body=body,
                    force=force_world[f].copy(),
                    torque=np.array([0.0, 0.0, fmz[f]], dtype=np.float64),
                    point=cop_world[f].copy(),
                )
            ]
        )
    return out


def merge_contacts(*per_foot: Sequence[Sequence[Contact]]) -> List[List[Contact]]:
    """Merge several per-foot per-frame contact lists into one per-frame list."""
    if not per_foot:
        return []
    F = len(per_foot[0])
    for pf in per_foot:
        if len(pf) != F:
            raise ValueError("all feet must have the same number of frames")
    merged: List[List[Contact]] = []
    for f in range(F):
        frame: List[Contact] = []
        for pf in per_foot:
            frame.extend(pf[f])
        merged.append(frame)
    return merged


def resample_to_frames(
    signal: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray
) -> np.ndarray:
    """Linearly resample a ``(N, ...)`` signal from ``src_times`` to ``dst_times``.

    Handles the variable-rate treadmill/analog base by interpolating each component;
    ``NaN`` samples (swing COP) are preserved as ``NaN`` in any interval they touch.
    """
    signal = np.asarray(signal, dtype=np.float64)
    src_times = np.asarray(src_times, dtype=np.float64)
    dst_times = np.asarray(dst_times, dtype=np.float64)
    flat = signal.reshape(signal.shape[0], -1)
    out = np.empty((dst_times.shape[0], flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst_times, src_times, flat[:, c])
    return out.reshape(dst_times.shape[0], *signal.shape[1:])


# ---------------------------------------------------------------------------
# Linear per-segment mass identification (Nimble's linear inertial solve, mass block)
# ---------------------------------------------------------------------------


def identify_masses(
    helper: ResidualHelper,
    qpos_t: np.ndarray,
    qvel_t: np.ndarray,
    qacc_t: np.ndarray,
    contacts_t: Sequence[Sequence[Contact]],
    body_names: Sequence[str],
    reg: float = 1e-2,
    min_mass: float = 1e-3,
    fd_step: float = 1.0,
) -> MassIdentificationResult:
    """Linear least-squares refinement of per-segment masses to null the root residual.

    Ports the mass block of Nimble's linear inertial identification. The stacked root
    residual ``r(m)`` is **exactly linear** in the per-body masses (validated), so the
    finite-difference regressor ``A[:, b] = ∂r/∂m_b`` is exact. We solve

        min_Δm  ‖A Δm + r₀‖² + reg ‖Δm‖²         (Tikhonov toward the initial masses)

    then take the step under a **backtracking line search** that keeps every mass
    ≥ ``min_mass`` and never increases the root-residual energy (damped Gauss-Newton
    with positivity). Dummy chain bodies are never varied — only ``body_names`` (the
    anatomical segments) are.
    """
    qpos_t = np.asarray(qpos_t, dtype=np.float64)
    qvel_t = np.asarray(qvel_t, dtype=np.float64)
    qacc_t = np.asarray(qacc_t, dtype=np.float64)
    F = qpos_t.shape[0]
    nb = len(body_names)

    report_before = helper.residual_report(qpos_t, qvel_t, qacc_t, contacts_t)

    m0 = helper.get_masses(body_names)

    def stacked_residual() -> np.ndarray:
        r = np.zeros(F * 6, dtype=np.float64)
        for f in range(F):
            r[f * 6 : f * 6 + 6] = helper.root_residual(
                qpos_t[f], qvel_t[f], qacc_t[f], contacts_t[f]
            )
        return r

    r0 = stacked_residual()

    # exact linear regressor via one +fd_step perturbation per body
    A = np.zeros((F * 6, nb), dtype=np.float64)
    for b in range(nb):
        helper.set_masses([body_names[b]], m0[b : b + 1] + fd_step)
        rb = stacked_residual()
        A[:, b] = (rb - r0) / fd_step
        helper.set_masses([body_names[b]], m0[b : b + 1])  # restore

    # Regularized least squares min ‖A Δm + r0‖² + reg‖Δm‖² solved as a stable
    # augmented lstsq (Tikhonov), not normal equations, so an ill-conditioned
    # regressor cannot blow up the mass update.
    aug_A = np.vstack([A, np.sqrt(reg) * np.eye(nb)])
    aug_b = np.concatenate([-r0, np.zeros(nb)])
    dm, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)

    # Backtracking line search: accept the largest step that keeps every mass
    # positive and does not increase the root-residual energy. Guarantees a
    # physically valid, non-worsening update (damped Gauss-Newton).
    e0 = float(r0 @ r0)
    m_fit = m0.copy()
    alpha = 1.0
    for _ in range(20):
        cand = m0 + alpha * dm
        if np.all(cand >= min_mass):
            helper.set_masses(body_names, cand)
            r = stacked_residual()
            if float(r @ r) <= e0 + 1e-9:
                m_fit = cand
                break
        alpha *= 0.5
    helper.set_masses(body_names, m_fit)
    report_after = helper.residual_report(qpos_t, qvel_t, qacc_t, contacts_t)

    return MassIdentificationResult(
        body_names=list(body_names),
        initial_mass=m0,
        fitted_mass=m_fit,
        residual_before=report_before,
        residual_after=report_after,
    )


# ---------------------------------------------------------------------------
# Full inertial identification (mass + COM + inertia; Nimble's linear solve)
# ---------------------------------------------------------------------------


@dataclass
class InertialIdentificationResult:
    """Result of the 10-param-per-body inertial identification."""

    body_names: List[str]
    initial_params: np.ndarray  # (nb, 10)
    fitted_params: np.ndarray  # (nb, 10)
    residual_before: ResidualReport
    residual_after: ResidualReport

    @property
    def initial_mass(self) -> np.ndarray:
        return self.initial_params[:, 0]

    @property
    def fitted_mass(self) -> np.ndarray:
        return self.fitted_params[:, 0]

    def com(self, params: np.ndarray) -> np.ndarray:
        """Per-body COM ``(nb, 3)`` from a param block (``m·c / m``)."""
        return params[:, 1:4] / params[:, 0:1]

    def summary(self) -> Dict[str, object]:
        d_com = self.com(self.fitted_params) - self.com(self.initial_params)
        return {
            "total_mass_before_kg": float(np.sum(self.initial_mass)),
            "total_mass_after_kg": float(np.sum(self.fitted_mass)),
            "max_abs_mass_change_kg": float(
                np.max(np.abs(self.fitted_mass - self.initial_mass))
            ),
            "max_abs_com_shift_m": float(np.max(np.abs(d_com))),
            "residual_before": self.residual_before.summary(),
            "residual_after": self.residual_after.summary(),
        }


# characteristic parameter scales for relative Tikhonov (mass, m·c, inertia)
_PARAM_FLOOR = np.array(
    [1.0, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64
)


def identify_inertial_params(
    helper: ResidualHelper,
    qpos_t: np.ndarray,
    qvel_t: np.ndarray,
    qacc_t: np.ndarray,
    contacts_t: Sequence[Sequence[Contact]],
    body_names: Sequence[str],
    reg: float = 1.0,
    fd_rel: float = 1e-3,
) -> InertialIdentificationResult:
    """Full 10-param-per-body inertial identification (mass + COM + inertia).

    Ports Nimble's linear inertial identification. The stacked root residual is
    **exactly linear** in the 10 inertial parameters per body
    (``[m, m·c, I_origin]``; validated to ~1e-13), so the finite-difference regressor
    ``A`` is exact even though each parameter is fed to MuJoCo through the nonlinear
    principal-axis (``ipos``/``inertia``/``iquat``) reparameterization.

    Solves ``min_Δφ  ‖A Δφ + r₀‖² + ‖W Δφ‖²`` with a **relative** Tikhonov
    ``W = sqrt(reg)/scale`` toward the anthropometric initial params (``scale`` = per-
    parameter magnitude floored by type), then takes the step under a backtracking
    line search that keeps every body's inertia **physically valid**
    (:func:`inertia_is_physical`) and never increases the residual energy. The root
    residual weakly observes per-segment params, so ``reg`` should stay > 0 for real
    data; use a tiny ``reg`` only for well-posed synthetic recovery.
    """
    qpos_t = np.asarray(qpos_t, dtype=np.float64)
    qvel_t = np.asarray(qvel_t, dtype=np.float64)
    qacc_t = np.asarray(qacc_t, dtype=np.float64)
    F = qpos_t.shape[0]
    nb = len(body_names)
    P = 10 * nb

    report_before = helper.residual_report(qpos_t, qvel_t, qacc_t, contacts_t)

    phi0 = helper.get_inertial_params(body_names)  # (nb, 10)
    flat0 = phi0.reshape(-1).copy()
    scale = np.maximum(np.abs(phi0), _PARAM_FLOOR[None, :]).reshape(-1)

    def apply(flat: np.ndarray) -> None:
        helper.set_inertial_params(body_names, flat.reshape(nb, 10))

    def stacked_residual() -> np.ndarray:
        r = np.zeros(F * 6, dtype=np.float64)
        for f in range(F):
            r[f * 6 : f * 6 + 6] = helper.root_residual(
                qpos_t[f], qvel_t[f], qacc_t[f], contacts_t[f]
            )
        return r

    r0 = stacked_residual()

    # exact linear regressor: one perturbation per parameter (relative step)
    A = np.zeros((F * 6, P), dtype=np.float64)
    for k in range(P):
        step = fd_rel * scale[k]
        pert = flat0.copy()
        pert[k] += step
        apply(pert)
        A[:, k] = (stacked_residual() - r0) / step
    apply(flat0)  # restore

    # relative Tikhonov toward the initial (anthropometric) params
    W = np.sqrt(reg) / scale
    aug_A = np.vstack([A, np.diag(W)])
    aug_b = np.concatenate([-r0, np.zeros(P)])
    dphi, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)

    e0 = float(r0 @ r0)
    flat_fit = flat0.copy()
    alpha = 1.0
    for _ in range(30):
        blocks = (flat0 + alpha * dphi).reshape(nb, 10)
        # project each body back onto the physical inertia cone, then test
        blocks = np.stack([project_physical(blocks[i]) for i in range(nb)])
        cand = blocks.reshape(-1)
        apply(cand)
        r = stacked_residual()
        if float(r @ r) <= e0 + 1e-9:
            flat_fit = cand
            break
        alpha *= 0.5
    apply(flat_fit)
    report_after = helper.residual_report(qpos_t, qvel_t, qacc_t, contacts_t)

    return InertialIdentificationResult(
        body_names=list(body_names),
        initial_params=flat0.reshape(nb, 10),
        fitted_params=flat_fit.reshape(nb, 10),
        residual_before=report_before,
        residual_after=report_after,
    )


# ---------------------------------------------------------------------------
# Kinematic RRA (Residual Reduction Algorithm) — adjust the root trajectory
# ---------------------------------------------------------------------------


@dataclass
class RRAResult:
    """Result of a kinematic residual-reduction pass."""

    qpos_t: np.ndarray  # (F, nq) adjusted pose trajectory
    qvel_t: np.ndarray  # (F, nv) re-derived velocities
    qacc_t: np.ndarray  # (F, nv) re-derived accelerations
    root_shift: np.ndarray  # (F, 3) net root-translation change vs the input
    residual_before: ResidualReport
    residual_after: ResidualReport
    objective_history: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {
            "max_abs_root_shift_m": float(np.max(np.abs(self.root_shift))),
            "mean_root_shift_m": float(np.mean(np.linalg.norm(self.root_shift, axis=1))),
            "residual_before": self.residual_before.summary(),
            "residual_after": self.residual_after.summary(),
        }


def _second_difference_matrix(F: int) -> np.ndarray:
    """``(F, F)`` second-difference operator (free ends) for a scalar time series."""
    D = np.zeros((F, F), dtype=np.float64)
    for f in range(1, F - 1):
        D[f, f - 1] = 1.0
        D[f, f] = -2.0
        D[f, f + 1] = 1.0
    return D


def rra_kinematics(
    helper: ResidualHelper,
    qpos_t: np.ndarray,
    contacts_t: Sequence[Sequence[Contact]],
    dt: float,
    *,
    iters: int = 8,
    track_weight: float = 1e-2,
    smooth_weight: float = 1e-1,
    fd_step: float = 1e-4,
    damping: float = 1e-8,
) -> RRAResult:
    """Reduce the floating-base root residual by adjusting the root translation.

    Port of the *kinematic* half of Nimble/OpenSim's Residual Reduction Algorithm. The
    root residual ``qfrc_inverse[:6] - Σ Jᵀ W`` is the "hand of god" force needed to make
    the reconstructed kinematics consistent with the **measured** ground reaction. RRA
    removes it by nudging the model kinematics (with the measured GRF held fixed) instead
    of by faking a root actuator. Here we optimize a smooth correction to the root
    *translation* trajectory (the dominant driver of the linear force residual), which
    directly changes the COM acceleration used by inverse dynamics.

    The residual at frame ``f`` depends on ``qpos`` at frames ``f-2 .. f+2`` (through the
    central finite-difference ``qvel``/``qacc``), so the Gauss-Newton Jacobian is banded;
    we build it by finite differences over the three root-translation ``qpos`` slots at
    each frame and solve a regularized least squares

        min_Δ  ‖J Δ + r‖² + track ‖(p - p0) + Δ‖² + smooth ‖D₂ (p + Δ)‖²

    where ``p`` is the current root translation, ``p0`` the input translation, and ``D₂``
    the second-difference (smoothness) operator. A backtracking line search accepts the
    largest step that decreases the total objective. Velocities/accelerations are
    re-derived on the manifold each outer iteration so the pass stays self-consistent.
    """
    qpos_t = np.asarray(qpos_t, dtype=np.float64).copy()
    F, nq = qpos_t.shape
    trans = [0, 1, 2]  # free-joint translation slots in qpos
    orig_trans = qpos_t[:, trans].copy()

    def kin(qp):
        qv = velocities_from_positions(helper, qp, dt)
        qa = accelerations_from_velocities(qv, dt)
        return qv, qa

    def stacked_residual(qp, qv, qa):
        r = np.zeros(F * 6, dtype=np.float64)
        for f in range(F):
            r[f * 6 : f * 6 + 6] = helper.root_residual(qp[f], qv[f], qa[f], contacts_t[f])
        return r

    qvel_t, qacc_t = kin(qpos_t)
    report_before = helper.residual_report(qpos_t, qvel_t, qacc_t, contacts_t)
    D2 = _second_difference_matrix(F)

    def objective(qp, qv, qa):
        r = stacked_residual(qp, qv, qa)
        dev = qp[:, trans] - orig_trans
        smooth = D2 @ qp[:, trans]
        e = float(r @ r) + track_weight * float((dev * dev).sum()) + smooth_weight * float(
            (smooth * smooth).sum()
        )
        return e, r

    history: List[float] = []
    e_cur, r_cur = objective(qpos_t, qvel_t, qacc_t)
    history.append(e_cur)

    for _ in range(iters):
        # Banded Jacobian J[:, (f, a)] = d r / d qpos[f, trans[a]] via finite differences,
        # evaluated only on the affected residual window f-2..f+2.
        J = np.zeros((F * 6, F * 3), dtype=np.float64)
        for f in range(F):
            for ai, a in enumerate(trans):
                qp2 = qpos_t.copy()
                qp2[f, a] += fd_step
                qv2, qa2 = kin(qp2)
                col = 3 * f + ai
                lo = max(0, f - 2)
                hi = min(F - 1, f + 2)
                for g in range(lo, hi + 1):
                    rg = helper.root_residual(qp2[g], qv2[g], qa2[g], contacts_t[g])
                    J[g * 6 : g * 6 + 6, col] = (rg - r_cur[g * 6 : g * 6 + 6]) / fd_step

        # Regularized least squares (augmented, stable).
        p = qpos_t[:, trans]
        dev0 = (p - orig_trans).reshape(-1)
        smooth0 = (D2 @ p).reshape(-1)  # (F*3,) axis-blocked below
        # smoothness operator acting on the flat (F*3) correction, per axis
        # build block-diagonal D2 over the 3 axes with the (f, axis) ordering used above
        S = np.zeros((F * 3, F * 3), dtype=np.float64)
        for a in range(3):
            for i in range(F):
                for j in range(F):
                    if D2[i, j] != 0.0:
                        S[3 * i + a, 3 * j + a] = D2[i, j]
        aug_A = np.vstack(
            [
                J,
                np.sqrt(track_weight) * np.eye(F * 3),
                np.sqrt(smooth_weight) * S,
                np.sqrt(damping) * np.eye(F * 3),
            ]
        )
        aug_b = np.concatenate(
            [
                -r_cur,
                -np.sqrt(track_weight) * dev0,
                -np.sqrt(smooth_weight) * smooth0,
                np.zeros(F * 3),
            ]
        )
        delta, *_ = np.linalg.lstsq(aug_A, aug_b, rcond=None)
        delta = delta.reshape(F, 3)

        # Backtracking line search on the total objective.
        alpha = 1.0
        accepted = False
        for _ls in range(20):
            cand = qpos_t.copy()
            cand[:, trans] += alpha * delta
            qv_c, qa_c = kin(cand)
            e_c, r_c = objective(cand, qv_c, qa_c)
            if e_c < e_cur - 1e-12:
                qpos_t = cand
                qvel_t, qacc_t = qv_c, qa_c
                e_cur, r_cur = e_c, r_c
                accepted = True
                break
            alpha *= 0.5
        history.append(e_cur)
        if not accepted:
            break

    report_after = helper.residual_report(qpos_t, qvel_t, qacc_t, contacts_t)
    return RRAResult(
        qpos_t=qpos_t,
        qvel_t=qvel_t,
        qacc_t=qacc_t,
        root_shift=qpos_t[:, trans] - orig_trans,
        residual_before=report_before,
        residual_after=report_after,
        objective_history=history,
    )


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DynamicsFitResult:
    mjcf_xml: str
    qpos_t: np.ndarray
    qvel_t: np.ndarray
    qacc_t: np.ndarray
    mass_result: MassIdentificationResult
    body_names: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {"mass_identification": self.mass_result.summary()}


class DynamicsFitter:
    """End-to-end M2e fit: fitted skeleton + q(t) + GRF -> dynamically consistent masses.

    Builds the M3 MJCF, derives free-joint-aware velocities/accelerations from the pose
    trajectory, maps the measured GRF/COP onto foot bodies, and runs the linear mass
    identification. This is the kinodynamic bridge from the kinematic fit (M2c/M2d) to
    the Newton ``SolverMuJoCo`` contact research (M5+).
    """

    def __init__(
        self,
        spec,
        group_scales: Optional[np.ndarray] = None,
        coupled_knee: str = "coupled",
    ):
        from biomech.export.mjcf import export_mjcf

        self.spec = spec
        self.group_scales = group_scales
        self.coupled_knee = coupled_knee
        self.export = export_mjcf(
            spec, group_scales=group_scales, coupled_knee=coupled_knee
        )
        self.helper = ResidualHelper(self.export.xml)
        # anatomical bodies present in the MJCF (dummies excluded)
        self.body_names = [
            b.name for b in spec.bodies if self.helper.has_body(b.name)
        ]

    def qpos_trajectory(self, q_t: np.ndarray) -> np.ndarray:
        from biomech.export.mjcf import dart_q_to_mjcf_qpos

        q_t = np.asarray(q_t, dtype=np.float64)
        if q_t.ndim == 1:
            q_t = q_t[None, :]
        return np.stack(
            [
                dart_q_to_mjcf_qpos(
                    self.spec, q_t[f], self.group_scales, self.coupled_knee
                )
                for f in range(q_t.shape[0])
            ]
        )

    def kinematics(
        self, q_t: np.ndarray, fps: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = 1.0 / float(fps)
        qpos_t = self.qpos_trajectory(q_t)
        qvel_t = velocities_from_positions(self.helper, qpos_t, dt)
        qacc_t = accelerations_from_velocities(qvel_t, dt)
        return qpos_t, qvel_t, qacc_t

    def fit_masses(
        self,
        q_t: np.ndarray,
        fps: float,
        contacts_t: Sequence[Sequence[Contact]],
        reg: float = 1e-2,
        backend: str = "cpu",
    ) -> DynamicsFitResult:
        """Refine per-segment masses from GRF.

        ``backend``: ``"cpu"`` (authoritative float64 ``mj_inverse``) or ``"gpu"``
        (``mujoco_warp`` batched over frames on the Newton MuJoCo GPU; float32).
        """
        qpos_t, qvel_t, qacc_t = self.kinematics(q_t, fps)
        if backend == "gpu":
            batched = BatchedResidualHelper(self.export.xml, qpos_t.shape[0])
            mass_result = identify_masses_batched(
                batched, qpos_t, qvel_t, qacc_t, contacts_t, self.body_names, reg=reg
            )
        elif backend == "cpu":
            mass_result = identify_masses(
                self.helper,
                qpos_t,
                qvel_t,
                qacc_t,
                contacts_t,
                self.body_names,
                reg=reg,
            )
        else:
            raise ValueError("backend must be 'cpu' or 'gpu'")
        return DynamicsFitResult(
            mjcf_xml=self.export.xml,
            qpos_t=qpos_t,
            qvel_t=qvel_t,
            qacc_t=qacc_t,
            mass_result=mass_result,
            body_names=list(self.body_names),
        )

    def fit_inertial_params(
        self,
        q_t: np.ndarray,
        fps: float,
        contacts_t: Sequence[Sequence[Contact]],
        reg: float = 1.0,
    ) -> InertialIdentificationResult:
        """Refine the full per-segment inertial params (mass + COM + inertia) from GRF.

        Uses the authoritative float64 CPU engine. The fitted params are also applied
        to ``self.helper.model`` so a subsequent MJCF/export reflects them.
        """
        qpos_t, qvel_t, qacc_t = self.kinematics(q_t, fps)
        return identify_inertial_params(
            self.helper,
            qpos_t,
            qvel_t,
            qacc_t,
            contacts_t,
            self.body_names,
            reg=reg,
        )

    def rra(
        self,
        q_t: np.ndarray,
        fps: float,
        contacts_t: Sequence[Sequence[Contact]],
        **kwargs,
    ) -> RRAResult:
        """Kinematic Residual Reduction: nudge the root trajectory to cut the residual.

        Builds the MJCF kinematics from ``q_t`` and runs :func:`rra_kinematics` against
        the measured ``contacts_t`` (fixed GRF). Extra keyword args are forwarded to
        :func:`rra_kinematics` (``iters``, ``track_weight``, ``smooth_weight``, ...).
        The returned ``qpos_t`` is the adjusted MJCF pose trajectory.
        """
        qpos_t = self.qpos_trajectory(q_t)
        return rra_kinematics(self.helper, qpos_t, contacts_t, 1.0 / float(fps), **kwargs)
