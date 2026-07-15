# SPDX-License-Identifier: MIT
#
# Full-skeleton contact-rich forward simulation (the research payload).
#
# ``forward_sim.ContactForwardSim`` proved a *single* rigid body settles onto the
# distributed foot contact at the analytic equilibrium. This module scales that up to the
# **whole M3-exported skeleton MJCF** driven by the **Newton MuJoCo solver**: the
# articulated forward dynamics run in MuJoCo (``mj_step1``/``mj_step2`` -- the same engine
# ``SolverMuJoCo`` integrates), the distributed foot contact is computed each step by the
# Warp/NumPy contact kernel and applied to ``calcn_r``/``calcn_l`` as an external wrench
# (``xfrc_applied``), and the (unactuated) joints are held/driven by a computed-torque PD
# servo written to ``qfrc_applied``. The ground reaction therefore *emerges* from the sim.
#
# "Use Newton as much as possible": MuJoCo does all the multibody dynamics + integration;
# Warp does the per-patch contact law; the servo is the only added control law.
#
# Frame handling (the crux). The exported MJCF is defined in the OpenSim **Y-up** frame
# (that is where its FK is bit-exact vs the Warp/DART skeleton). The contact kernels, the
# lab, gravity (MuJoCo default ``-Z``) and the measured GRF are all **Z-up**. We reconcile
# them by baking the Y-up->Z-up rotation ``R_OS2PM`` into the **free-root pose only**:
# because the root free joint carries the whole tree, prepending ``R_OS2PM`` to the root
# rotates every body rigidly into Z-up, so MuJoCo's body world poses equal
# ``export.motion.build_motion``'s Z-up poses -- exactly the poses the (already validated)
# contact pipeline drives the sole with. Non-root joint coordinates are frame-independent.
#
# Validation invariant: at a static standing equilibrium the only external forces are
# gravity and contact, so the net vertical force is zero -> the summed two-foot vertical
# GRF equals the model's total weight, regardless of the internal joint torques. This is a
# reconstruction-quality-independent check of the whole harness (frames, servo, contact).

"""Full-skeleton distributed-contact forward simulation on the Newton MuJoCo solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from biomech.contact.elastic_foundation import FootSole, _quat_rotate_np
from biomech.contact.forward_sim import (
    ContactForwardSim,
    ContactParams,
    FootContactModel,
)

# ---------------------------------------------------------------------------
# OpenSim Y-up pose -> Z-up MuJoCo qpos
# ---------------------------------------------------------------------------


def root_body_name(spec) -> str:
    """The free-floating root body (the joint whose parent is the world)."""
    for j in spec.joints:
        if j.parent_body is None:
            return j.child_body
    raise ValueError("no free root joint (parent_body is None) in spec")


def mjcf_qpos_zup(
    spec,
    q: np.ndarray,
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
) -> np.ndarray:
    """Map a DART pose ``q`` to the exported model's MuJoCo ``qpos`` in the **Z-up** world.

    The non-root joint coordinates are exactly ``dart_q_to_mjcf_qpos``'s (frame-invariant);
    the free-root 7 coords are the OpenSim root world pose rotated by ``R_OS2PM`` (Y-up ->
    Z-up) and written in MuJoCo's ``[px,py,pz, qw,qx,qy,qz]`` convention. Loading this into
    the exported MJCF places the whole skeleton upright under MuJoCo's default ``-Z``
    gravity, with every body pose equal to ``build_motion``'s Z-up pose.
    """
    from biomech.export.mjcf import _rotmat_to_quat_wxyz, dart_q_to_mjcf_qpos
    from biomech.export.motion import R_OS2PM
    from biomech.skeleton.skeleton import fk_numpy

    qp = dart_q_to_mjcf_qpos(spec, q, group_scales, coupled_knee)  # Newton joint_q (xyzw)

    root = root_body_name(spec)
    world, _ = fk_numpy(spec, np.asarray(q, dtype=np.float64).ravel(), group_scales)
    T = world[root]
    p_zup = R_OS2PM @ T[:3, 3]
    R_zup = R_OS2PM @ T[:3, :3]
    quat_wxyz = _rotmat_to_quat_wxyz(R_zup)

    qpos = qp.copy()
    qpos[0:3] = p_zup
    qpos[3:7] = quat_wxyz  # wxyz for MuJoCo
    return qpos


# ---------------------------------------------------------------------------
# Frozen (rigidly welded) skeleton for the standing-drop validation
# ---------------------------------------------------------------------------


def frozen_skeleton_xml(
    spec,
    q: np.ndarray,
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
) -> str:
    """Exported MJCF with every non-root joint locked at the reference pose ``q``.

    Each independent kinematic joint gets a single-joint ``<equality><joint>`` constraint
    pinning it to its reference value (``q1 = polycoef[0]``); the coupled-knee dependents
    are already tied to the independent knee, so locking the independents freezes the
    whole skeleton into one rigid body with a free floating base. Unlike a kinematic
    teleport, MuJoCo's constraint solver **transmits** the distal contact load up the
    (locked) leg chain to the root, so the free root settles at the true static
    equilibrium where the total two-foot vertical GRF equals body weight -- with no
    controller, hence unconditionally stable.
    """
    import mujoco

    from biomech.export.mjcf import export_mjcf

    export = export_mjcf(spec, group_scales=group_scales, coupled_knee=coupled_knee)
    xml = export.xml
    m0 = mujoco.MjModel.from_xml_string(xml)
    ref = mjcf_qpos_zup(spec, q, group_scales, coupled_knee)

    dependent = set()
    for e in range(int(m0.neq)):
        if m0.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            dependent.add(int(m0.eq_obj1id[e]))

    locks: List[str] = []
    for j in range(int(m0.njnt)):
        if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE or j in dependent:
            continue
        name = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)
        val = float(ref[int(m0.jnt_qposadr[j])])
        locks.append(f'    <joint joint1="{name}" polycoef="{val!r}"/>')
    block = "\n".join(locks)
    if "</equality>" in xml:
        xml = xml.replace("  </equality>", block + "\n  </equality>", 1)
    else:
        xml = xml.replace(
            "  </worldbody>",
            "  </worldbody>\n  <equality>\n" + block + "\n  </equality>",
            1,
        )
    return xml


# ---------------------------------------------------------------------------
# quaternion helpers (wxyz, for root orientation servo)
# ---------------------------------------------------------------------------


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis*angle) of a wxyz quaternion, shortest-arc."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    q = q / n
    if q[0] < 0.0:  # shortest arc
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-8:
        return 2.0 * q[1:4]  # small-angle limit
    angle = 2.0 * np.arccos(w)
    return q[1:4] / s * angle


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TrackingResult:
    """Per-step diagnostics of a tracking / settling run."""

    qpos: np.ndarray  # (F, nq)
    time: np.ndarray  # (F,)
    grf: Dict[str, np.ndarray] = field(default_factory=dict)  # body -> (F, 3)
    cop: Dict[str, np.ndarray] = field(default_factory=dict)  # body -> (F, 3)
    total_vertical_grf: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (F,)


# ---------------------------------------------------------------------------
# Servo gains
# ---------------------------------------------------------------------------


@dataclass
class ServoGains:
    """Computed-torque servo tuning as natural frequencies + damping ratios.

    Per-DOF PD gains are derived from the mass-matrix diagonal each step
    (``kp = M_jj * wn**2``, ``kd = M_jj * 2*zeta*wn``) so every joint -- from the heavy
    pelvis to the tiny forearm -- has the same closed-loop natural frequency and damping
    ratio. This keeps the explicit servo update stable (``wn*dt`` uniform and small)
    regardless of the huge inertia spread, and is why a fixed ``kp`` blows up the
    low-inertia joints. Gravity/coriolis are compensated separately, so modest ``wn``
    still holds a pose exactly.
    """

    joint_wn: float = 40.0
    joint_zeta: float = 1.0
    root_lin_wn: float = 25.0
    root_lin_zeta: float = 1.0
    root_rot_wn: float = 25.0
    root_rot_zeta: float = 1.0


# ---------------------------------------------------------------------------
# Full-skeleton tracking sim
# ---------------------------------------------------------------------------


class SkeletonTrackingSim(ContactForwardSim):
    """Newton MuJoCo forward dynamics of the exported skeleton with distributed foot
    contact (``xfrc_applied``) and a computed-torque PD joint servo (``qfrc_applied``).

    The free root is left partially free so a static equilibrium can be reached: in the
    standing-drop mode (``hold_root=True``) the root's horizontal position and orientation
    are held by the servo while its **vertical** DOF settles under gravity + contact. In
    full-tracking mode (``hold_root=False``) all six root DOFs are servoed toward a
    reference root pose.
    """

    def __init__(
        self,
        mjcf_xml: str,
        feet: Sequence[FootContactModel],
        ground_z: float = 0.0,
        gains: Optional[ServoGains] = None,
    ):
        super().__init__(mjcf_xml, feet, ground_z)
        self.gains = gains or ServoGains()
        mj = self._mj
        m = self.model
        self.nv = int(m.nv)
        self.nq = int(m.nq)

        # locate the single free (root) joint
        root_jnt = -1
        for j in range(m.njnt):
            if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                root_jnt = j
                break
        if root_jnt < 0:
            raise ValueError("model has no free (root) joint")
        self.root_qadr = int(m.jnt_qposadr[root_jnt])
        self.root_dadr = int(m.jnt_dofadr[root_jnt])

        # coupled-knee DOFs are driven by <equality><joint> constraints; servoing them
        # fights the constraint (instability), so exclude the dependent joints from the
        # actuated set -- the equality provides their coupling from the independent DOF.
        dependent_joints = set()
        for e in range(int(m.neq)):
            if m.eq_type[e] == mj.mjtEq.mjEQ_JOINT:
                dependent_joints.add(int(m.eq_obj1id[e]))
        self.dependent_joints = dependent_joints

        # actuated (non-root, non-dependent) joints: 1 dof each (hinge/slide)
        act_dof: List[int] = []
        act_qpos: List[int] = []
        for j in range(m.njnt):
            if j == root_jnt or j in dependent_joints:
                continue
            t = m.jnt_type[j]
            if t not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"unexpected non-free joint type {t} (expected 1-dof)")
            act_dof.append(int(m.jnt_dofadr[j]))
            act_qpos.append(int(m.jnt_qposadr[j]))
        self.act_dof = np.array(act_dof, dtype=np.int64)
        self.act_qpos = np.array(act_qpos, dtype=np.int64)
        # when the skeleton is rigidly frozen (equality joint locks), the joints must not
        # be servoed (that fights the constraints); only the free root is stabilized.
        self.servo_joints = True

        # non-root qpos/dof indices (for the kinematic-freeze standing drop)
        self.nonroot_qpos = np.array(
            [i for i in range(self.nq) if not (self.root_qadr <= i < self.root_qadr + 7)],
            dtype=np.int64,
        )
        self.nonroot_dof = np.array(
            [i for i in range(self.nv) if not (self.root_dadr <= i < self.root_dadr + 6)],
            dtype=np.int64,
        )

    # --- servo ------------------------------------------------------------
    def _mass_diag(self) -> np.ndarray:
        """Diagonal of the joint-space inertia matrix at the current state (nv,)."""
        M = np.zeros((self.nv, self.nv), dtype=np.float64)
        self._mj.mj_fullM(self.model, M, self.data.qM)
        return np.diag(M).copy()

    def _servo_qfrc(
        self,
        qpos_ref: np.ndarray,
        hold_root: bool,
        root_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computed-torque PD generalized force (inertia-scaled gains + gravity comp)."""
        d = self.data
        g = self.gains
        tau = np.zeros(self.nv, dtype=np.float64)
        bias = np.asarray(d.qfrc_bias, dtype=np.float64)
        Mdiag = self._mass_diag()

        # actuated joints: gravity/coriolis comp + inertia-scaled critically-damped PD
        if self.servo_joints and self.act_dof.size:
            kp = Mdiag[self.act_dof] * g.joint_wn ** 2
            kd = Mdiag[self.act_dof] * 2.0 * g.joint_zeta * g.joint_wn
            q = np.asarray(d.qpos)[self.act_qpos]
            qd = np.asarray(d.qvel)[self.act_dof]
            qref = np.asarray(qpos_ref)[self.act_qpos]
            tau[self.act_dof] = bias[self.act_dof] + kp * (qref - q) - kd * qd

        r0 = self.root_dadr
        rq = self.root_qadr
        rref = qpos_ref if root_ref is None else root_ref
        kp_lin = Mdiag[r0 : r0 + 3] * g.root_lin_wn ** 2
        kd_lin = Mdiag[r0 : r0 + 3] * 2.0 * g.root_lin_zeta * g.root_lin_wn
        held = (0, 1) if hold_root else (0, 1, 2)  # vertical free in standing-drop mode
        for k in held:
            pos_err = float(rref[rq + k] - d.qpos[rq + k])
            tau[r0 + k] = bias[r0 + k] + kp_lin[k] * pos_err - kd_lin[k] * d.qvel[r0 + k]
        self._servo_root_orientation(tau, rref, Mdiag)
        return tau

    def _servo_root_orientation(
        self, tau: np.ndarray, rref: np.ndarray, Mdiag: np.ndarray
    ) -> None:
        """PD torque on the root angular DOFs toward ``rref``'s quaternion (local frame)."""
        d = self.data
        g = self.gains
        r0 = self.root_dadr
        rq = self.root_qadr
        bias = np.asarray(d.qfrc_bias, dtype=np.float64)
        q_cur = np.asarray(d.qpos)[rq + 3 : rq + 7]  # wxyz
        q_tgt = np.asarray(rref)[rq + 3 : rq + 7]
        # error rotation expressed in the body-local frame (angular DOFs are local)
        q_err = _quat_mul(_quat_conj(q_cur), q_tgt)
        rotvec = _quat_to_rotvec(q_err)  # local-frame rotation to reach target
        kp = Mdiag[r0 + 3 : r0 + 6] * g.root_rot_wn ** 2
        kd = Mdiag[r0 + 3 : r0 + 6] * 2.0 * g.root_rot_zeta * g.root_rot_wn
        for i in range(3):
            tau[r0 + 3 + i] = (
                bias[r0 + 3 + i] + kp[i] * rotvec[i] - kd[i] * d.qvel[r0 + 3 + i]
            )

    # --- stepping (mj_step1 / mj_step2 control pattern) -------------------
    def _freeze_nonroot(self, qpos_ref: np.ndarray, hold_root: bool) -> None:
        """Kinematically pin every non-root DOF (and, if ``hold_root``, the root's
        horizontal + orientation DOFs) to the reference, leaving the root **vertical**
        DOF free. The frozen DOFs act as ideal welds -- their reaction forces are
        internal, so the vertical force balance still yields total GRF == weight at
        equilibrium, with no controller (unconditionally stable)."""
        d = self.data
        d.qpos[self.nonroot_qpos] = qpos_ref[self.nonroot_qpos]
        d.qvel[self.nonroot_dof] = 0.0
        if hold_root:
            rq, r0 = self.root_qadr, self.root_dadr
            d.qpos[rq + 0] = qpos_ref[rq + 0]
            d.qpos[rq + 1] = qpos_ref[rq + 1]
            d.qpos[rq + 3 : rq + 7] = qpos_ref[rq + 3 : rq + 7]
            d.qvel[r0 + 0] = 0.0
            d.qvel[r0 + 1] = 0.0
            d.qvel[r0 + 3 : r0 + 6] = 0.0
            # rq+2 / r0+2 (vertical) left free -> settles under gravity + contact

    def step_track(
        self,
        qpos_ref: np.ndarray,
        dt: Optional[float] = None,
        hold_mode: str = "kinematic",
        hold_root: bool = True,
        root_ref: Optional[np.ndarray] = None,
    ) -> None:
        """One step: forward pass -> distributed contact (+ optional servo) -> integrate.

        ``hold_mode``:
          - ``"kinematic"`` (default): freeze the non-root pose after integrating (the
            robust frozen standing drop). No controller.
          - ``"servo"``: apply the computed-torque PD servo as ``qfrc_applied`` (for
            dynamic tracking; requires well-conditioned gains).
        """
        mj = self._mj
        m, d = self.model, self.data
        if dt is not None:
            m.opt.timestep = float(dt)
        mj.mj_step1(m, d)  # position + velocity forward pass (xpos, cvel, qfrc_bias)
        self.apply_contacts()  # xfrc_applied from current body state (Warp/NumPy kernel)
        if hold_mode == "servo":
            d.qfrc_applied[:] = self._servo_qfrc(qpos_ref, hold_root, root_ref)
        mj.mj_step2(m, d)  # actuation + integrate
        if hold_mode == "kinematic":
            self._freeze_nonroot(np.asarray(qpos_ref, dtype=np.float64), hold_root)

    def settle(
        self,
        qpos_ref: np.ndarray,
        n_steps: int,
        dt: float = 5.0e-4,
        hold_root: bool = True,
        hold_mode: str = "kinematic",
        record: bool = True,
    ) -> TrackingResult:
        """Run ``n_steps`` holding ``qpos_ref`` (standing drop) and record diagnostics."""
        qpos_ref = np.asarray(qpos_ref, dtype=np.float64)
        self.set_qpos(qpos_ref)
        nq = self.nq
        qpos_hist = np.zeros((n_steps, nq)) if record else np.zeros((0, nq))
        time_hist = np.zeros(n_steps) if record else np.zeros(0)
        grf_hist: Dict[str, List[np.ndarray]] = {f.body: [] for f in self.feet}
        cop_hist: Dict[str, List[np.ndarray]] = {f.body: [] for f in self.feet}
        total_fz = np.zeros(n_steps) if record else np.zeros(0)
        for i in range(n_steps):
            self.step_track(qpos_ref, dt=dt, hold_mode=hold_mode, hold_root=hold_root)
            if record:
                qpos_hist[i] = self.data.qpos
                time_hist[i] = self.data.time
                s = 0.0
                for f in self.feet:
                    gr = self.last_grf[f.body].copy()
                    grf_hist[f.body].append(gr)
                    cop_hist[f.body].append(self.last_cop[f.body].copy())
                    s += float(gr[2])
                total_fz[i] = s
        return TrackingResult(
            qpos=qpos_hist,
            time=time_hist,
            grf={b: np.array(v) for b, v in grf_hist.items()},
            cop={b: np.array(v) for b, v in cop_hist.items()},
            total_vertical_grf=total_fz,
        )

    def track(
        self,
        qpos_traj: np.ndarray,
        dt: float,
        substeps: int = 1,
        hold_root: bool = False,
    ) -> TrackingResult:
        """Servo a full ``qpos`` trajectory (Z-up), predicting emergent contact GRF.

        Each reference frame is held for ``substeps`` integration steps of ``dt/substeps``.
        With ``hold_root=False`` all six root DOFs track the reference root pose; the
        distributed contact provides the ground reaction, which is recorded per foot.

        NOTE: this uses the computed-torque joint servo (``hold_mode="servo"``), which is
        NOT yet numerically stable for the full articulated body (diagonal PD diverges on
        the ill-conditioned floating base). It is scaffolding for dynamic walk-phase
        tracking; a full computed torque or MuJoCo actuator + implicit path is still
        needed. The validated payload today is the frozen standing drop (``settle``).
        """
        qpos_traj = np.asarray(qpos_traj, dtype=np.float64)
        F = qpos_traj.shape[0]
        self.set_qpos(qpos_traj[0])
        sub_dt = dt / max(1, substeps)
        qpos_hist = np.zeros((F, self.nq))
        time_hist = np.zeros(F)
        grf_hist: Dict[str, List[np.ndarray]] = {f.body: [] for f in self.feet}
        cop_hist: Dict[str, List[np.ndarray]] = {f.body: [] for f in self.feet}
        total_fz = np.zeros(F)
        for i in range(F):
            ref = qpos_traj[i]
            for _ in range(substeps):
                self.step_track(
                    ref, dt=sub_dt, hold_mode="servo", hold_root=hold_root, root_ref=ref
                )
            qpos_hist[i] = self.data.qpos
            time_hist[i] = self.data.time
            s = 0.0
            for f in self.feet:
                gr = self.last_grf[f.body].copy()
                grf_hist[f.body].append(gr)
                cop_hist[f.body].append(self.last_cop[f.body].copy())
                s += float(gr[2])
            total_fz[i] = s
        return TrackingResult(
            qpos=qpos_hist,
            time=time_hist,
            grf={b: np.array(v) for b, v in grf_hist.items()},
            cop={b: np.array(v) for b, v in cop_hist.items()},
            total_vertical_grf=total_fz,
        )

    # --- helpers ----------------------------------------------------------
    @property
    def total_mass(self) -> float:
        """Total model mass (kg) -- what gravity acts on."""
        return float(np.asarray(self.model.body_mass).sum())

    @property
    def gravity(self) -> float:
        return float(-self.model.opt.gravity[2])

    def sole_world_lowest_z(self, fcm: FootContactModel) -> float:
        """Lowest world Z of a foot's sole patches at the current (forwarded) state."""
        bid = self._body_id[fcm.body]
        xpos, quat, _, _, _ = self._foot_state(bid)
        pw = xpos[None, :] + _quat_rotate_np(
            np.broadcast_to(quat, (fcm.sole.n, 4)), fcm.sole.points
        )
        return float(pw[:, 2].min())


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_skeleton_tracking_sim(
    spec,
    q: np.ndarray,
    soles: Dict[str, FootSole],
    params: Dict[str, ContactParams],
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
    law: str = "hydroelastic",
    backend: str = "numpy",
    gains: Optional[ServoGains] = None,
    ground_gap: float = 0.0,
    freeze: bool = False,
) -> Tuple[SkeletonTrackingSim, np.ndarray]:
    """Assemble a full-skeleton tracking sim at pose ``q`` with subject soles + contact.

    Args:
        spec: fitted Rajagopal ``SkeletonSpec``.
        q: DART pose (radians/meters, OpenSim order) defining the standing configuration.
        soles: ``{"R": FootSole, "L": FootSole}`` in the ``calcn`` body frame.
        params: ``{"R": ContactParams, "L": ContactParams}`` matching ``law``.
        group_scales: fitted per-group scales (defaults to unit).
        law: ``"hydroelastic"`` or ``"elastic"``.
        backend: contact-kernel backend (``"numpy"`` or ``"warp"``).
        ground_gap: gap (m) between the initial lowest sole point and the ground plane
            (0 = zero initial penetration; the body then settles into contact). A negative
            value starts with initial penetration (useful for the frozen drop).
        freeze: if True, lock every non-root joint at ``q`` via ``<equality>`` so the whole
            skeleton is one rigid body with a free floating base (the standing-drop
            validation; run with ``hold_mode="none"``).

    Returns:
        ``(sim, qpos_zup)`` -- the assembled sim (ground registered under both feet) and
        the Z-up MuJoCo ``qpos`` of the standing pose.
    """
    from biomech.export.mjcf import export_mjcf

    if freeze:
        xml = frozen_skeleton_xml(spec, q, group_scales, coupled_knee)
    else:
        xml = export_mjcf(
            spec, group_scales=group_scales, coupled_knee=coupled_knee
        ).xml
    feet = []
    for side, body in (("R", "calcn_r"), ("L", "calcn_l")):
        if side not in soles or side not in params:
            continue
        feet.append(
            FootContactModel(
                body=body, sole=soles[side], params=params[side],
                law=law, backend=backend,
            )
        )
    sim = SkeletonTrackingSim(xml, feet, ground_z=0.0, gains=gains)
    if freeze:
        sim.servo_joints = False  # joints are locked by <equality>; only servo the root

    qpos = mjcf_qpos_zup(spec, q, group_scales, coupled_knee)
    sim.set_qpos(qpos)

    # register a single flat ground plane just below the lowest sole point of either foot
    lowest = min(sim.sole_world_lowest_z(f) for f in sim.feet)
    sim.ground_z = lowest - ground_gap
    sim.set_qpos(qpos)  # reset velocities after the diagnostic forward
    return sim, qpos
