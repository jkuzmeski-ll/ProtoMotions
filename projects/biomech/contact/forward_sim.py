# SPDX-License-Identifier: MIT
#
# Distributed contact inside a Newton MuJoCo forward simulation.
#
# The prescribed-kinematics path (M5-M7 + pipeline) predicts GRF from a fixed q(t). This
# module closes the loop: it runs the **MuJoCo solver's forward dynamics** (the same
# engine Newton's ``SolverMuJoCo`` integrates) and, each step, computes the distributed
# foot-contact wrench with the **Warp contact kernel** from the body's current world
# pose + spatial velocity and applies it as an external force (``xfrc_applied``) on the
# foot body. So the contact reaction *emerges* from the simulation instead of being read
# from data -- the foundation for contact-rich biomechanics research (settling, RRA,
# forward tracking) on the subject's own plantar geometry.
#
# "Use Newton as much as possible": MuJoCo (Newton's multibody solver) does the
# articulated forward dynamics + integration; Warp does the per-patch contact law. The
# contact is applied as an explicit force computed from the step's starting state (a
# compliant elastic-foundation/hydroelastic law, so an explicit update is stable at a
# reasonable dt with the model's own damping).
#
# Conventions: world Z-up, SI. MuJoCo free-joint quats are wxyz; the contact kernels use
# xyzw -- converted at the boundary. ``xfrc_applied`` is a world-frame wrench at the body
# COM (``xipos``); the contact kernel returns per-patch world forces at world points,
# reduced here to a net force + torque about the COM.

"""Distributed foot contact in a Newton MuJoCo forward simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from biomech.contact.elastic_foundation import ElasticFoundationParams, FootSole
from biomech.contact.hydroelastic import HydroelasticParams

ContactParams = Union[ElasticFoundationParams, HydroelasticParams]


# ---------------------------------------------------------------------------
# Foot contact model attached to a body
# ---------------------------------------------------------------------------


@dataclass
class FootContactModel:
    """A distributed contact law bound to a MuJoCo body.

    ``law`` selects the per-patch force law: ``"hydroelastic"`` (pressure-field, M7) or
    ``"elastic"`` (Winkler foundation, M5). ``params`` is the matching params dataclass
    (:class:`HydroelasticParams` / :class:`ElasticFoundationParams`). ``backend`` is the
    contact-kernel backend (``"warp"`` for the GPU kernel, ``"numpy"`` reference).
    """

    body: str
    sole: FootSole
    params: ContactParams
    law: str = "hydroelastic"
    backend: str = "numpy"
    device: str = "cuda"


def _point_forces(fcm: FootContactModel, pos, quat, linvel, angvel, ground_z):
    """Dispatch to the selected law's per-patch force kernel (single frame -> (N,3))."""
    if fcm.law == "hydroelastic":
        from biomech.contact.hydroelastic import point_forces_numpy, point_forces_warp
    elif fcm.law == "elastic":
        from biomech.contact.elastic_foundation import (
            point_forces_numpy,
            point_forces_warp,
        )
    else:
        raise ValueError(f"unknown contact law {fcm.law!r}")
    params = fcm.params  # law and params must match (runtime contract)
    args = (fcm.sole, params, pos[None], quat[None], linvel[None], angvel[None])
    if fcm.backend == "warp":
        pf, pw = point_forces_warp(*args, ground_z=ground_z, device=fcm.device)  # type: ignore[arg-type]
    else:
        pf, pw = point_forces_numpy(*args, ground_z=ground_z)  # type: ignore[arg-type]
    return pf[0], pw[0]  # (N,3), (N,3)


# ---------------------------------------------------------------------------
# MJCF helper for a standalone rigid-body drop test
# ---------------------------------------------------------------------------


def single_body_mjcf(
    mass: float,
    diaginertia: Tuple[float, float, float] = (0.01, 0.01, 0.01),
    com: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: float = 9.81,
    body_name: str = "foot",
    timestep: float = 1.0e-3,
) -> str:
    """A single free rigid body with explicit inertia and no collision geoms.

    Contact is supplied externally (via the distributed model), so the body carries no
    geoms -- only an ``<inertial>``. Gravity is ``-gravity`` along world Z.
    """
    cx, cy, cz = com
    ix, iy, iz = diaginertia
    px, py, pz = start_pos
    return f"""<mujoco model="single_body">
  <option timestep="{timestep}" gravity="0 0 {-gravity}"/>
  <worldbody>
    <body name="{body_name}" pos="{px} {py} {pz}">
      <freejoint/>
      <inertial pos="{cx} {cy} {cz}" mass="{mass}" diaginertia="{ix} {iy} {iz}"/>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Forward-simulation harness (CPU mj_step; Warp contact kernel)
# ---------------------------------------------------------------------------


@dataclass
class SimStepResult:
    """Per-step diagnostics recorded during a run."""

    qpos: np.ndarray  # (F, nq)
    time: np.ndarray  # (F,)
    grf: Dict[str, np.ndarray] = field(default_factory=dict)  # body -> (F, 3)
    cop: Dict[str, np.ndarray] = field(default_factory=dict)  # body -> (F, 3)


class ContactForwardSim:
    """MuJoCo forward dynamics with distributed foot contact as an applied external force.

    Builds a ``mujoco.MjModel`` from an MJCF (e.g. the M3 export, or
    :func:`single_body_mjcf`), and each :meth:`step` computes every foot's distributed
    contact wrench from the body's current world state and writes it to ``xfrc_applied``
    before integrating with ``mj_step`` (the MuJoCo/Newton solver's forward dynamics).
    """

    def __init__(
        self,
        mjcf_xml: str,
        feet: Sequence[FootContactModel],
        ground_z: float = 0.0,
    ):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self.data = mujoco.MjData(self.model)
        self.ground_z = float(ground_z)
        self.feet = list(feet)
        self._body_id: Dict[str, int] = {}
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name:
                self._body_id[name] = b
        for fcm in self.feet:
            if fcm.body not in self._body_id:
                raise KeyError(f"body {fcm.body!r} not in model")
        self.last_grf: Dict[str, np.ndarray] = {}
        self.last_cop: Dict[str, np.ndarray] = {}
        mujoco.mj_forward(self.model, self.data)

    # --- state -------------------------------------------------------------
    def set_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.data.qvel[:] = 0.0
        self._mj.mj_forward(self.model, self.data)

    def forward(self) -> None:
        """Recompute kinematics/dynamics for the current ``qpos``/``qvel`` (no integration).

        Call after writing ``data.qpos``/``data.qvel`` directly so that a subsequent
        :meth:`apply_contacts` reads up-to-date body poses and velocities.
        """
        self._mj.mj_forward(self.model, self.data)

    def _foot_state(self, bid: int):
        """World (xpos, xquat_xyzw, xipos, linvel_at_xpos, angvel) for a body."""
        d = self.data
        xpos = d.xpos[bid].copy()
        xipos = d.xipos[bid].copy()
        wxyz = d.xquat[bid]
        quat = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)
        res = np.zeros(6)
        self._mj.mj_objectVelocity(
            self.model, d, self._mj.mjtObj.mjOBJ_BODY, bid, res, 0
        )
        angvel = res[:3].copy()
        v_com = res[3:].copy()
        # objectVelocity gives velocity at the COM; the sole points are in the body
        # frame (relative to xpos), so shift to the velocity at the frame origin.
        linvel = v_com + np.cross(angvel, xpos - xipos)
        return xpos, quat, xipos, linvel, angvel

    def _foot_wrench(self, fcm: FootContactModel):
        """Net world force + torque about COM, plus net GRF/COP, for one foot."""
        bid = self._body_id[fcm.body]
        xpos, quat, xipos, linvel, angvel = self._foot_state(bid)
        pf, pw = _point_forces(fcm, xpos, quat, linvel, angvel, self.ground_z)
        net_force = pf.sum(axis=0)
        torque_com = np.cross(pw - xipos, pf).sum(axis=0)
        # centre of pressure (world) for reporting
        fz = pf[:, 2]
        tot = float(fz.sum())
        if tot > 1e-9:
            cop = np.array([
                float((pw[:, 0] * fz).sum() / tot),
                float((pw[:, 1] * fz).sum() / tot),
                self.ground_z,
            ])
        else:
            cop = np.full(3, np.nan)
        return net_force, torque_com, cop

    def apply_contacts(self) -> None:
        """Compute every foot wrench from the current state and set ``xfrc_applied``."""
        d = self.data
        d.xfrc_applied[:] = 0.0
        for fcm in self.feet:
            bid = self._body_id[fcm.body]
            force, torque_com, cop = self._foot_wrench(fcm)
            d.xfrc_applied[bid, :3] = force
            d.xfrc_applied[bid, 3:] = torque_com
            self.last_grf[fcm.body] = force
            self.last_cop[fcm.body] = cop

    def step(self, dt: Optional[float] = None) -> None:
        if dt is not None:
            self.model.opt.timestep = float(dt)
        self.apply_contacts()
        self._mj.mj_step(self.model, self.data)

    def run(
        self, n_steps: int, dt: Optional[float] = None, record: bool = True
    ) -> SimStepResult:
        if dt is not None:
            self.model.opt.timestep = float(dt)
        nq = int(self.model.nq)
        qpos_hist = np.zeros((n_steps, nq)) if record else np.zeros((0, nq))
        time_hist = np.zeros(n_steps) if record else np.zeros(0)
        grf_hist: Dict[str, List[np.ndarray]] = {f.body: [] for f in self.feet}
        cop_hist: Dict[str, List[np.ndarray]] = {f.body: [] for f in self.feet}
        for i in range(n_steps):
            self.step()
            if record:
                qpos_hist[i] = self.data.qpos
                time_hist[i] = self.data.time
                for f in self.feet:
                    grf_hist[f.body].append(self.last_grf[f.body].copy())
                    cop_hist[f.body].append(self.last_cop[f.body].copy())
        return SimStepResult(
            qpos=qpos_hist,
            time=time_hist,
            grf={b: np.array(v) for b, v in grf_hist.items()},
            cop={b: np.array(v) for b, v in cop_hist.items()},
        )
