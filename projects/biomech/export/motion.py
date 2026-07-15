# SPDX-License-Identifier: MIT
#
# Milestone M3 (motion half) — fitted gold-standard pose trajectory q(t) -> a
# ProtoMotions ``.motion`` clip (a ``RobotState.to_dict``-compatible dict of
# ``rigid_body_{pos,rot,vel,ang_vel}`` + ``dof_{pos,vel}`` + ``fps``), consumed by
# ``protomotions.components.motion_lib.MotionLib`` (fields ``gts/grs/gvs/gavs/dps/dvs``).
#
# The global body transforms are computed from the **float64 Warp/DART skeleton FK**
# (``biomech.skeleton.WarpSkeleton``) — the same gold-standard kinematics the fit
# produces — NOT routed through the 18-keypoint / PyRoki path. Frames are converted from
# OpenSim (Y-up) to the ProtoMotions/Newton world (Z-up) via ``R_OS2PM`` (the
# ``R_OS2PM`` referenced in ``biomech.frames``). Quaternions are xyzw (COMMON state).
#
# Body set: the 20 **anatomical** OpenSim bodies (``spec.bodies`` order). The MJCF
# exporter adds massless dummy bodies so Newton doesn't merge multi-DOF joints; those
# are Newton-internal and carry no anatomy, so they are excluded here. Wiring this clip
# to a concrete ProtoMotions robot (body-index/DOF map, dummy handling) is the M3->train
# integration step and needs that robot's config; this module produces the clip.

"""Fitted ``q(t)`` -> ProtoMotions ``.motion`` clip (gold-standard, Z-up) (M3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from biomech.osim.spec import SkeletonSpec

# OpenSim Y-up -> ProtoMotions/Newton Z-up (rotate +90 deg about X):
#   world_x = os_x (forward), world_z = os_y (up), world_y = -os_z.
R_OS2PM = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


@dataclass
class MotionExportResult:
    data: dict  # RobotState.to_dict-compatible (torch tensors + fps)
    body_names: list[str]  # anatomical body order (== gts/grs body axis)
    dof_names: list[str]  # sim DOF order (== dps/dvs axis)
    fps: float


def _matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Batched (..., 3, 3) rotation matrices -> (..., 4) xyzw unit quaternions."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    m = R.reshape(-1, 3, 3)
    n = m.shape[0]
    q = np.empty((n, 4), dtype=np.float64)  # x, y, z, w
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(n):
        M = m[i]
        tr = t[i]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (M[2, 1] - M[1, 2]) / s
            y = (M[0, 2] - M[2, 0]) / s
            z = (M[1, 0] - M[0, 1]) / s
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2.0
            w = (M[2, 1] - M[1, 2]) / s
            x = 0.25 * s
            y = (M[0, 1] + M[1, 0]) / s
            z = (M[0, 2] + M[2, 0]) / s
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2.0
            w = (M[0, 2] - M[2, 0]) / s
            x = (M[0, 1] + M[1, 0]) / s
            y = 0.25 * s
            z = (M[1, 2] + M[2, 1]) / s
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2.0
            w = (M[1, 0] - M[0, 1]) / s
            x = (M[0, 2] + M[2, 0]) / s
            y = (M[1, 2] + M[2, 1]) / s
            z = 0.25 * s
        q[i] = (x, y, z, w)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*shp, 4)


def _finite_diff_lin(pos: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference of positions (F, ...) -> velocities (F, ...)."""
    v = np.zeros_like(pos)
    if pos.shape[0] < 2:
        return v
    v[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    v[0] = (pos[1] - pos[0]) / dt
    v[-1] = (pos[-1] - pos[-2]) / dt
    return v


def _angular_velocity(Rm: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity from a rotation-matrix trajectory (F, N, 3, 3)."""
    F, N = Rm.shape[0], Rm.shape[1]
    w = np.zeros((F, N, 3), dtype=np.float64)
    if F < 2:
        return w

    def omega(Ra, Rb, h):
        # relative rotation Rb Ra^T over time h -> axis-angle / h (world frame)
        dR = np.einsum("nij,nkj->nik", Rb, Ra)  # Rb @ Ra^T
        cos = np.clip((dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
        ang = np.arccos(cos)
        ax = np.stack(
            [
                dR[:, 2, 1] - dR[:, 1, 2],
                dR[:, 0, 2] - dR[:, 2, 0],
                dR[:, 1, 0] - dR[:, 0, 1],
            ],
            axis=1,
        )
        norm = np.linalg.norm(ax, axis=1, keepdims=True)
        safe = norm[:, 0] > 1e-12
        out = np.zeros((ax.shape[0], 3), dtype=np.float64)
        out[safe] = ax[safe] / norm[safe] * (ang[safe, None] / h)
        return out

    w[1:-1] = omega(Rm[:-2].reshape(-1, 3, 3), Rm[2:].reshape(-1, 3, 3), 2.0 * dt).reshape(F - 2, N, 3)
    w[0] = omega(Rm[0], Rm[1], dt)
    w[-1] = omega(Rm[-2], Rm[-1], dt)
    return w


def build_motion(
    spec: SkeletonSpec,
    q_t: np.ndarray,
    fps: float,
    group_scales: Optional[np.ndarray] = None,
    coupled_knee: str = "coupled",
    up_convert: bool = True,
    device: str = "cpu",
) -> MotionExportResult:
    """Build a ProtoMotions ``.motion`` clip from a fitted pose trajectory.

    Args:
        spec: fitted skeleton.
        q_t: ``(F, ndof)`` gold-standard DART poses (radians / meters, OpenSim order).
        fps: frames per second of the trajectory.
        group_scales: fitted per-group scales ``(3G,)`` (defaults to unit).
        coupled_knee: DOF layout for ``dof_pos`` (must match the exported MJCF).
        up_convert: convert OpenSim Y-up -> ProtoMotions Z-up (recommended).

    Returns:
        ``MotionExportResult`` whose ``data`` is a ``RobotState.to_dict``-compatible
        dict (COMMON convention, xyzw quaternions) ready to ``torch.save`` as ``.motion``.
    """
    import torch

    from biomech.export.mjcf import dart_q_to_mjcf_qpos
    from biomech.skeleton.skeleton import WarpSkeleton

    q_t = np.asarray(q_t, dtype=np.float64)
    if q_t.ndim == 1:
        q_t = q_t[None, :]
    F = q_t.shape[0]
    dt = 1.0 / float(fps)

    ws = WarpSkeleton(spec, device=device)
    world, _ = ws.forward(q_t, group_scales)  # (F, Nb, 4, 4), OpenSim Y-up
    world = np.asarray(world, dtype=np.float64)

    pos = world[:, :, :3, 3].copy()  # (F, Nb, 3)
    rot = world[:, :, :3, :3].copy()  # (F, Nb, 3, 3)

    if up_convert:
        pos = np.einsum("ij,fnj->fni", R_OS2PM, pos)
        rot = np.einsum("ij,fnjk->fnik", R_OS2PM, rot)

    quat = _matrix_to_quat_xyzw(rot)  # (F, Nb, 4) xyzw
    lin_vel = _finite_diff_lin(pos, dt)
    ang_vel = _angular_velocity(rot, dt)

    # sim DOF trajectory (non-root joint coords), matching the exported MJCF order
    qpos = np.stack(
        [dart_q_to_mjcf_qpos(spec, q_t[f], group_scales, coupled_knee) for f in range(F)]
    )
    dof_pos = qpos[:, 7:]  # drop the 7-coord free root
    dof_vel = _finite_diff_lin(dof_pos, dt)

    def t32(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32))

    data = {
        "rigid_body_pos": t32(pos),
        "rigid_body_rot": t32(quat),
        "rigid_body_vel": t32(lin_vel),
        "rigid_body_ang_vel": t32(ang_vel),
        "dof_pos": t32(dof_pos),
        "dof_vel": t32(dof_vel),
        "fps": float(fps),
    }

    # dof names in export order (skip locked; slides before hinges), for metadata
    from biomech.export.mjcf import _joint_dofs, _scaled_frames, _ScaleMap

    sm = _ScaleMap(spec, group_scales)
    dof_names: list[str] = []
    for j in spec.joints:
        if j.parent_body is None:
            continue
        _, Tc = _scaled_frames(j, sm)
        for d in _joint_dofs(j, Tc[:3, :3], Tc[:3, 3], coupled_knee):
            dof_names.append(d.name)

    return MotionExportResult(
        data=data,
        body_names=[b.name for b in spec.bodies],
        dof_names=dof_names,
        fps=float(fps),
    )


def write_motion(path, spec, q_t, fps, **kwargs) -> MotionExportResult:
    """Build and ``torch.save`` a ``.motion`` clip. Returns the result."""
    import os

    import torch

    res = build_motion(spec, q_t, fps, **kwargs)
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    torch.save(res.data, str(path))
    return res
