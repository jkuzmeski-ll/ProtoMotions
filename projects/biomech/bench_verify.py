# SPDX-License-Identifier: MIT
"""Verify the exported S001 asset + .motion clip load together and mass-match.

Checks (the "Next steps" gate from the GPU-conversion handoff):
  1. the written biomech_rajagopal.xml loads into robot_config("biomech"),
  2. the clip's body order == cfg.kinematic_info.body_names (1:1),
  3. the clip's DOF count == cfg.number_of_actions,
  4. the MJCF's total simulated mass == the S001 subject mass (81.65 kg).

Usage:  python projects/biomech/bench_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from biomech.fitting.anthropometry import read_mp
from biomech.tests import SUBJECT_MP

_REPO = Path(__file__).resolve().parents[2]
_ASSET = _REPO / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"
_MOTION = _REPO / "projects" / "data" / "S001" / "Trial101_full.motion"


def main():
    from protomotions.robot_configs.factory import robot_config

    print(f"asset : {_ASSET}  ({_ASSET.stat().st_size/1e3:.1f} kB)")
    print(f"motion: {_MOTION}  ({_MOTION.stat().st_size/1e6:.1f} MB)")

    cfg = robot_config("biomech")
    ki = cfg.kinematic_info
    cfg_bodies = list(ki.body_names)
    print(f"\nrobot 'biomech': {len(cfg_bodies)} bodies, {cfg.number_of_actions} DOFs")

    clip = torch.load(str(_MOTION), weights_only=False)
    nb_clip = clip["rigid_body_pos"].shape[1]
    ndof_clip = clip["dof_pos"].shape[1]
    F = clip["rigid_body_pos"].shape[0]
    print(f"clip          : {nb_clip} bodies, {ndof_clip} DOFs, {F} frames")

    # ---- body/DOF count parity ----
    ok_bodies = nb_clip == len(cfg_bodies)
    ok_dofs = ndof_clip == cfg.number_of_actions
    print(f"\n  body count match : {'PASS' if ok_bodies else 'FAIL'} "
          f"({nb_clip} vs {len(cfg_bodies)})")
    print(f"  DOF count match  : {'PASS' if ok_dofs else 'FAIL'} "
          f"({ndof_clip} vs {cfg.number_of_actions})")

    # ---- body order parity (rebuild sim body order from the SAME MJCF via MuJoCo) ----
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    sim_bodies = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(1, model.nbody)
    ]
    ok_order = sim_bodies == cfg_bodies
    print(f"  body order match : {'PASS' if ok_order else 'FAIL'}")
    if not ok_order:
        for i, (a, b) in enumerate(zip(sim_bodies, cfg_bodies)):
            if a != b:
                print(f"      [{i}] clip/mjcf={a!r}  cfg={b!r}")

    # ---- total simulated mass ----
    subject_mass = float(read_mp(str(SUBJECT_MP)).get("Bodymass", 0.0))
    sim_mass = float(np.sum(model.body_mass[1:]))  # exclude world
    ok_mass = abs(sim_mass - subject_mass) < 1e-3
    print(f"\n  subject mass     : {subject_mass:.3f} kg")
    print(f"  sim (MJCF) mass  : {sim_mass:.3f} kg")
    print(f"  mass match       : {'PASS' if ok_mass else 'FAIL'} "
          f"(|Δ|={abs(sim_mass-subject_mass)*1e3:.3f} g)")

    # ---- geometry parity: does the robot reproduce the motion's body layout? ----
    # Feed the clip's DOFs back through MuJoCo FK on the on-disk asset and compare
    # ROOT-RELATIVE, DE-ROTATED body positions. That quantity is invariant to the
    # free-root placement AND to the global up-axis convention, so it isolates the
    # kinematic geometry itself -- bone lengths (geom sizes) and joint positions
    # (body offsets). A mismatch in any segment length would surface as a residual
    # on that body and its descendants.
    nq = model.nq
    ok_nq = nq == 7 + ndof_clip
    dof_pos = clip["dof_pos"].numpy().astype(np.float64)
    pos_clip = clip["rigid_body_pos"].numpy().astype(np.float64)
    quat_clip = clip["rigid_body_rot"].numpy().astype(np.float64)  # xyzw

    def rmat_xyzw(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    data = mujoco.MjData(model)
    sample = np.linspace(0, F - 1, min(F, 50)).astype(int)
    max_mm = 0.0
    for f in sample:
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0  # identity wxyz root quat
        data.qpos[7:] = dof_pos[f]
        mujoco.mj_kinematics(model, data)
        xp = data.xpos[1:].copy()
        Rr = data.xmat[1].reshape(3, 3)
        rel_asset = (xp - xp[0]) @ Rr  # R_root^T (p_i - p_0)
        Rc = rmat_xyzw(quat_clip[f, 0])
        rel_clip = (pos_clip[f] - pos_clip[f, 0]) @ Rc
        err = np.linalg.norm(rel_asset - rel_clip, axis=1).max()
        max_mm = max(max_mm, err * 1e3)
    # float32 storage of rigid_body_pos quantizes at ~|pos|*2^-23; after tm2og the
    # overground positions can reach hundreds of metres, so the geometry tolerance
    # must track that ULP floor rather than a fixed sub-micron bound. A few ULP of
    # the largest coordinate is the smallest difference two stored positions can
    # resolve; anything under that is storage precision, not a geometry mismatch.
    max_abs = float(np.abs(pos_clip).max())
    ulp_floor_mm = max_abs * (2.0 ** -23) * 1e3
    tol_mm = max(1e-3, 16.0 * ulp_floor_mm)
    ok_geom = max_mm < tol_mm
    print(f"\n  nq == 7+ndof     : {'PASS' if ok_nq else 'FAIL'} ({nq} vs {7+ndof_clip})")
    print(f"  geometry match   : {'PASS' if ok_geom else 'FAIL'} "
          f"(max root-relative body error over {len(sample)} frames = {max_mm:.2e} mm; "
          f"float32 tol = {tol_mm:.2e} mm @ |pos|<={max_abs:.1f} m)")

    all_ok = ok_bodies and ok_dofs and ok_order and ok_mass and ok_nq and ok_geom
    print(f"\nRESULT: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
