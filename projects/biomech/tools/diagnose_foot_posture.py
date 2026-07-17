# SPDX-License-Identifier: MIT
"""Authoritative foot-posture measure: world-z of heel-most vs toe-most plantar points.

Avoids axis-convention pitfalls: takes the calcn/toes collision geoms, drives them with
the exported .motion via the verified rotation, and reports, over the measured loaded
stance, the world height of the single most-posterior (heel) and most-anterior (toe)
plantar corner. dz = heel_z - toe_z is the honest toe-down signal (>0 means heel above
toe). Also prints the forward-axis and plantar-normal sagittal pitches for reference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_ASSET_ROOT = _REPO / "protomotions" / "data" / "assets"
_ASSET = _ASSET_ROOT / "mjcf" / "biomech_rajagopal_boxes.xml"
_MOTION = _BIOMECH / "data" / "motions" / "biomech_s001_walk.motion"
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"


def q2m(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main() -> int:
    import mujoco
    import torch
    from biomech.contact.kinematics import foot_trajectory_from_motion
    from biomech.export.protomotions_robot import build_biomech_robot_config

    model = mujoco.MjModel.from_xml_path(str(_ASSET))

    def corners(name):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        gp = model.geom_pos[gid].astype(np.float64)
        gs = model.geom_size[gid].astype(np.float64)
        s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        return gp + s * gs

    data = torch.load(str(_MOTION), weights_only=False)
    cfg = build_biomech_robot_config(
        asset_file_name="mjcf/biomech_rajagopal_boxes.xml", asset_root=str(_ASSET_ROOT))
    clip = SimpleNamespace(body_names=cfg.kinematic_info.body_names, data=data)

    cpos, cquat, _, _ = foot_trajectory_from_motion(clip, "calcn_r")
    tpos, tquat, _, _ = foot_trajectory_from_motion(clip, "toes_r")
    cc = corners("col_calcn_r_box")
    tc = corners("col_toes_r_box")
    # heel = most-posterior bottom corner of calcn box; toe = most-anterior bottom of toes
    heel_local = cc[cc[:, 1] < cc[:, 1].mean()]
    heel_local = heel_local[np.argmin(heel_local[:, 0])]
    toe_local = tc[tc[:, 1] < tc[:, 1].mean()]
    toe_local = toe_local[np.argmax(toe_local[:, 0])]

    F = cpos.shape[0]
    heel_z = np.array([(cpos[f] + q2m(cquat[f]) @ heel_local)[2] for f in range(F)])
    toe_z = np.array([(tpos[f] + q2m(tquat[f]) @ toe_local)[2] for f in range(F)])
    # forward-axis and plantar-normal sagittal pitch of calcn
    fwd_pitch = np.array([np.degrees(np.arcsin(np.clip(-(q2m(cquat[f]) @ [1, 0, 0])[2], -1, 1)))
                          for f in range(F)])
    nrm = np.array([q2m(cquat[f]) @ [0, 1, 0] for f in range(F)])
    nrm_pitch = np.degrees(np.arctan2(nrm[:, 0], nrm[:, 2]))

    cache = np.load(_CACHE, allow_pickle=True)
    grf = None
    for k in ("grf_R", "grf_r"):
        if k in cache:
            gg = np.asarray(cache[k]); grf = gg[:, 2] if gg.ndim == 2 else gg
    loaded = (grf > max(50.0, 0.3 * np.nanmax(grf))) if grf is not None else np.ones(F, bool)
    li = np.where(loaded)[0]

    print(" frame  heel_mm  toe_mm   dz_mm  fwd_pitch  nrm_pitch  grf")
    for f in li[::3]:
        print(f" {f:4d}  {heel_z[f]*1e3:7.1f}  {toe_z[f]*1e3:6.1f}  {(heel_z[f]-toe_z[f])*1e3:6.1f}"
              f"    {fwd_pitch[f]:6.1f}     {nrm_pitch[f]:6.1f}   {grf[f]:5.0f}")
    dz = (heel_z - toe_z)[loaded]
    print(f"\nover loaded stance: heel-toe dz mean={dz.mean()*1e3:.1f} mm "
          f"(>0 = heel above toe = toe-down)")
    print(f"  forward-axis pitch mean={fwd_pitch[loaded].mean():.1f} deg; "
          f"plantar-normal pitch mean={nrm_pitch[loaded].mean():.1f} deg")
    print(f"  heel_z min={heel_z[loaded].min()*1e3:.1f} mm; toe_z min={toe_z[loaded].min()*1e3:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
