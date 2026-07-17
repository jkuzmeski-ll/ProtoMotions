# SPDX-License-Identifier: MIT
"""Clean heel-contact check using the trusted foot_trajectory_from_motion path.

Replaces the earlier hand-rolled quaternion version (which had a frame bug that faked a
~10 deg toe-down). Reports, over the measured right-foot LOADED stance (cache GRF), the
world height of the calcn_r collision box heel vs forefoot corners and the lowest of all
right-foot collision geoms -- the honest "does the flat foot plant on the floor?" check.
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


def main() -> int:
    import mujoco
    import torch
    from biomech.contact.elastic_foundation import _quat_rotate_np
    from biomech.contact.kinematics import foot_trajectory_from_motion
    from biomech.export.protomotions_robot import build_biomech_robot_config

    data = torch.load(str(_MOTION), weights_only=False)
    cfg = build_biomech_robot_config(
        asset_file_name="mjcf/biomech_rajagopal_boxes.xml",
        asset_root=str(_ASSET_ROOT))
    clip = SimpleNamespace(body_names=cfg.kinematic_info.body_names, data=data)

    model = mujoco.MjModel.from_xml_path(str(_ASSET))

    def geom_corners(name):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        gp = model.geom_pos[gid].astype(np.float64)
        gs = model.geom_size[gid].astype(np.float64)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)], dtype=np.float64)
        return gp + signs * gs

    def world_z(body, pts):
        pos, quat, _, _ = foot_trajectory_from_motion(clip, body)
        F = pos.shape[0]
        pl = np.broadcast_to(pts, (F, pts.shape[0], 3))
        return (pos[:, None, :] + _quat_rotate_np(quat[:, None, :], pl))[:, :, 2]

    calc = geom_corners("col_calcn_r_box")
    x = calc[:, 0]; y = calc[:, 1]
    bottom = y < y.mean()
    heel = calc[bottom & (x < x.mean())]
    fore = calc[bottom & (x > x.mean())]
    heel_z = world_z("calcn_r", heel).min(axis=1)
    fore_z = world_z("calcn_r", fore).min(axis=1)

    # all right-foot colliding geoms
    all_z = np.full(heel_z.shape[0], np.inf)
    for body in ("calcn_r", "toes_r"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_contype[g] == 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            all_z = np.minimum(all_z, world_z(body, geom_corners(name)).min(axis=1))

    cache = np.load(_CACHE, allow_pickle=True)
    grf = None
    for k in ("grf_R", "grf_r"):
        if k in cache:
            gg = np.asarray(cache[k]); grf = gg[:, 2] if gg.ndim == 2 else gg
    F = heel_z.shape[0]
    loaded = (grf > max(50.0, 0.3 * np.nanmax(grf))) if grf is not None else np.ones(F, bool)

    print("=== right foot vs floor over MEASURED loaded stance (trusted FK path) ===")
    li = np.where(loaded)[0]
    print(f"loaded frames {li.min()}..{li.max()} ({loaded.sum()} frames)\n")
    print(" frame  heel_mm  fore_mm  allgeom_mm  grf_N")
    for f in li[::3]:
        print(f" {f:4d}  {heel_z[f]*1e3:7.1f}  {fore_z[f]*1e3:7.1f}  "
              f"{all_z[f]*1e3:8.1f}   {grf[f]:5.0f}")
    print(f"\nheel corner over loaded stance: min={heel_z[loaded].min()*1e3:.1f} mm, "
          f"mean={heel_z[loaded].mean()*1e3:.1f} mm")
    print(f"forefoot corner over loaded stance: min={fore_z[loaded].min()*1e3:.1f} mm, "
          f"mean={fore_z[loaded].mean()*1e3:.1f} mm")
    print(f"lowest of all geoms over loaded stance: min={all_z[loaded].min()*1e3:.1f} mm, "
          f"mean={all_z[loaded].mean()*1e3:.1f} mm  (want ~0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
