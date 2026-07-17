# SPDX-License-Identifier: MIT
"""Verify the rearfoot collision box lands its heel on the floor during stance.

Drives the *boxes* collision asset's ``col_calcn_r_box`` (what a mimic character actually
contacts the floor with) through the exported, ground-registered .motion and reports the
world height (Z-up, floor=0) of the box's posterior-bottom (heel) corners vs its
anterior-bottom (forefoot) corners over the clip and during stance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_ASSET = _REPO / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal_boxes.xml"
_MOTION = _BIOMECH / "data" / "motions" / "biomech_s001_walk.motion"


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def main() -> int:
    import mujoco
    import torch

    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    data = mujoco.MjData(model)

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "col_calcn_r_box")
    bid = model.geom_bodyid[gid]
    body_name = model.body(bid).name
    gpos = model.geom_pos[gid].astype(np.float64)   # box center in calcn frame (Y-up)
    gsize = model.geom_size[gid].astype(np.float64)  # half-extents
    # 8 corners in the calcn (body) frame
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float64)
    corners = gpos + signs * gsize  # (8,3), Y-up calcn frame
    # heel corners = most posterior (min x); forefoot = most anterior (max x); bottom = min y
    x, y = corners[:, 0], corners[:, 1]
    bottom = y < y.mean()
    heel = corners[bottom & (x < x.mean())]
    fore = corners[bottom & (x > x.mean())]
    print(f"box on {body_name}: center={gpos}, half={gsize}")
    print(f"  heel bottom corners: {heel.shape[0]}, forefoot bottom corners: {fore.shape[0]}")

    clip = torch.load(str(_MOTION), weights_only=False)
    pos = np.asarray(clip["rigid_body_pos"])   # (F,B,3) Z-up, registered
    rot = np.asarray(clip["rigid_body_rot"])   # (F,B,4) quat xyzw

    from biomech.export.protomotions_robot import build_biomech_robot_config
    cfg = build_biomech_robot_config(
        asset_file_name="mjcf/biomech_rajagopal_boxes.xml",
        asset_root=str(_REPO / "protomotions" / "data" / "assets"))
    ci = cfg.kinematic_info.body_names.index(body_name)

    F = pos.shape[0]

    def corner_z(cs):
        z = np.empty(F)
        for f in range(F):
            R = _quat_xyzw_to_mat(rot[f, ci])
            z[f] = (cs @ R.T + pos[f, ci])[:, 2].min()
        return z

    heel_z = corner_z(heel)
    fore_z = corner_z(fore)
    print("\n=== rearfoot box heel-corner world height (Z-up, floor=0) ===")
    print(f"heel corners: min={heel_z.min()*1e3:.1f} mm (frame {int(heel_z.argmin())})")
    print(f"fore corners: min={fore_z.min()*1e3:.1f} mm (frame {int(fore_z.argmin())})")

    # sagittal foot pitch per frame from the box (heel-up positive)
    x_span = float(heel[:, 0].mean() - fore[:, 0].mean())  # <0 (heel behind fore)
    span = abs(x_span)
    pitch = np.degrees(np.arctan2(heel_z - fore_z, span))  # + = heel above fore (toe-down)
    f_flat = int(np.argmin(np.abs(pitch)))
    print("\n=== sagittal foot pitch (deg, + = heel-up / toe-down) ===")
    print(f"min/mean/max = {pitch.min():.1f}/{pitch.mean():.1f}/{pitch.max():.1f}")
    print(f"flattest foot at frame {f_flat}: pitch={pitch[f_flat]:.1f} deg, "
          f"heel={heel_z[f_flat]*1e3:.1f} mm, fore={fore_z[f_flat]*1e3:.1f} mm")

    stance = np.minimum(heel_z, fore_z) < 0.010
    if stance.any():
        idx = np.where(stance)[0]
        print(f"\nstance frames (box within 10 mm): {idx[0]}..{idx[-1]} ({stance.sum()} frames)")
        print(f"heel-corner min during stance: {heel_z[stance].min()*1e3:.1f} mm")
        print(f"heel strike (first stance frame {idx[0]}): "
              f"heel={heel_z[idx[0]]*1e3:.1f} mm, fore={fore_z[idx[0]]*1e3:.1f} mm")

    # --- correlate with measured right-foot GRF (cache) to find loaded stance ---
    cache = np.load(_BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz", allow_pickle=True)
    grf = None
    for key in ("grf_R", "grf_r"):
        if key in cache:
            g = np.asarray(cache[key])
            grf = g[:, 2] if g.ndim == 2 else g
            break
    if grf is not None and grf.shape[0] == F:
        loaded = grf > max(50.0, 0.3 * float(np.nanmax(grf)))
        print(f"\nmeasured GRF_R loaded frames: {int(loaded.sum())} "
              f"(peak {np.nanmax(grf):.0f} N)")
        if loaded.any():
            li = np.where(loaded)[0]
            print(f"  loaded window {li[0]}..{li[-1]}")
            # flattest loaded frame
            lp = np.abs(pitch)[loaded]
            fflat = li[int(np.argmin(lp))]
            print(f"  flattest LOADED frame {fflat}: pitch={pitch[fflat]:.1f} deg, "
                  f"heel={heel_z[fflat]*1e3:.1f} mm, fore={fore_z[fflat]*1e3:.1f} mm")
            print(f"  min box corner during load: "
                  f"{np.minimum(heel_z, fore_z)[loaded].min()*1e3:.1f} mm")

    # --- lowest point over ALL right-foot collision geoms (calcn + toes) ---
    all_min = np.full(F, np.inf)
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        bn = model.body(b).name
        if bn not in ("calcn_r", "toes_r"):
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual only
        gp = model.geom_pos[g].astype(np.float64)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            gs = model.geom_size[g].astype(np.float64)
            pts = gp + signs * gs
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(model.geom_size[g][0])
            pts = gp[None, :] + np.array([[0, -r, 0]])
        else:
            continue
        gi = cfg.kinematic_info.body_names.index(bn)
        for f in range(F):
            R = _quat_xyzw_to_mat(rot[f, gi])
            all_min[f] = min(all_min[f], (pts @ R.T + pos[f, gi])[:, 2].min())
    if grf is not None and grf.shape[0] == F and loaded.any():
        print(f"\n  lowest of ALL right-foot collision geoms during load: "
              f"{all_min[loaded].min()*1e3:.1f} mm (should be ~0)")
    print("\n frame  heel_mm  fore_mm  pitch  grf_N")
    for f in range(0, F, 3):
        g = f"{grf[f]:6.0f}" if grf is not None and grf.shape[0] == F else "   n/a"
        print(f" {f:4d}  {heel_z[f]*1e3:7.1f}  {fore_z[f]*1e3:7.1f}  {pitch[f]:5.1f}  {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
