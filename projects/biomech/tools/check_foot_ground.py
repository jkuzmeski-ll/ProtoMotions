# SPDX-License-Identifier: MIT

"""Diagnose whether the exported bone-mesh foot ever brings the calcaneus to the ground.

Drives the committed S001 MJCF (`protomotions/data/assets/mjcf/biomech_rajagopal.xml`)
through the cached reconstructed walk motion (`docs/figures/_s001_ik_cache.npz`) with
`mj_kinematics`, then, per frame, computes in the MJCF's native (OpenSim Y-up) world
frame:

  * the lowest vertex of the calcaneus mesh (`r_foot` on `calcn_r`),
  * the lowest vertex of the forefoot/toes mesh (`r_bofoot` on `toes_r`),
  * the whole right-foot lowest vertex (= the effective ground-contact level),
  * the ankle dorsi/plantarflexion coordinate.

If the calcaneus never approaches the whole-foot minimum while the toes always sit at it,
the foot is stuck plantarflexed (kinematic). If the ankle DOES dorsiflex yet the heel mesh
still floats, the mesh geom is mis-placed (rendering/frame bug).

Run from the repo root::

    .venv/Scripts/python.exe projects/biomech/tools/check_foot_ground.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # projects/

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"
_ASSET = _REPO / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"


def _mesh_geoms_for_body(model, body_id):
    """Return list of (geom_id, mesh_vertices(local)) for mesh geoms on a body."""
    import mujoco

    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        adr = model.mesh_vertadr[mid]
        num = model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr : adr + num].reshape(-1, 3).astype(np.float64)
        out.append((g, verts))
    return out


def _world_verts(data, gid, verts):
    R = data.geom_xmat[gid].reshape(3, 3)
    return verts @ R.T + data.geom_xpos[gid]


def main() -> int:
    import mujoco

    from biomech.export.mjcf import dart_q_to_mjcf_qpos

    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)  # (F,37) DART q
    scales = np.asarray(cache["scales"], dtype=np.float64)
    F = poses.shape[0]

    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    data = mujoco.MjData(model)

    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("pelvis", "calcn_r", "toes_r")}
    calcn_meshes = _mesh_geoms_for_body(model, bid["calcn_r"])
    toes_meshes = _mesh_geoms_for_body(model, bid["toes_r"])
    print(f"calcn_r mesh geoms: {[model.geom(g).name or g for g,_ in calcn_meshes]}")
    print(f"toes_r  mesh geoms: {[model.geom(g).name or g for g,_ in toes_meshes]}")

    # Marker sites: these are fitted to the measured capture markers to ~1 cm, so their
    # world height is a faithful proxy for what the REAL foot markers do.
    def _sids(names):
        out = []
        for nm in names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
            if sid >= 0:
                out.append(sid)
        return out

    heel_sites = _sids(["mk_RCAL", "mk_RCAL2", "mk_RCAL3"])
    fore_sites = _sids(["mk_RMT1", "mk_RMT5", "mk_RTOE", "mk_RTOE_TIP"])
    print(f"heel sites: {[model.site(s).name for s in heel_sites]}")
    print(f"fore sites: {[model.site(s).name for s in fore_sites]}")

    # ankle coordinate index in DART q vector
    dof_names = list(spec.dof_names)
    ankle_idx = dof_names.index("ankle_angle_r") if "ankle_angle_r" in dof_names else None

    heel_min = np.empty(F)      # lowest vertex of calcaneus mesh (vertical)
    toe_min = np.empty(F)       # lowest vertex of forefoot mesh
    foot_min = np.empty(F)      # lowest vertex of whole right foot
    heel_site = np.empty(F)     # lowest heel marker site
    fore_site = np.empty(F)     # lowest forefoot marker site
    pelvis_v = np.empty(F)
    ankle = np.empty(F)
    foot_pitch = np.empty(F)   # foot-segment pitch vs horizontal (+ = heel-down/toes-up)

    # Detect vertical axis from neutral: the axis where feet sit far below pelvis.
    mujoco.mj_kinematics(model, data)  # qpos=0 neutral
    d0 = data.xpos[bid["calcn_r"]] - data.xpos[bid["pelvis"]]
    vax = int(np.argmax(np.abs(d0)))  # axis with largest pelvis->foot separation
    # "down" points from pelvis toward the foot. We want height() so that smaller ==
    # closer to the ground. If down is -axis (foot has smaller coord), raw coord works
    # (sign +1); if down is +axis, flip so smaller == lower.
    sign = 1.0 if d0[vax] < 0 else -1.0
    axname = "XYZ"[vax]
    print(f"neutral pelvis->calcn_r delta = {np.round(d0,3)} -> vertical axis = "
          f"{axname} (down = {'-' if d0[vax] < 0 else '+'}{axname})")
    # We measure "height" as sign-corrected component so that smaller = closer to ground.
    def height(p):
        return sign * p[..., vax]

    for f in range(F):
        qp = dart_q_to_mjcf_qpos(spec, poses[f], scales, "coupled")
        x, y, z, w = qp[3:7]
        qp[3:7] = (w, x, y, z)
        data.qpos[:] = qp
        mujoco.mj_kinematics(model, data)

        hv = min(height(_world_verts(data, g, v)).min() for g, v in calcn_meshes)
        tv = min(height(_world_verts(data, g, v)).min() for g, v in toes_meshes)
        heel_min[f] = hv
        toe_min[f] = tv
        foot_min[f] = min(hv, tv)
        heel_site[f] = min(height(data.site_xpos[s]) for s in heel_sites)
        fore_site[f] = min(height(data.site_xpos[s]) for s in fore_sites)
        pelvis_v[f] = height(data.xpos[bid["pelvis"]])
        ankle[f] = np.degrees(poses[f, ankle_idx]) if ankle_idx is not None else np.nan
        # foot pitch: elevation of calcn_r body x-axis (toward toes) above horizontal.
        xaxis = data.xmat[bid["calcn_r"]].reshape(3, 3)[:, 0]
        vert_comp = sign * xaxis[vax]  # + component along "down"; toes-down if >0
        horiz = np.sqrt(max(1.0 - xaxis[vax] ** 2, 1e-12))
        foot_pitch[f] = np.degrees(np.arctan2(-vert_comp, horiz))  # + = toes up

    # Ground = lowest the whole foot ever reaches over the cycle.
    ground = foot_min.min()
    heel_clear = heel_min - ground   # how far the calcaneus stays above ground
    toe_clear = toe_min - ground

    print("\n=== over the walk cycle (units: m above cycle-min ground) ===")
    print(f"frames={F}  ground level (min whole-foot height) = {ground:.4f}")
    print(f"calcaneus clearance above ground: min={heel_clear.min()*1000:6.1f}mm "
          f"mean={heel_clear.mean()*1000:6.1f}mm max={heel_clear.max()*1000:6.1f}mm")
    print(f"forefoot  clearance above ground: min={toe_clear.min()*1000:6.1f}mm "
          f"mean={toe_clear.mean()*1000:6.1f}mm max={toe_clear.max()*1000:6.1f}mm")
    print(f"ankle_angle_r: min={np.nanmin(ankle):.1f} mean={np.nanmean(ankle):.1f} "
          f"max={np.nanmax(ankle):.1f} deg  (+ = dorsiflexion)")

    # --- marker-site view (proxy for the REAL measured foot, fit to ~1 cm) ---
    sground = min(heel_site.min(), fore_site.min())
    heel_s_clear = heel_site - sground
    fore_s_clear = fore_site - sground
    print("\n=== marker-site view (fitted to measured markers ~1cm) ===")
    print(f"heel markers   clearance: min={heel_s_clear.min()*1000:6.1f}mm "
          f"mean={heel_s_clear.mean()*1000:6.1f}mm")
    print(f"forefoot mkrs  clearance: min={fore_s_clear.min()*1000:6.1f}mm "
          f"mean={fore_s_clear.mean()*1000:6.1f}mm")
    heel_lowest_marker = heel_site <= fore_site + 1e-6
    print(f"frames where heel markers are lower than forefoot markers: "
          f"{int(heel_lowest_marker.sum())}/{F} "
          f"({100*heel_lowest_marker.mean():.0f}%)")
    # Vertical heel->toe drop of the mesh vs of the markers (persistent toe-down bias).
    print(f"mean heel-minus-toe height: mesh={np.mean(heel_min-toe_min)*1000:.1f}mm  "
          f"markers={np.mean(heel_site-fore_site)*1000:.1f}mm")

    # --- bone clearance against the SKIN/marker ground (single consistent datum) ---
    # Use the marker-defined ground so bone and marker clearances are comparable.
    print("\n=== bone-mesh clearance above the SKIN (marker) ground ===")
    print(f"calcaneus bone: min={ (heel_min-sground).min()*1000:6.1f}mm "
          f"mean={(heel_min-sground).mean()*1000:6.1f}mm  (heel-pad ~15-20mm expected)")
    print(f"forefoot bone : min={ (toe_min-sground).min()*1000:6.1f}mm "
          f"mean={(toe_min-sground).mean()*1000:6.1f}mm  (thin forefoot tissue expected)")

    # --- foot-segment pitch relative to world horizontal (heel-down = +) ---
    # calcn_r body x-axis points toward the toes; its elevation vs horizontal is the
    # foot pitch. Positive = toes up / heel down (dorsiflexed contact posture).
    print("\n=== foot-segment pitch vs ground (+ = toes-up / heel-down) ===")
    print(f"foot pitch: min={foot_pitch.min():.1f} mean={foot_pitch.mean():.1f} "
          f"max={foot_pitch.max():.1f} deg")
    print(f"frames with heel-down posture (pitch>0): "
          f"{int((foot_pitch>0).sum())}/{F} ({100*(foot_pitch>0).mean():.0f}%)")

    # Heel strike = the frame where the calcaneus is lowest.
    hs = int(np.argmin(heel_min))
    print(f"\nframe of lowest calcaneus (heel-strike candidate) = {hs}")
    print(f"  heel clearance={heel_clear[hs]*1000:.1f}mm  "
          f"toe clearance={toe_clear[hs]*1000:.1f}mm  ankle={ankle[hs]:.1f}deg")
    # Is the heel EVER the lowest part of the foot (true heel strike)?
    heel_is_lowest = heel_min <= toe_min + 1e-6
    n_heel = int(heel_is_lowest.sum())
    print(f"  frames where calcaneus is the lowest foot point: {n_heel}/{F} "
          f"({100*n_heel/F:.0f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
