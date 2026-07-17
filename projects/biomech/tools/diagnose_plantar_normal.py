# SPDX-License-Identifier: MIT
"""Compare the anchor-derived sole normal against the TRUE plantar plane (bone mesh).

The sole `up` axis from `_foot_axes(anchors)` is the normal of the skin-marker triangle
(heel/mt5/toe). This script extracts the actual plantar surface of the calcaneus+toe bone
meshes (lowest band of vertices in the calcn frame) and fits a plane to it, giving the
anatomical plantar normal in the calcn frame -- the reference `up` *should* match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"
_ASSET = _REPO / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"


def _fit_plane_normal(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[-1]
    return n / np.linalg.norm(n)


def main() -> int:
    import mujoco
    from biomech.contact.foot_geometry import calcn_anchors_from_spec, _foot_axes

    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    scales = np.asarray(cache["scales"], dtype=np.float64)

    anchors = calcn_anchors_from_spec(spec, "R", group_scales=scales)
    _, _, up_anchor = _foot_axes(anchors)

    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)  # zero config

    calcn_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")
    # collect calcn_r mesh vertices expressed in the calcn_r body frame
    Rb = data.xmat[calcn_id].reshape(3, 3)
    pb = data.xpos[calcn_id]
    verts_body = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != calcn_id:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        va = model.mesh_vertadr[mid]; vn = model.mesh_vertnum[mid]
        v = model.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float64)
        Rg = data.geom_xmat[g].reshape(3, 3); pg = data.geom_xpos[g]
        world = v @ Rg.T + pg
        body = (world - pb) @ Rb  # into calcn frame
        verts_body.append(body)
    V = np.concatenate(verts_body, axis=0)
    print(f"calcn_r mesh: {V.shape[0]} verts, "
          f"y-range=[{V[:,1].min():.4f}, {V[:,1].max():.4f}]")

    # plantar band = lowest 8 mm of vertices in the calcn frame (-y is plantar-down)
    ymin = V[:, 1].min()
    band = V[V[:, 1] < ymin + 0.008]
    print(f"plantar band (<{ymin+0.008:.4f}): {band.shape[0]} verts")

    n_bone = _fit_plane_normal(band)
    if n_bone[1] < 0:
        n_bone = -n_bone  # point up (+y-ish)

    def pitch_roll(n):
        pitch = np.degrees(np.arctan2(n[0], n[1]))  # sagittal, toe-down +
        roll = np.degrees(np.arctan2(n[2], n[1]))   # frontal, lateral +
        tilt = np.degrees(np.arccos(np.clip(n[1], -1, 1)))
        return pitch, roll, tilt

    print("\n=== plantar normal in the calcn_r frame ===")
    p, r, t = pitch_roll(up_anchor)
    print(f"anchor up   = {up_anchor}")
    print(f"   sagittal pitch={p:.1f}  frontal roll={r:.1f}  total tilt off +y={t:.1f} deg")
    p, r, t = pitch_roll(n_bone)
    print(f"bone plantar= {n_bone}")
    print(f"   sagittal pitch={p:.1f}  frontal roll={r:.1f}  total tilt off +y={t:.1f} deg")
    print(f"calcn +y    = [0,1,0]  (sagittal 0  roll 0  tilt 0)")

    ang = np.degrees(np.arccos(np.clip(up_anchor @ n_bone, -1, 1)))
    print(f"\nangle(anchor up, bone plantar normal) = {ang:.1f} deg  <-- sole tilt error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
