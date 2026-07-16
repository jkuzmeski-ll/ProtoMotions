# SPDX-License-Identifier: MIT
"""Side-view render of the right-foot BONE meshes at heel-strike and push-off frames.

Drives the committed MJCF through the cached motion, projects the ``r_foot`` +
``r_bofoot`` mesh triangles onto the sagittal (X vertical=Y) plane, and overlays the
foot marker sites and the skin/marker ground line so we can see whether the calcaneus
bone reaches the ground.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

_BIOMECH = Path(__file__).resolve().parents[1]
_REPO = _BIOMECH.parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"
_ASSET = _REPO / "protomotions" / "data" / "assets" / "mjcf" / "biomech_rajagopal.xml"
_OUT = _BIOMECH / "docs" / "figures" / "foot_bone_check.png"


def main() -> int:
    import mujoco
    from biomech.export.mjcf import dart_q_to_mjcf_qpos

    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)
    scales = np.asarray(cache["scales"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    data = mujoco.MjData(model)

    def body(nm):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)

    def foot_geoms():
        out = []
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            if model.body(b).name in ("calcn_r", "toes_r") and \
               model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                va = model.mesh_vertadr[mid]; vn = model.mesh_vertnum[mid]
                fa = model.mesh_faceadr[mid]; fn = model.mesh_facenum[mid]
                v = model.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float64)
                f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
                out.append((g, v, f))
        return out

    fg = foot_geoms()
    site_ids = {nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mk_" + nm)
                for nm in ("RCAL", "RCAL2", "RCAL3", "RTOE", "RMT5", "RMT1", "RTOE_TIP")}

    def drive(fr):
        qp = dart_q_to_mjcf_qpos(spec, poses[fr], scales, "coupled")
        x, y, z, w = qp[3:7]
        qp[3:7] = (w, x, y, z)
        data.qpos[:] = qp
        mujoco.mj_kinematics(model, data)

    # skin ground = lowest foot marker over the whole cycle (Y is vertical; down=-Y)
    gy = np.inf
    for fr in range(poses.shape[0]):
        drive(fr)
        for sid in site_ids.values():
            gy = min(gy, data.site_xpos[sid][1])

    # ground-registration offset = lowest foot bone-mesh vertex over the clip (MuJoCo Y).
    # Subtracting it puts the deepest foot contact on the sim floor (Z-up z == MuJoCo Y).
    z_off = np.inf
    for fr in range(poses.shape[0]):
        drive(fr)
        for g, v, _ in fg:
            R = data.geom_xmat[g].reshape(3, 3); p = data.geom_xpos[g]
            z_off = min(z_off, float((v @ R.T + p)[:, 1].min()))
    print(f"clip ground offset z_off={z_off:.4f} m (sim floor z=0)")

    def zup(y):  # MuJoCo Y-up height -> registered ProtoMotions Z-up height
        return y - z_off

    # Right-foot stance from the MEASURED heel marker (RHEE) height, which is the ground
    # truth for foot posture. grf_R is not reliably aligned to the right foot here.
    from biomech.session import load_session
    from biomech.tests import TRIAL_C3D, SPEEDCHANGE
    lo_pf, hi_pf = (int(v) for v in cache["window"])
    sess = load_session(str(TRIAL_C3D), speedchange_path=str(SPEEDCHANGE))
    labels = list(sess.marker_labels); P = np.asarray(sess.markers)

    def mcol(lbl):
        return (P[lo_pf:hi_pf, labels.index(lbl), 2] if lbl in labels
                else np.full(hi_pf - lo_pf, np.nan))

    rhee = mcol("RHEE")
    rfore = np.nanmean(np.stack([mcol("RMTH1"), mcol("RMTH5"), mcol("RTOE")]), axis=0)
    hs = int(np.nanargmin(rhee))                 # lowest heel = heel strike
    mid = int(np.nanargmin(rhee - rfore))        # heel most below forefoot = flat/loaded
    frames = {f"heel-strike (lowest meas. heel, frame {hs})": hs,
              f"flat-foot (heel below forefoot, frame {mid})": mid}
    print(f"measured heel-strike frame {hs} (heelZ={rhee[hs]:.3f}), "
          f"flat-foot frame {mid} (heel-fore={rhee[mid]-rfore[mid]:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (title, fr) in zip(axes, frames.items()):
        drive(fr)
        polys = []
        for g, v, f in fg:
            R = data.geom_xmat[g].reshape(3, 3); p = data.geom_xpos[g]
            w = v @ R.T + p
            for tri in f:
                xy = w[tri][:, [0, 1]].copy()
                xy[:, 1] = zup(xy[:, 1])  # registered Z-up height, floor at 0
                polys.append(xy)
        pc = PolyCollection(polys, facecolors=(0.85, 0.82, 0.70, 0.9),
                            edgecolors=(0.4, 0.38, 0.3, 0.35), linewidths=0.2)
        ax.add_collection(pc)
        for nm, sid in site_ids.items():
            sx, sy = data.site_xpos[sid][0], zup(data.site_xpos[sid][1])
            c = "crimson" if "CAL" in nm else "navy"
            ax.plot(sx, sy, "o", ms=6, color=c, zorder=5)
            ax.annotate(nm, (sx, sy), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.axhline(0.0, color="saddlebrown", lw=2, label="sim floor (z=0)")
        ax.set_title(f"{title}\nframe {fr}")
        ax.set_xlabel("fwd X (m)"); ax.set_ylabel("registered up z (m)")
        ax.set_aspect("equal"); ax.autoscale_view(); ax.legend(loc="upper right")
    fig.suptitle("S001 right foot bone mesh vs sim floor after ground registration "
                 "(red=heel markers)")
    fig.tight_layout()
    fig.savefig(str(_OUT), dpi=120)
    print("wrote", _OUT, "frames:", frames, "ground Y=", round(gy, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
