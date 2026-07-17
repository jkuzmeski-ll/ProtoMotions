# SPDX-License-Identifier: MIT
"""Ground-truth check: does the MEASURED S001 right foot go heel-down in this window?

Compares the raw captured marker heights (RHEE vs forefoot RMTH1/RMTH5/RTOE) over the
clip window against the FITTED model's marker-site heights driven by the cached poses.
If the measured heel drops below the forefoot at some frame, the foot really does
heel-strike and the fit should reproduce it; if the measured heel stays above the
forefoot throughout, the capture window simply has no right heel-down phase (forefoot/
flat contact) and the "heel never contacts" is biomechanical reality, not a bug.
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


def main() -> int:
    import mujoco
    from biomech.export.mjcf import dart_q_to_mjcf_qpos
    from biomech.session import load_session
    from biomech.tests import TRIAL_C3D, SPEEDCHANGE

    cache = np.load(_CACHE, allow_pickle=True)
    spec = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)
    scales = np.asarray(cache["scales"], dtype=np.float64)
    lo, hi = (int(v) for v in cache["window"])

    sess = load_session(str(TRIAL_C3D), speedchange_path=str(SPEEDCHANGE))
    labels = list(sess.marker_labels)
    P = np.asarray(sess.markers)

    def mcol(lbl):  # world Z-up height (capture frame, Z vertical)
        return (P[lo:hi, labels.index(lbl), 2] if lbl in labels
                else np.full(hi - lo, np.nan))

    m_heel = mcol("RHEE")
    m_fore = np.nanmean(np.stack([mcol("RMTH1"), mcol("RMTH5"), mcol("RTOE")]), axis=0)
    m_diff = m_heel - m_fore  # <0 => heel below forefoot (heel-down)

    print("=== MEASURED right-foot marker heights (mm, capture Z-up) ===")
    print(f"heel(RHEE):      min/mean/max = {np.nanmin(m_heel)*1e3:.0f}/"
          f"{np.nanmean(m_heel)*1e3:.0f}/{np.nanmax(m_heel)*1e3:.0f}")
    print(f"forefoot:        min/mean/max = {np.nanmin(m_fore)*1e3:.0f}/"
          f"{np.nanmean(m_fore)*1e3:.0f}/{np.nanmax(m_fore)*1e3:.0f}")
    print(f"heel - forefoot: min/mean/max = {np.nanmin(m_diff)*1e3:.0f}/"
          f"{np.nanmean(m_diff)*1e3:.0f}/{np.nanmax(m_diff)*1e3:.0f} mm")
    fmin = int(np.nanargmin(m_diff))
    print(f"most heel-down at frame {fmin}: heel-fore = {m_diff[fmin]*1e3:.0f} mm "
          f"({'HEEL BELOW forefoot' if m_diff[fmin] < 0 else 'heel above forefoot'})")

    # FITTED model marker sites for the same labels
    model = mujoco.MjModel.from_xml_path(str(_ASSET))
    data = mujoco.MjData(model)

    def site(nm):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mk_" + nm)

    sids = {nm: site(nm) for nm in ("RCAL", "RTOE", "RMT5", "RMT1")}
    F = poses.shape[0]
    f_heel = np.empty(F); f_fore = np.empty(F)
    for f in range(F):
        qp = dart_q_to_mjcf_qpos(spec, poses[f], scales, "coupled")
        x, y, z, w = qp[3:7]; qp[3:7] = (w, x, y, z)
        data.qpos[:] = qp
        mujoco.mj_kinematics(model, data)
        f_heel[f] = data.site_xpos[sids["RCAL"]][1]  # Y-up height
        f_fore[f] = np.mean([data.site_xpos[sids[n]][1] for n in ("RTOE", "RMT5", "RMT1")])
    f_diff = f_heel - f_fore

    print("\n=== FITTED model marker-site heights (mm, model Y-up) ===")
    print(f"heel(RCAL):      min/mean/max = {f_heel.min()*1e3:.0f}/"
          f"{f_heel.mean()*1e3:.0f}/{f_heel.max()*1e3:.0f}")
    print(f"forefoot:        min/mean/max = {f_fore.min()*1e3:.0f}/"
          f"{f_fore.mean()*1e3:.0f}/{f_fore.max()*1e3:.0f}")
    print(f"heel - forefoot: min/mean/max = {f_diff.min()*1e3:.0f}/"
          f"{f_diff.mean()*1e3:.0f}/{f_diff.max()*1e3:.0f} mm")

    # compare the two heel-fore signals (aligned; both length F over the same window)
    n = min(len(m_diff), len(f_diff))
    md, fd = m_diff[:n] * 1e3, f_diff[:n] * 1e3
    ok = np.isfinite(md)
    bias = float(np.nanmean(fd[ok] - md[ok]))
    print(f"\nfitted(heel-fore) - measured(heel-fore): mean bias = {bias:.1f} mm "
          f"(+ = fit holds heel higher than the capture => spurious plantarflexion)")
    print(f"correlation = {np.corrcoef(md[ok], fd[ok])[0,1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
