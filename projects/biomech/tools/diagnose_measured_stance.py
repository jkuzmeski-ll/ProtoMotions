# SPDX-License-Identifier: MIT
"""Model-free ground truth: is the MEASURED foot toe-down during loaded stance?

Uses only captured marker heights (capture Z-up vertical), no skeleton, no frames. At
static standing the foot is flat on the floor, giving a reference (RHEE - forefoot) height
offset that is purely marker mounting. If the dynamic loaded-stance offset is close to the
static one, the real foot is flat during stance; if it is much larger, the foot is
genuinely toe-down (heel up) during stance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"


def _heel_fore(session, lo, hi):
    labels = list(session.marker_labels)
    P = np.asarray(session.markers, dtype=np.float64)[lo:hi]

    def col(l):
        return P[:, labels.index(l), 2] if l in labels else np.full(hi - lo, np.nan)

    heel = col("RHEE")
    fore = np.nanmean(np.stack([col("RMTH1"), col("RMTH5"), col("RTOE")]), axis=0)
    return heel, fore


def main() -> int:
    from biomech.session import load_session
    from biomech.tests import CAL_C3D, SPEEDCHANGE, TRIAL_C3D

    # static standing reference (foot flat on floor)
    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    n = min(60, np.asarray(static.markers).shape[0])
    sh, sf = _heel_fore(static, 0, n)
    static_off = float(np.nanmean(sh - sf)) * 1e3
    print(f"STATIC standing (foot flat): heel-fore = {static_off:.0f} mm "
          f"(pure marker mounting offset; heel={np.nanmean(sh)*1e3:.0f}, "
          f"fore={np.nanmean(sf)*1e3:.0f})")
    # approximate foot length between markers for angle conversion
    labels = list(static.marker_labels)
    Ps = np.asarray(static.markers)[:n]
    heelxy = np.nanmean(Ps[:, labels.index("RHEE"), :2], axis=0)
    forexy = np.nanmean(np.stack([Ps[:, labels.index(l), :2] for l in
                                  ("RMTH1", "RMTH5", "RTOE")]).mean(axis=0), axis=0)
    foot_len = float(np.linalg.norm(forexy - heelxy))
    print(f"  (heel->fore horizontal span ~ {foot_len*1e3:.0f} mm)")

    # dynamic loaded stance
    cache = np.load(_CACHE, allow_pickle=True)
    lo, hi = (int(v) for v in cache["window"])
    grf = None
    for k in ("grf_R", "grf_r"):
        if k in cache:
            gg = np.asarray(cache[k]); grf = gg[:, 2] if gg.ndim == 2 else gg
    trial = load_session(str(TRIAL_C3D), speedchange_path=str(SPEEDCHANGE))
    dh, df = _heel_fore(trial, lo, hi)
    diff = dh - df
    if grf is not None:
        loaded = grf > max(50.0, 0.3 * np.nanmax(grf))
    else:
        loaded = np.ones(diff.shape[0], bool)
    dyn_off = float(np.nanmean(diff[loaded])) * 1e3
    print(f"\nDYNAMIC loaded stance: heel-fore = {dyn_off:.0f} mm "
          f"(min {np.nanmin(diff[loaded])*1e3:.0f}, max {np.nanmax(diff[loaded])*1e3:.0f})")
    extra = dyn_off - static_off
    ang = np.degrees(np.arctan2(extra * 1e-3, foot_len))
    print(f"\nEXTRA heel rise vs flat standing = {extra:.0f} mm "
          f"~= {ang:.1f} deg toe-down beyond flat")
    print("  (>~5 deg => the captured foot really is toe-down/forefoot-loaded in stance;\n"
          "   ~0 => the foot is flat in stance and any toe-down is a model/export artifact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
