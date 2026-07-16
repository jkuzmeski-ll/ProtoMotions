# SPDX-License-Identifier: MIT
"""Diagnose left/right ankle asymmetry in the enriched foot fit.

Checks (1) mirror symmetry of the statically-placed foot marker offsets and the ankle
neutral offsets, (2) symmetry of the static-trial fitted lower-limb angles, and (3)
whether the dynamic L/R ankle traces overlap over a *full* gait cycle when phase-shifted
(vs. the short-window artifact).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "projects")
from biomech.contact.pipeline import reconstruct_window  # noqa: E402
from biomech.fitting.ik import MarkerIKConfig  # noqa: E402
from biomech.fitting.marker_fitter import MarkerFitConfig  # noqa: E402
from biomech.fitting.marker_placement import place_foot_markers  # noqa: E402
from biomech.osim import parse_osim  # noqa: E402
from biomech.session import load_session  # noqa: E402

DATA = Path("projects/data/S001")
OSIM = "projects/biomech/models/rajagopal_data/Rajagopal2015.osim"
DEG = 180.0 / np.pi


def _config():
    return MarkerFitConfig(
        outer_iters=6,
        inner=MarkerIKConfig(max_iters=30),
        inner_first=MarkerIKConfig(max_iters=120),
        final_inner=MarkerIKConfig(max_iters=30),
    )


def main():
    device = "cuda"
    static = load_session(str(DATA / "Cal 101.v3d.c3d"), filter_cutoff_hz=None)
    trial = load_session(str(DATA / "Trial 101.v3d.c3d"),
                         speedchange_path=str(DATA / "Speedchange101.txt"))

    spec = parse_osim(OSIM)
    pl = place_foot_markers(spec, static, marker_config=_config(),
                            device=device, frame_range=(0, 60))

    # ---- (1) placed-offset mirror symmetry (right vs left; z flips) ----
    print("=== placed marker offset mirror check (R vs mirror(L)) ===")
    pairs = [("RCAL", "LCAL"), ("RCAL2", "LCAL2"), ("RCAL3", "LCAL3"),
             ("RMT1", "LMT1"), ("RMT5", "LMT5"), ("RTOE", "LTOE"),
             ("RTOE_TIP", "LTOE_TIP")]
    for r, l in pairs:
        ro = spec.marker(r).offset
        lo = spec.marker(l).offset
        lo_mir = lo * np.array([1, 1, -1.0])  # mirror left -> right (z flips)
        print(f"  {r:8s} {ro.round(4)}   mirror({l})={lo_mir.round(4)}   "
              f"dz-asym={np.linalg.norm(ro-lo_mir)*1e3:.1f} mm")
    print("  ankle_neutral (deg):",
          {k: round(v*DEG, 2) for k, v in pl.ankle_neutral.items()})

    # ---- (2) static-trial fitted angle symmetry ----
    dof = spec.dof_index_map()
    p = pl.poses
    print("\n=== static fit angles (deg), R vs L ===")
    for base in ("ankle_angle", "subtalar_angle", "knee_angle",
                 "hip_flexion", "hip_adduction", "hip_rotation"):
        r = p[:, dof[f"{base}_r"]].mean() * DEG
        l = p[:, dof[f"{base}_l"]].mean() * DEG
        print(f"  {base:16s} R {r:+7.2f}   L {l:+7.2f}   diff {r-l:+6.2f}")

    # ---- (3) full-gait-cycle dynamic L/R comparison ----
    lo, hi = trial.phase_window("walk")
    win = (lo, min(lo + 130, hi))  # ~1.3 s, > one gait cycle
    print(f"\n=== dynamic fit over full-cycle window {win} ===")
    res, _, _ = reconstruct_window(trial, spec, win, marker_config=_config(),
                                   device=device)
    ar = res.poses[:, dof["ankle_angle_r"]] * DEG
    al = res.poses[:, dof["ankle_angle_l"]] * DEG
    print(f"  ankle_r: mean {ar.mean():+6.2f}  min {ar.min():+6.2f}  max {ar.max():+6.2f}")
    print(f"  ankle_l: mean {al.mean():+6.2f}  min {al.min():+6.2f}  max {al.max():+6.2f}")
    # phase-align: find the lag maximizing corr(ar, shift(al))
    n = len(ar)
    best_lag, best_c = 0, -2.0
    for lag in range(-n // 2, n // 2):
        a = ar[max(0, lag):n + min(0, lag)]
        b = al[max(0, -lag):n - max(0, lag)]
        if len(a) < 20:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if c > best_c:
            best_c, best_lag = c, lag
    print(f"  best L/R phase lag = {best_lag} frames, corr = {best_c:.3f}")
    a = ar[max(0, best_lag):n + min(0, best_lag)]
    b = al[max(0, -best_lag):n - max(0, best_lag)]
    print(f"  phase-aligned mean diff (R - L) = {a.mean()-b.mean():+.2f} deg  "
          f"(rms {np.sqrt(np.mean((a-b)**2)):.2f} deg)")


if __name__ == "__main__":
    main()
