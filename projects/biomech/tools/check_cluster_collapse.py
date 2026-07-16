# SPDX-License-Identifier: MIT
"""A/B check: does collapsing the thigh/shank clusters to a centroid help the solve?

Fits the same enriched-foot S001 walk window twice -- individual cluster markers vs each
cluster collapsed to a single centroid -- and reports marker RMS plus the stability of the
weakly-observed lower-limb angles (long-axis rotations especially). Collapsing should lower
the residual dragged in by the soft-tissue plates and smooth the pose, at the cost of the
cluster's (unreliable) long-axis-rotation detail. Use it to decide before wiring in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "projects")
from biomech.contact.pipeline import reconstruct_window  # noqa: E402
from biomech.fitting.cluster_collapse import collapse_clusters  # noqa: E402
from biomech.fitting.ik import MarkerIKConfig  # noqa: E402
from biomech.fitting.marker_fitter import MarkerFitConfig  # noqa: E402
from biomech.fitting.marker_map import s001_marker_map  # noqa: E402
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


def _jerk(x):
    """RMS of the 2nd difference (deg) -- a proxy for frame-to-frame roughness."""
    return float(np.sqrt(np.mean(np.diff(x, n=2) ** 2)))


def _stats(spec, res, label):
    dof = spec.dof_index_map()
    p = res.poses
    print(f"  [{label}] marker RMS median: {np.nanmedian(res.marker_rms)*1e3:.2f} mm")
    for base in ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle",
                 "ankle_angle"):
        for side in ("r", "l"):
            a = p[:, dof[f"{base}_{side}"]] * DEG
            print(f"      {base}_{side}: mean {a.mean():+6.1f} "
                  f"[{a.min():+6.1f},{a.max():+6.1f}] rough {_jerk(a):4.2f}")


def main():
    device = "cuda"
    static = load_session(str(DATA / "Cal 101.v3d.c3d"), filter_cutoff_hz=None)
    trial = load_session(str(DATA / "Trial 101.v3d.c3d"),
                         speedchange_path=str(DATA / "Speedchange101.txt"))
    lo, hi = trial.phase_window("walk")
    win = (lo, min(lo + 130, hi))
    print(f"window {win}\n")

    print("=== individual cluster markers (current) ===")
    spec_a = parse_osim(OSIM)
    mm_a = s001_marker_map()
    place_foot_markers(spec_a, static, mapping=mm_a, marker_config=_config(),
                       device=device, frame_range=(0, 60))
    res_a, _, _ = reconstruct_window(trial, spec_a, win, mapping=mm_a,
                                     marker_config=_config(), device=device)
    _stats(spec_a, res_a, "individual")

    print("\n=== collapsed cluster centroids ===")
    spec_b = parse_osim(OSIM)
    mm_b = s001_marker_map()
    mm_b, added = collapse_clusters(spec_b, mm_b)
    print(f"  added centroids: {added}")
    place_foot_markers(spec_b, static, mapping=mm_b, marker_config=_config(),
                       device=device, frame_range=(0, 60))
    res_b, _, _ = reconstruct_window(trial, spec_b, win, mapping=mm_b,
                                     marker_config=_config(), device=device)
    _stats(spec_b, res_b, "collapsed")

    rms_a = np.nanmedian(res_a.marker_rms) * 1e3
    rms_b = np.nanmedian(res_b.marker_rms) * 1e3
    print(f"\nmarker RMS: individual {rms_a:.2f} -> collapsed {rms_b:.2f} mm "
          f"({rms_b - rms_a:+.2f} mm)")


if __name__ == "__main__":
    main()
