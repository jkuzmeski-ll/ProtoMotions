# SPDX-License-Identifier: MIT
"""Final: calcn forward-axis pitch (toe-down) at static standing vs dynamic stance.

Uses WarpSkeleton FK (OpenSim Y-up, vertical = +y) so the toe-down angle is the vertical
component of the calcn +x (forward/toe) axis -- unambiguous, independent of the walking
heading. If the fitted foot is already toe-down at static (foot provably flat on floor),
the calcn body frame is intrinsically tilted vs the real flat foot (a bone/marker-offset
issue); the dynamic extra over static is the real gait toe-down.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"
DEG = 180.0 / np.pi


def _fwd_pitch(Rc):  # toe-down deg from calcn +x vertical (OpenSim +y) component
    fy = (Rc @ np.array([0.0, 1.0, 0.0]))  # up axis
    fx = (Rc @ np.array([1.0, 0.0, 0.0]))  # forward axis
    return -np.degrees(np.arcsin(np.clip(fx[1], -1, 1))), \
        np.degrees(np.arccos(np.clip(fy[1], -1, 1)))


def main() -> int:
    from biomech.fitting.cluster_collapse import collapse_clusters
    from biomech.fitting.marker_fitter import MarkerFitConfig
    from biomech.fitting.marker_map import s001_marker_map
    from biomech.fitting.marker_placement import place_foot_markers
    from biomech.osim import parse_osim
    from biomech.session import load_session
    from biomech.skeleton.skeleton import WarpSkeleton
    from biomech.tests import CAL_C3D

    # --- static standing fit (foot flat on the floor) ---
    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    spec_s = parse_osim(str(_BIOMECH / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    mm, _ = collapse_clusters(spec_s, s001_marker_map())
    n = min(60, int(np.asarray(static.markers).shape[0]))
    pl = place_foot_markers(spec_s, static, mapping=mm,
                            marker_config=MarkerFitConfig(outer_iters=6), device="cpu",
                            frame_range=(0, n), register_neutral=False)
    skel_s = WarpSkeleton(spec_s, device="cpu")
    w_s, _ = skel_s.forward(np.asarray(pl.poses), np.asarray(pl.group_scales))
    ci_s = {b.name: i for i, b in enumerate(spec_s.bodies)}["calcn_r"]
    Rc_s = np.asarray(w_s)[:, ci_s, :3, :3]
    sp = np.array([_fwd_pitch(Rc_s[f])[0] for f in range(Rc_s.shape[0])])
    print(f"STATIC standing (foot flat): calcn forward-axis toe-down = "
          f"{sp.mean():.1f} deg  (should be ~0 if the bone frame matches the flat foot)")

    # --- dynamic (cached fit) ---
    cache = np.load(_CACHE, allow_pickle=True)
    spec_d = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)
    scales = np.asarray(cache["scales"], dtype=np.float64)
    grf = None
    for k in ("grf_R", "grf_r"):
        if k in cache:
            gg = np.asarray(cache[k]); grf = gg[:, 2] if gg.ndim == 2 else gg
    skel_d = WarpSkeleton(spec_d, device="cpu")
    w_d, _ = skel_d.forward(poses, scales)
    ci_d = {b.name: i for i, b in enumerate(spec_d.bodies)}["calcn_r"]
    Rc_d = np.asarray(w_d)[:, ci_d, :3, :3]
    dp = np.array([_fwd_pitch(Rc_d[f])[0] for f in range(Rc_d.shape[0])])
    loaded = (grf > max(50.0, 0.3 * np.nanmax(grf))) if grf is not None else np.ones(len(dp), bool)
    print(f"DYNAMIC loaded stance: calcn forward-axis toe-down = "
          f"{dp[loaded].mean():.1f} deg (min {dp[loaded].min():.1f}, max {dp[loaded].max():.1f})")
    print(f"\nDYNAMIC - STATIC = {dp[loaded].mean() - sp.mean():.1f} deg "
          f"(this is the real gait toe-down; the STATIC value is the intrinsic bone-frame "
          f"tilt vs the flat foot)")
    print("model-free measured extra toe-down in stance was ~3.4 deg for comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
