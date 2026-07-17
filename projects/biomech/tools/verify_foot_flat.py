# SPDX-License-Identifier: MIT
"""Verify compute_foot_flat_offset: static->flat, and applied to dynamic cache -> ~2 deg."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_BIOMECH = Path(__file__).resolve().parents[1]
_CACHE = _BIOMECH / "docs" / "figures" / "_s001_ik_cache.npz"


def _toe_down_deg(skel, poses, scales, bi):
    world, _ = skel.forward(poses, scales)
    Rc = np.asarray(world)[:, bi, :3, :3]
    fx = Rc @ np.array([1.0, 0.0, 0.0])
    return np.degrees(-np.arcsin(np.clip(fx[:, 1], -1, 1)))


def main() -> int:
    from biomech.fitting.cluster_collapse import collapse_clusters
    from biomech.fitting.marker_fitter import MarkerFitConfig
    from biomech.fitting.marker_map import s001_marker_map
    from biomech.fitting.marker_placement import compute_foot_flat_offset, place_foot_markers
    from biomech.osim import parse_osim
    from biomech.session import load_session
    from biomech.skeleton.skeleton import WarpSkeleton
    from biomech.tests import CAL_C3D

    # fresh static fit (unbaked frame) -> foot_flat
    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    spec = parse_osim(str(_BIOMECH / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    mm, _ = collapse_clusters(spec, s001_marker_map())
    n = min(60, int(np.asarray(static.markers).shape[0]))
    pl = place_foot_markers(spec, static, mapping=mm,
                            marker_config=MarkerFitConfig(outer_iters=6), device="cpu",
                            frame_range=(0, n), register_neutral=False)
    ff = pl.foot_flat
    DEG = 180.0 / np.pi
    print("foot_flat (deg):", {k: round(v * DEG, 2) for k, v in ff.items()})

    dof = spec.dof_index_map()
    skel = WarpSkeleton(spec, device="cpu")
    bi = {b.name: i for i, b in enumerate(spec.bodies)}["calcn_r"]
    p = np.asarray(pl.poses).copy()
    before = _toe_down_deg(skel, p, np.asarray(pl.group_scales), bi).mean()
    p[:, dof["ankle_angle_r"]] += ff["ankle_angle_r"]
    after = _toe_down_deg(skel, p, np.asarray(pl.group_scales), bi).mean()
    print(f"STATIC calcn +x toe-down: before {before:+.2f} deg -> after {after:+.2f} deg")

    # apply to dynamic cache (baked spec) loaded stance
    cache = np.load(_CACHE, allow_pickle=True)
    spec_d = cache["spec_pickle"].item()
    poses = np.asarray(cache["poses"], dtype=np.float64)
    scales = np.asarray(cache["scales"], dtype=np.float64)
    grf = np.asarray(cache["grf_R"]); grf = grf[:, 2] if grf.ndim == 2 else grf
    dof_d = spec_d.dof_index_map()
    skel_d = WarpSkeleton(spec_d, device="cpu")
    bi_d = {b.name: i for i, b in enumerate(spec_d.bodies)}["calcn_r"]
    loaded = grf > max(50.0, 0.3 * np.nanmax(grf))
    d_before = _toe_down_deg(skel_d, poses, scales, bi_d)
    pc = poses.copy(); pc[:, dof_d["ankle_angle_r"]] += ff["ankle_angle_r"]
    d_after = _toe_down_deg(skel_d, pc, scales, bi_d)
    print(f"DYNAMIC loaded stance toe-down: before {d_before[loaded].mean():+.2f} deg "
          f"-> after {d_after[loaded].mean():+.2f} deg "
          f"(min {d_after[loaded].min():+.2f}, max {d_after[loaded].max():+.2f})")
    print("target: static ~0 deg, dynamic loaded ~+2 deg (the real gait toe-down).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
