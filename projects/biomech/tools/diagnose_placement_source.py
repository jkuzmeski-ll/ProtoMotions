# SPDX-License-Identifier: MIT
"""Locate the SOURCE of the intrinsic ~14 deg calcn toe-down at static (foot-flat) fit.

diagnose_foot_pitch_final.py proved the fitted calcn +x axis is ~13.7 deg toe-down even
at static standing (real foot provably flat), and only ~2 deg extra during gait. So the
~14 deg is a CONSTANT baked into the static placement fit, not real motion.

Hypothesis (from marker_placement.py docstring): the placement fit (step 1) uses the STOCK
sparse foot markers, whose RTOE sits at the *toe tip* while the S001 capture RTOE is on the
met-2 head (~2.7 cm proximal). The least-squares fit plantarflexes the foot to reconcile
that, corrupting the foot-flat pose that every downstream offset + the dynamic trace inherit.

Test: rerun the static placement fit (a) as-is, (b) with the stock RTOE removed from the
map, and compare the calcn +x toe-down. Also report the model's anatomical neutral (q=0)
and where the stock RTOE model marker lands vs the measured RTOE in the fitted flat pose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_BIOMECH = Path(__file__).resolve().parents[1]
DEG = 180.0 / np.pi


def _fwd_toe_down(Rc):  # deg, from calcn +x vertical (OpenSim +y) component
    fx = Rc @ np.array([1.0, 0.0, 0.0])
    return -np.degrees(np.arcsin(np.clip(fx[1], -1, 1)))


def _static_pitch(spec, static, mm, MarkerFitConfig, reconstruct_window, WarpSkeleton, n):
    result, _obs, _anat = reconstruct_window(
        static, spec, (0, n), mapping=mm,
        marker_config=MarkerFitConfig(outer_iters=6), device="cpu",
    )
    skel = WarpSkeleton(spec, device="cpu")
    world, _ = skel.forward(np.asarray(result.poses), np.asarray(result.group_scales))
    ci = {b.name: i for i, b in enumerate(spec.bodies)}["calcn_r"]
    Rc = np.asarray(world)[:, ci, :3, :3]
    pit = np.array([_fwd_toe_down(Rc[f]) for f in range(Rc.shape[0])])
    return pit.mean(), result, world, ci


def main() -> int:
    from biomech.contact.pipeline import reconstruct_window
    from biomech.fitting.cluster_collapse import collapse_clusters
    from biomech.fitting.marker_fitter import MarkerFitConfig
    from biomech.fitting.marker_map import s001_marker_map
    from biomech.osim import parse_osim
    from biomech.session import load_session
    from biomech.skeleton.skeleton import WarpSkeleton
    from biomech.tests import CAL_C3D

    osim = str(_BIOMECH / "models" / "rajagopal_data" / "Rajagopal2015.osim")
    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    n = min(60, int(np.asarray(static.markers).shape[0]))

    # 0) model anatomical neutral (q=0): calcn +x toe-down of the bone frame itself.
    spec0 = parse_osim(osim)
    skel0 = WarpSkeleton(spec0, device="cpu")
    ndof = len(spec0.dof_index_map())
    w0, _ = skel0.forward(np.zeros((1, ndof)), np.ones((len(spec0.scale_groups) * 3,)))
    ci0 = {b.name: i for i, b in enumerate(spec0.bodies)}["calcn_r"]
    print(f"[0] model neutral (q=0) calcn +x toe-down = "
          f"{_fwd_toe_down(np.asarray(w0)[0, ci0, :3, :3]):+.1f} deg  "
          f"(the model's built-in anatomical foot-flat frame)")

    # 1) static placement fit AS-IS (stock RTOE at the toe tip).
    spec_a = parse_osim(osim)
    mm_a, _ = collapse_clusters(spec_a, s001_marker_map())
    pa, res_a, world_a, cia = _static_pitch(
        spec_a, static, mm_a, MarkerFitConfig, reconstruct_window, WarpSkeleton, n)
    print(f"[1] static fit WITH stock RTOE     calcn +x toe-down = {pa:+.1f} deg")

    # where does stock RTOE (model, toe tip) land vs measured RTOE, in the flat pose?
    try:
        m_rtoe = spec_a.marker("RTOE")
        gi = {name: g for g, grp in enumerate(spec_a.scale_groups) for name in grp}["calcn_r"]
        s3 = np.asarray(res_a.group_scales).reshape(-1, 3)[gi]
        T = np.asarray(world_a)[:, cia, :, :]
        model_rtoe_w = np.einsum("fij,j->fi", T[:, :3, :3], s3 * np.asarray(m_rtoe.offset)) \
            + T[:, :3, 3]
        from biomech.fitting.marker_map import R_PM2OS
        labs = list(static.marker_labels)
        cap = np.asarray(static.markers)[:n, labs.index("RTOE"), :]
        cap_os = np.einsum("ij,fj->fi", R_PM2OS, cap)
        d = np.nanmean(cap_os - model_rtoe_w, axis=0)
        print(f"    measured RTOE - model RTOE (flat pose, OpenSim m): "
              f"[{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}]  |.|={np.linalg.norm(d)*1e3:.0f} mm")
    except Exception as e:  # noqa: BLE001
        print(f"    (RTOE landing check skipped: {e})")

    # 2) static placement fit WITHOUT the stock RTOE (drop it from the map for the fit).
    spec_b = parse_osim(osim)
    mm_full, _ = collapse_clusters(spec_b, s001_marker_map())
    from biomech.fitting.marker_map import MarkerMap
    mm_b = MarkerMap(
        model_to_capture={k: v for k, v in mm_full.model_to_capture.items() if k != "RTOE"},
        anatomical={a for a in mm_full.anatomical if a != "RTOE"},
    )
    pb, _res_b, _w_b, _ci_b = _static_pitch(
        spec_b, static, mm_b, MarkerFitConfig, reconstruct_window, WarpSkeleton, n)
    print(f"[2] static fit WITHOUT stock RTOE  calcn +x toe-down = {pb:+.1f} deg")

    print(f"\nRTOE contribution to the toe-down = {pa - pb:+.1f} deg")
    print("If [2] is markedly flatter than [1], the stock RTOE-at-tip marker is the source; "
          "fixing the placement pose flattens the whole foot without touching the chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
