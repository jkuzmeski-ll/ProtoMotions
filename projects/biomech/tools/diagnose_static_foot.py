# SPDX-License-Identifier: MIT
"""Decisive test: is the fitted foot flat (plantar horizontal) at static standing?

Reproduces the static marker-placement fit and FKs the static (standing) pose to measure
the calcaneus plantar orientation (calcn +y, the anatomical plantar normal) in the world.
The subject is standing on the floor, so the REAL plantar surface is horizontal. If the
FITTED calcn +y is vertical (~0 deg) the foot bone is flat and the ~10 deg dynamic
toe-down is real gait; if calcn +y is ~10 deg off vertical, marker placement baked in a
constant foot-segment offset (the fixable root cause).

Also prints the measured static foot-marker pitch (RHEE -> forefoot) as an independent
ground-truth check, and the ankle-neutral offset the placement computed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BIOMECH = Path(__file__).resolve().parents[1]

DEG = 180.0 / np.pi
DEVICE = "cpu"


def main() -> int:
    from biomech.fitting.cluster_collapse import collapse_clusters
    from biomech.fitting.marker_fitter import MarkerFitConfig
    from biomech.fitting.marker_map import R_PM2OS, s001_marker_map
    from biomech.fitting.marker_placement import place_foot_markers
    from biomech.osim import parse_osim
    from biomech.session import load_session
    from biomech.skeleton.skeleton import WarpSkeleton
    from biomech.tests import CAL_C3D

    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    spec = parse_osim(str(_BIOMECH / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    mm = s001_marker_map()
    mm, _ = collapse_clusters(spec, mm)

    nfr = min(60, int(np.asarray(static.markers).shape[0]))
    # register_neutral=False so no ankle bake -> FK the raw static poses cleanly
    pl = place_foot_markers(
        spec, static, mapping=mm,
        marker_config=MarkerFitConfig(outer_iters=6), device=DEVICE,
        frame_range=(0, nfr), register_neutral=False,
    )
    poses = np.asarray(pl.poses, dtype=np.float64)   # (Fw, ndof) static standing poses
    scales = np.asarray(pl.group_scales, dtype=np.float64)
    dof = spec.dof_index_map()
    print(f"static fit: {poses.shape[0]} frames, {poses.shape[1]} DOFs")

    ai = dof["ankle_angle_r"]
    ank = float(np.nanmean(poses[:, ai])) * DEG
    print(f"ankle_angle_r at static (model neutral) = {ank:.2f} deg "
          f"(this is the RStaticPlantFlex the bake removes)")

    # FK the static poses -> calcn_r world orientation
    skel = WarpSkeleton(spec, device=DEVICE)
    world, _ = skel.forward(poses, scales)
    bidx = {b.name: i for i, b in enumerate(spec.bodies)}
    Rc = np.asarray(world)[:, bidx["calcn_r"], :3, :3]  # (Fw,3,3)

    def sag(nrm):  # sagittal pitch off vertical (OpenSim Y-up), toe-down +
        return float(np.degrees(np.arctan2(nrm[0], nrm[1])))

    calcn_y = np.array([Rc[f] @ np.array([0.0, 1.0, 0.0]) for f in range(poses.shape[0])])
    pitch = np.array([sag(n) for n in calcn_y])
    print(f"\nFITTED calcn +y pitch off vertical at static: "
          f"mean={pitch.mean():.2f} deg (min {pitch.min():.2f}, max {pitch.max():.2f})")
    print("  (~0 => foot bone flat at standing = placement OK; "
          "~+/-10 => baked foot offset)")

    # --- ground truth: measured static foot-marker pitch (RHEE -> forefoot) ---
    labels = list(static.marker_labels)
    P = np.asarray(static.markers, dtype=np.float64)[:nfr]

    def mmean(lbl):
        if lbl not in labels:
            return None
        col = P[:, labels.index(lbl), :]
        finite = np.isfinite(col).all(axis=1)
        return col[finite].mean(axis=0) if finite.any() else None

    heel = mmean("RHEE")
    fore_pts = [mmean(l) for l in ("RMTH1", "RMTH5", "RTOE")]
    fore_pts = [p for p in fore_pts if p is not None]
    if heel is not None and fore_pts:
        fore = np.mean(fore_pts, axis=0)
        # capture frame is Z-up: pitch of heel->forefoot off horizontal
        vec = fore - heel
        horiz = np.linalg.norm(vec[:2])
        meas_pitch = float(np.degrees(np.arctan2(-vec[2], horiz)))
        print(f"\nMEASURED static foot marker pitch (RHEE->forefoot, capture Z-up): "
              f"{meas_pitch:.2f} deg  (+ = toe-down; ~0 => real foot flat at standing)")
        # also in OpenSim frame for apples-to-apples
        vo = R_PM2OS @ vec
        print(f"  (heel z={heel[2]*1e3:.0f} mm, forefoot z={fore[2]*1e3:.0f} mm; "
              f"heel-fore={heel[2]*1e3 - fore[2]*1e3:+.0f} mm)")

    print("\n=> If FITTED calcn+y pitch >> MEASURED foot pitch, the placement rotated the "
          "foot bone relative to reality (fixable at the ankle only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
