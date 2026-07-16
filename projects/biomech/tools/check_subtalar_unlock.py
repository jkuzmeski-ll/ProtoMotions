# SPDX-License-Identifier: MIT
"""A/B check: should the subtalar (inversion/eversion) joint be unlocked?

The stock Rajagopal model ships ``subtalar_angle_{r,l}`` ``locked=true`` -- the standard
choice for a sparse Plug-in-Gait foot, where nothing resolves frontal-plane foot roll. But
the enriched S001 foot marker set (calcaneus triangle ``HEE/HEE2/HEE3`` + medial/lateral
met heads ``MTH1``/``MTH5``) *does* constrain hindfoot 3D orientation and foot roll, so the
subtalar becomes observable (same argument that justified unlocking the MTP once a toes
marker existed).

This reconstructs the same enriched-foot S001 walk window twice -- subtalar locked vs
unlocked -- and reports marker RMS, the recovered subtalar range/means, and whether the
ankle/MTP stay well-behaved. Use it to decide whether unlocking is a net improvement before
wiring it in.
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


def _unlock_subtalar(spec):
    unlocked = []
    for side in ("r", "l"):
        try:
            joint = spec.joint(f"subtalar_{side}")
        except KeyError:
            continue
        for c in joint.coordinates:
            if c.locked:
                c.locked = False
                unlocked.append(c.name)
    return unlocked


def _stats(spec, result, label):
    dof = spec.dof_index_map()
    p = result.poses
    print(f"  [{label}] marker RMS median: {np.nanmedian(result.marker_rms)*1e3:.2f} mm")
    for side in ("r", "l"):
        a = p[:, dof[f"ankle_angle_{side}"]] * DEG
        s = p[:, dof[f"subtalar_angle_{side}"]] * DEG
        rail = float(np.mean(np.abs(s) > 19.5) * 100.0)  # %% frames at the +/-20 limit
        row = (f"    {side}: ankle mean {a.mean():+6.1f} [{a.min():+6.1f},{a.max():+6.1f}]"
               f"  subtalar mean {s.mean():+6.1f} [{s.min():+6.1f},{s.max():+6.1f}]"
               f" rail@20={rail:4.0f}%")
        if f"mtp_angle_{side}" in dof:
            m = p[:, dof[f"mtp_angle_{side}"]] * DEG
            row += f"  mtp mean {m.mean():+6.1f} rng {m.max()-m.min():5.1f}"
        print(row)


def main():
    device = "cuda"
    static = load_session(str(DATA / "Cal 101.v3d.c3d"), filter_cutoff_hz=None)
    trial = load_session(str(DATA / "Trial 101.v3d.c3d"),
                         speedchange_path=str(DATA / "Speedchange101.txt"))
    lo, hi = trial.phase_window("walk")
    win = (lo, min(lo + 130, hi))  # ~1.3 s, > one gait cycle
    print(f"window {win} (subtalar range in model = +/-20 deg)\n")

    print("=== subtalar LOCKED (current) ===")
    spec_a = parse_osim(OSIM)
    place_foot_markers(spec_a, static, marker_config=_config(),
                       device=device, frame_range=(0, 60))
    res_a, _, _ = reconstruct_window(trial, spec_a, win,
                                     marker_config=_config(), device=device)
    _stats(spec_a, res_a, "locked")

    print("\n=== subtalar UNLOCKED ===")
    spec_b = parse_osim(OSIM)
    place_foot_markers(spec_b, static, marker_config=_config(),
                       device=device, frame_range=(0, 60))
    unlocked = _unlock_subtalar(spec_b)
    print(f"  unlocked: {unlocked}")
    res_b, _, _ = reconstruct_window(trial, spec_b, win,
                                     marker_config=_config(), device=device)
    _stats(spec_b, res_b, "unlocked")

    rms_a = np.nanmedian(res_a.marker_rms) * 1e3
    rms_b = np.nanmedian(res_b.marker_rms) * 1e3
    print(f"\nmarker RMS: locked {rms_a:.2f} mm -> unlocked {rms_b:.2f} mm "
          f"({rms_b - rms_a:+.2f} mm)")

    # Condition C: unlock BEFORE placement so the static fit sees a free subtalar, then
    # bake a subtalar static neutral (same correction we applied to the ankle). Tests
    # whether the railing is a missing-neutral artifact vs genuine error absorption.
    print("\n=== subtalar UNLOCKED + static neutral ===")
    from biomech.fitting.marker_placement import register_ankle_neutral
    spec_c = parse_osim(OSIM)
    _unlock_subtalar(spec_c)
    pl_c = place_foot_markers(spec_c, static, marker_config=_config(),
                             device=device, frame_range=(0, 60))
    sub_neutral = register_ankle_neutral(
        spec_c, pl_c.poses,
        coords=("subtalar_angle_r", "subtalar_angle_l"),
    )
    print(f"  subtalar neutral (deg): "
          f"{ {k: round(v*DEG, 2) for k, v in sub_neutral.items()} }")
    res_c, _, _ = reconstruct_window(trial, spec_c, win,
                                     marker_config=_config(), device=device)
    _stats(spec_c, res_c, "unlocked+neutral")
    rms_c = np.nanmedian(res_c.marker_rms) * 1e3
    print(f"\nmarker RMS: locked {rms_a:.2f} -> unlocked {rms_b:.2f} -> "
          f"unlocked+neutral {rms_c:.2f} mm")


if __name__ == "__main__":
    main()
