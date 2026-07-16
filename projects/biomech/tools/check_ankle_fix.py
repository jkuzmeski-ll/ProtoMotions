# SPDX-License-Identifier: MIT
"""A/B check: does static foot marker placement fix the ankle plantarflexion bias?

Reconstructs the same S001 walk window twice — with the stock (sparse) foot marker set
and with the enriched, statically-placed foot markers — and prints ankle/mtp angle stats.
The stock fit produces an entirely-negative (plantarflexed) ankle angle; the enriched fit
should recentre it toward zero and cross into dorsiflexion during stance.
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
WINDOW = (1469, 1519)  # 50-frame walk window (inside the walk phase)


def _config():
    return MarkerFitConfig(
        outer_iters=6,
        inner=MarkerIKConfig(max_iters=30),
        inner_first=MarkerIKConfig(max_iters=120),
        final_inner=MarkerIKConfig(max_iters=30),
    )


def _ankle_stats(spec, result, label):
    dof = spec.dof_index_map()
    p = result.poses
    for side in ("r", "l"):
        a = p[:, dof[f"ankle_angle_{side}"]] * DEG
        s = p[:, dof[f"subtalar_angle_{side}"]] * DEG
        row = (f"  ankle_{side}: mean {a.mean():+6.1f} min {a.min():+6.1f} max {a.max():+6.1f}"
               f"  | subtalar mean {s.mean():+6.1f}")
        if f"mtp_angle_{side}" in dof:
            m = p[:, dof[f"mtp_angle_{side}"]] * DEG
            row += f"  | mtp mean {m.mean():+6.1f} range {m.max()-m.min():5.1f}"
        print(row)
    print(f"  marker RMS median: {np.nanmedian(result.marker_rms)*1e3:.1f} mm")


def main():
    device = "cuda"
    static = load_session(str(DATA / "Cal 101.v3d.c3d"), filter_cutoff_hz=None)
    trial = load_session(str(DATA / "Trial 101.v3d.c3d"))

    print(f"=== BASELINE (stock foot markers), window {WINDOW} ===")
    spec_base = parse_osim(OSIM)
    res_base, _, _ = reconstruct_window(
        trial, spec_base, WINDOW, marker_config=_config(), device=device
    )
    _ankle_stats(spec_base, res_base, "baseline")

    print(f"\n=== ENRICHED (static-placed foot markers), window {WINDOW} ===")
    spec_rich = parse_osim(OSIM)
    placement = place_foot_markers(
        spec_rich, static, marker_config=_config(), device=device, frame_range=(0, 60)
    )
    print(f"  placed: added={placement.added}")
    res_rich, _, _ = reconstruct_window(
        trial, spec_rich, WINDOW, marker_config=_config(), device=device
    )
    _ankle_stats(spec_rich, res_rich, "enriched")


if __name__ == "__main__":
    main()
