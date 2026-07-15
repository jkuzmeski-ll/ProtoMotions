# SPDX-License-Identifier: MIT
"""Micro-profile S001 IK throughput (frames/s and frame-iterations/s)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik  # noqa: E402
from biomech.osim import parse_osim  # noqa: E402
from biomech.tools import benchmark_s001_ik as b  # noqa: E402


def main():
    spec = parse_osim("projects/biomech/models/rajagopal_data/Rajagopal2015.osim")
    R = b.load_npz(b.FAST_CACHE, spec)
    print("cached fast wall", R["wall_s"], "frames", len(R["t"]),
          "frames/s total", len(R["t"]) / R["wall_s"])
    _, _, skel, _, _, obs_all, window = b.load_common()
    lo, hi = window
    obs = obs_all[lo:hi]
    q_init = R["poses"].copy()
    scales = R["scales"]
    for it in [1, 2, 5, 10, 20, 40, 80]:
        t0 = time.perf_counter()
        res = solve_marker_ik(
            skel, obs, q_init, group_scales=scales,
            config=MarkerIKConfig(max_iters=it),
        )
        dt = time.perf_counter() - t0
        print(
            "iters", it,
            "wall", round(dt, 4),
            "frames/s", round(obs.shape[0] / dt, 1),
            "frame-iters/s", round(obs.shape[0] * it / dt, 1),
            "rmsmm", round(float(np.nanmedian(res.marker_rms)) * 1e3, 4),
        )


if __name__ == "__main__":
    main()
