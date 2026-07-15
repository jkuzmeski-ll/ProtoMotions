# SPDX-License-Identifier: MIT
"""Parity check against the ~16 mm S001 figure.

Reproduces the exact reconstruction methodology of
``tools/make_s001_ik_figures.py`` (walk-window calibrate-once / IK-per-frame),
then prints the median per-frame marker RMS and dumps the fitted scales + poses
so the old (host-loop scale step) and new (device scale-Jacobian) code paths can
be diffed for numerical parity.

Usage::

    python projects/biomech/bench_parity.py --tag new
    #  git stash ...
    python projects/biomech/bench_parity.py --tag old
    python projects/biomech/bench_parity.py --compare old new
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from biomech.contact.pipeline import pick_visible_window, reconstruct_window
from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik
from biomech.fitting.marker_fitter import MarkerFitConfig
from biomech.fitting.marker_map import observations_from_session, s001_marker_map
from biomech.osim import parse_osim
from biomech.session import load_session
from biomech.skeleton.skeleton import WarpSkeleton
from biomech.tests import SPEEDCHANGE, TRIAL_C3D

_ROOT = Path(__file__).resolve().parent
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"
_OUT = _ROOT / "docs" / "figures"

WIN_LEN = 150
CALIB_LEN = 60
DEVICE = "cuda"


def reconstruct():
    session = load_session(str(TRIAL_C3D), speedchange_path=str(SPEEDCHANGE))
    spec = parse_osim(str(_OSIM))
    skel = WarpSkeleton(spec, device=DEVICE)
    model_names = skel.marker_names()
    mm = s001_marker_map()
    obs_all, present = observations_from_session(session, model_names, mm)

    plo, phi = session.phase_window("walk")
    sub = obs_all[plo:phi]
    s = plo + pick_visible_window(sub, present, WIN_LEN)
    lo, hi = s, min(s + WIN_LEN, phi)
    obs = obs_all[lo:hi]
    F = obs.shape[0]

    c0 = lo + max(0, (F - CALIB_LEN) // 2)
    calib_win = (c0, c0 + min(CALIB_LEN, F))
    cfg = MarkerFitConfig(
        outer_iters=15,
        inner=MarkerIKConfig(max_iters=50),
        inner_first=MarkerIKConfig(max_iters=150),
    )
    fit, _, _ = reconstruct_window(
        session, spec, calib_win, mapping=mm, marker_config=cfg, device=DEVICE
    )
    scales = fit.group_scales

    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    res = solve_marker_ik(
        skel, obs, q_seed, group_scales=scales, config=MarkerIKConfig(max_iters=80)
    )
    return dict(
        window=(lo, hi),
        calib_win=calib_win,
        scales=scales,
        offsets=fit.marker_offsets,
        calib_poses=fit.poses,
        poses=res.q,
        marker_rms=res.marker_rms,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="run and save results under this tag")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        a, b = args.compare
        za = np.load(_OUT / f"_parity_{a}.npz")
        zb = np.load(_OUT / f"_parity_{b}.npz")
        print(f"comparing '{a}' vs '{b}'")
        for k in ["scales", "offsets", "calib_poses", "poses", "marker_rms"]:
            da = np.max(np.abs(za[k] - zb[k]))
            print(f"  max|Δ {k:12s}| = {da:.3e}")
        print(f"  median RMS {a}: {np.nanmedian(za['marker_rms'])*1e3:.4f} mm")
        print(f"  median RMS {b}: {np.nanmedian(zb['marker_rms'])*1e3:.4f} mm")
        return

    t0 = time.perf_counter()
    R = reconstruct()
    dt = time.perf_counter() - t0
    med = np.nanmedian(R["marker_rms"]) * 1e3
    print(f"window={R['window']} calib={R['calib_win']}")
    print(f"median per-frame marker RMS : {med:.4f} mm")
    print(f"scales range : [{R['scales'].min():.4f}, {R['scales'].max():.4f}]")
    print(f"elapsed : {dt:.1f} s")
    if args.tag:
        _OUT.mkdir(parents=True, exist_ok=True)
        np.savez(_OUT / f"_parity_{args.tag}.npz", **R)
        print(f"saved -> _parity_{args.tag}.npz")


if __name__ == "__main__":
    main()
