# SPDX-License-Identifier: MIT
"""Benchmark the biomech fitting pipeline on the real S001 *walk* window.

Reproduces the accepted S001 reconstruction methodology (the one behind the ~16 mm
figure): calibrate {group scales, marker offsets} once on a mid-window slice with the
bilevel ``MarkerFitter``, then run per-frame marker IK across the walk window with those
fixed. Each of the four stages is timed separately so a change can be measured
before/after (via ``git stash``), and the resulting per-frame marker RMS is asserted to
stay at or below the 16 mm real-data quality bar. Usage::

    python projects/biomech/bench_fit.py --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from biomech.contact.pipeline import pick_visible_window
from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik
from biomech.fitting.ik_initializer import IKInitializer
from biomech.fitting.marker_fitter import MarkerFitter, MarkerFitConfig
from biomech.fitting.marker_map import (
    anatomical_mask,
    observations_from_session,
    s001_marker_map,
)
from biomech.fitting.report import marker_errors
from biomech.osim import parse_osim
from biomech.session import load_session
from biomech.skeleton.skeleton import WarpSkeleton
from biomech.tests import SPEEDCHANGE, TRIAL_C3D

_ROOT = Path(__file__).resolve().parent
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

# Real-data quality bar for the S001 walk window (PiG <-> Rajagopal marker-set floor).
RMS_TARGET_MM = 16.0
CALIB_LEN = 60  # mid-window slice used to calibrate scales + offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=150,
                    help="length of the walk window to reconstruct")
    ap.add_argument("--outer", type=int, default=15,
                    help="MarkerFitter outer iterations for the calibration")
    args = ap.parse_args()

    session = load_session(str(TRIAL_C3D), speedchange_path=str(SPEEDCHANGE))
    spec = parse_osim(str(_OSIM))
    skel = WarpSkeleton(spec, device=args.device)
    names = skel.marker_names()
    model_offsets = skel.marker_offsets().copy()  # reset target for deterministic runs
    mm = s001_marker_map()
    obs_all, present = observations_from_session(session, names, mm)
    anat = anatomical_mask(names, mm)

    # Contiguous, best-visibility window inside the treadmill-walk phase.
    plo, phi = session.phase_window("walk")
    n = min(args.frames, phi - plo)
    s = plo + pick_visible_window(obs_all[plo:phi], present, n)
    lo, hi = s, min(s + n, phi)
    obs = np.ascontiguousarray(obs_all[lo:hi])
    F = obs.shape[0]

    # Mid-window calibration slice (relative to the window).
    c0 = max(0, (F - CALIB_LEN) // 2)
    obs_calib = np.ascontiguousarray(obs[c0:c0 + min(CALIB_LEN, F)])
    print(f"device={args.device} walk_window=({lo},{hi}) frames={F} "
          f"calib_frames={obs_calib.shape[0]} mapped={present.sum()}")

    calib_cfg = MarkerFitConfig(
        outer_iters=args.outer,
        inner=MarkerIKConfig(max_iters=50),
        inner_first=MarkerIKConfig(max_iters=150),
    )

    # --- stage functions (skeleton state reset so every run is identical) ----------
    def stage_init():
        skel.set_marker_offsets(model_offsets)
        return IKInitializer(skel, obs_calib, anatomical=anat).run(
            MarkerIKConfig(max_iters=40)
        )

    def stage_fit(seed):
        skel.set_marker_offsets(model_offsets)
        return MarkerFitter(skel, obs_calib, anatomical=anat).fit(
            init_scales=seed.group_scales, q_init=seed.poses, config=calib_cfg
        )

    def stage_perframe(fit):
        skel.set_marker_offsets(fit.marker_offsets)
        q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
        return solve_marker_ik(
            skel, obs, q_seed, group_scales=fit.group_scales,
            config=MarkerIKConfig(max_iters=80),
        )

    def stage_report(fit, res):
        skel.set_marker_offsets(fit.marker_offsets)
        return marker_errors(
            skel, obs, res.q,
            group_scales=fit.group_scales, marker_offsets=fit.marker_offsets,
        )

    # warm-up (kernel compilation, caches) + the reported reconstruction
    seed = stage_init()
    fit = stage_fit(seed)
    res = stage_perframe(fit)
    rep = stage_report(fit, res)

    def timeit(fn, k=3):
        best = float("inf")
        for _ in range(k):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_init = timeit(lambda: stage_init())
    t_fit = timeit(lambda: stage_fit(seed))
    t_pf = timeit(lambda: stage_perframe(fit))
    t_rep = timeit(lambda: stage_report(fit, res))

    # Two RMS conventions (they differ by exactly sqrt(3)):
    #   * per-coordinate  = sqrt(sum||d||^2 / (3*n_markers))  -- solve_marker_ik.marker_rms,
    #     the metric behind the accepted ~16 mm S001 figure.
    #   * per-marker Euclidean = sqrt(mean ||d||^2) -- marker_errors, the plain distance RMS.
    rms_pc = float(np.nanmedian(res.marker_rms)) * 1e3
    rms_eu = float(np.nanmedian(rep.per_frame_rms)) * 1e3
    print(f"  IKInitializer.run : {t_init*1e3:9.1f} ms")
    print(f"  MarkerFitter.fit  : {t_fit*1e3:9.1f} ms")
    print(f"  solve_marker_ik   : {t_pf*1e3:9.1f} ms")
    print(f"  marker_errors     : {t_rep*1e3:9.1f} ms")
    print(f"  TOTAL             : {(t_init+t_fit+t_pf+t_rep)*1e3:9.1f} ms")
    print(f"  median frame RMS  : {rms_pc:.2f} mm  (per-coordinate; ~16 mm figure metric)")
    print(f"  median Euclidean  : {rms_eu:.2f} mm  (per-marker distance = sqrt(3) x)")

    ok = rms_pc <= RMS_TARGET_MM
    print(f"  RMS <= {RMS_TARGET_MM:.0f} mm    : {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(
            f"median marker RMS {rms_pc:.2f} mm exceeds the "
            f"{RMS_TARGET_MM:.0f} mm bar (try a larger --outer)"
        )


if __name__ == "__main__":
    main()
