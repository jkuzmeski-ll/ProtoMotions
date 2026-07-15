# SPDX-License-Identifier: MIT
"""Benchmark the full S001 reconstruction pipeline end-to-end and write a .motion.

Runs the whole gold-standard chain on the real S001 capture and times each stage:

    load_session  ->  calibrate {scales, offsets} on a clean walk slice (MarkerFitter)
                  ->  per-frame marker IK over every frame of the chosen phase (chunked
                      so the batched FD Jacobian fits in GPU memory)
                  ->  build_motion (Warp FK, Y-up -> Z-up)  ->  torch.save(.motion)

Reports per-stage wall time, throughput (frames/s), the per-coordinate marker RMS (the
~16 mm figure metric), and the written .motion file path/size. Usage::

    python projects/biomech/bench_pipeline.py --device cuda:0 --phase all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from biomech.contact.pipeline import pick_visible_window, reconstruct_window
from biomech.export.protomotions_robot import build_simbody_motion
from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik
from biomech.fitting.marker_fitter import MarkerFitConfig
from biomech.fitting.marker_map import (
    anatomical_mask,
    observations_from_session,
    s001_marker_map,
)
from biomech.fitting.anthropometry import read_mp
from biomech.osim import parse_osim
from biomech.session import load_session
from biomech.skeleton.skeleton import WarpSkeleton
from biomech.tests import LEFT_BELT, RIGHT_BELT, SPEEDCHANGE, TRIAL_C3D, SUBJECT_MP

_ROOT = Path(__file__).resolve().parent
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

CALIB_LEN = 60          # clean walk frames used to calibrate scales + offsets
RMS_TARGET_MM = 16.0    # per-coordinate quality bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phase", default="walk", choices=("walk", "run", "all"))
    ap.add_argument("--frames", type=int, default=None,
                    help="cap the reconstructed frames to a subset of the phase")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="frames per per-frame-IK batch (GPU memory bound)")
    ap.add_argument("--outer", type=int, default=15,
                    help="MarkerFitter outer iterations for calibration")
    ap.add_argument("--ik-iters", type=int, default=80,
                    help="per-frame marker IK LM iterations")
    ap.add_argument("--out", type=Path,
                    default=Path("projects/data/S001/Trial101_full.motion"))
    ap.add_argument("--write-asset", action="store_true",
                    help="also (re)write the biomech robot MJCF with these scales so the "
                         "robot geometry matches the clip (overwrites the committed asset)")
    ap.add_argument("--tm2og", action="store_true",
                    help="map the clip from treadmill to overground using the instrumented "
                         "belt-speed logs (virtual-origin method)")
    args = ap.parse_args()

    # ------------------------------------------------------------------ load
    t0 = time.perf_counter()
    session = load_session(
        str(TRIAL_C3D),
        left_belt_path=str(LEFT_BELT),
        right_belt_path=str(RIGHT_BELT),
        speedchange_path=str(SPEEDCHANGE),
    )
    spec = parse_osim(str(_OSIM))
    skel = WarpSkeleton(spec, device=args.device)
    names = skel.marker_names()
    mm = s001_marker_map()
    obs_all, present = observations_from_session(session, names, mm)
    anat = anatomical_mask(names, mm)
    fps = session.point_rate
    t_load = time.perf_counter() - t0

    lo, hi = session.phase_window(args.phase)
    if args.frames is not None:
        # subset: pick the best-visibility contiguous block of the requested length
        n = min(args.frames, hi - lo)
        lo = lo + pick_visible_window(obs_all[lo:hi], present, n)
        hi = lo + n
    obs = np.ascontiguousarray(obs_all[lo:hi])
    F = obs.shape[0]

    # Per-frame belt speed for the exported window (mean of both split belts, as a
    # non-negative magnitude; tm2og infers the forward direction from stance feet).
    belt_speed = None
    if args.tm2og:
        sides = [session.belt_speed_point[s] for s in ("left", "right")
                 if s in session.belt_speed_point]
        if not sides:
            raise SystemExit("--tm2og requested but no belt-speed logs were loaded")
        belt_speed = np.abs(np.nanmean(np.stack(sides), axis=0))[lo:hi]

    # ----------------------------------------------------- calibrate scales/offsets
    # Always calibrate on the cleanest walk slice, even for a full-trial export.
    wlo, whi = session.phase_window("walk")
    cs = wlo + pick_visible_window(obs_all[wlo:whi], present, CALIB_LEN)
    calib_win = (cs, min(cs + CALIB_LEN, whi))
    t0 = time.perf_counter()
    fit, _, _ = reconstruct_window(
        session, spec, calib_win, mapping=mm, device=args.device,
        marker_config=MarkerFitConfig(
            outer_iters=args.outer,
            inner=MarkerIKConfig(max_iters=50),
            inner_first=MarkerIKConfig(max_iters=150),
        ),
    )
    t_cal = time.perf_counter() - t0
    scales = fit.group_scales
    print(f"device={args.device} phase={args.phase} window=({lo},{hi}) frames={F} "
          f"calib={calib_win} mapped={present.sum()}")

    # -------------------------------------------------- per-frame IK over all frames
    skel.set_marker_offsets(fit.marker_offsets)
    seed_row = np.mean(fit.poses, axis=0)
    poses = np.empty((F, skel.topo.num_dofs), dtype=np.float64)
    rms = np.empty(F, dtype=np.float64)
    ik_cfg = MarkerIKConfig(max_iters=args.ik_iters)
    t0 = time.perf_counter()
    for c0 in range(0, F, args.chunk):
        c1 = min(c0 + args.chunk, F)
        q_seed = np.repeat(seed_row[None], c1 - c0, axis=0)
        res = solve_marker_ik(
            skel, obs[c0:c1], q_seed, group_scales=scales, config=ik_cfg
        )
        poses[c0:c1] = res.q
        rms[c0:c1] = res.marker_rms
    t_ik = time.perf_counter() - t0

    # --------------------------------------------------------- build + write .motion
    # Use the *sim-body* clip (MuJoCo FK over the exported 38-body MJCF, incl. the
    # exporter's massless dummy bodies) so rigid_body_* align 1:1 with the registered
    # ``biomech`` robot. ``build_motion`` would emit only the 20 anatomical bodies and
    # fail to load ("motion has 20 bodies, robot expects 38").
    import torch

    t0 = time.perf_counter()
    mres = build_simbody_motion(
        spec, poses, fps=fps, group_scales=scales, belt_speed=belt_speed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mres.data, str(args.out))
    t_mo = time.perf_counter() - t0

    total = t_load + t_cal + t_ik + t_mo
    median_mm = float(np.nanmedian(rms)) * 1e3
    size_mb = args.out.stat().st_size / 1e6
    n_bodies = len(mres.body_names)
    print(f"  load_session      : {t_load*1e3:9.1f} ms")
    print(f"  calibrate (fit)   : {t_cal*1e3:9.1f} ms")
    print(f"  per-frame IK      : {t_ik*1e3:9.1f} ms  "
          f"({F / max(t_ik, 1e-9):8.1f} frames/s)")
    print(f"  build+write motion: {t_mo*1e3:9.1f} ms  (sim-body: {n_bodies} bodies)")
    print(f"  TOTAL             : {total*1e3:9.1f} ms  "
          f"({F / max(total, 1e-9):8.1f} frames/s)")
    print(f"  median frame RMS  : {median_mm:.2f} mm  (per-coordinate)")
    print(f"  motion            : {F} frames x {n_bodies} bodies @ {fps:.0f} fps")
    if belt_speed is not None:
        print(f"  tm2og             : ON  (belt {np.nanmin(belt_speed):.2f}"
              f"-{np.nanmax(belt_speed):.2f} m/s, travel {np.nansum(belt_speed)/fps:.1f} m)")
    print(f"  wrote             : {args.out}  ({size_mb:.1f} MB)")

    ok = median_mm <= RMS_TARGET_MM
    print(f"  RMS <= {RMS_TARGET_MM:.0f} mm    : {'PASS' if ok else 'FAIL'}")

    if args.write_asset:
        from biomech.export.protomotions_robot import write_biomech_asset
        subject_mass = float(read_mp(str(SUBJECT_MP)).get("Bodymass", 0.0)) or None
        asset = write_biomech_asset(spec, group_scales=scales, subject_mass=subject_mass)
        print(f"  wrote asset       : {asset}")
        if subject_mass:
            print(f"  subject mass      : {subject_mass:.2f} kg (anthropometric rescale)")
    else:
        print("  note: loads into the 'biomech' robot by body count/order; for "
              "scale+mass-consistent geometry re-run with --write-asset")


if __name__ == "__main__":
    main()
