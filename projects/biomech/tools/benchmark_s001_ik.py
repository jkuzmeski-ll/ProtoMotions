# SPDX-License-Identifier: MIT
#
# S001-only IK speed/accuracy benchmark.
#
# This keeps the first real-data S001 reconstruction as the baseline and compares every
# subsequent speed/accuracy change against it. It intentionally uses only S001 data.
#
# Current variant implemented here:
#   baseline_cached  : the existing `make_s001_ik_figures.py` result, loaded from cache.
#   anthropo_fixed_v1: subject .mp lower-body segment-length scales are fixed; scale FD
#                      optimization is skipped; anatomical markers are weighted higher
#                      than soft-tissue clusters; final noisy IK iterations are bounded.
#
# Usage:
#   .venv/Scripts/python projects/biomech/tools/benchmark_s001_ik.py
#   .venv/Scripts/python projects/biomech/tools/benchmark_s001_ik.py --fresh-anthro

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from biomech.contact.pipeline import measured_belt_grf, pick_visible_window  # noqa: E402
from biomech.export.motion import build_motion  # noqa: E402
from biomech.fitting.anthropometry import anthropometric_scale_prior  # noqa: E402
from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik  # noqa: E402
from biomech.fitting.ik_initializer import IKInitializer  # noqa: E402
from biomech.fitting.marker_fitter import MarkerFitConfig, MarkerFitter  # noqa: E402
from biomech.fitting.marker_map import (  # noqa: E402
    LOWER_BODY_MARKERS,
    S001_STATIC_CALIBRATION_MARKERS,
    anatomical_mask,
    observations_from_session,
    s001_marker_map,
)
from biomech.osim import parse_osim  # noqa: E402
from biomech.session import load_session  # noqa: E402
from biomech.skeleton.skeleton import WarpSkeleton  # noqa: E402
from biomech.tests import SPEEDCHANGE, SUBJECT_MP, TRIAL_C3D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BASE_CACHE = FIG / "_s001_ik_cache.npz"
FAST_CACHE = FIG / "_s001_ik_fast_bilevel_v1.npz"
WARM_200_CACHE = FIG / "_s001_ik_warp_warm_200_v1.npz"
NO_STATIC_CACHE = FIG / "_s001_ik_no_static_calib_v1.npz"
ROBUST_ANAT_CACHE = FIG / "_s001_ik_robust_anatomical_v1.npz"
ROBUST_BALANCED_CACHE = FIG / "_s001_ik_robust_balanced_v2.npz"
ANTHRO_CACHE = FIG / "_s001_ik_anthro_fixed_v1.npz"
ANTHRO_PRIOR_CACHE = FIG / "_s001_ik_anthro_prior_v1.npz"
METRICS_JSON = FIG / "s001_ik_benchmark_metrics.json"
DEG = 180.0 / np.pi
DEVICE = "cuda"
WIN_LEN = 150
CALIB_LEN = 60

KEY_DOFS = [
    "hip_flexion_r", "hip_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
]
# Wide, practical normal-walk sanity bands (deg) for a speed/accuracy progress figure.
# These are NOT a substitute for an OpenSim/Visual3D gold-standard comparison; they are a
# guardrail to catch obvious amplitude inflation while we improve marker correspondence.
GAIT_RANGE_BANDS = {
    "hip_flexion_r": (35.0, 55.0), "hip_flexion_l": (35.0, 55.0),
    "knee_angle_r": (50.0, 70.0), "knee_angle_l": (50.0, 70.0),
    "ankle_angle_r": (15.0, 35.0), "ankle_angle_l": (15.0, 35.0),
}


def load_common():
    session = load_session(str(TRIAL_C3D), speedchange_path=str(SPEEDCHANGE))
    spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    skel = WarpSkeleton(spec, device=DEVICE)
    names = skel.marker_names()
    mm = s001_marker_map()
    obs_all, present = observations_from_session(session, names, mm)
    plo, phi = session.phase_window("walk")
    lo = plo + pick_visible_window(obs_all[plo:phi], present, WIN_LEN)
    hi = min(lo + WIN_LEN, phi)
    return session, spec, skel, names, mm, obs_all, (lo, hi)


def marker_weights(names, mm):
    """S001-specific marker weights: bony landmarks high, clusters low."""
    anat = anatomical_mask(names, mm)
    w = np.zeros(len(names), dtype=np.float64)
    mapped = set(mm.model_to_capture)
    for i, n in enumerate(names):
        if n not in mapped:
            w[i] = 0.0
        elif anat[i]:
            w[i] = 4.0
        elif n in LOWER_BODY_MARKERS:
            # Thigh/shank tracking clusters are useful for phase, but soft-tissue artifact
            # should not be allowed to dominate the joint angles or segment scales.
            w[i] = 0.35
        else:
            # Upper body is present but secondary for the first lower-body milestone.
            w[i] = 0.15
    return w


def no_static_calibration_weights(names):
    """Uniform dynamic IK weights with medial knee/ankle calibration markers removed."""
    w = np.ones(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if n in S001_STATIC_CALIBRATION_MARKERS:
            w[i] = 0.0
    return w


def balanced_anatomical_weights(names, mm):
    """Best S001 v2 weighting from the local sweep: anatomy dominates, clusters still track."""
    anat = anatomical_mask(names, mm)
    mapped = set(mm.model_to_capture)
    w = np.zeros(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if n not in mapped:
            w[i] = 0.0
        elif anat[i]:
            w[i] = 4.0
        elif n in LOWER_BODY_MARKERS:
            w[i] = 0.5
        else:
            w[i] = 0.25
    return w


def per_marker_rms(skel, obs, poses, scales, offsets):
    skel.set_marker_offsets(offsets)
    _, mk = skel.forward(poses, scales)
    vis = np.isfinite(obs).all(axis=2)
    d = np.linalg.norm(np.where(vis[..., None], mk - np.nan_to_num(obs), 0.0), axis=2)
    return np.array([
        np.sqrt(np.mean(d[vis[:, m], m] ** 2)) if vis[:, m].any() else np.nan
        for m in range(d.shape[1])
    ])


def pack_result(spec, poses, scales, marker_offsets, marker_rms, per_marker, names, motion, grf, t, window, wall_s):
    return {
        "spec": spec,
        "poses": poses,
        "scales": scales,
        "marker_offsets": marker_offsets,
        "marker_rms": marker_rms,
        "per_marker_rms": per_marker,
        "marker_names": list(names),
        "rigid_body_pos": np.asarray(motion.data["rigid_body_pos"]),
        "body_names": list(motion.body_names),
        "grf": grf,
        "t": t,
        "window": tuple(window),
        "wall_s": float(wall_s),
    }


def save_npz(path, R):
    save = dict(
        poses=R["poses"], scales=R["scales"],
        marker_offsets=R.get("marker_offsets", np.empty((0, 3))),
        marker_rms=R["marker_rms"], per_marker_rms=R["per_marker_rms"],
        marker_names=np.array(R["marker_names"]),
        rigid_body_pos=R["rigid_body_pos"], body_names=np.array(R["body_names"]),
        t=R["t"], window=np.array(R["window"]), wall_s=np.array([R["wall_s"]]),
        ik_iters=np.array([R.get("ik_iters", -1)]),
    )
    for side in R["grf"]:
        save[f"grf_{side}"] = R["grf"][side]
    np.savez(str(path), **cast(dict[str, Any], save))


def load_npz(path, spec):
    z = np.load(path, allow_pickle=True)
    grf = {}
    if "grf_R" in z:
        grf["R"] = z["grf_R"]
    if "grf_L" in z:
        grf["L"] = z["grf_L"]
    wall_s = float(z["wall_s"][0]) if "wall_s" in z else 357.8  # first S001 baseline run
    return dict(
        spec=spec, poses=z["poses"], scales=z["scales"],
        marker_offsets=z["marker_offsets"] if "marker_offsets" in z else np.empty((0, 3)),
        marker_rms=z["marker_rms"], per_marker_rms=z["per_marker_rms"], marker_names=list(z["marker_names"]),
        rigid_body_pos=z["rigid_body_pos"], body_names=list(z["body_names"]),
        grf=grf, t=z["t"], window=tuple(z["window"]), wall_s=wall_s,
        ik_iters=int(z["ik_iters"][0]) if "ik_iters" in z else -1,
    )


def _common_fit_inputs():
    session, spec, skel, names, mm, obs_all, window = load_common()
    lo, hi = window
    obs = obs_all[lo:hi]
    F = obs.shape[0]
    c0 = lo + max(0, (F - CALIB_LEN) // 2)
    calib_win = (c0, c0 + min(CALIB_LEN, F))
    obs_cal = obs_all[calib_win[0]:calib_win[1]]
    anat = anatomical_mask(names, mm)
    return session, spec, skel, names, mm, obs, obs_cal, anat, window


def run_fast_bilevel(fresh=False):
    """Same model as the original baseline, but with the fast FD IK and bounded final IK."""
    if FAST_CACHE.exists() and not fresh:
        print(f"Loading cached fast_bilevel_v1 from {FAST_CACHE.name}")
        spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
        return load_npz(FAST_CACHE, spec)

    session, spec, skel, names, mm, obs, obs_cal, anat, window = _common_fit_inputs()
    lo, hi = window
    F = obs.shape[0]
    t0 = time.perf_counter()
    init = IKInitializer(skel, obs_cal, anatomical=anat)
    seed = init.run(MarkerIKConfig(max_iters=40))
    fitter = MarkerFitter(skel, obs_cal, anatomical=anat)
    cfg = MarkerFitConfig(
        outer_iters=15,
        inner_first=MarkerIKConfig(max_iters=150),
        inner=MarkerIKConfig(max_iters=50),
        final_inner=MarkerIKConfig(max_iters=80),
    )
    fit = fitter.fit(init_scales=seed.group_scales, q_init=seed.poses, config=cfg)
    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    ik = solve_marker_ik(
        skel, obs, q_seed, group_scales=fit.group_scales,
        config=MarkerIKConfig(max_iters=80),
    )
    wall_s = time.perf_counter() - t0
    pm = per_marker_rms(skel, obs, ik.q, fit.group_scales, fit.marker_offsets)
    motion = build_motion(spec, ik.q, fps=session.point_rate, group_scales=fit.group_scales)
    belt = measured_belt_grf(session)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    R = pack_result(spec, ik.q, fit.group_scales, fit.marker_offsets, ik.marker_rms, pm, names, motion, grf, t, window, wall_s)
    save_npz(FAST_CACHE, R)
    return R


def _run_weighted_dynamic_variant(cache_path, label, weight_fn, fresh=False):
    if cache_path.exists() and not fresh:
        print(f"Loading cached {label} from {cache_path.name}")
        spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
        return load_npz(cache_path, spec)
    session, spec, skel, names, mm, obs, obs_cal, anat, window = _common_fit_inputs()
    lo, hi = window
    F = obs.shape[0]
    weights = weight_fn(names, mm)
    t0 = time.perf_counter()
    init = IKInitializer(skel, obs_cal, anatomical=anat)
    seed = init.run(MarkerIKConfig(max_iters=30))
    fitter = MarkerFitter(skel, obs_cal, weights=weights, anatomical=anat)
    cfg = MarkerFitConfig(
        outer_iters=8,
        inner_first=MarkerIKConfig(max_iters=80),
        inner=MarkerIKConfig(max_iters=30),
        final_inner=MarkerIKConfig(max_iters=50),
        offset_prior_weight=2.0,
        anatomical_prior_factor=35.0,
        offset_max_delta=0.035,
    )
    fit = fitter.fit(init_scales=seed.group_scales, q_init=seed.poses, config=cfg)
    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    ik = solve_marker_ik(
        skel, obs, q_seed, group_scales=fit.group_scales, weights=weights,
        config=MarkerIKConfig(max_iters=50),
    )
    wall_s = time.perf_counter() - t0
    pm = per_marker_rms(skel, obs, ik.q, fit.group_scales, fit.marker_offsets)
    motion = build_motion(spec, ik.q, fps=session.point_rate, group_scales=fit.group_scales)
    belt = measured_belt_grf(session)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    R = pack_result(spec, ik.q, fit.group_scales, fit.marker_offsets, ik.marker_rms, pm, names, motion, grf, t, window, wall_s)
    save_npz(cache_path, R)
    return R


def run_robust_balanced(fresh=False):
    return _run_weighted_dynamic_variant(
        ROBUST_BALANCED_CACHE, "robust_balanced_v2", balanced_anatomical_weights, fresh=fresh
    )


def run_robust_anatomical(fresh=False):
    """Dynamic fit with anatomical markers high-weighted and soft-tissue clusters downweighted."""
    if ROBUST_ANAT_CACHE.exists() and not fresh:
        print(f"Loading cached robust_anatomical_v1 from {ROBUST_ANAT_CACHE.name}")
        spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
        return load_npz(ROBUST_ANAT_CACHE, spec)

    session, spec, skel, names, mm, obs, obs_cal, anat, window = _common_fit_inputs()
    lo, hi = window
    F = obs.shape[0]
    weights = marker_weights(names, mm)
    t0 = time.perf_counter()
    init = IKInitializer(skel, obs_cal, anatomical=anat)
    seed = init.run(MarkerIKConfig(max_iters=40))
    fitter = MarkerFitter(skel, obs_cal, weights=weights, anatomical=anat)
    cfg = MarkerFitConfig(
        outer_iters=15,
        inner_first=MarkerIKConfig(max_iters=150),
        inner=MarkerIKConfig(max_iters=50),
        final_inner=MarkerIKConfig(max_iters=80),
        offset_prior_weight=2.0,
        anatomical_prior_factor=40.0,
        offset_max_delta=0.035,
    )
    fit = fitter.fit(init_scales=seed.group_scales, q_init=seed.poses, config=cfg)
    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    ik = solve_marker_ik(
        skel, obs, q_seed, group_scales=fit.group_scales, weights=weights,
        config=MarkerIKConfig(max_iters=80),
    )
    wall_s = time.perf_counter() - t0
    pm = per_marker_rms(skel, obs, ik.q, fit.group_scales, fit.marker_offsets)
    motion = build_motion(spec, ik.q, fps=session.point_rate, group_scales=fit.group_scales)
    belt = measured_belt_grf(session)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    R = pack_result(spec, ik.q, fit.group_scales, fit.marker_offsets, ik.marker_rms, pm, names, motion, grf, t, window, wall_s)
    save_npz(ROBUST_ANAT_CACHE, R)
    return R


def run_no_static_calib(fresh=False):
    """S001 fit excluding medial knee/ankle calibration markers from dynamic IK."""
    if NO_STATIC_CACHE.exists() and not fresh:
        print(f"Loading cached no_static_calib_v1 from {NO_STATIC_CACHE.name}")
        spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
        return load_npz(NO_STATIC_CACHE, spec)

    session, spec, skel, names, mm, obs, obs_cal, anat, window = _common_fit_inputs()
    lo, hi = window
    F = obs.shape[0]
    weights = no_static_calibration_weights(names)
    t0 = time.perf_counter()
    init = IKInitializer(skel, obs_cal, anatomical=anat)
    seed = init.run(MarkerIKConfig(max_iters=40))
    fitter = MarkerFitter(skel, obs_cal, weights=weights, anatomical=anat)
    cfg = MarkerFitConfig(
        outer_iters=15,
        inner_first=MarkerIKConfig(max_iters=150),
        inner=MarkerIKConfig(max_iters=50),
        final_inner=MarkerIKConfig(max_iters=80),
    )
    fit = fitter.fit(init_scales=seed.group_scales, q_init=seed.poses, config=cfg)
    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    ik = solve_marker_ik(
        skel, obs, q_seed, group_scales=fit.group_scales, weights=weights,
        config=MarkerIKConfig(max_iters=80),
    )
    wall_s = time.perf_counter() - t0
    pm = per_marker_rms(skel, obs, ik.q, fit.group_scales, fit.marker_offsets)
    motion = build_motion(spec, ik.q, fps=session.point_rate, group_scales=fit.group_scales)
    belt = measured_belt_grf(session)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    R = pack_result(spec, ik.q, fit.group_scales, fit.marker_offsets, ik.marker_rms, pm, names, motion, grf, t, window, wall_s)
    R["excluded_markers"] = np.array(sorted(S001_STATIC_CALIBRATION_MARKERS))
    save_npz(NO_STATIC_CACHE, R)
    return R


def run_warp_warm_200(source_R, fresh=False):
    """Native Warp-FD IK throughput on 200 S001 frames with gait-continuation warm start.

    This is the realistic streaming/data-processing mode after subject calibration: use the
    previous solved gait window as the initial guess, then run a small fixed number of LM
    iterations. No Torch is used; FK/Jacobian are native Warp kernels.
    """
    if WARM_200_CACHE.exists() and not fresh:
        print(f"Loading cached warp_warm_200_v1 from {WARM_200_CACHE.name}")
        spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
        return load_npz(WARM_200_CACHE, spec)
    session, spec, skel, names, mm, obs_all, _ = load_common()
    lo, hi = 1469, 1669
    obs = obs_all[lo:hi]
    F = obs.shape[0]
    skel.set_marker_offsets(source_R["marker_offsets"])
    q_seed = np.zeros((F, source_R["poses"].shape[1]), dtype=np.float64)
    n = min(source_R["poses"].shape[0], F)
    q_seed[:n] = source_R["poses"][:n]
    q_seed[n:] = source_R["poses"][-1]
    t0 = time.perf_counter()
    ik = solve_marker_ik(
        skel, obs, q_seed, group_scales=source_R["scales"],
        config=MarkerIKConfig(max_iters=5),
    )
    wall_s = time.perf_counter() - t0
    pm = per_marker_rms(skel, obs, ik.q, source_R["scales"], source_R["marker_offsets"])
    motion = build_motion(spec, ik.q, fps=session.point_rate, group_scales=source_R["scales"])
    belt = measured_belt_grf(session)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    R = pack_result(spec, ik.q, source_R["scales"], source_R["marker_offsets"], ik.marker_rms, pm, names, motion, grf, t, (lo, hi), wall_s)
    R["ik_iters"] = ik.iters
    save_npz(WARM_200_CACHE, R)
    return R



def run_anthro_fixed(fresh=False):
    if ANTHRO_CACHE.exists() and not fresh:
        print(f"Loading cached anthropo_fixed_v1 from {ANTHRO_CACHE.name}")
        spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
        return load_npz(ANTHRO_CACHE, spec)

    session, spec, skel, names, mm, obs, obs_cal, anat, window = _common_fit_inputs()
    lo, hi = window
    F = obs.shape[0]
    weights = marker_weights(names, mm)

    scale_target, scale_w, diag = anthropometric_scale_prior(
        spec, SUBJECT_MP, length_weight=100.0, width_weight=20.0, include_upper_body=False
    )
    init_scales = np.ones_like(scale_target)
    init_scales[scale_w > 0] = scale_target[scale_w > 0]

    print("anthropo_fixed_v1 scale targets:")
    for k, v in sorted(diag.items()):
        if ":" in k:
            print(f"  {k}: {v:.3f}")

    t0 = time.perf_counter()
    # Seed pose only; ignore the seed scales because subject .mp defines the scale prior.
    init = IKInitializer(skel, obs_cal, anatomical=anat)
    seed = init.run(MarkerIKConfig(max_iters=30))

    fitter = MarkerFitter(skel, obs_cal, weights=weights, anatomical=anat)
    cfg = MarkerFitConfig(
        outer_iters=4,
        optimize_scales=False,
        offset_prior_weight=4.0,
        anatomical_prior_factor=40.0,
        offset_max_delta=0.025,
        inner_first=MarkerIKConfig(max_iters=60),
        inner=MarkerIKConfig(max_iters=25),
        final_inner=MarkerIKConfig(max_iters=50),
    )
    fit = fitter.fit(init_scales=init_scales, q_init=seed.poses, config=cfg)

    # Whole-window fixed-scale IK.
    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    ik = solve_marker_ik(
        skel, obs, q_seed, group_scales=fit.group_scales, weights=weights,
        config=MarkerIKConfig(max_iters=60),
    )
    wall_s = time.perf_counter() - t0

    pm = per_marker_rms(skel, obs, ik.q, fit.group_scales, fit.marker_offsets)
    motion = build_motion(spec, ik.q, fps=session.point_rate, group_scales=fit.group_scales)
    belt = measured_belt_grf(session)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    R = pack_result(spec, ik.q, fit.group_scales, fit.marker_offsets, ik.marker_rms, pm, names, motion, grf, t, window, wall_s)
    save_npz(ANTHRO_CACHE, R)
    return R


def run_anthro_prior(fresh=False):
    """Anthropometric scale prior, but scales may still move to fit S001 markers."""
    if ANTHRO_PRIOR_CACHE.exists() and not fresh:
        print(f"Loading cached anthropo_prior_v1 from {ANTHRO_PRIOR_CACHE.name}")
        spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
        return load_npz(ANTHRO_PRIOR_CACHE, spec)

    session, spec, skel, names, mm, obs, obs_cal, anat, window = _common_fit_inputs()
    lo, hi = window
    F = obs.shape[0]
    scale_target, scale_w, _ = anthropometric_scale_prior(
        spec, SUBJECT_MP, length_weight=50.0, width_weight=10.0, include_upper_body=False
    )
    init_scales = np.ones_like(scale_target)
    init_scales[scale_w > 0] = scale_target[scale_w > 0]

    t0 = time.perf_counter()
    init = IKInitializer(skel, obs_cal, anatomical=anat)
    seed = init.run(MarkerIKConfig(max_iters=30))

    # Keep marker weights uniform here: isolate the effect of the scale prior from marker
    # reweighting, so RMS remains comparable to the original baseline.
    fitter = MarkerFitter(skel, obs_cal, anatomical=anat)
    cfg = MarkerFitConfig(
        outer_iters=5,
        optimize_scales=True,
        scale_prior_target=scale_target,
        scale_prior_weights=scale_w,
        offset_prior_weight=2.0,
        anatomical_prior_factor=35.0,
        offset_max_delta=0.035,
        inner_first=MarkerIKConfig(max_iters=60),
        inner=MarkerIKConfig(max_iters=25),
        final_inner=MarkerIKConfig(max_iters=50),
    )
    fit = fitter.fit(init_scales=init_scales, q_init=seed.poses, config=cfg)

    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    ik = solve_marker_ik(
        skel, obs, q_seed, group_scales=fit.group_scales,
        config=MarkerIKConfig(max_iters=60),
    )
    wall_s = time.perf_counter() - t0

    pm = per_marker_rms(skel, obs, ik.q, fit.group_scales, fit.marker_offsets)
    motion = build_motion(spec, ik.q, fps=session.point_rate, group_scales=fit.group_scales)
    belt = measured_belt_grf(session)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    R = pack_result(spec, ik.q, fit.group_scales, fit.marker_offsets, ik.marker_rms, pm, names, motion, grf, t, window, wall_s)
    save_npz(ANTHRO_PRIOR_CACHE, R)
    return R


def load_baseline():
    if not BASE_CACHE.exists():
        raise FileNotFoundError(
            f"Missing baseline cache {BASE_CACHE}. Run make_s001_ik_figures.py first."
        )
    spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    return load_npz(BASE_CACHE, spec)


def metrics(label, R):
    spec = R["spec"]
    idx = spec.dof_index_map()
    q = R["poses"]
    ranges = {n: float((q[:, idx[n]].max() - q[:, idx[n]].min()) * DEG) for n in KEY_DOFS}
    peaks = {n: float(q[:, idx[n]].max() * DEG) for n in KEY_DOFS}
    # Penalty = distance outside practical normal-walk range bands.
    band_pen = 0.0
    for n, r in ranges.items():
        lo, hi = GAIT_RANGE_BANDS[n]
        band_pen += max(0.0, lo - r) + max(0.0, r - hi)
    frames = int(len(R["t"]))
    ik_iters = int(R.get("ik_iters", -1))
    names = list(R.get("marker_names", []))
    pm = np.asarray(R.get("per_marker_rms", []), dtype=np.float64) * 1e3
    mm = s001_marker_map()
    anat_mask = np.array([n in mm.anatomical for n in names], dtype=bool) if names else np.zeros(0, dtype=bool)
    mapped_mask = np.array([n in mm.model_to_capture for n in names], dtype=bool) if names else np.zeros(0, dtype=bool)
    cluster_mask = mapped_mask & (~anat_mask)
    return {
        "label": label,
        "window": [int(x) for x in R["window"]],
        "frames": frames,
        "wall_s": float(R["wall_s"]),
        "frames_per_second": float(frames / R["wall_s"]) if R["wall_s"] > 0 else None,
        "ik_iters": ik_iters,
        "frame_iters_per_second": float(frames * ik_iters / R["wall_s"]) if ik_iters > 0 and R["wall_s"] > 0 else None,
        "speedup_vs_baseline": None,
        "median_marker_rms_mm": float(np.nanmedian(R["marker_rms"]) * 1e3),
        "mean_marker_rms_mm": float(np.nanmean(R["marker_rms"]) * 1e3),
        "median_per_marker_rms_mm": float(np.nanmedian(pm)) if pm.size else None,
        "median_anatomical_marker_rms_mm": float(np.nanmedian(pm[anat_mask])) if pm.size and anat_mask.any() else None,
        "median_tracking_marker_rms_mm": float(np.nanmedian(pm[cluster_mask])) if pm.size and cluster_mask.any() else None,
        "joint_ranges_deg": ranges,
        "joint_peaks_deg": peaks,
        "range_band_penalty_deg": float(band_pen),
        "scale_min": float(np.nanmin(R["scales"])),
        "scale_max": float(np.nanmax(R["scales"])),
    }


def fig_compare_gait(results):
    base = results["baseline_cached"]
    fast = results["fast_bilevel_v1"]
    robust_balanced = results.get("robust_balanced_v2")
    robust = results.get("robust_anatomical_v1")
    no_static = results.get("no_static_calib_v1")
    spec = base["spec"]
    idx = spec.dof_index_map()
    t = base["t"]
    panels = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, n in zip(axes.ravel(), panels):
        ax.plot(t, base["poses"][:, idx[n]] * DEG, color="0.35", lw=1.5, label="baseline")
        ax.plot(t, fast["poses"][:, idx[n]] * DEG, color="tab:orange", lw=1.4, ls="--", label="fast_bilevel_v1")
        if robust_balanced is not None:
            ax.plot(t, robust_balanced["poses"][:, idx[n]] * DEG, color="tab:red", lw=1.6, label="robust balanced v2")
        if robust is not None:
            ax.plot(t, robust["poses"][:, idx[n]] * DEG, color="tab:brown", lw=1.1, alpha=0.8, label="robust anatomical v1")
        if no_static is not None:
            ax.plot(t, no_static["poses"][:, idx[n]] * DEG, color="tab:pink", lw=1.0, alpha=0.8, label="no static calib markers")
        lo, hi = GAIT_RANGE_BANDS[n]
        rng_b = (base["poses"][:, idx[n]].max() - base["poses"][:, idx[n]].min()) * DEG
        rng_f = (fast["poses"][:, idx[n]].max() - fast["poses"][:, idx[n]].min()) * DEG
        ax.set_title(f"{n}\nrange {rng_b:.1f}° -> {rng_f:.1f}° (band {lo:.0f}-{hi:.0f})", fontsize=9)
        ax.set_ylabel("angle (deg)")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("S001 IK accuracy progress: robust marker weighting improves gait ranges", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "12_s001_ik_baseline_vs_fast.png", dpi=130)
    plt.close(fig)


def fig_anthro_attempts(results):
    base = results["baseline_cached"]
    spec = base["spec"]
    idx = spec.dof_index_map()
    t = base["t"]
    panels = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, n in zip(axes.ravel(), panels):
        ax.plot(t, base["poses"][:, idx[n]] * DEG, color="0.35", lw=1.5, label="baseline")
        ax.plot(t, results["anthropo_fixed_v1"]["poses"][:, idx[n]] * DEG,
                color="tab:green", lw=1.2, label="anthro_fixed_v1")
        ax.plot(t, results["anthropo_prior_v1"]["poses"][:, idx[n]] * DEG,
                color="tab:blue", lw=1.2, ls="--", label="anthro_prior_v1")
        ax.set_title(n, fontsize=9)
        ax.set_ylabel("angle (deg)")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("S001 anthropometric-prior attempts (v1 rejected: worse marker RMS/ranges)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "14_s001_ik_anthro_attempts.png", dpi=130)
    plt.close(fig)


def fig_speed_accuracy(metrics_list):
    labels = [m["label"].replace("_", "\n") for m in metrics_list]
    wall = [m["wall_s"] for m in metrics_list]
    fps = [m["frames_per_second"] for m in metrics_list]
    rms = [m["median_marker_rms_mm"] for m in metrics_list]
    pen = [m["range_band_penalty_deg"] for m in metrics_list]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    colors = ["0.45", "tab:orange", "tab:red", "tab:brown", "tab:pink", "tab:purple", "tab:green", "tab:blue"][:len(labels)]
    axes[0].bar(labels, wall, color=colors)
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_title("Runtime")
    axes[1].bar(labels, fps, color=colors)
    axes[1].set_ylabel("frames/s")
    axes[1].set_title("Throughput")
    axes[2].bar(labels, rms, color=colors)
    axes[2].set_ylabel("median marker RMS (mm)")
    axes[2].set_title("Marker fit")
    axes[3].bar(labels, pen, color=colors)
    axes[3].set_ylabel("joint-range band penalty (deg)")
    axes[3].set_title("Gait-amplitude guardrail")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("S001-only IK benchmark: wall time, frames/s, accuracy", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "13_s001_ik_speed_accuracy.png", dpi=130)
    plt.close(fig)


def main():
    fresh = "--fresh-anthro" in sys.argv
    baseline = load_baseline()
    fast = run_fast_bilevel(fresh=fresh)
    robust_balanced = run_robust_balanced(fresh=fresh)
    robust = run_robust_anatomical(fresh=fresh)
    no_static = run_no_static_calib(fresh=fresh)
    warm_200 = run_warp_warm_200(fast, fresh=fresh)
    anth_fixed = run_anthro_fixed(fresh=fresh)
    anth_prior = run_anthro_prior(fresh=fresh)
    results = {
        "baseline_cached": baseline,
        "fast_bilevel_v1": fast,
        "robust_balanced_v2": robust_balanced,
        "robust_anatomical_v1": robust,
        "no_static_calib_v1": no_static,
        "warp_warm_200_v1": warm_200,
        "anthropo_fixed_v1": anth_fixed,
        "anthropo_prior_v1": anth_prior,
    }
    mets = [metrics(k, v) for k, v in results.items()]
    base_wall = mets[0]["wall_s"]
    for m in mets:
        m["speedup_vs_baseline"] = base_wall / m["wall_s"] if m["wall_s"] > 0 else None
    METRICS_JSON.write_text(json.dumps(mets, indent=2))
    fig_compare_gait(results)
    fig_speed_accuracy(mets)
    fig_anthro_attempts(results)
    print(json.dumps(mets, indent=2))
    print("Wrote", METRICS_JSON)
    print("Wrote figures 12_s001_ik_baseline_vs_fast.png, 13_s001_ik_speed_accuracy.png, and 14_s001_ik_anthro_attempts.png")


if __name__ == "__main__":
    main()
