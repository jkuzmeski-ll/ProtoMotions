# SPDX-License-Identifier: MIT
#
# Real-data pipeline — stitch the whole gold-standard chain on a captured session:
#
#   load_session (markers + split-belt GRF)
#     -> observations_from_session (Plug-in-Gait -> Rajagopal, lab Z-up -> OpenSim Y-up)
#     -> IKInitializer (seed scales/poses)  -> MarkerFitter (scales + per-marker offsets
#        + poses; drives marker RMS down)
#     -> build_motion (fitted q(t) -> Z-up per-body world pose trajectory)
#     -> build_subject_sole (subject plantar geometry from the static C3D, in calcn frame)
#     -> per-stance ground registration (from the kinematics, independent of the absolute
#        fit height) and the measured belt GRF/COP (right belt = right foot)
#     -> calibrate the (hydroelastic) distributed contact model against the measured GRF.
#
# "Use Newton as much as possible": the pose fit is Warp-driven (WarpSkeleton IK), the
# contact forward + calibration run through the Warp contact kernels, and the whole thing
# stays on the local GPU. This module is the orchestration + diagnostics layer over the
# already-validated pieces; it favours robustness/reporting over asserting a
# publication-grade fit (reconstruction quality gates the numbers).
#
# Belt/foot convention: the split-belt plates are separated along lab +x/-x; by default
# the +x plate is the subject's right foot (facing +y). Override with ``right_plate_x_sign``.

"""End-to-end real-data reconstruction + contact calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from biomech.contact.calibration import (
    CalibrationConfig,
    CalibrationResult,
    CalibrationTarget,
    calibrate_hydroelastic,
)
from biomech.contact.elastic_foundation import FootSole
from biomech.contact.foot_geometry import subject_sole_from_session
from biomech.contact.kinematics import foot_trajectory_from_motion
from biomech.contact.stance import (
    flat_foot_mask,
    register_ground_flatfoot,
    sole_world_z,
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class FootContactFit:
    side: str
    body: str
    sole: FootSole
    ground_z: float
    stance_mask: np.ndarray  # (F,) measured-stance frames used for calibration
    measured_grf: np.ndarray  # (F, 3) on the subject (world)
    measured_cop: np.ndarray  # (F, 3) world (NaN in swing)
    calibration: Optional[CalibrationResult]


@dataclass
class SubjectPipelineResult:
    window: Tuple[int, int]
    group_scales: np.ndarray
    marker_rms_median: float
    marker_rms: np.ndarray
    motion: object  # MotionExportResult
    feet: Dict[str, FootContactFit] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def pick_visible_window(obs: np.ndarray, present: np.ndarray, n: int) -> int:
    """Start index of a length-``n`` window maximizing mapped-marker visibility."""
    vis = np.isfinite(obs[:, present, :]).all(axis=2).mean(axis=1)
    F = obs.shape[0]
    if F <= n:
        return 0
    step = max(1, F // 300)
    best_s, best = 0, -1.0
    for s in range(0, F - n, step):
        sc = float(vis[s:s + n].mean())
        if sc > best:
            best, best_s = sc, s
    return best_s


def reconstruct_window(
    session,
    spec,
    window: Tuple[int, int],
    mapping=None,
    marker_config=None,
    device: str = "cpu",
):
    """Fit scales/offsets/poses on a frame window; return (MarkerFitResult, obs, anat)."""
    from biomech.fitting.ik import MarkerIKConfig
    from biomech.fitting.ik_initializer import IKInitializer
    from biomech.fitting.marker_fitter import MarkerFitter
    from biomech.fitting.marker_map import (
        anatomical_mask,
        observations_from_session,
        s001_marker_map,
    )
    from biomech.skeleton.skeleton import WarpSkeleton

    mm = mapping or s001_marker_map()
    model_names = WarpSkeleton(spec, device=device).marker_names()
    obs_all, present = observations_from_session(session, model_names, mm)
    anat = anatomical_mask(model_names, mm)

    lo, hi = window
    obs = obs_all[lo:hi]

    skel = WarpSkeleton(spec, device=device)
    # closed-form seed
    init = IKInitializer(skel, obs, anatomical=anat)
    seed = init.run(MarkerIKConfig(max_iters=40))

    # full bilevel fit (scales + per-marker offsets + poses)
    fitter = MarkerFitter(skel, obs, anatomical=anat)
    result = fitter.fit(
        init_scales=seed.group_scales,
        q_init=seed.poses,
        config=marker_config,
    )
    return result, obs, anat


# ---------------------------------------------------------------------------
# Ground registration + measured GRF
# ---------------------------------------------------------------------------


def register_ground_z(
    sole: FootSole,
    pos: np.ndarray,
    quat: np.ndarray,
    stance_mask: np.ndarray,
    penetration: float = 0.005,
    contact_percentile: float = 80.0,
) -> float:
    """Register the effective ground plane from the kinematics (robustly).

    Contact needs the ground *above* a sole patch (penetration ``d = ground_z - z > 0``).
    We take a high percentile of the per-frame lowest sole patch over the measured-stance
    frames and place the ground ``penetration`` meters above it, so the large majority of
    stance frames register contact while no single noisy, deeply-dipping frame dominates.
    This removes the dependence on the absolute vertical fit offset / analytic
    ``plantar_drop`` when comparing to the measured GRF.

    NOTE: predicting instantaneous GRF from *prescribed* (noisy) kinematics with a stiff
    distributed spring is inherently sensitive to this vertical registration and to
    centimeter-level reconstruction noise; the calibration below is best-effort. Aggregate
    objectives / smoothed kinematics are the robust path (future work).
    """
    zsole = sole_world_z(sole, pos, quat)  # (F, N)
    per_frame_min = np.min(zsole, axis=1)  # (F,)
    sel = per_frame_min[stance_mask] if stance_mask.any() else per_frame_min
    plane = float(np.percentile(sel, contact_percentile))
    return plane + penetration


def measured_belt_grf(
    session, right_plate_x_sign: int = 1
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-foot measured GRF/COP on the point timeline (right belt = right foot).

    Returns ``{"R": (grf, cop), "L": (grf, cop)}`` each ``(F, 3)`` world.
    """
    forces = session.forces_on_point_timeline()
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for plate in session.force_plates:
        side = "R" if plate.x_sign == right_plate_x_sign else "L"
        grf = forces[f"plate{plate.index}_grf"]
        cop = forces[f"plate{plate.index}_cop"]
        out[side] = (np.asarray(grf), np.asarray(cop))
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_subject_pipeline(
    trial_session,
    static_session,
    spec,
    window: Optional[Tuple[int, int]] = None,
    window_len: int = 60,
    phase: Optional[str] = None,
    mapping=None,
    marker_config=None,
    right_plate_x_sign: int = 1,
    fz_threshold: float = 50.0,
    sole_nx: int = 14,
    sole_ny: int = 6,
    contact_backend: str = "numpy",
    device: str = "cpu",
    calibrate: bool = True,
    free_params: Tuple[str, ...] = ("k_bed", "hc_alpha"),
    registration: str = "percentile",
    objective: str = "perframe",
    enrich_foot_markers: bool = True,
    placement_window_len: int = 60,
    collapse_lower_clusters: bool = True,
) -> SubjectPipelineResult:
    """Run reconstruction + subject-sole contact calibration on a captured session.

    Args:
        trial_session: the dynamic (walking/running) ``CaptureSession`` with belt GRF.
        static_session: the static (calibration) ``CaptureSession`` for foot dimensions.
        spec: parsed Rajagopal ``SkeletonSpec``.
        window: explicit ``(lo, hi)`` frame window; else auto-picked (length ``window_len``).
        phase: optional protocol phase (``"walk"``/``"run"``/``"all"``) — when given
            (and ``window`` is None) the window is restricted to that phase using the
            session's Speedchange protocol, then a visibility-best ``window_len`` sub-window
            is picked inside it. Requires the trial to be loaded with ``speedchange_path``.
        right_plate_x_sign: which plate ``x_sign`` is the right foot (default +x).
        fz_threshold: measured vertical GRF (N) above which a frame counts as stance.
        free_params: hydroelastic params to calibrate. Default ``("k_bed", "hc_alpha")``
            — the *vertical* stiffness/dissipation, which the measured Fz determines well.
            Friction ``mu`` is intentionally excluded: during planted stance the sliding
            velocity is ~0, so static friction is physically indeterminate and shear
            cannot be recovered from prescribed kinematics (it needs sliding phases).
        registration: ground-plane registration mode. ``"percentile"`` (default) uses a
            high percentile of the per-frame lowest sole point over measured stance;
            ``"flatfoot"`` registers only from genuinely planted (flat, low-speed,
            high-load) frames — more robust on dynamic windows where the foot rolls.
        objective: calibration objective, ``"perframe"`` (default) or ``"aggregate"``
            (per-stance-phase mean + peak; robust to per-frame kinematic noise, recommended
            for walk/run windows).
    """
    from biomech.export.motion import build_motion
    from biomech.fitting.marker_map import (
        observations_from_session,
        s001_marker_map,
    )
    from biomech.skeleton.skeleton import WarpSkeleton

    mm = mapping or s001_marker_map()
    if collapse_lower_clusters:
        from biomech.fitting.cluster_collapse import collapse_clusters
        mm, _ = collapse_clusters(spec, mm)
    if enrich_foot_markers:
        from biomech.fitting.marker_placement import place_foot_markers
        n_static = int(np.asarray(static_session.markers).shape[0])
        place_foot_markers(
            spec, static_session, mapping=mm, marker_config=marker_config,
            device=device, frame_range=(0, min(placement_window_len, n_static)),
        )
    model_names = WarpSkeleton(spec, device=device).marker_names()
    obs_all, present = observations_from_session(trial_session, model_names, mm)

    if window is None:
        if phase is not None:
            plo, phi = trial_session.phase_window(phase)
            sub = obs_all[plo:phi]
            _, pres = observations_from_session(trial_session, model_names, mm)
            s = plo + pick_visible_window(sub, pres, window_len)
            window = (s, min(s + window_len, phi))
        else:
            s = pick_visible_window(obs_all, present, window_len)
            window = (s, s + window_len)
    lo, hi = window

    fit, _, _ = reconstruct_window(
        trial_session, spec, window, mapping=mm,
        marker_config=marker_config, device=device,
    )
    fps = trial_session.point_rate
    motion = build_motion(spec, fit.poses, fps=fps, group_scales=fit.group_scales)

    result = SubjectPipelineResult(
        window=window,
        group_scales=fit.group_scales,
        marker_rms_median=float(np.nanmedian(fit.marker_rms)),
        marker_rms=fit.marker_rms,
        motion=motion,
    )

    belt = measured_belt_grf(trial_session, right_plate_x_sign)
    for side, body in (("R", "calcn_r"), ("L", "calcn_l")):
        if side not in belt:
            continue
        grf_all, cop_all = belt[side]
        grf = grf_all[lo:hi]
        cop = cop_all[lo:hi]
        stance = grf[:, 2] > fz_threshold

        sole = subject_sole_from_session(
            static_session, spec, side, group_scales=fit.group_scales,
            nx=sole_nx, ny=sole_ny,
        )
        pos, quat, linvel, angvel = foot_trajectory_from_motion(motion, body)
        if registration == "flatfoot":
            flat = flat_foot_mask(
                sole, pos, quat, linvel=linvel, fz=grf[:, 2],
                fz_threshold=fz_threshold,
            )
            ground_z = register_ground_flatfoot(
                sole, pos, quat, flat, fallback=stance
            )
        else:
            ground_z = register_ground_z(sole, pos, quat, stance)

        calib = None
        if calibrate and stance.any():
            target = CalibrationTarget(grf=grf, cop=cop)
            # horizontal_weight=0: fit the vertical GRF only (mu unobservable in stance)
            cfg = CalibrationConfig(
                max_iters=50, fz_threshold=fz_threshold, horizontal_weight=0.0,
                objective=objective,
            )
            calib = calibrate_hydroelastic(
                sole, pos, quat, linvel, angvel, target,
                free_params=free_params,
                ground_z=ground_z, backend=contact_backend, config=cfg,
            )

        result.feet[side] = FootContactFit(
            side=side,
            body=body,
            sole=sole,
            ground_z=ground_z,
            stance_mask=stance,
            measured_grf=grf,
            measured_cop=cop,
            calibration=calib,
        )

    return result
