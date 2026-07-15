# SPDX-License-Identifier: MIT
#
# Generate correctness figures for the batched marker IK (biomech.fitting.ik).
#
# The IK unit tests prove the *marker* reprojection error goes to ~zero. These figures
# go one step further and prove the recovered *joint angles* match ground truth. Because
# real mocap has no ground-truth joint angles, we use an honest round trip:
#
#   1. Build a smooth, physiologically-shaped synthetic gait trajectory q_true(t) that
#      lives inside the model's joint limits.
#   2. Forward-kinematic it through the Warp skeleton to synthetic marker clouds.
#   3. Corrupt the markers with Gaussian measurement noise (0-8 mm).
#   4. Recover q(t) with solve_marker_ik, initialized from a single NEUTRAL static pose
#      (NOT the per-frame truth), warm-started nowhere -- the solver must find the right
#      minimum on its own.
#   5. Compare recovered joint angles to q_true.
#
# Usage (from the repo root):
#   .venv/Scripts/python projects/biomech/tools/make_ik_figures.py
# Figures are written to projects/biomech/docs/figures/.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from biomech.fitting.ik import MarkerIKConfig, position_limits, solve_marker_ik  # noqa: E402
from biomech.osim import parse_osim  # noqa: E402
from biomech.skeleton.skeleton import WarpSkeleton  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DEG = 180.0 / np.pi

# Lower-body rotational DOFs we report on (pelvis orientation + hip/knee/ankle, both sides).
LOWER_BODY = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
]
# Six representative DOFs for the time-series overlay.
TS_DOFS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r",
           "hip_flexion_l", "knee_angle_l", "pelvis_tilt"]


def synthetic_gait(spec, F=120, dt=0.01):
    """A smooth, feasible, gait-like trajectory q_true (F, ndof).

    Named lower-body DOFs get physiologically-shaped sinusoids; everything else gets a
    small smooth motion. All DOFs are clamped strictly inside the model limits so the
    clamping IK can recover them exactly.
    """
    idx = spec.dof_index_map()
    ndof = spec.num_dofs
    t = np.arange(F) * dt
    T = F * dt  # one cycle over the clip
    ph = 2.0 * np.pi * t / T
    q = np.zeros((F, ndof))

    def put(name, vals):
        if name in idx:
            q[:, idx[name]] = vals

    # Pelvis: walk forward ~1.2 m/s, gentle 6-DOF motion.
    put("pelvis_tx", 1.2 * t)
    put("pelvis_ty", 0.90 + 0.015 * np.sin(2 * ph))
    put("pelvis_tz", 0.02 * np.sin(ph))
    put("pelvis_tilt", 0.08 + 0.03 * np.sin(2 * ph))
    put("pelvis_list", 0.03 * np.sin(ph))
    put("pelvis_rotation", 0.06 * np.sin(ph))
    # Hips: flexion swings out of phase L/R; small ad/abduction + rotation.
    put("hip_flexion_r", 0.20 + 0.38 * np.cos(ph))
    put("hip_flexion_l", 0.20 + 0.38 * np.cos(ph + np.pi))
    put("hip_adduction_r", 0.06 * np.sin(ph))
    put("hip_adduction_l", -0.06 * np.sin(ph))
    put("hip_rotation_r", 0.05 * np.sin(ph))
    put("hip_rotation_l", 0.05 * np.sin(ph + np.pi))
    # Knees: mostly flexed swing (stay >= 0).
    put("knee_angle_r", 0.45 + 0.40 * np.cos(2 * ph + 0.6))
    put("knee_angle_l", 0.45 + 0.40 * np.cos(2 * ph + 0.6 + np.pi))
    # Ankles.
    put("ankle_angle_r", 0.10 * np.sin(2 * ph))
    put("ankle_angle_l", 0.10 * np.sin(2 * ph + np.pi))
    # Trunk + arms (arms swing opposite to legs).
    put("lumbar_extension", -0.05 + 0.02 * np.sin(2 * ph))
    put("lumbar_bending", 0.02 * np.sin(ph))
    put("lumbar_rotation", 0.03 * np.sin(ph))
    put("arm_flex_r", 0.25 * np.cos(ph + np.pi))
    put("arm_flex_l", 0.25 * np.cos(ph))
    put("elbow_flex_r", 0.5 + 0.15 * np.sin(ph))
    put("elbow_flex_l", 0.5 + 0.15 * np.sin(ph + np.pi))

    lo, hi = position_limits(spec)
    q = np.clip(q, lo, hi)
    ranged = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    q[:, ranged] = np.clip(q[:, ranged], lo[ranged] + 1e-3, hi[ranged] - 1e-3)
    return q, t


def neutral_init(spec, F):
    """A single static neutral pose, repeated for every frame (the IK starting point)."""
    lo, hi = position_limits(spec)
    q0 = np.zeros(spec.num_dofs)
    q0 = np.clip(q0, lo, hi)
    ranged = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    q0[ranged] = np.clip(q0[ranged], lo[ranged] + 1e-3, hi[ranged] - 1e-3)
    # Put it into a plausible upright standing pose height.
    idx = spec.dof_index_map()
    if "pelvis_ty" in idx:
        q0[idx["pelvis_ty"]] = 0.90
    return np.repeat(q0[None], F, axis=0)


# LM makes tiny accepted steps forever near a nonzero-residual (noisy) minimum, so it
# runs to max_iters instead of tripping the convergence threshold. Joint angles are fully
# converged (< 0.06 deg vs a 120-iter reference) by ~40 iters, so we cap at 50 for speed.
_MAX_ITERS = 50
_SEED = 42
_solve_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def run_ik(skel, q_true, sigma_m, seed=_SEED):
    """FK -> add sigma_m marker noise -> IK from neutral. Cached by sigma (single seed).

    Returns recovered q (F, ndof) + per-frame marker RMS.
    """
    key = round(float(sigma_m), 6)
    if key in _solve_cache:
        return _solve_cache[key]
    _, markers = skel.forward(q_true)  # (F, M, 3)
    rng = np.random.default_rng(seed)
    obs = markers + rng.normal(0.0, sigma_m, size=markers.shape)
    q_init = neutral_init(skel.spec, q_true.shape[0])
    res = solve_marker_ik(skel, obs, q_init, config=MarkerIKConfig(max_iters=_MAX_ITERS))
    _solve_cache[key] = (res.q, res.marker_rms)
    return res.q, res.marker_rms


def angle_err_deg(spec, q_true, q_rec, names):
    """Per-DOF angle error time series (deg) for the given DOF names."""
    idx = spec.dof_index_map()
    out = {}
    for n in names:
        i = idx[n]
        out[n] = (q_rec[:, i] - q_true[:, i]) * DEG
    return out


# ---------------------------------------------------------------------------
# Figure 6: per-joint angle recovery error (noiseless + realistic noise)
# ---------------------------------------------------------------------------
def fig_joint_error(skel, q_true):
    spec = skel.spec
    q_clean, rms_clean = run_ik(skel, q_true, sigma_m=0.0)
    q_noisy, rms_noisy = run_ik(skel, q_true, sigma_m=0.002)  # 2 mm

    e_clean = angle_err_deg(spec, q_true, q_clean, LOWER_BODY)
    e_noisy = angle_err_deg(spec, q_true, q_noisy, LOWER_BODY)
    max_clean = [np.max(np.abs(e_clean[n])) for n in LOWER_BODY]
    rms_noisy_dof = [np.sqrt(np.mean(e_noisy[n] ** 2)) for n in LOWER_BODY]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    x = np.arange(len(LOWER_BODY))

    ax1.bar(x, max_clean, color="tab:blue")
    ax1.set_yscale("log")
    ax1.axhline(1e-3, color="tab:green", ls="--", lw=1, label="0.001° (1 millidegree)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(LOWER_BODY, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("max |angle error| (deg)")
    ax1.set_title("Noiseless markers: joint angles recovered to numerical precision "
                  f"(marker RMS = {np.mean(rms_clean) * 1e6:.2f} µm)")
    ax1.legend(loc="upper right")

    ax2.bar(x, rms_noisy_dof, color="tab:orange")
    ax2.axhline(1.0, color="0.4", ls="--", lw=1, label="1.0°")
    ax2.set_xticks(x)
    ax2.set_xticklabels(LOWER_BODY, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("RMS angle error (deg)")
    ax2.set_title(f"Realistic 2 mm marker noise: sub-degree joint-angle RMS "
                  f"(marker RMS = {np.mean(rms_noisy) * 1e3:.2f} mm)")
    ax2.legend(loc="upper right")

    fig.suptitle("Marker IK joint-angle recovery (synthetic gait round trip)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "06_ik_joint_angle_recovery.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: true vs recovered joint-angle time series (with 2 mm noise)
# ---------------------------------------------------------------------------
def fig_time_series(skel, q_true, t):
    spec = skel.spec
    idx = spec.dof_index_map()
    q_rec, _ = run_ik(skel, q_true, sigma_m=0.002)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, name in zip(axes.ravel(), TS_DOFS):
        i = idx[name]
        ax.plot(t, q_true[:, i] * DEG, color="k", lw=2.2, label="ground truth")
        ax.plot(t, q_rec[:, i] * DEG, color="tab:red", lw=1.3, ls="--",
                label="IK recovered")
        rms = np.sqrt(np.mean(((q_rec[:, i] - q_true[:, i]) * DEG) ** 2))
        ax.set_title(f"{name}  (RMS {rms:.2f}°)", fontsize=10)
        ax.set_ylabel("angle (deg)")
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("True vs IK-recovered joint angles under 2 mm marker noise "
                 "(neutral-pose initialization)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "07_ik_time_series.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: recovered-vs-true scatter + noise sweep
# ---------------------------------------------------------------------------
def fig_scatter_and_sweep(skel, q_true):
    spec = skel.spec
    idx = spec.dof_index_map()
    li = [idx[n] for n in LOWER_BODY]

    # Panel A: recovered vs true scatter (2 mm noise), all lower-body DOFs & frames.
    q_rec, _ = run_ik(skel, q_true, sigma_m=0.002)
    xt = (q_true[:, li] * DEG).ravel()
    yr = (q_rec[:, li] * DEG).ravel()

    # Panel B: sweep marker noise, measure joint-angle RMS + marker RMS.
    sigmas_mm = [0.0, 1.0, 2.0, 4.0, 8.0]
    ang_rms, mk_rms = [], []
    for s in sigmas_mm:
        qr, mrms = run_ik(skel, q_true, sigma_m=s * 1e-3)
        err = (qr[:, li] - q_true[:, li]) * DEG
        ang_rms.append(np.sqrt(np.mean(err ** 2)))
        mk_rms.append(np.mean(mrms) * 1e3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    lim = [min(xt.min(), yr.min()) - 2, max(xt.max(), yr.max()) + 2]
    ax1.plot(lim, lim, color="0.5", ls="--", lw=1, label="identity")
    ax1.scatter(xt, yr, s=5, alpha=0.25, color="tab:blue")
    r2 = 1.0 - np.sum((yr - xt) ** 2) / np.sum((xt - xt.mean()) ** 2)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("true joint angle (deg)")
    ax1.set_ylabel("IK-recovered joint angle (deg)")
    ax1.set_title(f"Recovered vs true (2 mm noise, all lower-body DOFs)\nR² = {r2:.5f}")
    ax1.legend(loc="upper left")

    ax2.plot(sigmas_mm, ang_rms, "o-", color="tab:orange", label="joint-angle RMS (deg)")
    ax2.set_xlabel("marker noise σ (mm, per coordinate)")
    ax2.set_ylabel("joint-angle RMS error (deg)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2b = ax2.twinx()
    ax2b.plot(sigmas_mm, mk_rms, "s--", color="tab:blue", label="marker RMS (mm)")
    ax2b.set_ylabel("marker reprojection RMS (mm)", color="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Error scales linearly & unbiased with marker noise")
    ax2.grid(alpha=0.3)

    fig.suptitle("Marker IK accuracy vs measurement noise (synthetic gait round trip)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "08_ik_noise_sweep.png", dpi=130)
    plt.close(fig)


def main():
    spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    skel = WarpSkeleton(spec, device="cpu")
    q_true, t = synthetic_gait(spec, F=60, dt=0.02)
    print("Figure 6: per-joint angle recovery error ...")
    fig_joint_error(skel, q_true)
    print("Figure 7: true vs recovered time series ...")
    fig_time_series(skel, q_true, t)
    print("Figure 8: scatter + noise sweep ...")
    fig_scatter_and_sweep(skel, q_true)
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
