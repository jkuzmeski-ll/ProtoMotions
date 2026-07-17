# SPDX-License-Identifier: MIT
#
# Real-data IK figures on the S001 capture (instrumented-treadmill walking).
#
# Real mocap has NO ground-truth joint angles, so unlike the synthetic round trip
# (make_ik_figures.py) we cannot plot recovered-vs-true. What we CAN show, and what these
# figures show, is that the IK reconstructs *physiologically correct, self-consistent*
# joint angles from the real markers:
#
#   * recovered lower-body joint-angle waveforms have the right shape, range and timing
#     for treadmill gait, and the two legs are bilaterally symmetric with the expected
#     ~50%-cycle phase lag;
#   * the swing/stance timing of the recovered kinematics lines up with the INDEPENDENTLY
#     measured split-belt vertical GRF (shaded), i.e. the feet are reconstructed as loaded
#     exactly when the plates say they are;
#   * the marker reprojection residual is reported honestly (per frame + per marker) --
#     it is limited by the Plug-in-Gait <-> Rajagopal marker-set mismatch, not the solver.
#
# Methodology (standard scale-once / IK-per-frame workflow):
#   1. Calibrate {group scales, marker offsets} once with the full bilevel MarkerFitter on
#      a short mid-window slice.
#   2. With those fixed, run per-frame Warp marker IK across a contiguous walk window.
#
# Usage (from the repo root):
#   .venv/Scripts/python projects/biomech/tools/make_s001_ik_figures.py
# Figures are written to projects/biomech/docs/figures/. Requires the S001 data + torch.

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from biomech.contact.pipeline import (  # noqa: E402
    detect_right_plate_x_sign,
    measured_belt_grf,
    pick_visible_window,
    reconstruct_window,
)
from biomech.export.motion import build_motion  # noqa: E402
from biomech.fitting.ik import MarkerIKConfig, solve_marker_ik  # noqa: E402
from biomech.fitting.marker_fitter import MarkerFitConfig  # noqa: E402
from biomech.fitting.cluster_collapse import collapse_clusters  # noqa: E402
from biomech.fitting.marker_map import (  # noqa: E402
    observations_from_session,
    s001_marker_map,
)
from biomech.osim import parse_osim  # noqa: E402
from biomech.session import load_session  # noqa: E402
from biomech.skeleton.skeleton import WarpSkeleton  # noqa: E402
from biomech.fitting.marker_placement import place_foot_markers  # noqa: E402
from biomech.tests import CAL_C3D, SPEEDCHANGE, TRIAL_C3D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_s001_ik_cache.npz"
DEG = 180.0 / np.pi
DEVICE = "cuda"

WIN_LEN = 150          # contiguous walk frames to reconstruct (~1.5 s at 100 Hz)
CALIB_LEN = 60         # mid-window slice used to calibrate scales + offsets

# Bilateral joint pairs for the gait-angle grid (full enriched lower-body DOF set).
# The MTP is now fit (unlocked once a toes/hallux marker was added); the subtalar is
# intentionally left locked -- surface markers can't cleanly resolve inversion/eversion
# (it rails at the anatomical limit if freed), so it is shown flat at 0 for honesty.
JOINT_PAIRS = [
    ("hip flexion", "hip_flexion_r", "hip_flexion_l"),
    ("hip adduction", "hip_adduction_r", "hip_adduction_l"),
    ("hip rotation", "hip_rotation_r", "hip_rotation_l"),
    ("knee angle", "knee_angle_r", "knee_angle_l"),
    ("ankle angle", "ankle_angle_r", "ankle_angle_l"),
    ("subtalar angle (locked)", "subtalar_angle_r", "subtalar_angle_l"),
    ("MTP angle", "mtp_angle_r", "mtp_angle_l"),
]


def reconstruct():
    """Load S001, calibrate scales/offsets once, run per-frame IK over a walk window.

    Returns a dict with poses, group_scales, per-frame + per-marker marker RMS,
    measured per-foot GRF on the window, the motion clip, spec, and time axis.
    """
    session = load_session(str(TRIAL_C3D), speedchange_path=str(SPEEDCHANGE))
    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    mm = s001_marker_map()
    # collapse the thigh/shank soft-tissue tracking clusters to one centroid each so the
    # noisy plate markers stop dragging the pose (they carry the largest residual).
    mm, centroids = collapse_clusters(spec, mm)
    print(f"  collapsed clusters -> centroids: {centroids}")
    # enrich the sparse stock foot marker set from the static trial (adds the
    # calcaneus cluster / met-1 / hallux markers, re-seats TOE, unlocks the MTP, and
    # re-zeros the ankle at the standing neutral) before building the skeleton.
    pl = place_foot_markers(
        spec, static, mapping=mm,
        marker_config=MarkerFitConfig(outer_iters=6), device=DEVICE,
        frame_range=(0, min(60, int(np.asarray(static.markers).shape[0]))),
    )
    print(f"  enriched foot markers: +{pl.added}  reseated={pl.reseated}  "
          f"unlocked={pl.unlocked}  ankle_neutral(deg)="
          f"{ {k: round(v * DEG, 2) for k, v in pl.ankle_neutral.items()} }")
    skel = WarpSkeleton(spec, device=DEVICE)
    model_names = skel.marker_names()
    obs_all, present = observations_from_session(session, model_names, mm)

    # contiguous, best-visibility walk window
    plo, phi = session.phase_window("walk")
    sub = obs_all[plo:phi]
    s = plo + pick_visible_window(sub, present, WIN_LEN)
    lo, hi = s, min(s + WIN_LEN, phi)
    obs = obs_all[lo:hi]
    F = obs.shape[0]

    # 1) calibrate {scales, offsets} on a mid-window slice (bounded config for speed)
    c0 = lo + max(0, (F - CALIB_LEN) // 2)
    calib_win = (c0, c0 + min(CALIB_LEN, F))
    print(f"  calibrating scales/offsets on frames {calib_win} ...")
    cfg = MarkerFitConfig(
        outer_iters=15,
        inner=MarkerIKConfig(max_iters=50),
        inner_first=MarkerIKConfig(max_iters=150),
    )
    fit, _, _ = reconstruct_window(
        session, spec, calib_win, mapping=mm, marker_config=cfg, device=DEVICE
    )
    scales = fit.group_scales

    # 2) per-frame IK over the whole window with fixed scales + offsets
    print(f"  per-frame IK over frames {(lo, hi)} ...")
    skel.set_marker_offsets(fit.marker_offsets)
    q_seed = np.repeat(np.mean(fit.poses, axis=0)[None], F, axis=0)
    res = solve_marker_ik(
        skel, obs, q_seed, group_scales=scales, config=MarkerIKConfig(max_iters=80)
    )
    poses = res.q

    # per-marker RMS residual (mm) over visible frames
    skel.set_marker_offsets(fit.marker_offsets)
    _, mk = skel.forward(poses, scales)
    vis = np.isfinite(obs).all(axis=2)  # (F, M)
    d = np.linalg.norm(np.where(vis[..., None], mk - np.nan_to_num(obs), 0.0), axis=2)
    per_marker_rms = np.array([
        np.sqrt(np.mean(d[vis[:, m], m] ** 2)) if vis[:, m].any() else np.nan
        for m in range(d.shape[1])
    ])

    motion = build_motion(spec, poses, fps=session.point_rate, group_scales=scales)
    # Auto-detect which force plate is the right foot (S001's right foot is on the -x
    # plate; the default sign=+1 would swap R/L GRF). Robust to lab/capture convention.
    sign = detect_right_plate_x_sign(
        session, static, spec, motion, (lo, hi), group_scales=scales
    )
    print(f"  belt->foot: right_plate_x_sign={sign:+d}")
    belt = measured_belt_grf(session, sign)
    grf = {side: belt[side][0][lo:hi] for side in belt}
    t = np.arange(F) / session.point_rate
    rbp = np.asarray(motion.data["rigid_body_pos"])  # (F, B, 3) Z-up
    body_names = list(motion.body_names)

    return dict(
        spec=spec, poses=poses, scales=scales, marker_rms=res.marker_rms,
        per_marker_rms=per_marker_rms, marker_names=model_names,
        rigid_body_pos=rbp, body_names=body_names,
        grf=grf, t=t, window=(lo, hi), fps=session.point_rate,
        ankle_neutral=pl.ankle_neutral,
    )


def load_or_reconstruct(fresh: bool = False):
    """Reconstruct (slow) and cache to disk, or reload the cached arrays."""
    if CACHE.exists() and not fresh:
        print(f"Loading cached reconstruction from {CACHE.name}")
        z = np.load(CACHE, allow_pickle=True)
        if "spec_pickle" in z:
            spec = z["spec_pickle"].item()
        else:
            spec = parse_osim(
                str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim")
            )
        grf = {}
        if "grf_R" in z:
            grf["R"] = z["grf_R"]
        if "grf_L" in z:
            grf["L"] = z["grf_L"]
        return dict(
            spec=spec, poses=z["poses"], scales=z["scales"],
            marker_rms=z["marker_rms"], per_marker_rms=z["per_marker_rms"],
            marker_names=list(z["marker_names"]), rigid_body_pos=z["rigid_body_pos"],
            body_names=list(z["body_names"]), grf=grf, t=z["t"],
            window=tuple(z["window"]), fps=float(z["fps"]),
        )
    R = reconstruct()
    save = dict(
        poses=R["poses"], scales=R["scales"], marker_rms=R["marker_rms"],
        per_marker_rms=R["per_marker_rms"], marker_names=np.array(R["marker_names"]),
        rigid_body_pos=R["rigid_body_pos"], body_names=np.array(R["body_names"]),
        t=R["t"], window=np.array(R["window"]), fps=R["fps"],
        spec_pickle=np.array(R["spec"], dtype=object),
        ankle_neutral=np.array(R["ankle_neutral"], dtype=object),
    )
    for side in R["grf"]:
        save[f"grf_{side}"] = R["grf"][side]
    np.savez(CACHE, **save)
    return R


def _shade_stance(ax, t, fz, color, thr=50.0):
    """Shade the frames where the measured belt vertical GRF marks stance."""
    stance = fz > thr
    in_seg = False
    x0 = 0.0
    for i, s_ in enumerate(stance):
        if s_ and not in_seg:
            in_seg, x0 = True, t[i]
        elif not s_ and in_seg:
            in_seg = False
            ax.axvspan(x0, t[i], color=color, alpha=0.10, lw=0)
    if in_seg:
        ax.axvspan(x0, t[-1], color=color, alpha=0.10, lw=0)


# ---------------------------------------------------------------------------
# Figure 9: recovered lower-body joint angles + measured-GRF stance
# ---------------------------------------------------------------------------
def fig_gait_angles(R):
    spec, poses, t = R["spec"], R["poses"], R["t"]
    idx = spec.dof_index_map()
    fzr = R["grf"]["R"][:, 2] if "R" in R["grf"] else None
    fzl = R["grf"]["L"][:, 2] if "L" in R["grf"] else None

    n = len(JOINT_PAIRS)
    ncols = 4
    nrows = int(np.ceil((n + 1) / ncols))  # +1 panel for the legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows),
                             sharex=True)
    axes = axes.ravel()
    for ax, (title, rn, ln) in zip(axes, JOINT_PAIRS):
        if fzr is not None:
            _shade_stance(ax, t, fzr, "tab:red")
        if fzl is not None:
            _shade_stance(ax, t, fzl, "tab:blue")
        ax.plot(t, poses[:, idx[rn]] * DEG, color="tab:red", lw=1.8, label="right")
        ax.plot(t, poses[:, idx[ln]] * DEG, color="tab:blue", lw=1.8, ls="--",
                label="left")
        ax.set_title(title)
        ax.set_ylabel("angle (deg)")
        ax.grid(alpha=0.3)
    # legend panel right after the last joint; hide any trailing empties
    legend_ax = axes[n]
    legend_ax.axis("off")
    handles = [
        plt.Line2D([], [], color="tab:red", lw=2, label="right joint angle"),
        plt.Line2D([], [], color="tab:blue", lw=2, ls="--", label="left joint angle"),
        plt.Rectangle((0, 0), 1, 1, color="tab:red", alpha=0.10, label="right foot stance (measured GRF)"),
        plt.Rectangle((0, 0), 1, 1, color="tab:blue", alpha=0.10, label="left foot stance (measured GRF)"),
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=11, frameon=False)
    for k in range(n + 1, len(axes)):
        axes[k].axis("off")
    # x-axis label on the lowest joint panel of each column (sharex hides the rest)
    for c in range(ncols):
        col = [i for i in range(n) if i % ncols == c]
        if col:
            axes[max(col)].set_xlabel("time (s)")
    fig.suptitle("S001 treadmill walk: IK-reconstructed lower-body joint angles\n"
                 "(full enriched DOF set incl. MTP; subtalar locked; phase-locked to "
                 "the independently measured GRF)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "09_s001_gait_angles.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: marker-fit residual (per frame + per marker)
# ---------------------------------------------------------------------------
def fig_marker_fit(R):
    t = R["t"]
    rms = R["marker_rms"] * 1e3  # mm
    pm = R["per_marker_rms"] * 1e3
    names = R["marker_names"]
    ok = np.isfinite(pm)
    order = np.argsort(pm[ok])
    pm_s = pm[ok][order]
    nm_s = [names[i] for i in np.where(ok)[0][order]]

    # colour each bar by anatomical region, calling out the newly-added foot markers.
    body_of = {m.name: m.body for m in R["spec"].markers}
    foot_bodies = {"calcn_r", "calcn_l", "toes_r", "toes_l"}
    lower_bodies = foot_bodies | {
        "pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l",
        "talus_r", "talus_l", "patella_r", "patella_l",
    }
    new_foot = {
        "RCAL2", "RCAL3", "RMT1", "RTOE_TIP",
        "LCAL2", "LCAL3", "LMT1", "LTOE_TIP",
    }

    def _bar_color(name):
        if name in new_foot:
            return "crimson"
        b = body_of.get(name)
        if b in foot_bodies:
            return "darkorange"
        if b in lower_bodies:
            return "teal"
        return "0.6"

    colors = [_bar_color(n) for n in nm_s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [1, 2.2]})
    ax1.plot(t, rms, color="tab:purple", lw=1.5)
    ax1.axhline(np.median(rms), color="k", ls="--", lw=1,
                label=f"median {np.median(rms):.1f} mm")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("marker RMS (mm)")
    ax1.set_ylim(0, max(30.0, rms.max() * 1.1))
    ax1.set_title("Per-frame marker reprojection RMS")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    x = np.arange(len(pm_s))
    ax2.bar(x, pm_s, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nm_s, rotation=90, fontsize=6)
    for lbl, nm in zip(ax2.get_xticklabels(), nm_s):
        if nm in new_foot:
            lbl.set_color("crimson")
            lbl.set_fontweight("bold")
    ax2.set_ylabel("per-marker RMS (mm)")
    ax2.set_title("Per-marker reprojection RMS (sorted) — "
                  "residual is dominated by PiG↔Rajagopal marker-set offsets")
    ax2.grid(alpha=0.3, axis="y")
    region_handles = [
        plt.Rectangle((0, 0), 1, 1, color="crimson", label="new foot marker (added)"),
        plt.Rectangle((0, 0), 1, 1, color="darkorange", label="foot marker (re-seated)"),
        plt.Rectangle((0, 0), 1, 1, color="teal", label="other lower body"),
        plt.Rectangle((0, 0), 1, 1, color="0.6", label="upper body"),
    ]
    ax2.legend(handles=region_handles, loc="upper left", fontsize=9, frameon=True)

    fig.suptitle("S001 marker-fit quality (real capture)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "10_s001_marker_fit.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11: reconstruction filmstrip over a stride
# ---------------------------------------------------------------------------
def fig_filmstrip(R):
    spec = R["spec"]
    pos = np.asarray(R["rigid_body_pos"])  # (F, B, 3) Z-up
    body_idx = {name: i for i, name in enumerate(R["body_names"])}
    bones = [(j.parent_body, j.child_body) for j in spec.joints if j.parent_body]
    F = pos.shape[0]
    n_snap = 6
    snaps = np.linspace(0, F - 1, n_snap).astype(int)

    # lateral view (x forward, z up); slide each snapshot along +x for a filmstrip
    dx = 0.55
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for k, f in enumerate(snaps):
        off = k * dx
        col = plt.cm.viridis(k / (n_snap - 1))
        for p, c in bones:
            a, b = pos[f, body_idx[p]], pos[f, body_idx[c]]
            ax.plot([a[0] + off, b[0] + off], [a[2], b[2]], "-", color=col, lw=2)
        ax.scatter(pos[f, :, 0] + off, pos[f, :, 2], s=10, color=col, zorder=3)
        ax.text(off, pos[f, :, 2].max() + 0.05, f"t={R['t'][f]:.2f}s",
                ha="center", fontsize=9, color=col)
    zmin = pos[:, :, 2].min()
    ax.axhline(zmin, color="saddlebrown", lw=2)
    ax.set_aspect("equal")
    ax.set_xlabel("filmstrip offset (m) — subject walks in place on the treadmill")
    ax.set_ylabel("z (m, up)")
    ax.set_yticks(np.round(np.arange(np.floor(zmin), pos[:, :, 2].max() + 0.2, 0.4), 1))
    ax.set_title("S001 reconstructed skeleton across a stride (side view, Z-up)")
    fig.tight_layout()
    fig.savefig(OUT / "11_s001_filmstrip.png", dpi=130)
    plt.close(fig)


def main():
    fresh = "--fresh" in sys.argv
    print("Reconstructing S001 walk window (slow, cached after first run) ...")
    t0 = time.perf_counter()
    R = load_or_reconstruct(fresh=fresh)
    elapsed = time.perf_counter() - t0
    R["elapsed_s"] = elapsed
    print(f"  window {R['window']}, marker RMS median "
          f"{np.nanmedian(R['marker_rms']) * 1e3:.1f} mm, "
          f"scales [{R['scales'].min():.3f}, {R['scales'].max():.3f}], "
          f"elapsed {elapsed:.1f} s")
    print("Figure 9: recovered gait angles ...")
    fig_gait_angles(R)
    print("Figure 10: marker-fit residual ...")
    fig_marker_fit(R)
    print("Figure 11: reconstruction filmstrip ...")
    fig_filmstrip(R)
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
