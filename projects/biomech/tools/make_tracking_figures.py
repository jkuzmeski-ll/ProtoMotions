# SPDX-License-Identifier: MIT
#
# Generate correctness figures for the full-skeleton distributed-contact forward sim
# (biomech.contact.tracking). Runs the frozen standing drop on the Newton MuJoCo solver
# and plots: the body-weight invariant convergence, the Z-up skeleton standing on the
# ground, the plantar pressure distribution + COP, backend/law parity, and the Y-up->Z-up
# frame-consistency check vs build_motion.
#
# Usage (from the repo root):
#   .venv/Scripts/python projects/biomech/tools/make_tracking_figures.py
# Figures are written to projects/biomech/docs/figures/.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from biomech.contact.elastic_foundation import ElasticFoundationParams  # noqa: E402
from biomech.contact.foot_geometry import (  # noqa: E402
    FootDimensions,
    build_subject_sole,
    calcn_anchors_from_spec,
)
from biomech.contact.forward_sim import _point_forces  # noqa: E402
from biomech.contact.hydroelastic import HydroelasticParams  # noqa: E402
from biomech.contact.kinematics import foot_trajectory_from_motion  # noqa: E402
from biomech.contact.tracking import build_skeleton_tracking_sim  # noqa: E402
from biomech.export.mjcf import export_mjcf  # noqa: E402
from biomech.export.motion import build_motion  # noqa: E402
from biomech.osim import parse_osim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def make_soles(spec, nx=14, ny=6):
    soles = {}
    for side in ("R", "L"):
        anchors = calcn_anchors_from_spec(spec, side)
        dims = FootDimensions(
            side=side, heel_width=0.05, forefoot_width=0.09,
            foot_length=0.26, heel_to_ball=0.18, toe_length=0.05,
        )
        soles[side] = build_subject_sole(dims, anchors, nx=nx, ny=ny)
    return soles


def build_drop(spec, soles, law="elastic", backend="numpy", k=1.0e6, c=3.0e5):
    A = sum(s.total_area for s in soles.values())
    init_pen = 2.0 * 737.0 / (k * A)
    if law == "elastic":
        params = {s: ElasticFoundationParams(k_bed=k, c_bed=c, mu=0.9, v_eps=1e-3)
                  for s in ("R", "L")}
    else:
        params = {s: HydroelasticParams(k_bed=k, stiffen_b=0.0, hc_alpha=200.0,
                                        mu_d=0.9, mu_s=0.9, v_stribeck=0.05, v_eps=1e-3)
                  for s in ("R", "L")}
    sim, qpos = build_skeleton_tracking_sim(
        spec, np.zeros(spec.num_dofs), soles, params, law=law, backend=backend,
        ground_gap=-init_pen, freeze=True,
    )
    return sim, qpos, params


# ---------------------------------------------------------------------------
# Figure 1: body-weight invariant convergence
# ---------------------------------------------------------------------------
def fig_convergence(spec, soles):
    sim, qpos, _ = build_drop(spec, soles)
    res = sim.settle(qpos, n_steps=3000, dt=5.0e-4, hold_mode="servo")
    w = sim.total_mass * sim.gravity
    t = res.time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.axhspan(0.95, 1.05, color="tab:green", alpha=0.12, label="±5% band")
    ax1.axhline(1.0, color="tab:green", ls="--", lw=1)
    ax1.plot(t, res.total_vertical_grf / w, color="tab:blue", lw=1.5)
    mean_tail = float(np.mean(res.total_vertical_grf[-400:])) / w
    ax1.set_title(f"Body-weight invariant\n(tail mean = {mean_tail:.3f} × weight)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("total vertical GRF / body weight")
    ax1.set_ylim(0, 2.0)
    ax1.legend(loc="upper right")

    ax2.axhline(w / 2.0, color="k", ls=":", lw=1, label="½ body weight")
    ax2.plot(t, res.grf["calcn_r"][:, 2], color="tab:red", lw=1.3, label="right foot")
    ax2.plot(t, res.grf["calcn_l"][:, 2], color="tab:orange", lw=1.3, label="left foot")
    ax2.set_title("Per-foot vertical GRF")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("vertical GRF (N)")
    ax2.legend(loc="lower right")

    fig.suptitle("Frozen standing drop on the Newton MuJoCo solver "
                 "(full skeleton, distributed contact)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "01_body_weight_invariant.png", dpi=130)
    plt.close(fig)
    return sim, res, w


# ---------------------------------------------------------------------------
# Figure 2: Z-up skeleton standing on the ground
# ---------------------------------------------------------------------------
def fig_skeleton(spec, sim, soles):
    d = sim.data
    def xp(name):
        return np.asarray(d.xpos[sim._body_id[name]])
    bones = [(j.parent_body, j.child_body) for j in spec.joints if j.parent_body]
    gz = sim.ground_z

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, (ax_i, ax_j, title, xl) in zip(
        axes, [(0, 2, "Side view (x–z)", "x (m, fwd)"),
               (1, 2, "Front view (y–z)", "y (m, lat)")]):
        for p, c in bones:
            a, b = xp(p), xp(c)
            ax.plot([a[ax_i], b[ax_i]], [a[ax_j], b[ax_j]], "-", color="0.4", lw=2, zorder=1)
        pts = np.array([xp(bn.name) for bn in spec.bodies])
        ax.scatter(pts[:, ax_i], pts[:, ax_j], s=18, color="tab:blue", zorder=2)
        # sole patches in world
        for side, col in (("R", "tab:red"), ("L", "tab:orange")):
            fcm = next(f for f in sim.feet if f.body == f"calcn_{side.lower()}")
            xpos, quat, _, _, _ = sim._foot_state(sim._body_id[fcm.body])
            from biomech.contact.elastic_foundation import _quat_rotate_np
            pw = xpos[None] + _quat_rotate_np(
                np.broadcast_to(quat, (fcm.sole.n, 4)), fcm.sole.points)
            ax.scatter(pw[:, ax_i], pw[:, ax_j], s=4, color=col, alpha=0.5, zorder=3)
        ax.axhline(gz, color="saddlebrown", lw=2, label="ground plane")
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel("z (m, up)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    fig.suptitle("Reconstructed skeleton in Z-up world, standing on the registered ground\n"
                 "(free-root pose carries the Y-up→Z-up rotation; feet in contact)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "02_skeleton_zup.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: plantar pressure distribution + COP
# ---------------------------------------------------------------------------
def fig_pressure(spec, sim, res):
    from biomech.contact.elastic_foundation import _quat_rotate_np
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, side in zip(axes, ("R", "L")):
        fcm = next(f for f in sim.feet if f.body == f"calcn_{side.lower()}")
        bid = sim._body_id[fcm.body]
        xpos, quat, _, linvel, angvel = sim._foot_state(bid)
        pf, pw = _point_forces(fcm, xpos, quat, linvel, angvel, sim.ground_z)
        fn = pf[:, 2]
        press = fn / fcm.sole.areas / 1000.0  # kPa
        sc = ax.scatter(pw[:, 0], pw[:, 1], c=press, s=90, cmap="viridis",
                        marker="s", edgecolors="none")
        cop = np.nanmean(res.cop[fcm.body][-400:], axis=0)
        ax.plot(cop[0], cop[1], "*", ms=18, color="red", mec="k", zorder=5)
        ax.annotate("COP", (cop[0], cop[1]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="red", weight="bold")
        ax.set_title(f"{side} foot plantar pressure  (Fz = {fn.sum():.0f} N)")
        ax.set_xlabel("world x (m, fwd)")
        ax.set_ylabel("world y (m, lat)")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="pressure (kPa)")
    fig.suptitle("Distributed plantar pressure at static equilibrium (subject sole geometry)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "03_plantar_pressure.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: backend / law parity
# ---------------------------------------------------------------------------
def fig_parity(spec, soles):
    import warp as wp
    labels, ratios = [], []
    for law, backend, lab in [("elastic", "numpy", "elastic\n(NumPy)"),
                              ("hydroelastic", "numpy", "hydroelastic\n(NumPy)")]:
        sim, qpos, _ = build_drop(spec, soles, law=law, backend=backend)
        res = sim.settle(qpos, n_steps=3000, dt=5.0e-4, hold_mode="servo")
        w = sim.total_mass * sim.gravity
        labels.append(lab)
        ratios.append(float(np.mean(res.total_vertical_grf[-400:])) / w)
    if wp.is_cuda_available():
        sim, qpos, _ = build_drop(spec, soles, law="elastic", backend="warp")
        res = sim.settle(qpos, n_steps=3000, dt=5.0e-4, hold_mode="servo")
        w = sim.total_mass * sim.gravity
        labels.append("elastic\n(Warp/GPU)")
        ratios.append(float(np.mean(res.total_vertical_grf[-400:])) / w)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, ratios, color=["tab:blue", "tab:cyan", "tab:green"][:len(labels)])
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.axhspan(0.95, 1.05, color="tab:green", alpha=0.12)
    ax.set_ylim(0.8, 1.2)
    ax.set_ylabel("settled GRF / body weight")
    ax.set_title("Contact law + backend parity (all recover body weight)")
    for b, r in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.01, f"{r:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "04_backend_law_parity.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Y-up -> Z-up frame consistency vs build_motion
# ---------------------------------------------------------------------------
def fig_frame_consistency(spec):
    import mujoco
    from biomech.contact.tracking import mjcf_qpos_zup
    q = np.zeros(spec.num_dofs)
    xml = export_mjcf(spec, coupled_knee="coupled").xml
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = mjcf_qpos_zup(spec, q)
    mujoco.mj_forward(m, d)
    motion = build_motion(spec, q[None], fps=100.0)

    names, errs = [], []
    for bn in [b.name for b in spec.bodies]:
        pos, _, _, _ = foot_trajectory_from_motion(motion, bn)
        bid = m.body(bn).id
        names.append(bn)
        errs.append(max(float(np.abs(np.asarray(d.xpos[bid]) - pos[0]).max()), 1e-17))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(names)), errs, color="tab:purple")
    ax.set_yscale("log")
    ax.axhline(1e-6, color="tab:green", ls="--", lw=1, label="1 µm tolerance")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("max |MuJoCo xpos − build_motion| (m)")
    ax.set_title("Frame consistency: MuJoCo body poses vs gold-standard build_motion (Z-up)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_frame_consistency.png", dpi=130)
    plt.close(fig)


def main():
    spec = parse_osim(str(ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    soles = make_soles(spec)
    print("Figure 1: body-weight invariant convergence ...")
    sim, res, w = fig_convergence(spec, soles)
    print("Figure 2: Z-up skeleton ...")
    fig_skeleton(spec, sim, soles)
    print("Figure 3: plantar pressure ...")
    fig_pressure(spec, sim, res)
    print("Figure 4: backend/law parity ...")
    fig_parity(spec, soles)
    print("Figure 5: frame consistency ...")
    fig_frame_consistency(spec)
    print("Wrote figures to", OUT)


if __name__ == "__main__":
    main()
