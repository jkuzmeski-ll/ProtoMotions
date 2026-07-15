# SPDX-License-Identifier: MIT

"""Tests for the full-skeleton distributed-contact forward sim (research payload).

Validates ``biomech.contact.tracking`` -- the M3-exported skeleton MJCF stepped by the
Newton MuJoCo solver with the Warp/NumPy distributed foot contact applied to
``calcn_r``/``calcn_l`` each step:

- **Frame consistency**: setting the free-root qpos to the Z-up pelvis pose makes MuJoCo's
  body world poses equal ``export.motion.build_motion``'s Z-up poses (the same poses the
  contact pipeline drives the sole with).
- **Body-weight invariant**: a frozen standing drop (all non-root joints locked by
  ``<equality>``, only the root vertical DOF free) settles to a static equilibrium where
  the summed two-foot vertical GRF equals the model's total weight, with the COP under
  each foot. This is independent of reconstruction quality -- it is pure force balance.
- The **hydroelastic** law reaches the same weight equilibrium.
- The **Warp** contact backend reproduces the NumPy result (skipped without CUDA).

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.tests import SkipTest

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_SPEC = None


def _spec():
    global _SPEC
    if _SPEC is None:
        from biomech.osim import parse_osim
        _SPEC = parse_osim(str(_OSIM))
    return _SPEC


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"mujoco not available: {exc}")


def _require_warp_cuda():
    try:
        import warp as wp
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"warp not available: {exc}")
    if not wp.is_cuda_available():
        raise SkipTest("no CUDA device available for warp")


def _synthetic_soles(spec, group_scales=None, nx=10, ny=5):
    from biomech.contact.foot_geometry import (
        FootDimensions,
        build_subject_sole,
        calcn_anchors_from_spec,
    )

    soles = {}
    for side in ("R", "L"):
        anchors = calcn_anchors_from_spec(spec, side, group_scales)
        dims = FootDimensions(
            side=side, heel_width=0.05, forefoot_width=0.09,
            foot_length=0.26, heel_to_ball=0.18, toe_length=0.05,
        )
        soles[side] = build_subject_sole(dims, anchors, nx=nx, ny=ny)
    return soles


# ---------------------------------------------------------------------------
# frame consistency
# ---------------------------------------------------------------------------


def test_zup_pose_matches_build_motion():
    """MuJoCo body poses at the Z-up qpos equal build_motion's Z-up per-body poses."""
    _require_mujoco()
    import mujoco

    from biomech.contact.tracking import mjcf_qpos_zup
    from biomech.contact.kinematics import foot_trajectory_from_motion
    from biomech.export.mjcf import export_mjcf
    from biomech.export.motion import build_motion

    spec = _spec()
    q = np.zeros(spec.num_dofs)
    xml = export_mjcf(spec, coupled_knee="coupled").xml
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = mjcf_qpos_zup(spec, q)
    mujoco.mj_forward(m, d)

    motion = build_motion(spec, q[None], fps=100.0)
    for body in ("calcn_r", "calcn_l", "pelvis", "tibia_r"):
        pos, _, _, _ = foot_trajectory_from_motion(motion, body)
        bid = m.body(body).id
        err = float(np.abs(np.asarray(d.xpos[bid]) - pos[0]).max())
        assert err < 1e-6, (body, err)


def test_zup_pose_upright():
    """The Z-up standing pose puts the pelvis above the feet (feet at the bottom)."""
    _require_mujoco()
    import mujoco

    from biomech.contact.tracking import mjcf_qpos_zup
    from biomech.export.mjcf import export_mjcf

    spec = _spec()
    xml = export_mjcf(spec).xml
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = mjcf_qpos_zup(spec, np.zeros(spec.num_dofs))
    mujoco.mj_forward(m, d)
    z_pelvis = d.xpos[m.body("pelvis").id][2]
    z_foot = d.xpos[m.body("calcn_r").id][2]
    assert z_pelvis > z_foot + 0.5  # pelvis ~1 m above the foot


# ---------------------------------------------------------------------------
# body-weight invariant (the payload)
# ---------------------------------------------------------------------------


def _standing_drop(law, params, backend, n_steps=3000):
    from biomech.contact.tracking import build_skeleton_tracking_sim

    spec = _spec()
    q = np.zeros(spec.num_dofs)
    soles = _synthetic_soles(spec)
    A = sum(s.total_area for s in soles.values())
    # start with initial penetration to shorten the settling transient
    weight = 737.0
    k = params["R"].k_bed
    init_pen = 2.0 * weight / (k * A)
    sim, qpos = build_skeleton_tracking_sim(
        spec, q, soles, params, law=law, backend=backend,
        ground_gap=-init_pen, freeze=True,
    )
    res = sim.settle(qpos, n_steps=n_steps, dt=5.0e-4, hold_mode="servo")
    w = sim.total_mass * sim.gravity
    return sim, res, w


def test_frozen_standing_drop_equals_body_weight():
    _require_mujoco()
    from biomech.contact.elastic_foundation import ElasticFoundationParams

    params = {
        s: ElasticFoundationParams(k_bed=1.0e6, c_bed=3.0e5, mu=0.9, v_eps=1e-3)
        for s in ("R", "L")
    }
    sim, res, w = _standing_drop("elastic", params, "numpy")
    fz = res.total_vertical_grf
    # time-averaged over the settled tail (small residual sway is averaged out)
    mean_ratio = float(np.mean(fz[-400:])) / w
    assert 0.95 < mean_ratio < 1.05, mean_ratio
    # both feet load, symmetric, COP under each foot
    fr = res.grf["calcn_r"][-400:, 2].mean()
    fl = res.grf["calcn_l"][-400:, 2].mean()
    assert fr > 0.2 * w and fl > 0.2 * w
    assert abs(fr - fl) < 0.15 * w  # symmetric stance
    cop_r = np.nanmean(res.cop["calcn_r"][-400:], axis=0)
    bid = sim._body_id["calcn_r"]
    foot_x = float(sim.data.xpos[bid][0])
    assert abs(cop_r[0] - foot_x) < 0.2  # COP near the foot in x


def test_hydroelastic_standing_drop_equals_body_weight():
    _require_mujoco()
    from biomech.contact.hydroelastic import HydroelasticParams

    params = {
        s: HydroelasticParams(
            k_bed=1.0e6, stiffen_b=0.0, hc_alpha=200.0,
            mu_d=0.9, mu_s=0.9, v_stribeck=0.05, v_eps=1e-3,
        )
        for s in ("R", "L")
    }
    _sim, res, w = _standing_drop("hydroelastic", params, "numpy", n_steps=3000)
    mean_ratio = float(np.mean(res.total_vertical_grf[-400:])) / w
    assert 0.95 < mean_ratio < 1.05, mean_ratio


def test_warp_backend_matches_numpy_standing_drop():
    _require_mujoco()
    _require_warp_cuda()
    from biomech.contact.elastic_foundation import ElasticFoundationParams

    def params():
        return {
            s: ElasticFoundationParams(k_bed=1.0e6, c_bed=3.0e5, mu=0.9, v_eps=1e-3)
            for s in ("R", "L")
        }

    _s1, res_np, w = _standing_drop("elastic", params(), "numpy", n_steps=1500)
    _s2, res_wp, _ = _standing_drop("elastic", params(), "warp", n_steps=1500)
    r_np = float(np.mean(res_np.total_vertical_grf[-200:])) / w
    r_wp = float(np.mean(res_wp.total_vertical_grf[-200:])) / w
    assert abs(r_np - r_wp) < 0.02, (r_np, r_wp)
