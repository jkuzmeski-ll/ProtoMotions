# SPDX-License-Identifier: MIT

"""Tests for distributed foot contact inside a Newton MuJoCo forward simulation.

Validates ``biomech.contact.forward_sim.ContactForwardSim`` — the MuJoCo solver's
forward dynamics with the Warp/NumPy distributed-contact wrench applied each step as an
external force:

- a rigid body dropped onto the belt settles to the analytic elastic-foundation
  equilibrium (GRF == weight, penetration == weight/(k*A), COP at the sole centre),
- the hydroelastic law reduces to the same equilibrium in its linear limit,
- sliding produces a friction force opposing the motion, bounded by mu*Fn,
- the Warp backend reproduces the NumPy equilibrium (skipped without CUDA).

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

import numpy as np

from biomech.contact.elastic_foundation import (
    ElasticFoundationParams,
    sample_flat_sole,
)
from biomech.contact.forward_sim import (
    ContactForwardSim,
    FootContactModel,
    single_body_mjcf,
)
from biomech.tests import SkipTest

_G = 9.81
_MASS = 2.0


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


def _drop_sim(backend="numpy", law="elastic", k=1.0e6, c=5.0e4,
              hc_alpha=200.0, start_z=0.005, timestep=5e-4):
    sole = sample_flat_sole(0.25, 0.10, 10, 5)  # area 0.025 m^2, z=0 in body frame
    if law == "elastic":
        params = ElasticFoundationParams(k_bed=k, c_bed=c, mu=0.9, v_eps=1e-3)
    else:
        from biomech.contact.hydroelastic import HydroelasticParams
        # linear normal law (no stiffening); Hunt-Crossley dissipation only settles the
        # transient -- it vanishes at rest (vn=0), so the static equilibrium is the same
        # Winkler d* = weight/(k*A).
        params = HydroelasticParams(k_bed=k, stiffen_b=0.0, hc_alpha=hc_alpha,
                                    mu_d=0.9, mu_s=0.9, v_stribeck=0.05, v_eps=1e-3)
    xml = single_body_mjcf(mass=_MASS, start_pos=(0.0, 0.0, start_z),
                           gravity=_G, timestep=timestep)
    fcm = FootContactModel(body="foot", sole=sole, params=params, law=law,
                           backend=backend)
    sim = ContactForwardSim(xml, [fcm], ground_z=0.0)
    return sim, sole, k


def test_single_body_mjcf_builds():
    _require_mujoco()
    import mujoco
    xml = single_body_mjcf(mass=1.5)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nq == 7  # free joint
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")
    assert bid >= 0
    assert abs(m.body_mass[bid] - 1.5) < 1e-9


def test_drop_settles_to_elastic_equilibrium():
    _require_mujoco()
    sim, sole, k = _drop_sim(law="elastic")
    res = sim.run(3000, dt=5e-4)
    A = sole.total_area
    d_star = _MASS * _G / (k * A)
    z = res.qpos[:, 2]
    grf_z = res.grf["foot"][:, 2]
    # GRF balances weight at equilibrium
    assert abs(grf_z[-1] - _MASS * _G) < 1e-2
    # body settles at the analytic penetration depth (below ground z=0)
    assert abs(z[-1] + d_star) < 5e-5
    # settled (near-zero residual vertical velocity)
    vel_tail = np.abs(np.diff(z[-50:])).max() / 5e-4
    assert vel_tail < 1e-3
    # COP at the (symmetric) sole centre
    cop = res.cop["foot"][-1]
    assert abs(cop[0]) < 1e-4 and abs(cop[1]) < 1e-4


def test_hydroelastic_linear_limit_matches_elastic_drop():
    _require_mujoco()
    sim, sole, k = _drop_sim(law="hydroelastic", hc_alpha=200.0,
                             start_z=0.002, timestep=2e-4)
    res = sim.run(6000, dt=2e-4)
    A = sole.total_area
    d_star = _MASS * _G / (k * A)
    assert abs(res.grf["foot"][-1, 2] - _MASS * _G) < 1e-2
    assert abs(res.qpos[-1, 2] + d_star) < 5e-5


def test_sliding_friction_opposes_motion_and_is_bounded():
    _require_mujoco()
    mu = 0.5
    sole = sample_flat_sole(0.25, 0.10, 10, 5)
    k = 1.0e6
    A = sole.total_area
    d_star = _MASS * _G / (k * A)
    params = ElasticFoundationParams(k_bed=k, c_bed=0.0, mu=mu, v_eps=1e-4)
    xml = single_body_mjcf(mass=_MASS, start_pos=(0.0, 0.0, 0.0),
                           gravity=_G, timestep=1e-4)
    fcm = FootContactModel(body="foot", sole=sole, params=params, law="elastic",
                           backend="numpy")
    sim = ContactForwardSim(xml, [fcm], ground_z=0.0)
    # place at equilibrium penetration, sliding in +x well above v_eps
    sim.data.qpos[:] = 0.0
    sim.data.qpos[2] = -d_star
    sim.data.qpos[3] = 1.0  # quat w (wxyz)
    sim.data.qvel[:] = 0.0
    sim.data.qvel[0] = 0.5
    sim.forward()          # refresh xpos/velocities for the manual state
    sim.apply_contacts()   # compute the contact wrench from that state
    gx = sim.last_grf["foot"][0]
    gz = sim.last_grf["foot"][2]
    assert gz > 0.0
    assert gx < 0.0  # friction opposes +x motion
    # bounded by the Coulomb cone (near-saturated at v >> v_eps)
    assert abs(gx) <= mu * gz * 1.02
    assert abs(gx) > 0.9 * mu * gz  # essentially saturated


def test_warp_backend_matches_numpy_equilibrium():
    _require_mujoco()
    _require_warp_cuda()
    sim_np, sole, k = _drop_sim(backend="numpy", law="elastic")
    res_np = sim_np.run(2000, dt=5e-4)
    sim_wp, _, _ = _drop_sim(backend="warp", law="elastic")
    res_wp = sim_wp.run(2000, dt=5e-4)
    # same settled height + GRF to float32 tolerance
    assert abs(res_np.qpos[-1, 2] - res_wp.qpos[-1, 2]) < 1e-4
    assert abs(res_np.grf["foot"][-1, 2] - res_wp.grf["foot"][-1, 2]) < 1e-1
