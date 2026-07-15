# SPDX-License-Identifier: MIT

"""M2e DynamicsFitter tests — GRF residual + mass identification on the MuJoCo solver.

Validates the ``ResidualForceHelper`` port (``biomech.fitting.dynamics_fitter``):

- The inverse-dynamics engine (``mj_inverse``) is self-consistent: applying the computed
  ``qfrc_inverse`` in forward dynamics reproduces the prescribed ``qacc``.
- The GRF/COP -> generalized-force mapping (``mj_applyFT``) gives the expected
  free-joint root force for a world wrench.
- The root residual is **exactly linear** in per-segment mass (Nimble's linear inertial
  identification), so the finite-difference regressor is exact.
- Synthetic round-trip: with contacts that make the residual zero at the *true* masses,
  the linear identification recovers those masses / nulls the residual from a wrong guess.
- The free-joint-aware kinematics (``mj_differentiatePos``) recover a known velocity.
- The GRF adapter drops swing (NaN COP) frames and builds the correct wrench.
- The high-level ``DynamicsFitter`` runs end-to-end and reduces the residual.

Requires ``mujoco`` (pinned 3.5.0); tests skip if absent. Frame is the model's native
OpenSim (Y-up, meters) frame; gravity is MuJoCo's default -Z (irrelevant to the linear
identities being checked).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.osim import parse_osim
from biomech.tests import SkipTest

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_SPEC = None


def _spec():
    global _SPEC
    if _SPEC is None:
        _SPEC = parse_osim(str(_OSIM))
    return _SPEC


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"mujoco not available: {exc}")
    return __import__("mujoco")


def _require_mujoco_warp():
    _require_mujoco()
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"mujoco_warp/warp not available: {exc}")


def _helper():
    from biomech.export.mjcf import export_mjcf
    from biomech.fitting.dynamics_fitter import ResidualHelper

    res = export_mjcf(_spec(), coupled_knee="coupled")
    return ResidualHelper(res.xml)


def _random_state(helper, rng):
    m = helper.model
    qpos = np.array(helper.data.qpos, dtype=np.float64)
    qpos += 0.05 * rng.standard_normal(m.nq)
    qpos[3:7] /= np.linalg.norm(qpos[3:7])  # renormalize free-joint quat
    qvel = 0.1 * rng.standard_normal(m.nv)
    qacc = 0.2 * rng.standard_normal(m.nv)
    return qpos, qvel, qacc


# ---------------------------------------------------------------------------
# low-level engine
# ---------------------------------------------------------------------------


def test_inverse_dynamics_roundtrip():
    mj = _require_mujoco()
    h = _helper()
    rng = np.random.default_rng(0)
    qpos, qvel, qacc = _random_state(h, rng)
    qfrc = h.inverse_dynamics(qpos, qvel, qacc)
    # forward dynamics with that force must reproduce qacc
    d = h.data
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.qfrc_applied[:] = qfrc
    mj.mj_forward(h.model, d)
    d.qfrc_applied[:] = 0.0
    assert np.abs(d.qacc - qacc).max() < 1e-8


def test_free_joint_force_mapping():
    _require_mujoco()
    h = _helper()
    rng = np.random.default_rng(1)
    qpos, qvel, qacc = _random_state(h, rng)
    from biomech.fitting.dynamics_fitter import Contact

    bid = h.body_id("calcn_r")
    d = h.data
    d.qpos[:] = qpos
    h._mj.mj_kinematics(h.model, d)
    h._mj.mj_comPos(h.model, d)
    force = np.array([10.0, -20.0, 500.0])
    point = d.xpos[bid].copy()
    c = Contact(body="calcn_r", force=force, torque=np.zeros(3), point=point)
    Fs = h.contact_generalized_force([c])
    # free-joint translational generalized force == world force
    assert np.allclose(Fs[:3], force, atol=1e-9)


def test_residual_is_id_minus_contact():
    _require_mujoco()
    h = _helper()
    rng = np.random.default_rng(2)
    qpos, qvel, qacc = _random_state(h, rng)
    from biomech.fitting.dynamics_fitter import Contact

    bid = h.body_id("calcn_l")
    d = h.data
    d.qpos[:] = qpos
    h._mj.mj_kinematics(h.model, d)
    h._mj.mj_comPos(h.model, d)
    c = Contact(
        body="calcn_l",
        force=np.array([5.0, 3.0, 400.0]),
        torque=np.array([0.0, 0.0, 1.5]),
        point=d.xpos[bid].copy(),
    )
    qfrc = h.inverse_dynamics(qpos, qvel, qacc)
    Fs = h.contact_generalized_force([c])
    r = h.root_residual(qpos, qvel, qacc, [c])
    assert np.allclose(r, qfrc[:6] - Fs[:6], atol=1e-9)


# ---------------------------------------------------------------------------
# linearity in mass + regressor
# ---------------------------------------------------------------------------


def test_root_residual_linear_in_mass():
    _require_mujoco()
    h = _helper()
    rng = np.random.default_rng(3)
    qpos, qvel, qacc = _random_state(h, rng)

    def r():
        return h.root_residual(qpos, qvel, qacc, [])

    base = r()
    m0 = h.get_masses(["femur_r"])
    h.set_masses(["femur_r"], m0 + 1.0)
    d1 = r() - base
    h.set_masses(["femur_r"], m0 + 2.0)
    d2 = r() - base
    h.set_masses(["femur_r"], m0)  # restore
    assert np.abs(d2 - 2.0 * d1).max() < 1e-9


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------


def test_velocities_from_positions_translation():
    _require_mujoco()
    h = _helper()
    from biomech.fitting.dynamics_fitter import velocities_from_positions

    dt = 0.01
    v_world = np.array([0.3, -0.1, 0.2])
    q0 = np.array(h.data.qpos, dtype=np.float64)
    q0[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    F = 5
    qpos_t = np.tile(q0, (F, 1))
    for f in range(F):
        qpos_t[f, :3] = q0[:3] + v_world * (f * dt)
    qvel = velocities_from_positions(h, qpos_t, dt)
    # translational DOFs of the free joint == constant world velocity
    assert np.allclose(qvel[1:-1, :3], v_world, atol=1e-9)
    assert np.allclose(qvel[:, 3:], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# GRF adapter
# ---------------------------------------------------------------------------


def test_contacts_from_grf_drops_swing():
    from biomech.fitting.dynamics_fitter import contacts_from_grf

    force = np.array(
        [[0.0, 0.0, 0.0], [10.0, 5.0, 600.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    cop = np.array(
        [[np.nan, np.nan, np.nan], [0.1, 0.2, 0.0], [np.nan, np.nan, np.nan]]
    )
    fmz = np.array([np.nan, 2.0, np.nan])
    contacts = contacts_from_grf(force, cop, "calcn_r", free_moment_z=fmz)
    assert contacts[0] == [] and contacts[2] == []
    assert len(contacts[1]) == 1
    c = contacts[1][0]
    assert c.body == "calcn_r"
    assert np.allclose(c.force, [10.0, 5.0, 600.0])
    assert np.allclose(c.torque, [0.0, 0.0, 2.0])
    assert np.allclose(c.point, [0.1, 0.2, 0.0])


def test_merge_and_resample():
    from biomech.fitting.dynamics_fitter import (
        Contact,
        merge_contacts,
        resample_to_frames,
    )

    a = [[Contact("calcn_r", np.zeros(3), np.zeros(3), np.zeros(3))], []]
    b = [[], [Contact("calcn_l", np.zeros(3), np.zeros(3), np.zeros(3))]]
    merged = merge_contacts(a, b)
    assert len(merged[0]) == 1 and len(merged[1]) == 1

    src_t = np.array([0.0, 1.0, 2.0])
    sig = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    dst_t = np.array([0.5, 1.5])
    out = resample_to_frames(sig, src_t, dst_t)
    assert np.allclose(out[:, 0], [5.0, 15.0])


# ---------------------------------------------------------------------------
# synthetic mass recovery (round-trip)
# ---------------------------------------------------------------------------


def _realize_root_wrench(h, qpos, target6):
    """A single root-body Contact whose Fs[:6] equals ``target6`` (calibrated 6x6)."""
    from biomech.fitting.dynamics_fitter import Contact

    root = _spec().bodies[0].name  # pelvis
    d = h.data
    d.qpos[:] = qpos
    h._mj.mj_kinematics(h.model, d)
    h._mj.mj_comPos(h.model, d)
    origin = np.zeros(3)
    K = np.zeros((6, 6))
    for i in range(6):
        Fs = np.zeros(h.nv)
        f = np.zeros(3)
        t = np.zeros(3)
        if i < 3:
            f[i] = 1.0
        else:
            t[i - 3] = 1.0
        h._mj.mj_applyFT(h.model, d, f, t, origin, h.body_id(root), Fs)
        K[:, i] = Fs[:6]
    w = np.linalg.solve(K, target6)
    return [Contact(body=root, force=w[:3], torque=w[3:], point=origin)]


def _smooth_trajectory(spec, rng, F=6):
    lo = np.array([c.limit_lo for j in spec.joints for c in j.coordinates])
    hi = np.array([c.limit_hi for j in spec.joints for c in j.coordinates])
    locked = np.array([c.locked for j in spec.joints for c in j.coordinates])
    lo = np.where(np.isfinite(lo), lo, -0.5)
    hi = np.where(np.isfinite(hi), hi, 0.5)
    a = rng.uniform(lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo))
    b = rng.uniform(lo + 0.3 * (hi - lo), hi - 0.3 * (hi - lo))
    a[locked] = 0.0
    b[locked] = 0.0
    q_t = np.zeros((F, len(a)))
    for f in range(F):
        s = 0.5 - 0.5 * np.cos(np.pi * f / (F - 1))  # smooth 0->1
        q_t[f] = (1 - s) * a + s * b
    return q_t


def test_mass_identification_recovers_masses():
    _require_mujoco()
    from biomech.fitting.dynamics_fitter import (
        DynamicsFitter,
        identify_masses,
    )

    spec = _spec()
    rng = np.random.default_rng(7)
    fitter = DynamicsFitter(spec, coupled_knee="coupled")
    h = fitter.helper
    body_names = fitter.body_names

    q_t = _smooth_trajectory(spec, rng, F=6)
    fps = 100.0
    qpos_t, qvel_t, qacc_t = fitter.kinematics(q_t, fps)
    F = qpos_t.shape[0]

    # ground-truth masses, and contacts that null the residual at those masses
    m0 = h.get_masses(body_names)
    m_true = m0 * rng.uniform(0.85, 1.15, size=m0.shape)
    h.set_masses(body_names, m_true)
    contacts_t = []
    for f in range(F):
        target = h.inverse_dynamics(qpos_t[f], qvel_t[f], qacc_t[f])[:6]
        contacts_t.append(_realize_root_wrench(h, qpos_t[f], target))

    # start from the wrong (initial) masses and identify
    h.set_masses(body_names, m0)
    result = identify_masses(
        h, qpos_t, qvel_t, qacc_t, contacts_t, body_names, reg=1e-6
    )

    before = result.residual_before
    after = result.residual_after
    assert after.mean_force_norm < 0.05 * before.mean_force_norm + 1e-6
    assert after.mean_torque_norm < 0.05 * before.mean_torque_norm + 1e-6
    # total mass recovered well (per-body may have nullspace)
    assert abs(np.sum(result.fitted_mass) - np.sum(m_true)) < 0.05 * np.sum(m_true)


def test_dynamics_fitter_end_to_end():
    _require_mujoco()
    from biomech.fitting.dynamics_fitter import DynamicsFitter, contacts_from_grf

    spec = _spec()
    rng = np.random.default_rng(11)
    fitter = DynamicsFitter(spec, coupled_knee="coupled")

    # a quasi-static pose (repeated frames -> zero velocity/acceleration) so the root
    # residual is the physical gravity-vs-GRF balance that mass identification resolves
    ndof = spec.num_dofs
    locked = np.array([c.locked for j in spec.joints for c in j.coordinates])
    q_pose = 0.03 * rng.standard_normal(ndof)
    q_pose[locked] = 0.0
    F = 5
    q_t = np.tile(q_pose, (F, 1))
    fps = 100.0
    qpos_t, qvel_t, qacc_t = fitter.kinematics(q_t, fps)

    # split vertical GRF slightly below body weight so the fit must reduce total mass
    h = fitter.helper
    cr = []
    cl = []
    for f in range(F):
        h.inverse_dynamics(qpos_t[f], qvel_t[f], qacc_t[f])  # populate kinematics
        cr.append(h.data.xpos[h.body_id("calcn_r")].copy())
        cl.append(h.data.xpos[h.body_id("calcn_l")].copy())
    total_weight = float(np.sum(h.get_masses(fitter.body_names))) * 9.81
    fr = np.tile([0.0, 0.0, 0.45 * total_weight], (F, 1))
    fl = np.tile([0.0, 0.0, 0.45 * total_weight], (F, 1))
    from biomech.fitting.dynamics_fitter import merge_contacts

    ct = merge_contacts(
        contacts_from_grf(fr, np.array(cr), "calcn_r"),
        contacts_from_grf(fl, np.array(cl), "calcn_l"),
    )
    res = fitter.fit_masses(q_t, fps, ct, reg=1e-3)
    before = res.mass_result.residual_before
    after = res.mass_result.residual_after
    # the solve minimizes the combined (force + torque) root-residual energy
    e_before = float(np.sum(before.force_norm**2 + before.torque_norm**2))
    e_after = float(np.sum(after.force_norm**2 + after.torque_norm**2))
    assert e_after <= e_before + 1e-6
    assert res.mass_result.fitted_mass.min() > 0.0
    assert len(res.body_names) == 20


# ---------------------------------------------------------------------------
# full inertial identification (mass + COM + inertia)
# ---------------------------------------------------------------------------


def test_inertial_param_roundtrip():
    _require_mujoco()
    h = _helper()
    names = [b.name for b in _spec().bodies if h.has_body(b.name)]
    phi = h.get_inertial_params(names)
    h.set_inertial_params(names, phi)
    assert np.abs(h.get_inertial_params(names) - phi).max() < 1e-9
    assert phi.shape == (20, 10)


def test_inertial_params_linear_in_residual():
    _require_mujoco()
    h = _helper()
    rng = np.random.default_rng(31)
    qpos, qvel, qacc = _random_state(h, rng)
    phi0 = h.get_inertial_params(["femur_r"])[0]

    def r():
        return h.root_residual(qpos, qvel, qacc, [])

    base = r()
    max_nl = 0.0
    for k in range(10):
        step = 1e-3 * max(abs(phi0[k]), 1e-3)
        p = phi0.copy()
        p[k] += step
        h.set_inertial_params(["femur_r"], p[None])
        d1 = r() - base
        p = phi0.copy()
        p[k] += 2 * step
        h.set_inertial_params(["femur_r"], p[None])
        d2 = r() - base
        max_nl = max(max_nl, np.abs(d2 - 2 * d1).max())
        h.set_inertial_params(["femur_r"], phi0[None])
    assert max_nl < 1e-9


def test_inertia_is_physical():
    from biomech.fitting.dynamics_fitter import inertia_is_physical

    # a solid sphere-ish body: valid
    good = np.array([2.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0])
    assert inertia_is_physical(good)
    bad_mass = good.copy()
    bad_mass[0] = -1.0
    assert not inertia_is_physical(bad_mass)
    # violate the triangle inequality (one moment too large)
    bad_tri = np.array([2.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.1, 0.0, 0.0, 0.0])
    assert not inertia_is_physical(bad_tri)


def test_inertial_identification_recovers():
    _require_mujoco()
    from biomech.fitting.dynamics_fitter import (
        DynamicsFitter,
        identify_inertial_params,
        inertia_is_physical,
    )

    spec = _spec()
    rng = np.random.default_rng(37)
    fitter = DynamicsFitter(spec, coupled_knee="coupled")
    h = fitter.helper
    names = fitter.body_names
    q_t = _smooth_trajectory(spec, rng, F=6)
    fps = 100.0
    qpos_t, qvel_t, qacc_t = fitter.kinematics(q_t, fps)
    F = qpos_t.shape[0]

    phi0 = h.get_inertial_params(names)
    # perturbed *true* params (mass, COM, inertia), kept physically valid
    phi_true = phi0.copy()
    phi_true[:, 0] *= rng.uniform(0.9, 1.1, size=phi0.shape[0])  # mass
    phi_true[:, 1:4] += phi_true[:, 0:1] * 0.01 * rng.standard_normal((phi0.shape[0], 3))
    phi_true[:, 4:7] *= rng.uniform(0.95, 1.05, size=(phi0.shape[0], 3))
    for i in range(phi0.shape[0]):
        if not inertia_is_physical(phi_true[i]):
            phi_true[i] = phi0[i]

    # contacts that null the residual at the true params
    h.set_inertial_params(names, phi_true)
    contacts_t = []
    for f in range(F):
        target = h.inverse_dynamics(qpos_t[f], qvel_t[f], qacc_t[f])[:6]
        contacts_t.append(_realize_root_wrench(h, qpos_t[f], target))

    # reset to the initial params and identify
    h.set_inertial_params(names, phi0)
    result = identify_inertial_params(
        h, qpos_t, qvel_t, qacc_t, contacts_t, names, reg=1e-8
    )
    before = result.residual_before
    after = result.residual_after
    assert after.mean_force_norm < 0.1 * before.mean_force_norm + 1e-6
    assert after.mean_torque_norm < 0.5 * before.mean_torque_norm + 1e-6
    # every fitted body remains physically valid
    for i in range(len(names)):
        assert inertia_is_physical(result.fitted_params[i])


# ---------------------------------------------------------------------------
# GPU-batched residual (mujoco_warp / Newton MuJoCo)
# ---------------------------------------------------------------------------


def test_batched_residual_matches_cpu():
    _require_mujoco_warp()
    from biomech.export.mjcf import export_mjcf
    from biomech.fitting.dynamics_fitter import (
        BatchedResidualHelper,
        Contact,
        ResidualHelper,
    )

    xml = export_mjcf(_spec(), coupled_knee="coupled").xml
    cpu = ResidualHelper(xml)
    rng = np.random.default_rng(21)
    F = 4
    qpos = np.zeros((F, cpu.nq))
    qvel = np.zeros((F, cpu.nv))
    qacc = np.zeros((F, cpu.nv))
    for f in range(F):
        qp, qv, qa = _random_state(cpu, rng)
        qpos[f], qvel[f], qacc[f] = qp, qv, qa

    # a moving foot contact each frame at the current foot position
    bid = cpu.body_id("calcn_r")
    contacts_t = []
    for f in range(F):
        cpu.inverse_dynamics(qpos[f], qvel[f], qacc[f])
        p = cpu.data.xpos[bid].copy()
        contacts_t.append(
            [Contact("calcn_r", np.array([8.0, -4.0, 420.0]), np.array([0.0, 0.0, 1.0]), p)]
        )

    cpu_res = np.stack(
        [cpu.root_residual(qpos[f], qvel[f], qacc[f], contacts_t[f]) for f in range(F)]
    )
    gpu = BatchedResidualHelper(xml, F)
    gpu_res = gpu.root_residual_batch(qpos, qvel, qacc, contacts_t)
    # float32 GPU vs float64 CPU
    assert np.abs(gpu_res - cpu_res).max() < 1e-2


def test_batched_mass_identification_recovers_masses():
    _require_mujoco_warp()
    from biomech.fitting.dynamics_fitter import (
        DynamicsFitter,
        identify_masses_batched,
        BatchedResidualHelper,
    )

    spec = _spec()
    rng = np.random.default_rng(23)
    fitter = DynamicsFitter(spec, coupled_knee="coupled")
    q_t = _smooth_trajectory(spec, rng, F=6)
    fps = 100.0
    qpos_t, qvel_t, qacc_t = fitter.kinematics(q_t, fps)
    F = qpos_t.shape[0]
    body_names = fitter.body_names

    # contacts that null the residual at the *true* masses (built on the CPU helper)
    h = fitter.helper
    m0 = h.get_masses(body_names)
    m_true = m0 * rng.uniform(0.85, 1.15, size=m0.shape)
    h.set_masses(body_names, m_true)
    contacts_t = []
    for f in range(F):
        target = h.inverse_dynamics(qpos_t[f], qvel_t[f], qacc_t[f])[:6]
        contacts_t.append(_realize_root_wrench(h, qpos_t[f], target))
    h.set_masses(body_names, m0)  # reset CPU helper (unused hereafter)

    # GPU-batched identification starting from the wrong (initial) masses
    gpu = BatchedResidualHelper(fitter.export.xml, F)
    result = identify_masses_batched(
        gpu, qpos_t, qvel_t, qacc_t, contacts_t, body_names, reg=1e-6
    )
    before = result.residual_before
    after = result.residual_after
    assert after.mean_force_norm < 0.2 * before.mean_force_norm + 1e-3
    assert after.mean_torque_norm < 0.2 * before.mean_torque_norm + 1e-3


# ---------------------------------------------------------------------------
# Kinematic RRA (residual reduction by adjusting the root trajectory)
# ---------------------------------------------------------------------------


def test_rra_reduces_residual_from_perturbed_root():
    """Perturb the root translation off a residual-consistent trajectory; RRA both
    cuts the root residual and pulls the root trajectory back toward the true one."""
    _require_mujoco()
    from biomech.fitting.dynamics_fitter import DynamicsFitter

    spec = _spec()
    rng = np.random.default_rng(31)
    fitter = DynamicsFitter(spec, coupled_knee="coupled")
    h = fitter.helper

    q_t = _smooth_trajectory(spec, rng, F=8)
    fps = 100.0
    qpos_t, qvel_t, qacc_t = fitter.kinematics(q_t, fps)
    F = qpos_t.shape[0]

    # contacts that null the residual at the ORIGINAL trajectory
    contacts_t = []
    for f in range(F):
        target = h.inverse_dynamics(qpos_t[f], qvel_t[f], qacc_t[f])[:6]
        contacts_t.append(_realize_root_wrench(h, qpos_t[f], target))

    true_trans = qpos_t[:, :3].copy()

    # add a smooth root-translation bump (zero at the endpoints) -> nonzero residual
    bump = 0.03 * np.sin(np.pi * np.arange(F) / (F - 1))
    perturbed = qpos_t.copy()
    perturbed[:, 2] += bump

    from biomech.fitting.dynamics_fitter import rra_kinematics

    res = rra_kinematics(
        h, perturbed, contacts_t, 1.0 / fps, iters=6, track_weight=1e-3, smooth_weight=1e-2
    )

    e_before = float(
        np.sum(res.residual_before.force_norm**2 + res.residual_before.torque_norm**2)
    )
    e_after = float(
        np.sum(res.residual_after.force_norm**2 + res.residual_after.torque_norm**2)
    )
    assert e_after < 0.5 * e_before, (e_before, e_after)
    # adjusted root translation is closer to the true (unperturbed) trajectory
    err_in = np.max(np.abs(perturbed[:, 2] - true_trans[:, 2]))
    err_out = np.max(np.abs(res.qpos_t[:, 2] - true_trans[:, 2]))
    assert err_out < err_in, (err_in, err_out)


def test_rra_leaves_consistent_trajectory_alone():
    """On an already residual-consistent trajectory, RRA barely moves the root."""
    _require_mujoco()
    from biomech.fitting.dynamics_fitter import DynamicsFitter, rra_kinematics

    spec = _spec()
    rng = np.random.default_rng(37)
    fitter = DynamicsFitter(spec, coupled_knee="coupled")
    h = fitter.helper

    q_t = _smooth_trajectory(spec, rng, F=6)
    fps = 100.0
    qpos_t, qvel_t, qacc_t = fitter.kinematics(q_t, fps)
    F = qpos_t.shape[0]
    contacts_t = []
    for f in range(F):
        target = h.inverse_dynamics(qpos_t[f], qvel_t[f], qacc_t[f])[:6]
        contacts_t.append(_realize_root_wrench(h, qpos_t[f], target))

    res = rra_kinematics(h, qpos_t, contacts_t, 1.0 / fps, iters=4)
    assert np.max(np.abs(res.root_shift)) < 5e-3
