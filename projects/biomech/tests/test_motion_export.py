# SPDX-License-Identifier: MIT

"""M3 motion-export tests — fitted q(t) -> ProtoMotions ``.motion`` clip.

Validates ``biomech.export.motion.build_motion``:
- shapes / dtypes / unit-norm quaternions,
- OpenSim Y-up -> ProtoMotions Z-up conversion (root height lands on +Z),
- gold-standard body positions come straight from the Warp FK (Z-up rotated),
- velocities vanish for a static clip and are correct for constant motion,
- ``dof_pos`` matches the exported-MJCF sim DOF layout,
- the produced dict loads as a ``RobotState`` (COMMON convention) when ProtoMotions
  is importable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.export.motion import R_OS2PM, build_motion
from biomech.osim import parse_osim
from biomech.skeleton.skeleton import fk_numpy
from biomech.tests import SkipTest

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_SPEC = None


def _spec():
    global _SPEC
    if _SPEC is None:
        _SPEC = parse_osim(str(_OSIM))
    return _SPEC


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"torch not available: {exc}")


def _feasible(spec, rng, n):
    lo = np.array([c.limit_lo for j in spec.joints for c in j.coordinates])
    hi = np.array([c.limit_hi for j in spec.joints for c in j.coordinates])
    locked = np.array([c.locked for j in spec.joints for c in j.coordinates])
    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    Q = rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), size=(n, len(lo)))
    Q[:, locked] = 0.0
    return Q


def test_shapes_and_unit_quats():
    _require_torch()
    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(0), 5)
    res = build_motion(spec, Q, fps=100.0)
    d = res.data
    nb = len(res.body_names)
    assert d["rigid_body_pos"].shape == (5, nb, 3)
    assert d["rigid_body_rot"].shape == (5, nb, 4)
    assert d["rigid_body_vel"].shape == (5, nb, 3)
    assert d["rigid_body_ang_vel"].shape == (5, nb, 3)
    assert d["dof_pos"].shape[0] == 5
    assert d["dof_vel"].shape == d["dof_pos"].shape
    assert d["fps"] == 100.0
    norms = np.linalg.norm(d["rigid_body_rot"].numpy(), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_zup_conversion_root_height():
    _require_torch()
    spec = _spec()
    # neutral pose except the pelvis_ty default (~0.94 m up in OpenSim Y)
    q = np.zeros(spec.num_dofs)
    dof = {n: i for i, n in enumerate(spec.dof_names)}
    q[dof["pelvis_ty"]] = 0.94
    res = build_motion(spec, np.tile(q, (3, 1)), fps=100.0)
    pos = res.data["rigid_body_pos"].numpy()
    # root (pelvis) is body 0; up axis is Z (index 2) after conversion
    assert pos[0, 0, 2] > 0.9
    assert abs(pos[0, 0, 1]) < 1e-5  # Y not up


def test_positions_match_warp_fk_rotated():
    _require_torch()
    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(2), 4)
    res = build_motion(spec, Q, fps=100.0)
    pos = res.data["rigid_body_pos"].numpy()
    for f in range(Q.shape[0]):
        world, _ = fk_numpy(spec, Q[f])
        for bi, bn in enumerate(res.body_names):
            expected = R_OS2PM @ world[bn][:3, 3]
            assert np.allclose(pos[f, bi], expected, atol=1e-5), (bn, f)


def test_static_clip_has_zero_velocity():
    _require_torch()
    spec = _spec()
    q = _feasible(spec, np.random.default_rng(3), 1)[0]
    res = build_motion(spec, np.tile(q, (6, 1)), fps=120.0)
    assert np.abs(res.data["rigid_body_vel"].numpy()).max() < 1e-6
    assert np.abs(res.data["rigid_body_ang_vel"].numpy()).max() < 1e-6
    assert np.abs(res.data["dof_vel"].numpy()).max() < 1e-6


def test_constant_root_translation_velocity():
    _require_torch()
    spec = _spec()
    dof = {n: i for i, n in enumerate(spec.dof_names)}
    fps = 100.0
    F = 5
    Q = np.zeros((F, spec.num_dofs))
    # translate along OpenSim +X (forward) by 0.5 m/s -> 0.005 m/frame
    Q[:, dof["pelvis_tx"]] = 0.005 * np.arange(F)
    res = build_motion(spec, Q, fps=fps)
    vel = res.data["rigid_body_vel"].numpy()
    # OpenSim X maps to ProtoMotions X (forward); interior frames -> 0.5 m/s
    assert np.allclose(vel[2, 0, 0], 0.5, atol=1e-4)
    assert np.abs(vel[2, 0, 1:]).max() < 1e-4


def test_dof_pos_matches_mjcf_qpos():
    _require_torch()
    from biomech.export.mjcf import dart_q_to_mjcf_qpos

    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(4), 3)
    res = build_motion(spec, Q, fps=100.0)
    dp = res.data["dof_pos"].numpy()
    for f in range(Q.shape[0]):
        qpos = dart_q_to_mjcf_qpos(spec, Q[f])
        assert np.allclose(dp[f], qpos[7:], atol=1e-5)


def test_loads_as_robotstate():
    _require_torch()
    try:
        from protomotions.simulator.base_simulator.simulator_state import (
            RobotState,
            StateConversion,
        )
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"protomotions not importable: {exc}")

    spec = _spec()
    Q = _feasible(spec, np.random.default_rng(5), 4)
    res = build_motion(spec, Q, fps=100.0)
    rs = RobotState.from_dict(res.data, state_conversion=StateConversion.COMMON)
    rs.fps = res.fps
    assert rs.num_bodies == len(res.body_names)
    assert rs.motion_num_frames == 4
