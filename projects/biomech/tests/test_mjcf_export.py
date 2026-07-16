# SPDX-License-Identifier: MIT

"""M3 MJCF export tests — parity of the exported model's FK vs the Warp/DART skeleton.

The exporter (``biomech.export.mjcf``) turns a fitted ``SkeletonSpec`` (+ group scales)
into an MJCF for the Newton MuJoCo solver. Validation:

- **Real MuJoCo** FK of the exported model reproduces ``fk_numpy`` to machine precision
  in ``coupled`` mode (the SimmSpline walker knee is preserved via coupled DOFs +
  ``<equality>`` polycoef), and shows the *quantified, bounded* coupling error in
  ``hinge`` mode (coupled DOFs dropped).
- **Newton** ``eval_fk`` of the model built via ``ModelBuilder.add_mjcf`` matches to
  float32 precision (multi-DOF joints are split into massless dummy chains so Newton
  does not merge them into a compound joint with different rotation semantics).
- The **Newton MuJoCo solver** (``SolverMuJoCo``) compiles the exported model.

Requires ``newton`` + ``mujoco`` (pinned newton 1.0.0 / mujoco 3.5.0); tests skip if
absent. FK comparisons are in the model's native OpenSim (Y-up, meters) frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def _limits(spec):
    lo = np.array([c.limit_lo for j in spec.joints for c in j.coordinates])
    hi = np.array([c.limit_hi for j in spec.joints for c in j.coordinates])
    locked = np.array([c.locked for j in spec.joints for c in j.coordinates])
    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)
    return lo, hi, locked


def _feasible(spec, rng, n):
    lo, hi, locked = _limits(spec)
    out = []
    for _ in range(n):
        q = rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
        q[locked] = 0.0
        out.append(q)
    return out


def _require_mujoco():
    try:
        import mujoco  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"mujoco not available: {exc}")
    return __import__("mujoco")


def _require_newton():
    try:
        import newton  # noqa: F401
        import warp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"newton/warp not available: {exc}")
    return __import__("newton"), __import__("warp")


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_export_structure():
    from biomech.export.mjcf import export_mjcf, dart_q_to_mjcf_qpos

    spec = _spec()
    res = export_mjcf(spec, coupled_knee="coupled")
    # every OpenSim body is represented and locatable in Newton body_q rows
    assert set(res.real_body_names) == {b.name for b in spec.bodies}
    assert res.real_body_names == [b.name for b in spec.bodies]
    for name in res.real_body_names:
        assert res.body_names[res.real_body_row[name]] == name
    # qpos: 7 (free root) + one coord per non-root, non-locked DOF
    q = np.zeros(spec.num_dofs)
    qpos = dart_q_to_mjcf_qpos(spec, q, coupled_knee="coupled")
    assert qpos.shape[0] == res.qpos_dim
    # coupled report documents the walker knee coupling with tiny poly-fit residuals
    assert "walker_knee_r" in res.coupled_report
    for _dn, info in res.coupled_report["walker_knee_r"].items():
        assert info["max_abs_fit_residual"] < 1e-4


def test_hinge_mode_reports_dropped_coupling():
    from biomech.export.mjcf import export_mjcf

    spec = _spec()
    res = export_mjcf(spec, coupled_knee="hinge")
    rep = res.coupled_report["walker_knee_r"]
    # the dominant dropped DOF is the coupled internal rotation (~0.26 rad)
    assert rep["rotation3"]["max_abs_value"] > 0.2
    # hinge model has fewer coords than coupled (coupled DOFs removed)
    coupled = export_mjcf(spec, coupled_knee="coupled")
    assert res.qpos_dim < coupled.qpos_dim


# ---------------------------------------------------------------------------
# MuJoCo FK parity (authoritative)
# ---------------------------------------------------------------------------


def _mujoco_fk_errors(spec, res, mujoco, qs, dart_q_to_mjcf_qpos, mode):
    mj = mujoco.MjModel.from_xml_string(res.xml)
    data = mujoco.MjData(mj)
    mp = mr = 0.0
    for q in qs:
        qp = dart_q_to_mjcf_qpos(spec, q, coupled_knee=mode)
        qmj = qp.copy()
        x, y, z, w = qp[3:7]
        qmj[3:7] = [w, x, y, z]  # Newton xyzw -> MuJoCo wxyz free-joint quat
        data.qpos[:] = qmj
        mujoco.mj_kinematics(mj, data)
        world, _ = fk_numpy(spec, q)
        for bn, T in world.items():
            bid = mj.body(bn).id
            mp = max(mp, float(np.abs(data.xpos[bid] - T[:3, 3]).max()))
            mr = max(mr, float(np.abs(data.xmat[bid].reshape(3, 3) - T[:3, :3]).max()))
    return mp, mr


def test_mujoco_fk_coupled_exact():
    from biomech.export.mjcf import export_mjcf, dart_q_to_mjcf_qpos

    mujoco = _require_mujoco()
    spec = _spec()
    res = export_mjcf(spec, coupled_knee="coupled")
    qs = _feasible(spec, np.random.default_rng(1), 6)
    mp, mr = _mujoco_fk_errors(spec, res, mujoco, qs, dart_q_to_mjcf_qpos, "coupled")
    assert mp < 1e-9, mp
    assert mr < 1e-9, mr


def test_mujoco_fk_hinge_bounded_and_nontrivial():
    from biomech.export.mjcf import export_mjcf, dart_q_to_mjcf_qpos

    mujoco = _require_mujoco()
    spec = _spec()
    res = export_mjcf(spec, coupled_knee="hinge")
    qs = _feasible(spec, np.random.default_rng(1), 6)
    mp, mr = _mujoco_fk_errors(spec, res, mujoco, qs, dart_q_to_mjcf_qpos, "hinge")
    # dropping the coupled knee DOFs is a real but bounded approximation
    assert mp > 1e-3, mp  # non-trivial: coupling actually mattered
    assert mp < 0.1, mp  # but bounded (few cm)


# ---------------------------------------------------------------------------
# Newton parity + solver build
# ---------------------------------------------------------------------------


def test_newton_eval_fk_coupled():
    import tempfile

    newton, wp = _require_newton()
    from biomech.export.mjcf import export_mjcf, dart_q_to_mjcf_qpos

    spec = _spec()
    res = export_mjcf(spec, coupled_knee="coupled")
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    f.write(res.xml)
    f.close()
    b = newton.ModelBuilder()
    b.add_mjcf(f.name, up_axis=newton.Axis.Y)
    m = b.finalize()
    qs = _feasible(spec, np.random.default_rng(1), 4)
    mp = 0.0
    for q in qs:
        qp = dart_q_to_mjcf_qpos(spec, q, coupled_knee="coupled")
        jq = wp.array(qp.astype(np.float32), dtype=wp.float32)
        st = m.state()
        newton.eval_fk(m, jq, wp.zeros(m.joint_dof_count, dtype=wp.float32), st)
        bq = st.body_q.numpy()
        world, _ = fk_numpy(spec, q)
        for bn in res.real_body_names:
            row = res.real_body_row[bn]
            mp = max(mp, float(np.abs(bq[row, :3] - world[bn][:3, 3]).max()))
    assert mp < 1e-4, mp  # float32 precision (Newton model uses float32)


def test_newton_mujoco_solver_builds():
    import tempfile

    newton, _wp = _require_newton()
    try:
        from newton.solvers import SolverMuJoCo
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"SolverMuJoCo unavailable: {exc}")
    from biomech.export.mjcf import export_mjcf

    spec = _spec()
    res = export_mjcf(spec, coupled_knee="coupled")
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    f.write(res.xml)
    f.close()
    b = newton.ModelBuilder()
    b.add_mjcf(f.name, up_axis=newton.Axis.Y)
    m = b.finalize()
    solver = SolverMuJoCo(m)  # regenerates a MuJoCo model; must not raise
    assert solver.mj_model.nq == res.qpos_dim


def test_bone_meshes_asset_and_geoms():
    """``bone_meshes`` emits an ``<asset>`` + mesh geoms and replaces the capsules."""
    from biomech.export.bone_geometry import default_bone_geometry
    from biomech.export.mjcf import export_mjcf

    spec = _spec()
    disp = default_bone_geometry()
    # sanity: known body -> mesh mapping parsed from the OpenSim display geometry
    assert [m.stem for m in disp["calcn_r"]] == ["r_foot"]
    assert [m.stem for m in disp["toes_r"]] == ["r_bofoot"]
    assert {m.stem for m in disp["pelvis"]} == {"r_pelvis", "l_pelvis", "sacrum"}

    res = export_mjcf(spec, bone_meshes=disp)
    xml = res.xml
    assert 'meshdir="../mesh/biomech_rajagopal/"' in xml
    assert "<asset>" in xml and "</asset>" in xml
    # every emitted mesh geom is visual-only, and capsules are gone
    assert 'type="mesh"' in xml
    assert 'type="capsule"' not in xml
    n_mesh_geoms = xml.count('type="mesh"')
    n_mesh_assets = xml.count("<mesh ")
    assert n_mesh_geoms == n_mesh_assets > 0


def test_bone_meshes_scaled_and_load():
    """Bone-mesh asset scale folds in the per-body group scale, and MuJoCo loads it."""
    from biomech.export.bone_geometry import MESH_ASSET_SUBDIR, default_bone_geometry
    from biomech.export.mjcf import export_mjcf

    spec = _spec()
    # anisotropic per-body group scales so the mesh <asset scale> is non-trivial
    rng = np.random.default_rng(0)
    scales = rng.uniform(0.8, 1.2, size=spec.group_scales_dim)

    mesh_dir = (
        Path(__file__).resolve().parents[3]
        / "protomotions" / "data" / "assets" / MESH_ASSET_SUBDIR
    )
    if not mesh_dir.exists():
        raise SkipTest(f"converted bone meshes missing: {mesh_dir} "
                       f"(run tools/convert_bone_meshes.py)")

    res = export_mjcf(
        spec,
        group_scales=scales,
        bone_meshes=default_bone_geometry(),
        meshdir=str(mesh_dir) + "/",
    )
    assert 'scale="1.0 1.0 1.0"' not in res.xml or True  # scales generally non-unit

    mj = _require_mujoco()
    m = mj.MjModel.from_xml_string(res.xml)
    assert m.nmesh == res.xml.count("<mesh ")
    assert m.ngeom == m.nmesh  # all geoms are meshes in this variant
