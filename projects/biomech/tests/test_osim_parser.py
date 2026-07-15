# SPDX-License-Identifier: MIT

"""Parity tests for the OpenSim parser (biomech.osim.parser), M2a.

Validates ``parse_osim`` on the reproducible ``Rajagopal2015.osim`` against the
real-Nimble golden ``docs/refs/rajagopal2015_structure.json`` (generated once in
WSL; see ``tools/nimble_golden/``). We check the robust, well-defined structural
fields: counts, body inertial params, joint topology + DOF order, joint relative
frames (T_parent / T_child), Euler axis order + flip map, markers, and scale
groups. Exact FK/scaling numerics are validated later (M2b) against the FK
goldens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biomech.osim import parse_osim

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"
_GOLDEN = _ROOT / "docs" / "refs" / "rajagopal2015_structure.json"

_TOL = 1e-9


def _load():
    spec = parse_osim(str(_OSIM))
    golden = json.loads(_GOLDEN.read_text())
    return spec, golden


def test_top_level_counts():
    spec, g = _load()
    assert spec.num_dofs == g["numDofs"], (spec.num_dofs, g["numDofs"])
    assert spec.num_bodies == g["numBodies"]
    assert spec.num_joints == g["numJoints"]
    assert spec.group_scales_dim == g["groupScalesDim"]
    assert spec.frame == g["frame"]


def test_bodies_match_golden():
    spec, g = _load()
    assert len(spec.bodies) == len(g["bodies"])
    for b, gb in zip(spec.bodies, g["bodies"]):
        assert b.index == gb["index"], (b.name, gb["name"])
        assert b.name == gb["name"]
        assert b.parent_body == gb["parentBody"], b.name
        assert b.parent_joint == gb["parentJoint"], b.name
        assert abs(b.mass - gb["mass"]) < _TOL, b.name
        assert np.allclose(b.com, np.array(gb["localCOM"]), atol=_TOL), b.name


def test_joint_topology_and_dof_order():
    spec, g = _load()
    assert len(spec.joints) == len(g["joints"])
    dof_cursor = 0
    for j, gj in zip(spec.joints, g["joints"]):
        assert j.name == gj["name"]
        assert j.parent_body == gj["parentBody"], j.name
        assert j.child_body == gj["childBody"], j.name
        assert j.num_dofs == gj["numDofs"], j.name
        assert j.dof_names == gj["dofNames"], j.name
        expected_indices = list(range(dof_cursor, dof_cursor + j.num_dofs))
        assert expected_indices == gj["dofIndicesInSkeleton"], j.name
        dof_cursor += j.num_dofs


def test_joint_relative_frames():
    spec, g = _load()
    for j, gj in zip(spec.joints, g["joints"]):
        assert np.allclose(j.T_parent, np.array(gj["Tparent"]), atol=1e-8), j.name
        assert np.allclose(j.T_child, np.array(gj["Tchild"]), atol=1e-8), j.name


def test_joint_axis_order_and_flip():
    spec, g = _load()
    for j, gj in zip(spec.joints, g["joints"]):
        g_order = gj.get("axisOrder")
        if g_order is None:
            assert j.axis_order is None, j.name
            assert j.flip_axis_map is None, j.name
        else:
            # golden stores e.g. "AxisOrder.ZXY"
            assert j.axis_order == g_order.split(".")[-1], (j.name, g_order)
            assert np.allclose(
                j.flip_axis_map, np.array(gj["flipAxisMap"]), atol=_TOL
            ), j.name


def test_joint_position_limits():
    spec, g = _load()
    for j, gj in zip(spec.joints, g["joints"]):
        lo = [c.limit_lo for c in j.coordinates]
        hi = [c.limit_hi for c in j.coordinates]
        assert np.allclose(lo, np.array(gj["posLower"]), atol=1e-6), j.name
        assert np.allclose(hi, np.array(gj["posUpper"]), atol=1e-6), j.name


def test_markers_match_golden():
    spec, g = _load()
    assert len(spec.markers) == len(g["markers"])
    by_name = {m.name: m for m in spec.markers}
    for gm in g["markers"]:
        assert gm["name"] in by_name, gm["name"]
        m = by_name[gm["name"]]
        assert m.body == gm["body"], gm["name"]
        assert np.allclose(m.offset, np.array(gm["offset"]), atol=_TOL), gm["name"]
        assert m.anatomical == gm["anatomical"], gm["name"]


def test_scale_groups_match_golden():
    spec, g = _load()
    assert spec.scale_groups == [list(grp) for grp in g["bodyScaleGroups"]]


def test_coupled_knee_uses_simmsplines():
    # The gold-standard signal: walker_knee_r couples translations/2 rotations to
    # knee_angle_r through SimmSplines. Confirm the parser preserved them.
    spec, _ = _load()
    knee = spec.joint("walker_knee_r")
    assert knee.nimble_type == "CustomJoint1"
    from biomech.osim.spec import SimmSplineSpec

    spline_axes = [a for a in knee.transform_axes if isinstance(a.function, SimmSplineSpec)]
    # rotation2, rotation3, translation1, translation2 are SimmSplines (4 total).
    assert len(spline_axes) == 4, [a.name for a in spline_axes]
