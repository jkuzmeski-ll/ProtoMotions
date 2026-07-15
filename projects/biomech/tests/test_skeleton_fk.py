# SPDX-License-Identifier: MIT

"""Parity tests for the Warp skeleton FK (biomech.skeleton.skeleton), M2b.

Validates forward kinematics on the reproducible ``Rajagopal2015.osim`` against the
real-Nimble goldens (generated once in WSL; see ``tools/nimble_golden/``):

- ``rajagopal2015_fk.json``               body world transforms + marker positions
                                          at q=0 and random poses (unit scale).
- ``rajagopal2015_customjoint_sweep.json`` per-CustomJoint driving-DOF sweep child
                                          body transforms (SimmSpline knee coupling).
- ``rajagopal2015_scaling.json``          body transforms under random anisotropic
                                          group scales.

Both the NumPy reference (``fk_numpy``) and the batched Warp kernel (``WarpSkeleton``)
are checked. All comparisons are in Nimble's native OpenSim (Y-up, meters) frame.

Note on markers under scaling: the golden's marker field for the scaling cases was
dumped with an *unscaled* local offset (a documented unit-scale fallback in the
golden generator), whereas Nimble's authoritative ``getMarkerWorldPositions`` scales
the offset (``scale ⊙ offset``). Our FK follows the authoritative formula, so the
scaling-case marker check reconstructs the expected value from the golden body
transforms and the case scales rather than comparing to the (unscaled) golden field.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biomech.osim import parse_osim
from biomech.skeleton.skeleton import WarpSkeleton, fk_numpy

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"
_REFS = _ROOT / "docs" / "refs"

# Nimble is float64; our FK reproduces it to ~machine precision.
_TOL = 1e-9


def _spec():
    return parse_osim(str(_OSIM))


def _body_group_map(spec):
    m = {}
    for gi, group in enumerate(spec.scale_groups):
        for name in group:
            m[name] = gi
    return m


# ---------------------------------------------------------------------------
# NumPy reference FK
# ---------------------------------------------------------------------------


def test_fk_numpy_matches_golden():
    spec = _spec()
    g = json.loads((_REFS / "rajagopal2015_fk.json").read_text())
    max_b = max_m = 0.0
    for case in g["cases"]:
        world, markers = fk_numpy(spec, np.array(case["q"]))
        for name, T in case["bodyTransforms"].items():
            max_b = max(max_b, np.abs(world[name] - np.array(T)).max())
        for name, p in case["markers"].items():
            max_m = max(max_m, np.abs(markers[name] - np.array(p)).max())
    assert max_b < _TOL, max_b
    assert max_m < _TOL, max_m


def test_fk_numpy_customjoint_sweep():
    spec = _spec()
    ndof = spec.num_dofs
    g = json.loads((_REFS / "rajagopal2015_customjoint_sweep.json").read_text())
    max_e = 0.0
    for _jname, info in g.items():
        di = info["dofIndexInSkeleton"]
        for s in info["samples"]:
            q = np.zeros(ndof)
            q[di] = s["x"]
            world, _ = fk_numpy(spec, q)
            max_e = max(max_e, np.abs(world[info["childBody"]] - np.array(s["childBodyWorld"])).max())
    assert max_e < _TOL, max_e


def test_fk_numpy_scaling_body_transforms():
    spec = _spec()
    g = json.loads((_REFS / "rajagopal2015_scaling.json").read_text())
    max_b = 0.0
    for case in g["cases"]:
        world, _ = fk_numpy(spec, np.array(case["q"]), np.array(case["groupScales"]))
        for name, T in case["bodyTransforms"].items():
            max_b = max(max_b, np.abs(world[name] - np.array(T)).max())
    assert max_b < _TOL, max_b


def test_fk_numpy_scaling_markers_authoritative():
    """Markers under scaling follow Nimble's ``scale ⊙ offset`` formula."""
    spec = _spec()
    bg = _body_group_map(spec)
    mmap = {m.name: m for m in spec.markers}
    g = json.loads((_REFS / "rajagopal2015_scaling.json").read_text())
    max_m = 0.0
    for case in g["cases"]:
        gs = np.array(case["groupScales"]).reshape(-1, 3)
        _, markers = fk_numpy(spec, np.array(case["q"]), np.array(case["groupScales"]))
        for name in case["markers"]:
            m = mmap[name]
            T = np.array(case["bodyTransforms"][m.body])
            s = gs[bg[m.body]]
            expected = T[:3, :3] @ (s * m.offset) + T[:3, 3]
            max_m = max(max_m, np.abs(markers[name] - expected).max())
    assert max_m < _TOL, max_m


# ---------------------------------------------------------------------------
# Warp kernel FK (batched)
# ---------------------------------------------------------------------------


def test_fk_warp_matches_golden():
    spec = _spec()
    ws = WarpSkeleton(spec, device="cpu")
    bn, mn = ws.body_names(), ws.marker_names()
    g = json.loads((_REFS / "rajagopal2015_fk.json").read_text())
    Q = np.array([c["q"] for c in g["cases"]])
    world, markers = ws.forward(Q)
    max_b = max_m = 0.0
    for ci, case in enumerate(g["cases"]):
        for name, T in case["bodyTransforms"].items():
            max_b = max(max_b, np.abs(world[ci, bn.index(name)] - np.array(T)).max())
        for name, p in case["markers"].items():
            max_m = max(max_m, np.abs(markers[ci, mn.index(name)] - np.array(p)).max())
    assert max_b < _TOL, max_b
    assert max_m < _TOL, max_m


def test_fk_warp_customjoint_sweep():
    spec = _spec()
    ws = WarpSkeleton(spec, device="cpu")
    bn = ws.body_names()
    ndof = spec.num_dofs
    g = json.loads((_REFS / "rajagopal2015_customjoint_sweep.json").read_text())
    max_e = 0.0
    for _jname, info in g.items():
        di = info["dofIndexInSkeleton"]
        Q = np.zeros((len(info["samples"]), ndof))
        for k, s in enumerate(info["samples"]):
            Q[k, di] = s["x"]
        world, _ = ws.forward(Q)
        bi = bn.index(info["childBody"])
        for k, s in enumerate(info["samples"]):
            max_e = max(max_e, np.abs(world[k, bi] - np.array(s["childBodyWorld"])).max())
    assert max_e < _TOL, max_e


def test_fk_warp_scaling():
    spec = _spec()
    ws = WarpSkeleton(spec, device="cpu")
    bn, mn = ws.body_names(), ws.marker_names()
    bg = _body_group_map(spec)
    mmap = {m.name: m for m in spec.markers}
    g = json.loads((_REFS / "rajagopal2015_scaling.json").read_text())
    max_b = max_m = 0.0
    for case in g["cases"]:
        gs = np.array(case["groupScales"]).reshape(-1, 3)
        world, markers = ws.forward(np.array(case["q"]), np.array(case["groupScales"]))
        for name, T in case["bodyTransforms"].items():
            max_b = max(max_b, np.abs(world[0, bn.index(name)] - np.array(T)).max())
        for name in case["markers"]:
            m = mmap[name]
            T = np.array(case["bodyTransforms"][m.body])
            s = gs[bg[m.body]]
            expected = T[:3, :3] @ (s * m.offset) + T[:3, 3]
            max_m = max(max_m, np.abs(markers[0, mn.index(name)] - expected).max())
    assert max_b < _TOL, max_b
    assert max_m < _TOL, max_m


def test_marker_jacobian_wrt_q_matches_finite_difference():
    """Autodiff marker Jacobian dq matches finite differences (flows through SimmSpline)."""
    spec = _spec()
    ws = WarpSkeleton(spec, device="cpu")
    ndof = spec.num_dofs
    rng = np.random.default_rng(11)
    q = rng.uniform(-0.4, 0.4, size=ndof)
    jac = ws.marker_jacobian_wrt_q(q)  # (M, 3, ndof)

    eps = 1e-6
    _, m0 = ws.forward(q)
    m0 = m0[0]
    fd = np.zeros_like(jac)
    for i in range(ndof):
        qp = q.copy()
        qp[i] += eps
        _, mp = ws.forward(qp)
        fd[:, :, i] = (mp[0] - m0) / eps
    err = np.abs(jac - fd).max()
    assert err < 1e-5, err
    # sanity: the Jacobian is non-trivial
    assert np.linalg.norm(jac) > 1.0


def test_fk_warp_matches_numpy():
    """Warp kernel and NumPy reference agree on a batch of random poses."""
    spec = _spec()
    ws = WarpSkeleton(spec, device="cpu")
    bn, mn = ws.body_names(), ws.marker_names()
    rng = np.random.default_rng(7)
    Q = rng.uniform(-1.0, 1.0, size=(16, spec.num_dofs))
    world, markers = ws.forward(Q)
    max_b = max_m = 0.0
    for f in range(Q.shape[0]):
        w_np, m_np = fk_numpy(spec, Q[f])
        for name in bn:
            max_b = max(max_b, np.abs(world[f, bn.index(name)] - w_np[name]).max())
        for name in mn:
            max_m = max(max_m, np.abs(markers[f, mn.index(name)] - m_np[name]).max())
    assert max_b < _TOL, max_b
    assert max_m < _TOL, max_m
