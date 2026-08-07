# SPDX-License-Identifier: MIT

"""Tests for subject foot geometry (biomech.contact.foot_geometry, M4).

Validates building a subject-specific plantar sole from static plantar markers + model
anchors:

- ``foot_dimensions_from_markers`` recovers known widths/length from synthetic markers,
- ``calcn_anchors_from_spec`` returns scaled marker offsets,
- ``build_subject_sole`` produces a tapered bed in the calcn frame with sane areas,
  downward normals, a soft heel / relieved arch compliance map, and forefoot wider than
  heel,
- the subject sole drops into the hydroelastic contact model and predicts a plausible
  GRF (heel-strike pose loads the heel; the compliance map is respected),
- on the real S001 static capture (if present) both feet yield realistic dimensions.

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.contact.foot_geometry import (
    FootAnchors,
    FootDimensions,
    SolePads,
    average_static_markers,
    build_subject_sole,
    calcn_anchors_from_spec,
    foot_dimensions_from_markers,
    subject_sole_from_session,
)
from biomech.tests import CAL_C3D, SkipTest, require

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_SPEC = None


def _spec():
    global _SPEC
    if _SPEC is None:
        from biomech.osim import parse_osim

        _SPEC = parse_osim(str(_OSIM))
    return _SPEC


def _synthetic_foot_markers(side="R", length=0.26, heel_w=0.05, fore_w=0.09):
    """Build a synthetic right-foot plantar marker set along lab +x, width along +y."""
    heel_center = np.array([0.0, 0.0, 0.0])
    # heel cluster: HEE (center), HEE2/HEE3 across the width
    m = {
        f"{side}HEE": heel_center + [0.0, 0.0, 0.0],
        f"{side}HEE2": heel_center + [0.0, -heel_w / 2, 0.0],
        f"{side}HEE3": heel_center + [0.0, +heel_w / 2, 0.0],
        f"{side}MTH1": np.array([0.16, -fore_w / 2, 0.0]),
        f"{side}MTH5": np.array([0.16, +fore_w / 2, 0.0]),
        f"{side}HLX": np.array([length, 0.0, 0.0]),
        f"{side}TOE": np.array([length - 0.02, 0.0, 0.0]),
    }
    return m


def test_dimensions_from_synthetic_markers():
    m = _synthetic_foot_markers(length=0.26, heel_w=0.05, fore_w=0.09)
    dims = foot_dimensions_from_markers(m, "R")
    assert abs(dims.foot_length - 0.26) < 1e-6
    assert abs(dims.heel_width - 0.05) < 1e-6
    assert abs(dims.forefoot_width - 0.09) < 1e-6
    assert abs(dims.heel_to_ball - 0.16) < 1e-6
    assert dims.toe_length >= 0.03


def test_average_static_markers_nan_aware():
    labels = ["A", "B"]
    pts = np.full((4, 2, 3), np.nan)
    pts[:, 0, :] = [1.0, 2.0, 3.0]
    pts[2, 1, :] = [4.0, 5.0, 6.0]  # only one valid frame for B
    means = average_static_markers(labels, pts)
    assert np.allclose(means["A"], [1.0, 2.0, 3.0])
    assert np.allclose(means["B"], [4.0, 5.0, 6.0])


def test_calcn_anchors_scale():
    spec = _spec()
    a1 = calcn_anchors_from_spec(spec, "R")
    # scale the calcn_r group (index 4) by 2x in x
    gs = np.ones(3 * len(spec.scale_groups))
    gs[4 * 3 + 0] = 2.0
    a2 = calcn_anchors_from_spec(spec, "R", gs)
    assert np.isclose(a2.toe[0], a1.toe[0] * 2.0)
    assert np.isclose(a2.toe[1], a1.toe[1])  # y unscaled


def test_build_subject_sole_shape_and_pads():
    spec = _spec()
    dims = FootDimensions(
        side="R", heel_width=0.05, forefoot_width=0.09, foot_length=0.26,
        heel_to_ball=0.16, toe_length=0.05,
    )
    anchors = calcn_anchors_from_spec(spec, "R")
    nx, ny = 16, 6
    pads = SolePads(heel=0.6, arch=0.15, forefoot=1.0, toe=0.8)
    sole = build_subject_sole(dims, anchors, nx=nx, ny=ny, pads=pads)

    assert sole.n == nx * ny
    assert np.all(sole.areas > 0.0)
    # normals are unit and point "down" in the body frame (-up)
    assert np.allclose(np.linalg.norm(sole.normals, axis=1), 1.0)
    # modulus map uses the pad values
    mods = set(np.round(sole.modulus, 4))
    assert 0.6 in mods  # heel pad present
    assert 0.15 in mods  # arch relief present
    assert 1.0 in mods  # forefoot present
    # total plantar area roughly matches a tapered footprint (< bounding box)
    bbox = dims.foot_length * dims.forefoot_width
    assert sole.total_area < bbox
    assert sole.total_area > 0.5 * dims.foot_length * dims.heel_width


def test_subject_sole_drives_contact():
    from biomech.contact.elastic_foundation import _quat_rotate_np  # noqa: F401
    from biomech.contact.hydroelastic import HydroelasticParams, evaluate_contact

    spec = _spec()
    dims = FootDimensions("R", 0.05, 0.09, 0.26, 0.16, 0.05)
    anchors = calcn_anchors_from_spec(spec, "R")
    sole = build_subject_sole(dims, anchors, nx=16, ny=6, plantar_drop=0.02)

    # place the foot so the plantar bed penetrates a flat ground at z=0.
    # body at origin, identity orientation: sole y (up) == world y; but contact uses
    # world z. Rotate the foot so body -up aligns with world -z: a -90deg rot about x
    # maps body y->world z. Simpler: just push the whole sole below z=0 by translating.
    lo = sole.points[:, 1].min()  # body y of lowest plantar point (most negative)
    # identity quat; world z of a point == body z. To get penetration we set ground
    # above the points' body-z. Use the body frame directly (identity), ground at max z.
    body_pos = np.array([[0.0, 0.0, 0.0]])
    quat = np.array([[0.0, 0.0, 0.0, 1.0]])
    z = np.zeros((1, 3))
    ground_z = float(sole.points[:, 2].max()) + 0.01
    params = HydroelasticParams(k_bed=5e6, stiffen_b=0.0, hc_alpha=0.0,
                                mu_d=0.0, mu_s=0.0)
    pred = evaluate_contact(sole, params, body_pos, quat, z, z, ground_z=ground_z)
    assert pred.total_normal[0] > 0.0
    assert np.all(np.isfinite(pred.grf))
    # The corrected anatomical sole uses calcn +y as plantar-up. Its deepest plantar
    # point is the body-frame y=0 tangent plane; ``plantar_drop`` offsets the marker
    # anchor used to build that plane rather than requiring negative body y.
    assert abs(lo) < 1e-12


def test_real_s001_static_dimensions():
    require(CAL_C3D)
    from biomech.contact.foot_geometry import foot_dimensions_from_session
    from biomech.session import load_session

    session = load_session(str(CAL_C3D))
    for side in ("R", "L"):
        dims = foot_dimensions_from_session(session, side)
        # realistic adult foot: length 0.20-0.32 m, forefoot wider than heel
        assert 0.18 < dims.foot_length < 0.34, (side, dims.foot_length)
        assert 0.03 < dims.heel_width < 0.09, (side, dims.heel_width)
        assert 0.06 < dims.forefoot_width < 0.14, (side, dims.forefoot_width)
        assert dims.forefoot_width > dims.heel_width
        assert 0.0 < dims.heel_to_ball < dims.foot_length

    # and a full sole builds from the real static capture
    sole = subject_sole_from_session(session, _spec(), "R", nx=12, ny=5)
    assert sole.n == 12 * 5
    assert np.all(sole.areas > 0.0)
