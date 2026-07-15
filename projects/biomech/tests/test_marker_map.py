# SPDX-License-Identifier: MIT

"""Tests for the real-data marker bridge (biomech.fitting.marker_map).

Validates the S001 Plug-in-Gait -> Rajagopal2015 marker adapter:

- ``build_observations`` reorders into model-marker order, NaN for unmapped markers,
- the lab Z-up -> OpenSim Y-up rotation is applied correctly,
- the S001 map covers the lower-body landmarks and leaves virtual joint centres unmapped,
- on the real S001 capture (if present) coverage is sane and observations are finite for
  mapped markers.

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.marker_map import (
    LOWER_BODY_MARKERS,
    R_PM2OS,
    build_observations,
    mapping_coverage,
    observations_from_session,
    s001_marker_map,
)
from biomech.tests import SkipTest, TRIAL_C3D, require

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"


def _model_marker_names():
    from biomech.osim import parse_osim
    from biomech.skeleton.skeleton import WarpSkeleton

    spec = parse_osim(str(_OSIM))
    return WarpSkeleton(spec).marker_names()


def test_build_observations_order_and_nan():
    model_names = ["RASI", "LASI", "RHJC", "RKNE_fake"]  # RHJC virtual, last unmapped
    cap_labels = ["LASI", "RASI", "JUNK"]
    F = 3
    pts = np.zeros((F, 3, 3))
    pts[:, 0, :] = [1.0, 2.0, 3.0]  # LASI
    pts[:, 1, :] = [4.0, 5.0, 6.0]  # RASI
    mm = s001_marker_map()
    obs, present = build_observations(
        cap_labels, pts, model_names, mm, to_opensim=False
    )
    assert obs.shape == (F, 4, 3)
    # RASI (model idx 0) should pull capture RASI = [4,5,6]
    assert np.allclose(obs[:, 0, :], [4.0, 5.0, 6.0])
    assert np.allclose(obs[:, 1, :], [1.0, 2.0, 3.0])  # LASI
    # RHJC is virtual (not in map) -> NaN; fake marker -> NaN
    assert np.all(np.isnan(obs[:, 2, :]))
    assert np.all(np.isnan(obs[:, 3, :]))
    assert present.tolist() == [True, True, False, False]


def test_opensim_rotation_applied():
    model_names = ["RASI"]
    cap_labels = ["RASI"]
    pts = np.array([[[1.0, 2.0, 3.0]]])  # lab Z-up
    mm = s001_marker_map()
    obs, _ = build_observations(cap_labels, pts, model_names, mm, to_opensim=True)
    # os = R_PM2OS @ lab
    expected = R_PM2OS @ np.array([1.0, 2.0, 3.0])
    assert np.allclose(obs[0, 0, :], expected)
    # sanity: lab_z (up) -> os_y (up)
    assert np.isclose(obs[0, 0, 1], 3.0)
    assert np.isclose(obs[0, 0, 2], -2.0)


def test_map_covers_lower_body_landmarks():
    mm = s001_marker_map()
    # every mapped anatomical marker should also be in the map
    for a in mm.anatomical:
        assert a in mm.model_to_capture
    # virtual joint centres are NOT mapped
    for jc in ["RHJC", "LHJC", "RKJC", "LKJC", "RAJC", "LAJC", "RSJC", "REJC"]:
        assert jc not in mm.model_to_capture
    # lower-body foot/ankle/knee landmarks are mapped
    for m in ["RCAL", "RTOE", "RMT5", "RLMAL", "RMMAL", "RLFC", "RMFC"]:
        assert m in mm.model_to_capture


def test_lower_body_restrict():
    full = s001_marker_map(lower_body_only=False)
    lb = s001_marker_map(lower_body_only=True)
    assert len(lb.model_to_capture) < len(full.model_to_capture)
    # upper-body markers dropped
    assert "C7" not in lb.model_to_capture
    assert "RELB" not in lb.model_to_capture
    # lower-body kept
    assert "RASI" in lb.model_to_capture
    assert "RCAL" in lb.model_to_capture
    for m in lb.model_to_capture:
        assert m in LOWER_BODY_MARKERS


def test_real_s001_coverage():
    require(TRIAL_C3D)
    from biomech.session import load_session

    session = load_session(str(TRIAL_C3D))
    model_names = _model_marker_names()
    mm = s001_marker_map()
    cov = mapping_coverage(model_names, session.marker_labels, mm)
    # we should map a healthy chunk of the model markers to real labels
    assert len(cov["mapped"]) >= 30
    # all mapped markers are finite for at least some frames
    obs, present = observations_from_session(session, model_names, mm)
    assert obs.shape == (session.n_frames, len(model_names), 3)
    mapped_idx = [i for i, n in enumerate(model_names) if n in set(cov["mapped"])]
    finite_any = np.isfinite(obs[:, mapped_idx, :]).any(axis=(0, 2))
    assert finite_any.all()
    # lower-body anatomical landmarks are present
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    for m in ["RASI", "LASI", "RCAL", "LCAL", "RLMAL", "LLMAL"]:
        assert present[name_to_idx[m]]
