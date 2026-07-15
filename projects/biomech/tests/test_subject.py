# SPDX-License-Identifier: MIT

"""Tests for the serialized fitted-subject bundle (biomech.export.subject)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from biomech.export.subject import FittedSubject, load_subject, save_subject
from biomech.fitting.marker_fitter import MarkerFitResult
from biomech.osim import parse_osim

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_spec_cache = None


def _spec():
    global _spec_cache
    if _spec_cache is None:
        _spec_cache = parse_osim(str(_OSIM))
    return _spec_cache


def _synthetic_subject(spec):
    rng = np.random.default_rng(0)
    G = len(spec.scale_groups)
    M = len(spec.markers)
    F, ndof = 7, spec.num_dofs
    nb = len(spec.bodies)
    return FittedSubject(
        group_scales=rng.uniform(0.9, 1.1, size=3 * G),
        marker_offsets=rng.normal(0, 0.01, size=(M, 3)),
        marker_names=[m.name for m in spec.markers],
        poses=rng.normal(0, 0.2, size=(F, ndof)),
        fps=250.0,
        marker_rms=rng.uniform(0, 0.02, size=F),
        anatomical=np.array([m.anatomical for m in spec.markers], dtype=bool),
        body_names=[b.name for b in spec.bodies],
        inertial_params=rng.uniform(0.5, 2.0, size=(nb, 10)),
        inertial_body_names=[b.name for b in spec.bodies],
        mjcf_xml="<mujoco><worldbody/></mujoco>",
        osim_path=str(_OSIM),
        metadata={"subject": "S001", "phase": "walk", "trial": 101},
    )


def test_round_trip_preserves_everything():
    spec = _spec()
    subj = _synthetic_subject(spec)
    with tempfile.TemporaryDirectory() as tmp:
        p = save_subject(subj, Path(tmp) / "s001")
        assert p.suffix == ".npz" and p.exists()
        got = load_subject(p)

    assert np.allclose(got.group_scales, subj.group_scales)
    assert np.allclose(got.marker_offsets, subj.marker_offsets)
    assert np.allclose(got.poses, subj.poses)
    assert np.allclose(got.marker_rms, subj.marker_rms)
    assert np.array_equal(got.anatomical, subj.anatomical)
    assert np.allclose(got.inertial_params, subj.inertial_params)
    assert got.marker_names == subj.marker_names
    assert got.body_names == subj.body_names
    assert got.inertial_body_names == subj.inertial_body_names
    assert got.fps == subj.fps
    assert got.mjcf_xml == subj.mjcf_xml
    assert got.coupled_knee == subj.coupled_knee
    assert got.osim_path == subj.osim_path
    assert got.metadata == subj.metadata


def test_optional_fields_absent():
    spec = _spec()
    G = len(spec.scale_groups)
    M = len(spec.markers)
    subj = FittedSubject(
        group_scales=np.ones(3 * G),
        marker_offsets=np.zeros((M, 3)),
        marker_names=[m.name for m in spec.markers],
        poses=np.zeros((3, spec.num_dofs)),
        fps=100.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = save_subject(subj, Path(tmp) / "min.npz")
        got = load_subject(p)
    assert got.marker_rms is None
    assert got.anatomical is None
    assert got.inertial_params is None
    assert got.mjcf_xml is None


def test_from_marker_fit():
    spec = _spec()
    G = len(spec.scale_groups)
    M = len(spec.markers)
    F = 4
    result = MarkerFitResult(
        group_scales=np.ones(3 * G),
        marker_offsets=np.zeros((M, 3)),
        poses=np.zeros((F, spec.num_dofs)),
        marker_rms=np.full(F, 0.01),
    )
    subj = FittedSubject.from_marker_fit(
        spec, result, fps=200.0, osim_path=str(_OSIM), metadata={"subject": "S001"}
    )
    assert len(subj.marker_names) == M
    assert subj.anatomical.shape == (M,)
    assert subj.fps == 200.0
    assert subj.metadata["subject"] == "S001"


def test_to_mjcf_uses_stored_xml():
    spec = _spec()
    subj = _synthetic_subject(spec)
    assert subj.to_mjcf() == "<mujoco><worldbody/></mujoco>"


def test_to_motion_builds_clip():
    spec = _spec()
    G = len(spec.scale_groups)
    M = len(spec.markers)
    subj = FittedSubject(
        group_scales=np.ones(3 * G),
        marker_offsets=np.zeros((M, 3)),
        marker_names=[m.name for m in spec.markers],
        poses=np.zeros((4, spec.num_dofs)),
        fps=100.0,
        osim_path=str(_OSIM),
    )
    res = subj.to_motion(spec=spec)
    assert res.data["rigid_body_pos"].shape[0] == 4
    assert len(res.body_names) == len(spec.bodies)
    assert res.fps == 100.0
