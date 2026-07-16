# SPDX-License-Identifier: MIT

"""Tests for OpenSim-style foot marker placement (fitting/marker_placement.py).

Verifies that placing the rich S001 foot markers from the static trial (a) adds the
missing calcaneus-cluster / 1st-met / hallux markers on the anatomically-correct bodies,
(b) reproduces the measured static foot markers via FK (the placement is self-consistent),
and (c) makes the MTP angle observable — a marker on the ``toes`` segment moves with
``mtp_angle`` while the ``calcn`` markers do not.

Skipped if the S001 static capture or torch is unavailable. No pytest dependency:
run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.tests import CAL_C3D, SkipTest, require

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"torch not available: {exc}")


def _light_config():
    from biomech.fitting.ik import MarkerIKConfig
    from biomech.fitting.marker_fitter import MarkerFitConfig

    return MarkerFitConfig(
        outer_iters=2,
        inner=MarkerIKConfig(max_iters=15),
        inner_first=MarkerIKConfig(max_iters=40),
        final_inner=MarkerIKConfig(max_iters=15),
    )


def _place():
    require(CAL_C3D)
    _require_torch()
    from biomech.fitting.marker_placement import place_foot_markers
    from biomech.osim import parse_osim
    from biomech.session import load_session

    static = load_session(str(CAL_C3D), filter_cutoff_hz=None)
    spec = parse_osim(str(_OSIM))
    placement = place_foot_markers(
        spec, static, marker_config=_light_config(),
        device="cpu", frame_range=(0, 6),
    )
    return spec, static, placement


def test_placement_adds_expected_foot_markers():
    spec, _static, placement = _place()

    expect_added = {
        "RCAL2", "RCAL3", "RMT1", "RTOE_TIP",
        "LCAL2", "LCAL3", "LMT1", "LTOE_TIP",
    }
    assert set(placement.added) == expect_added
    assert set(placement.reseated) == {
        "RCAL", "RTOE", "RMT5", "LCAL", "LTOE", "LMT5",
    }

    # New markers exist on the anatomically-correct bodies.
    body_of = {m.name: m.body for m in spec.markers}
    assert body_of["RTOE_TIP"] == "toes_r" and body_of["LTOE_TIP"] == "toes_l"
    for name in ("RCAL2", "RCAL3", "RMT1"):
        assert body_of[name] == "calcn_r"

    # Offsets are finite and physically plausible (within ~30 cm of the body origin).
    for name, off in placement.offsets.items():
        assert np.all(np.isfinite(off)), name
        assert np.linalg.norm(off) < 0.3, (name, off)

    # Placement residual (rigid-cluster spread) is small on the static trial.
    for name, resid in placement.residual_mm.items():
        assert resid < 20.0, (name, resid)


def test_placement_roundtrip_reproduces_static_markers():
    """FK of the placed markers at the fit pose reproduces the measured static positions."""
    spec, static, placement = _place()

    from biomech.fitting.marker_placement import _capture_positions_opensim
    from biomech.skeleton.skeleton import WarpSkeleton

    skel = WarpSkeleton(spec, device="cpu")
    names = skel.marker_names()
    _world, markers = skel.forward(placement.poses, placement.group_scales)  # (Fw, M, 3)
    idx = {n: i for i, n in enumerate(names)}
    lo, hi = placement.window

    capture_of = {
        "RCAL2": "RHEE2", "RCAL3": "RHEE3", "RMT1": "RMTH1", "RTOE_TIP": "RHLX",
        "RTOE": "RTOE",
    }
    for model_name, label in capture_of.items():
        obs = _capture_positions_opensim(static, label)[lo:hi]  # (Fw, 3)
        fk = markers[:, idx[model_name], :]
        err = np.linalg.norm(fk - obs, axis=1)
        # Real (noisy) markers on a rigid segment: mean reprojection well under 2.5 cm.
        assert np.nanmean(err) < 0.025, (model_name, float(np.nanmean(err)))


def test_ankle_neutral_preserves_world_geometry():
    """Re-zeroing the ankle at a static neutral must not move any body in the world."""
    _require_torch()
    from biomech.fitting.marker_placement import register_ankle_neutral
    from biomech.osim import parse_osim
    from biomech.skeleton.skeleton import WarpSkeleton

    spec = parse_osim(str(_OSIM))
    dof = spec.dof_index_map()
    ndof = spec.num_dofs

    rng = np.random.default_rng(0)
    q = rng.uniform(-0.25, 0.25, size=(3, ndof))

    skel0 = WarpSkeleton(spec, device="cpu")
    world0, markers0 = skel0.forward(q)

    # pretend the static trial sat at a constant ankle plantarflexion offset
    static = np.zeros((10, ndof), dtype=np.float64)
    static[:, dof["ankle_angle_r"]] = -0.18
    static[:, dof["ankle_angle_l"]] = -0.20
    offsets = register_ankle_neutral(spec, static)
    assert set(offsets) == {"ankle_angle_r", "ankle_angle_l"}

    # apply q' = q - off on the ankle dofs; every body pose must be unchanged
    q2 = q.copy()
    q2[:, dof["ankle_angle_r"]] -= offsets["ankle_angle_r"]
    q2[:, dof["ankle_angle_l"]] -= offsets["ankle_angle_l"]
    skel1 = WarpSkeleton(spec, device="cpu")
    world1, markers1 = skel1.forward(q2)

    assert np.allclose(world0, world1, atol=1e-9), np.abs(world0 - world1).max()
    assert np.allclose(markers0, markers1, atol=1e-9)


def test_placement_registers_ankle_neutral():
    """The full placement re-zeros the ankle by the subject's static plantarflexion."""
    _spec, _static, placement = _place()
    assert set(placement.ankle_neutral) == {"ankle_angle_r", "ankle_angle_l"}
    # S001 static plantarflexion is ~ -10 deg (PiG RStaticPlantFlex ~ 9-10 deg).
    for coord, off in placement.ankle_neutral.items():
        assert -0.30 < off < -0.05, (coord, off)


def test_mtp_becomes_observable():
    """A toes-segment marker moves with mtp_angle; calcn markers do not."""
    spec, _static, _placement = _place()

    from biomech.skeleton.skeleton import WarpSkeleton

    skel = WarpSkeleton(spec, device="cpu")
    names = skel.marker_names()
    idx = {n: i for i, n in enumerate(names)}
    dof = spec.dof_index_map()

    q0 = np.zeros((1, spec.num_dofs), dtype=np.float64)
    q1 = q0.copy()
    q1[0, dof["mtp_angle_r"]] = 0.3  # ~17 deg toe extension

    _w0, m0 = skel.forward(q0)
    _w1, m1 = skel.forward(q1)

    toe = np.linalg.norm(m1[0, idx["RTOE_TIP"]] - m0[0, idx["RTOE_TIP"]])
    calcn = np.linalg.norm(m1[0, idx["RMT1"]] - m0[0, idx["RMT1"]])
    assert toe > 0.01, toe          # toes marker clearly moves with MTP
    assert calcn < 1e-9, calcn      # calcn marker is unaffected by MTP
