# SPDX-License-Identifier: MIT

"""Tests for the kinematics->contact bridge (biomech.contact.kinematics, M5).

Validates that a foot body's world pose + spatial velocity can be sliced out of a
gold-standard ``MotionExportResult`` and fed to the elastic-foundation contact law:

- ``foot_trajectory_from_motion`` returns the correct per-body trajectory,
- driving a foot into the ground produces a nonzero GRF/COP for that foot,
- a foot held clearly above the ground carries no load,
- ``evaluate_both_feet_from_motion`` returns a prediction per foot.

Needs torch (for ``build_motion``); skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.contact.elastic_foundation import ElasticFoundationParams, sample_flat_sole
from biomech.contact.kinematics import (
    evaluate_both_feet_from_motion,
    evaluate_foot_contact_from_motion,
    foot_trajectory_from_motion,
)
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


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"torch not available: {exc}")


def _neutral_clip(spec, n=4, fps=100.0):
    from biomech.export.motion import build_motion

    q = np.zeros(spec.num_dofs)
    Q = np.tile(q, (n, 1))
    return build_motion(spec, Q, fps=fps)


def test_foot_trajectory_slice_shapes():
    _require_torch()
    spec = _spec()
    res = _neutral_clip(spec, n=6)
    pos, quat, linvel, angvel = foot_trajectory_from_motion(res, "calcn_r")
    assert pos.shape == (6, 3)
    assert quat.shape == (6, 4)
    assert linvel.shape == (6, 3)
    assert angvel.shape == (6, 3)
    # slice matches the raw per-body field
    bi = res.body_names.index("calcn_r")
    ref = np.asarray(res.data["rigid_body_pos"])[:, bi, :]
    assert np.allclose(pos, ref, atol=1e-5)


def test_unknown_body_raises():
    _require_torch()
    res = _neutral_clip(_spec(), n=2)
    try:
        foot_trajectory_from_motion(res, "not_a_body")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown body")


def test_foot_in_ground_produces_load():
    _require_torch()
    spec = _spec()
    res = _neutral_clip(spec, n=4)
    # find the right foot's world height in the neutral clip and set the ground just
    # above it so the flat sole penetrates.
    pos, _, _, _ = foot_trajectory_from_motion(res, "calcn_r")
    ground_z = float(pos[0, 2]) + 0.01  # 1 cm of penetration

    sole = sample_flat_sole(0.2, 0.09, 8, 4)
    params = ElasticFoundationParams()
    pred = evaluate_foot_contact_from_motion(
        res, "calcn_r", sole, params, ground_z=ground_z, backend="numpy"
    )
    assert np.all(pred.total_normal > 0.0)
    assert np.all(pred.grf[:, 2] > 0.0)
    # loaded frames get a finite COP near the foot in x/y
    assert np.all(np.isfinite(pred.cop))


def test_foot_above_ground_no_load():
    _require_torch()
    spec = _spec()
    res = _neutral_clip(spec, n=3)
    pos, _, _, _ = foot_trajectory_from_motion(res, "calcn_r")
    ground_z = float(pos[0, 2]) - 0.2  # ground well below the foot
    sole = sample_flat_sole(0.2, 0.09, 6, 4)
    pred = evaluate_foot_contact_from_motion(
        res, "calcn_r", sole, ground_z=ground_z, backend="numpy"
    )
    assert np.allclose(pred.total_normal, 0.0)
    assert np.all(np.isnan(pred.cop))


def test_both_feet():
    _require_torch()
    spec = _spec()
    res = _neutral_clip(spec, n=3)
    posr, _, _, _ = foot_trajectory_from_motion(res, "calcn_r")
    posl, _, _, _ = foot_trajectory_from_motion(res, "calcn_l")
    ground_z = float(min(posr[0, 2], posl[0, 2])) + 0.01
    sole = sample_flat_sole(0.2, 0.09, 6, 4)
    right, left = evaluate_both_feet_from_motion(
        res, sole, sole, ground_z=ground_z, backend="numpy"
    )
    assert right.grf.shape == (3, 3)
    assert left.grf.shape == (3, 3)
    assert np.all(right.total_normal > 0.0)
    assert np.all(left.total_normal > 0.0)
