# SPDX-License-Identifier: MIT

"""Tests for the real-data reconstruction + contact-calibration pipeline (M-pipeline).

Unit-tests the ground-registration + belt-mapping helpers on synthetic data, and runs
the whole pipeline on the real S001 capture (skipped if data/torch absent):

    load -> observations -> IKInitializer + MarkerFitter -> build_motion -> subject sole
      -> per-stance ground registration -> calibrate vertical GRF (hydroelastic k, alpha)

The real-data assertions are deliberately loose on absolutes but do check the two things
that matter: the full-fit marker RMS is much better than the initializer alone (cm-level),
and the *vertical* GRF calibration lands close to the measured Fz (mu is not calibrated
from planted stance by design).

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.contact.elastic_foundation import sample_flat_sole
from biomech.contact.pipeline import (
    register_ground_z,
    sole_world_z,
)
from biomech.tests import (
    CAL_C3D,
    LEFT_BELT,
    RIGHT_BELT,
    SkipTest,
    TRIAL_C3D,
    require,
)

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_QID = np.array([0.0, 0.0, 0.0, 1.0])


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"torch not available: {exc}")


def test_sole_world_z_identity():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)  # points in z=0 plane
    F = 3
    pos = np.zeros((F, 3))
    pos[:, 2] = [0.1, 0.2, 0.3]
    quat = np.tile(_QID, (F, 1))
    zsole = sole_world_z(sole, pos, quat)
    assert zsole.shape == (F, sole.n)
    # identity rotation: world z == body-origin z (sole is at body z=0)
    assert np.allclose(zsole[0], 0.1)
    assert np.allclose(zsole[1], 0.2)


def test_register_ground_z_percentile():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    F = 10
    pos = np.zeros((F, 3))
    # foot sits at z=0 for most frames, with a couple of noisy deep dips
    pos[:, 2] = 0.0
    pos[3, 2] = -0.05  # noisy deep frame
    pos[7, 2] = -0.04
    quat = np.tile(_QID, (F, 1))
    stance = np.ones(F, dtype=bool)
    gz = register_ground_z(sole, pos, quat, stance, penetration=0.005,
                           contact_percentile=80.0)
    # the deep outliers must NOT drag the plane far down; it should sit near z=0 + pen
    assert -0.01 < gz < 0.02
    # ground is above the typical sole height so typical frames make contact
    assert gz > 0.0


def test_pipeline_real_s001():
    require(TRIAL_C3D)
    require(CAL_C3D)
    _require_torch()

    from biomech.contact.pipeline import run_subject_pipeline
    from biomech.fitting.marker_fitter import MarkerFitConfig
    from biomech.osim import parse_osim
    from biomech.session import load_session

    trial = load_session(
        str(TRIAL_C3D),
        left_belt_path=str(LEFT_BELT) if LEFT_BELT.exists() else None,
        right_belt_path=str(RIGHT_BELT) if RIGHT_BELT.exists() else None,
    )
    static = load_session(str(CAL_C3D))
    spec = parse_osim(str(_OSIM))

    res = run_subject_pipeline(
        trial, static, spec, window_len=40,
        marker_config=MarkerFitConfig(outer_iters=6),
    )

    # reconstruction: full fit is cm-level (vs ~10 cm initializer-only)
    assert res.marker_rms_median < 0.03
    assert np.all(np.isfinite(res.group_scales))

    # both feet reconstructed + measured GRF present
    assert set(res.feet.keys()) == {"R", "L"}
    for side, foot in res.feet.items():
        assert foot.measured_grf.shape[0] == 40
        # this capture opens with quiet standing -> both belts loaded
        assert foot.stance_mask.any()
        c = foot.calibration
        assert c is not None
        # calibrated stiffness/dissipation are physical
        assert c.params.k_bed > 1e5
        assert c.params.hc_alpha >= 0.0
        # vertical GRF fit is close to the measured Fz over the window
        meas_mean = float(np.mean(foot.measured_grf[foot.stance_mask, 2]))
        assert meas_mean > 50.0
        assert c.vertical_rms < 0.1 * meas_mean  # < 10% of mean vertical load
