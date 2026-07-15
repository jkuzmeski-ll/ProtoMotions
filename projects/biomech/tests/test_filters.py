# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the zero-phase Butterworth preprocessing filter (kinematics + kinetics)."""

import numpy as np

from biomech.tests import SkipTest


def _require_scipy():
    try:
        import scipy.signal  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SkipTest(f"scipy required for Butterworth filtering: {exc}")


def test_lowpass_removes_high_freq_keeps_low_freq():
    _require_scipy()
    from biomech.io.filters import lowpass_dense

    fs = 100.0
    t = np.arange(0, 5.0, 1.0 / fs)
    low = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz, in-band
    high = 0.5 * np.sin(2 * np.pi * 40.0 * t)  # 40 Hz, above 20 Hz cutoff
    y = lowpass_dense(low + high, cutoff_hz=20.0, fs_hz=fs)
    # Ignore filter edge transients; interior should recover the 1 Hz component.
    interior = slice(50, -50)
    err = np.max(np.abs(y[interior] - low[interior]))
    assert err < 0.05, err


def test_lowpass_is_zero_phase():
    _require_scipy()
    from biomech.io.filters import lowpass_dense

    fs = 100.0
    t = np.arange(0, 5.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 2.0 * t)
    y = lowpass_dense(x, cutoff_hz=20.0, fs_hz=fs)
    # Zero-phase: peak of a clean in-band sine should not shift in time.
    interior = slice(60, len(t) - 60)
    lag = np.argmax(np.correlate(y[interior], x[interior], mode="full")) - (
        len(x[interior]) - 1
    )
    assert lag == 0, f"nonzero phase lag: {lag} samples"


def test_marker_filter_preserves_nan_gaps_and_shape():
    _require_scipy()
    from biomech.io.filters import filter_markers

    fs = 100.0
    F, M = 400, 3
    t = np.arange(F) / fs
    markers = np.empty((F, M, 3), dtype=np.float64)
    for m in range(M):
        for c in range(3):
            markers[:, m, c] = np.sin(2 * np.pi * 1.5 * t + m + c)
    # punch an occlusion gap into marker 1
    markers[100:130, 1, :] = np.nan

    out = filter_markers(markers, fs_hz=fs, cutoff_hz=20.0)
    assert out.shape == markers.shape
    # gaps preserved exactly
    assert np.all(np.isnan(out[100:130, 1, :]))
    # finite elsewhere
    assert np.isfinite(out[:100, 1, :]).all()
    assert np.isfinite(out[130:, 1, :]).all()
    # a clean in-band signal (no gap) is recovered in the interior
    assert np.max(np.abs(out[60:-60, 0, 0] - markers[60:-60, 0, 0])) < 0.05


def test_marker_filter_only_touches_finite_runs():
    _require_scipy()
    from biomech.io.filters import filter_markers

    F = 200
    markers = np.full((F, 1, 3), np.nan, dtype=np.float64)
    # a single short finite run shorter than the filter taps is left as-is
    markers[10:13, 0, :] = 1.0
    out = filter_markers(markers, fs_hz=100.0, cutoff_hz=20.0)
    assert np.allclose(out[10:13, 0, :], 1.0)
    assert np.all(np.isnan(out[:10, 0, :]))


def test_cutoff_above_nyquist_raises():
    _require_scipy()
    from biomech.io.filters import design_butter_lowpass

    try:
        design_butter_lowpass(cutoff_hz=60.0, fs_hz=100.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for cutoff above Nyquist")


def test_session_filter_metadata_and_bodyweight():
    """On real S001: filtering is recorded and GRF still integrates to ~body weight."""
    _require_scipy()
    from biomech.session import load_session, read_subject_mp
    from biomech.tests import SUBJECT_MP, TRIAL_C3D, require

    require(TRIAL_C3D)
    s = load_session(c3d_path=TRIAL_C3D, filter_cutoff_hz=20.0)
    assert s.filter_info is not None
    assert s.filter_info["cutoff_hz"] == 20.0
    assert s.filter_info["order"] == 4
    assert s.filter_info["markers_filtered"] is True
    assert s.filter_info["kinetics_filtered"] is True
    assert any("kinetics low-pass filtered" in w for w in s.warnings)

    if SUBJECT_MP.exists():
        meta = read_subject_mp(SUBJECT_MP)
        if "mass_kg" in meta:
            expected = meta["mass_kg"] * 9.81
            total = np.zeros(s.n_analog)
            for plate in s.force_plates:
                total = total + np.clip(plate.grf[:, 2], 0.0, None)
            mean_total = float(total.mean())
            assert 0.6 * expected < mean_total < 1.4 * expected


def test_disabling_filter_leaves_raw_markers():
    _require_scipy()
    from biomech.io.c3d import read_c3d
    from biomech.session import load_session
    from biomech.tests import TRIAL_C3D, require

    require(TRIAL_C3D)
    raw = read_c3d(require(TRIAL_C3D)).points
    s = load_session(c3d_path=TRIAL_C3D, filter_cutoff_hz=None)
    assert s.filter_info is None
    # unfiltered session markers equal the raw C3D points (NaN-aware compare)
    both_nan = np.isnan(raw) & np.isnan(s.markers)
    assert np.allclose(np.where(both_nan, 0.0, raw), np.where(both_nan, 0.0, s.markers))
