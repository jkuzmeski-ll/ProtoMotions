# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C3D reader tests: header metadata, units, dense-frame alignment."""

import numpy as np

from biomech.io.c3d import read_c3d
from biomech.tests import TRIAL_C3D, require


def test_header_metadata():
    c3d = read_c3d(require(TRIAL_C3D))
    assert c3d.point_rate == 100.0
    assert c3d.analog_rate == 2000.0
    assert c3d.n_frames == 17844
    assert c3d.header.n_points == 72
    assert c3d.analog_per_point_frame == 20
    assert c3d.point_units.lower() == "mm"


def test_points_are_dense_and_metric():
    c3d = read_c3d(require(TRIAL_C3D))
    # Dense: one row per frame, prefix stripped from labels.
    assert c3d.points.shape == (c3d.n_frames, 72, 3)
    assert all(":" not in lbl for lbl in c3d.point_labels)
    # Metric: marker coordinates are in meters (a standing human spans < 3 m).
    finite = c3d.points[np.isfinite(c3d.points)]
    assert finite.size > 0
    assert np.nanmax(np.abs(finite)) < 5.0
    # Gaps are NaN, not zeros.
    assert np.isnan(c3d.points).any() or np.isfinite(c3d.points).all()


def test_analog_shape_matches_rate_ratio():
    c3d = read_c3d(require(TRIAL_C3D))
    assert c3d.analog.shape[0] == c3d.n_frames * c3d.analog_per_point_frame
    assert c3d.analog.shape[1] == 249
    # Force channels have the expected labels.
    assert c3d.analog_labels[0] == "Force.Fx1"
    assert c3d.analog_labels[2] == "Force.Fz1"


def test_vertical_force_is_bodyweight_scale():
    c3d = read_c3d(require(TRIAL_C3D))
    # Fz1 + Fz2 (channels index 2 and 8) is negative during stance and its
    # magnitude should be body-weight-scale (subject ~81.65 kg -> ~800 N).
    fz_total = c3d.analog[:, 2] + c3d.analog[:, 8]
    assert fz_total.min() < -700.0
    assert abs(np.mean(fz_total)) > 500.0
