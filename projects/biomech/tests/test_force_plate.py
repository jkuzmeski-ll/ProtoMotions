# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force-plate tests: frames, GRF sign, and COP inside plate bounds."""

import numpy as np

from biomech.io.c3d import read_c3d
from biomech.io.force_plate import compute_force_plates
from biomech.tests import TRIAL_C3D, require


def _plates():
    return compute_force_plates(read_c3d(require(TRIAL_C3D)), fz_threshold=20.0)


def test_two_plates_axis_aligned():
    plates = _plates()
    assert len(plates) == 2
    for plate in plates:
        # Corners share a constant z (axis-aligned treadmill plate).
        assert np.ptp(plate.corners_world[:, 2]) < 1e-3
        # Geometry converted to meters (plate spans ~0.5 x 1.7 m).
        span = plate.corners_world.max(axis=0) - plate.corners_world.min(axis=0)
        assert 0.3 < span[0] < 0.7
        assert 1.0 < span[1] < 2.0


def test_grf_is_upward_on_subject_during_stance():
    plates = _plates()
    for plate in plates:
        # GRF (on subject) = -measured; vertical GRF should peak upward and be
        # body-weight-scale.
        peak_up = np.nanmax(plate.grf[:, 2])
        assert peak_up > 700.0
        # Measured vertical force is negative in stance (force onto plate).
        assert plate.force_measured[:, 2].min() < -700.0


def test_cop_lands_inside_plate_bounds():
    plates = _plates()
    for plate in plates:
        cop = plate.cop_world
        # COP is only physically trustworthy when the plate is well loaded;
        # near the contact threshold M/Fz divides by a tiny Fz and is noisy.
        well_loaded = plate.grf[:, 2] > 100.0
        valid = well_loaded & np.isfinite(cop[:, 0])
        assert valid.sum() > 1000  # plenty of stance samples
        xmin, ymin, _ = plate.corners_world.min(axis=0)
        xmax, ymax, _ = plate.corners_world.max(axis=0)
        cx = cop[valid, 0]
        cy = cop[valid, 1]
        # The bulk of the well-loaded COP must sit on the plate. Tails can cross
        # the shared inner edge (x=0) on a split-belt treadmill during weight
        # transfer / low-load phases, which is a real effect, not a bug.
        assert xmin <= np.median(cx) <= xmax
        assert ymin <= np.median(cy) <= ymax
        margin = 0.05
        inside = (
            (cx >= xmin - margin)
            & (cx <= xmax + margin)
            & (cy >= ymin - margin)
            & (cy <= ymax + margin)
        )
        assert inside.mean() > 0.9
        # COP z sits on the plate surface.
        assert np.allclose(cop[valid, 2], plate.surface_z)


def test_plates_split_by_x_sign():
    plates = _plates()
    signs = sorted(p.x_sign for p in plates)
    assert signs == [-1, 1]  # one belt at +x, one at -x


def test_cop_is_nan_during_swing():
    plates = _plates()
    for plate in plates:
        swing = np.abs(plate.force_measured[:, 2]) < plate.fz_threshold
        assert np.isnan(plate.cop_world[swing, 0]).all()
