# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end capture-session tests: alignment, frames, physical sanity."""

import numpy as np

from biomech.frames import UpAxis
from biomech.session import load_session, read_subject_mp
from biomech.tests import (
    LEFT_BELT,
    RIGHT_BELT,
    SPEEDCHANGE,
    SUBJECT_MP,
    TRIAL_C3D,
    require,
)


def _session():
    require(TRIAL_C3D)
    return load_session(
        c3d_path=TRIAL_C3D,
        left_belt_path=LEFT_BELT if LEFT_BELT.exists() else None,
        right_belt_path=RIGHT_BELT if RIGHT_BELT.exists() else None,
        subject_mp_path=SUBJECT_MP if SUBJECT_MP.exists() else None,
        speedchange_path=SPEEDCHANGE if SPEEDCHANGE.exists() else None,
    )


def test_timelines_are_consistent():
    s = _session()
    assert s.n_frames == 17844
    assert s.n_analog == s.n_frames * s.analog_per_point_frame
    # Analog and point timelines share t=0 and cover the same duration.
    assert abs(s.t_point[0] - s.t_analog[0]) < 1e-9
    assert np.all(np.diff(s.t_point) > 0)
    assert np.all(np.diff(s.t_analog) > 0)
    dt = 1.0 / s.point_rate
    assert np.allclose(np.diff(s.t_point), dt)


def test_frames_metadata_is_explicit_and_z_up():
    s = _session()
    assert s.frames.lab_up_axis == UpAxis.Z
    assert s.frames.world_up_axis == UpAxis.Z
    assert s.frames.length_unit == "m"
    assert s.frames.force_unit == "N"
    assert s.frames.moment_unit == "N*m"
    # lab == world for this Z-up capture.
    assert np.allclose(s.frames.lab_to_world, np.eye(3))


def test_belt_resampled_onto_both_timelines():
    s = _session()
    if s.treadmill is None:
        return
    for side in ("left", "right"):
        assert s.belt_speed_point[side].shape[0] == s.n_frames
        assert s.belt_speed_analog[side].shape[0] == s.n_analog
        assert np.nanmax(s.belt_speed_point[side]) <= 3.0 + 1e-3
    # Belt-rate inference should be flagged in warnings.
    assert any("belt sample rate inferred" in w for w in s.warnings)


def test_total_vertical_grf_matches_bodyweight():
    s = _session()
    meta = read_subject_mp(SUBJECT_MP) if SUBJECT_MP.exists() else {}
    if "mass_kg" not in meta:
        return
    expected = meta["mass_kg"] * 9.81
    # Sum upward GRF across both belts, averaged over the recording. During
    # steady locomotion mean total vertical GRF ~= body weight.
    total = np.zeros(s.n_analog)
    for plate in s.force_plates:
        total = total + np.clip(plate.grf[:, 2], 0.0, None)
    mean_total = float(total.mean())
    assert 0.6 * expected < mean_total < 1.4 * expected


def test_forces_downsample_to_point_timeline():
    s = _session()
    forces = s.forces_on_point_timeline()
    for plate in s.force_plates:
        assert forces[f"plate{plate.index}_grf"].shape == (s.n_frames, 3)
        assert forces[f"plate{plate.index}_cop"].shape == (s.n_frames, 3)


def test_protocol_phase_windows_align_with_timeline():
    s = _session()
    require(SPEEDCHANGE)
    assert s.protocol is not None
    # walk phase maps to a mid-trial frame window, within bounds and ordered.
    lo, hi = s.phase_window("walk")
    assert 0 <= lo < hi <= s.n_frames
    assert lo == 1469 and hi == 7469
    # the reported protocol carries the six named boundaries.
    rep = s.report()
    assert set(rep["protocol"]["events"]) == {
        "START", "WALK_START", "WALK_END", "RUN_START", "RUN_END", "END",
    }
