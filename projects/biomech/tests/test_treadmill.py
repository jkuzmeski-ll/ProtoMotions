# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Treadmill belt reader/resampler tests."""

import numpy as np

from biomech.io.treadmill import (
    PROTOCOL_ITEM_MAP,
    TreadmillProtocol,
    load_treadmill,
    read_belt_file,
    read_speedchange,
)
from biomech.tests import LEFT_BELT, RIGHT_BELT, SPEEDCHANGE, require


def test_belt_parses_and_ramps_to_target_speed():
    belt = read_belt_file(require(LEFT_BELT), "left", rate_hz=100.0)
    assert belt.n_samples > 50000
    assert belt.speed.min() >= 0.0
    assert abs(belt.speed.max() - 3.0) < 1e-3  # ramps to 3.0 m/s


def test_rate_inferred_from_reference_duration():
    require(LEFT_BELT)
    require(RIGHT_BELT)
    tm = load_treadmill(
        left_path=LEFT_BELT,
        right_path=RIGHT_BELT,
        rate_hz=None,
        reference_duration_s=178.44,  # 17844 point frames @ 100 Hz
    )
    assert tm is not None
    assert tm.rate_inferred
    # ~53224 samples over ~178 s -> roughly 300 Hz.
    assert 250.0 < tm.rate_hz < 350.0


def test_resample_endpoints_and_monotonic_time():
    belt = read_belt_file(require(LEFT_BELT), "left", rate_hz=300.0)
    t = belt.time_s()
    assert np.all(np.diff(t) > 0)
    # Resampling onto the belt's own timeline is (nearly) identity.
    resampled = belt.resample_to(t)
    assert np.allclose(resampled, belt.speed, atol=1e-6)
    # Out-of-range times clamp to endpoints.
    assert belt.resample_to(np.array([-1.0]))[0] == belt.speed[0]
    assert belt.resample_to(np.array([1e9]))[0] == belt.speed[-1]


def test_speedchange_named_events_and_windows():
    proto = read_speedchange(require(SPEEDCHANGE))
    assert isinstance(proto, TreadmillProtocol)
    # all six named boundaries present
    assert set(proto.events) == set(PROTOCOL_ITEM_MAP)
    # S001 protocol values (seconds): items 1,3,4,5,6,7
    assert abs(proto.event_time("START") - 0.0) < 1e-6
    assert abs(proto.event_time("WALK_START") - 14.69) < 1e-2
    assert abs(proto.event_time("WALK_END") - 74.69) < 1e-2
    assert abs(proto.event_time("RUN_START") - 84.70) < 1e-2
    assert abs(proto.event_time("RUN_END") - 144.70) < 1e-2
    assert abs(proto.event_time("END") - 174.71) < 1e-2
    # monotonic protocol
    assert (
        proto.event_time("START") < proto.event_time("WALK_START")
        < proto.event_time("WALK_END") < proto.event_time("RUN_START")
        < proto.event_time("RUN_END") < proto.event_time("END")
    )
    # phase windows in seconds
    assert proto.phase_window_s("walk") == (
        proto.event_time("WALK_START"), proto.event_time("WALK_END")
    )


def test_speedchange_phase_windows_in_frames():
    proto = read_speedchange(require(SPEEDCHANGE))
    # 100 Hz point rate, trial starts at t=0, 17844 frames
    lo, hi = proto.phase_window_frames("walk", rate_hz=100.0, n_frames=17844)
    assert lo == 1469
    assert hi == 7469
    lo, hi = proto.phase_window_frames("run", rate_hz=100.0, n_frames=17844)
    assert lo == 8470
    assert hi == 14470
    # clamping respects n_frames
    lo, hi = proto.phase_window_frames("all", rate_hz=100.0, n_frames=5000)
    assert hi == 5000


def test_speedchange_synthetic_item_map():
    import tempfile
    from pathlib import Path

    text = (
        "\tZ:\\some\\Trial.c3d\t\n"
        "\tSPEEDCHANGE\t\n"
        "\tMETRIC\t\n"
        "\tPROCESSED\t\n"
        "ITEM\t0\n"
        "1\t0\n"
        "2\t5.0\n"
        "3\t10.0\n"
        "4\t20.0\n"
        "5\t30.0\n"
        "6\t40.0\n"
        "7\t50.0\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Speedchange.txt"
        p.write_text(text)
        proto = read_speedchange(p)
    assert proto.item_times.size == 7
    assert proto.event_time("WALK_START") == 10.0
    assert proto.event_time("RUN_END") == 40.0
