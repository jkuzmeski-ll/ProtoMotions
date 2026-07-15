# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gold-standard biomechanics capture ingestion for ProtoMotions.

This package turns raw mocap + instrumented-treadmill captures into a single,
explicitly-framed, unit-checked, time-aligned :class:`CaptureSession`.

Milestone 1 scope (this module): local C3D + treadmill sync loader.
Everything runs locally with no third-party C3D dependency.
"""

from .frames import (
    MM_TO_M,
    NMM_TO_NM,
    UpAxis,
    CoordinateFrame,
    Frames,
    lab_to_world_rotation,
)
from .session import (
    CaptureSession,
    GaitEvent,
    load_session,
    read_subject_mp,
)

__all__ = [
    "MM_TO_M",
    "NMM_TO_NM",
    "UpAxis",
    "CoordinateFrame",
    "Frames",
    "lab_to_world_rotation",
    "CaptureSession",
    "GaitEvent",
    "load_session",
    "read_subject_mp",
]
