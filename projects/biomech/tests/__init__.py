# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the biomech capture loader.

No pytest dependency is required: run ``python projects/biomech/run_tests.py``.
Tests that need the S001 capture raise :class:`SkipTest` if the data is absent.
"""

from pathlib import Path


class SkipTest(Exception):
    """Raised to signal a test should be skipped (e.g. missing data)."""


# Repository root is three levels up from this file:
# <root>/projects/biomech/tests/__init__.py
REPO_ROOT = Path(__file__).resolve().parents[3]
S001 = REPO_ROOT / "projects" / "data" / "S001"
TRIAL_C3D = S001 / "Trial 101.v3d.c3d"
CAL_C3D = S001 / "Cal 101.v3d.c3d"
LEFT_BELT = S001 / "LeftBelt101.txt"
RIGHT_BELT = S001 / "RightBelt101.txt"
SUBJECT_MP = S001 / "S001.mp"
SPEEDCHANGE = S001 / "Speedchange101.txt"


def require(path: Path) -> Path:
    if not path.exists():
        raise SkipTest(f"missing data file: {path}")
    return path
