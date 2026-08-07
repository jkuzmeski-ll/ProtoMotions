# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal test runner for the biomech package (no pytest dependency).

Usage (from the repository root)::

    python projects/biomech/run_tests.py

Discovers ``test_*`` functions in ``projects/biomech/tests/test_*.py``, runs
each, and reports pass / skip / fail. Compatible with pytest too (the test
functions are plain ``assert``-based), but this runner needs nothing installed.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Make `import biomech` resolve when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biomech.tests import SkipTest  # noqa: E402

TEST_MODULES = [
    "biomech.tests.test_c3d",
    "biomech.tests.test_force_plate",
    "biomech.tests.test_filters",
    "biomech.tests.test_treadmill",
    "biomech.tests.test_session",
    "biomech.tests.test_simmspline",
    "biomech.tests.test_osim_parser",
    "biomech.tests.test_skeleton_fk",
    "biomech.tests.test_closed_form",
    "biomech.tests.test_marker_ik",
    "biomech.tests.test_ik_initializer",
    "biomech.tests.test_marker_fitter",
    "biomech.tests.test_marker_fixer",
    "biomech.tests.test_report",
    "biomech.tests.test_priors",
    "biomech.tests.test_mjcf_export",
    "biomech.tests.test_motion_export",
    "biomech.tests.test_subject",
    "biomech.tests.test_protomotions_bridge",
    "biomech.tests.test_tm2og",
    "biomech.tests.test_dynamics_fitter",
    "biomech.tests.test_elastic_foundation",
    "biomech.tests.test_contact_kinematics",
    "biomech.tests.test_contact_calibration",
    "biomech.tests.test_stance",
    "biomech.tests.test_hydroelastic",
    "biomech.tests.test_marker_map",
    "biomech.tests.test_marker_placement",
    "biomech.tests.test_cluster_collapse",
    "biomech.tests.test_s001_end_to_end",
    "biomech.tests.test_foot_geometry",
    "biomech.tests.test_contact_pipeline",
    "biomech.tests.test_forward_sim",
    "biomech.tests.test_tracking",
    "biomech.tests.test_unified_pipeline",
]


def main() -> int:
    passed = skipped = failed = 0
    failures = []

    for mod_name in TEST_MODULES:
        module = importlib.import_module(mod_name)
        tests = sorted(
            name for name in dir(module) if name.startswith("test_")
        )
        for name in tests:
            fn = getattr(module, name)
            if not callable(fn):
                continue
            label = f"{mod_name}.{name}"
            try:
                fn()
            except SkipTest as exc:
                skipped += 1
                print(f"SKIP  {label}  ({exc})")
            except Exception:  # noqa: BLE001
                failed += 1
                failures.append((label, traceback.format_exc()))
                print(f"FAIL  {label}")
            else:
                passed += 1
                print(f"PASS  {label}")

    print("\n" + "=" * 60)
    print(f"passed={passed} skipped={skipped} failed={failed}")
    for label, tb in failures:
        print("\n" + "-" * 60)
        print(f"FAILURE: {label}")
        print(tb)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
