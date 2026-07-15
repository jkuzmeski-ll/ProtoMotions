# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI: load a capture session and print / save a Milestone-1 report.

Run from the repository root::

    python projects/biomech/load_capture.py \
        --c3d "projects/data/S001/Trial 101.v3d.c3d" \
        --left-belt "projects/data/S001/LeftBelt101.txt" \
        --right-belt "projects/data/S001/RightBelt101.txt" \
        --subject-mp "projects/data/S001/S001.mp" \
        --report projects/data/S001/Trial101_session_report.json

Defaults point at ``projects/data/S001`` so it can be run with no arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `import biomech` work when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biomech.session import load_session  # noqa: E402

_S001 = Path("projects/data/S001")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c3d", type=Path, default=_S001 / "Trial 101.v3d.c3d")
    p.add_argument("--left-belt", type=Path, default=_S001 / "LeftBelt101.txt")
    p.add_argument("--right-belt", type=Path, default=_S001 / "RightBelt101.txt")
    p.add_argument("--subject-mp", type=Path, default=_S001 / "S001.mp")
    p.add_argument("--belt-rate-hz", type=float, default=None)
    p.add_argument("--fz-threshold", type=float, default=20.0)
    p.add_argument("--subject-id", type=str, default=None)
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the JSON report.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    def _opt(path: Path) -> Path | None:
        return path if path and Path(path).exists() else None

    session = load_session(
        c3d_path=args.c3d,
        left_belt_path=_opt(args.left_belt),
        right_belt_path=_opt(args.right_belt),
        belt_rate_hz=args.belt_rate_hz,
        subject_mp_path=_opt(args.subject_mp),
        subject_id=args.subject_id,
        fz_threshold=args.fz_threshold,
    )

    report = session.report()
    print(json.dumps(report, indent=2))

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nWrote report to {args.report}")


if __name__ == "__main__":
    main()
