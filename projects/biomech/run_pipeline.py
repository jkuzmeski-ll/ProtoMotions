# SPDX-License-Identifier: MIT

"""CLI: run the full real-data reconstruction + contact-calibration pipeline.

Wraps :func:`biomech.contact.pipeline.run_subject_pipeline`: loads a dynamic trial
(markers + split-belt GRF) and a static calibration trial, fits the Rajagopal skeleton
(IKInitializer seed -> MarkerFitter), exports the gold-standard motion, builds the
subject plantar geometry, registers the ground plane, and calibrates the hydroelastic
distributed-contact model against the measured vertical GRF. Prints a JSON report and
optionally writes it to disk.

Run from the repository root::

    python projects/biomech/run_pipeline.py \
        --trial "projects/data/S001/Trial 101.v3d.c3d" \
        --static "projects/data/S001/Cal 101.v3d.c3d" \
        --left-belt "projects/data/S001/LeftBelt101.txt" \
        --right-belt "projects/data/S001/RightBelt101.txt" \
        --window-len 40 --outer-iters 6 \
        --report projects/data/S001/Trial101_pipeline_report.json

Defaults point at ``projects/data/S001`` so it can be run with no arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `import biomech` work when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

_S001 = Path("projects/data/S001")
_OSIM = Path(__file__).resolve().parent / "models" / "rajagopal_data" / "Rajagopal2015.osim"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", type=Path, default=_S001 / "Trial 101.v3d.c3d")
    p.add_argument("--static", type=Path, default=_S001 / "Cal 101.v3d.c3d")
    p.add_argument("--left-belt", type=Path, default=_S001 / "LeftBelt101.txt")
    p.add_argument("--right-belt", type=Path, default=_S001 / "RightBelt101.txt")
    p.add_argument("--speedchange", type=Path, default=_S001 / "Speedchange101.txt")
    p.add_argument("--osim", type=Path, default=_OSIM)
    p.add_argument("--window-len", type=int, default=40)
    p.add_argument(
        "--phase", type=str, default=None, choices=("walk", "run", "all"),
        help="Restrict the window to a protocol phase (needs --speedchange).",
    )
    p.add_argument(
        "--window", type=int, nargs=2, default=None, metavar=("LO", "HI"),
        help="Explicit frame window; overrides --window-len auto-pick.",
    )
    p.add_argument("--outer-iters", type=int, default=6,
                   help="MarkerFitter outer iterations (speed vs accuracy).")
    p.add_argument("--fz-threshold", type=float, default=50.0)
    p.add_argument("--registration", type=str, default="percentile",
                   choices=("percentile", "flatfoot"),
                   help="Ground-plane registration mode.")
    p.add_argument("--objective", type=str, default="perframe",
                   choices=("perframe", "aggregate"),
                   help="Contact calibration objective (aggregate=robust on dynamic windows).")
    p.add_argument("--right-plate-x-sign", type=int, default=None, choices=(1, -1),
                   help="Which plate x_sign is the right foot (default: auto-detect from "
                        "kinematics; robust to captures whose right foot is on the -x plate).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip contact calibration (reconstruction + geometry only).")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional path to write the JSON report.")
    return p.parse_args()


def _opt(path: Path) -> Path | None:
    return path if path and Path(path).exists() else None


def build_report(res, args) -> dict:
    report: dict = {
        "window": list(res.window),
        "marker_rms_median_m": round(float(res.marker_rms_median), 6),
        "marker_rms_max_m": round(float(np.nanmax(res.marker_rms)), 6),
        "n_group_scales": int(res.group_scales.size),
        "feet": {},
    }
    for side, foot in res.feet.items():
        stance = foot.stance_mask
        entry: dict = {
            "body": foot.body,
            "ground_z_m": round(float(foot.ground_z), 6),
            "n_stance_frames": int(stance.sum()),
            "n_frames": int(stance.size),
        }
        if stance.any():
            entry["measured_fz_mean_N"] = round(
                float(np.mean(foot.measured_grf[stance, 2])), 3
            )
            entry["measured_fz_peak_N"] = round(
                float(np.max(foot.measured_grf[stance, 2])), 3
            )
        c = foot.calibration
        if c is not None:
            entry["calibration"] = {
                "vertical_rms_N": round(float(c.vertical_rms), 4),
                "force_rms_N": round(float(getattr(c, "force_rms", np.nan)), 4),
                "params": {
                    k: round(float(v), 6)
                    for k, v in vars(c.params).items()
                    if isinstance(v, (int, float))
                },
            }
        report["feet"][side] = entry
    return report


def main() -> None:
    args = parse_args()

    from biomech.contact.pipeline import run_subject_pipeline
    from biomech.fitting.marker_fitter import MarkerFitConfig
    from biomech.osim import parse_osim
    from biomech.session import load_session

    trial = load_session(
        c3d_path=args.trial,
        left_belt_path=_opt(args.left_belt),
        right_belt_path=_opt(args.right_belt),
        speedchange_path=_opt(args.speedchange),
    )
    static = load_session(c3d_path=args.static)
    spec = parse_osim(str(args.osim))

    window = tuple(args.window) if args.window is not None else None
    res = run_subject_pipeline(
        trial, static, spec,
        window=window,
        window_len=args.window_len,
        phase=args.phase,
        marker_config=MarkerFitConfig(outer_iters=args.outer_iters),
        right_plate_x_sign=args.right_plate_x_sign,
        fz_threshold=args.fz_threshold,
        registration=args.registration,
        objective=args.objective,
        device=args.device,
        calibrate=not args.no_calibrate,
    )

    report = build_report(res, args)
    print(json.dumps(report, indent=2))

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nWrote report to {args.report}")


if __name__ == "__main__":
    main()
