# SPDX-License-Identifier: MIT

"""CLI for the repeatable C3D-to-ProtoMotions biomechanics pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biomech.pipeline import (  # noqa: E402
    PipelineConfig,
    PipelineQualityError,
    QualityThresholds,
    run_c3d_to_protomotions,
)


_S001 = Path("projects/data/S001")
_OSIM = Path(__file__).resolve().parent / "models" / "rajagopal_data" / "Rajagopal2015.osim"


def _boolean_pair(parser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=_S001 / "Trial 101.v3d.c3d")
    parser.add_argument("--static", type=Path, default=_S001 / "Cal 101.v3d.c3d")
    parser.add_argument("--osim", type=Path, default=_OSIM)
    parser.add_argument("--left-belt", type=Path, default=_S001 / "LeftBelt101.txt")
    parser.add_argument("--right-belt", type=Path, default=_S001 / "RightBelt101.txt")
    parser.add_argument("--speedchange", type=Path, default=_S001 / "Speedchange101.txt")
    parser.add_argument("--subject-mp", type=Path, default=_S001 / "S001.mp")
    parser.add_argument("--subject-id")
    parser.add_argument("--subject-mass", type=float)
    parser.add_argument("--allow-model-mass", action="store_true")
    parser.add_argument("--belt-rate-hz", type=float)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/biomech"))
    parser.add_argument("--phase", choices=("walk", "run", "all"), default="walk")
    parser.add_argument("--window", type=int, nargs=2, metavar=("LO", "HI"))
    parser.add_argument("--frames", type=int, help="Best-visibility subset of the phase")
    parser.add_argument("--calibration-phase", choices=("walk", "run", "all"), default="walk")
    parser.add_argument("--calibration-window", type=int, nargs=2, metavar=("LO", "HI"))
    parser.add_argument("--calibration-frames", type=int, default=60)
    parser.add_argument("--placement-frames", type=int, default=60)
    parser.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--outer-iters", type=int, default=15)
    parser.add_argument("--ik-iters", type=int, default=80)
    parser.add_argument(
        "--marker-weights",
        choices=("robust-anatomical", "balanced-anatomical", "uniform"),
        default="robust-anatomical",
    )
    parser.add_argument("--travel-direction", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--fz-threshold", type=float, default=50.0)
    parser.add_argument("--contact-height-mm", type=float, default=20.0)
    parser.add_argument("--contact-load-fraction", type=float, default=0.0)
    parser.add_argument("--right-plate-x-sign", type=int, choices=(-1, 1))
    parser.add_argument("--bone-meshes", action="store_true")
    parser.add_argument(
        "--anthropometric-prior",
        action="store_true",
        help="Experimental .mp segment-length scale prior; disabled by default",
    )
    parser.add_argument("--dynamics-diagnostics", action="store_true")
    parser.add_argument("--max-marker-rms-mm", type=float, default=25.0)
    parser.add_argument("--max-anatomical-rms-mm", type=float, default=12.0)
    parser.add_argument("--max-delivered-marker-rms-mm", type=float, default=35.0)
    parser.add_argument("--max-delivered-anatomical-rms-mm", type=float, default=22.0)
    parser.add_argument("--min-visible-markers", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and verify scientific equivalence to an immutable bundle",
    )
    _boolean_pair(parser, "clean-markers", True, "Reject spikes/outliers and fill short gaps")
    _boolean_pair(parser, "bilateral-scale-symmetry", True, "Share left/right segment scales")
    _boolean_pair(parser, "collapse-clusters", True, "Collapse soft-tissue marker clusters")
    _boolean_pair(parser, "enrich-foot-markers", True, "Use static foot placement and MTP")
    _boolean_pair(parser, "treadmill-to-overground", True, "Apply measured belt displacement")
    _boolean_pair(parser, "ground-register", True, "Place measured stance sole on z=0")
    _boolean_pair(parser, "measured-contacts", True, "Bake measured GRF-gated foot contacts")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        trial_c3d=args.trial,
        static_c3d=args.static,
        osim_path=args.osim,
        left_belt=args.left_belt,
        right_belt=args.right_belt,
        speedchange=args.speedchange,
        subject_mp=args.subject_mp,
        subject_id=args.subject_id,
        subject_mass_kg=args.subject_mass,
        allow_model_mass=args.allow_model_mass,
        belt_rate_hz=args.belt_rate_hz,
        output_root=args.output_root,
        phase=args.phase,
        window=tuple(args.window) if args.window else None,
        max_frames=args.frames,
        calibration_phase=args.calibration_phase,
        calibration_window=(tuple(args.calibration_window) if args.calibration_window else None),
        calibration_frames=args.calibration_frames,
        placement_frames=args.placement_frames,
        clean_markers=args.clean_markers,
        bilateral_scale_symmetry=args.bilateral_scale_symmetry,
        collapse_clusters=args.collapse_clusters,
        enrich_foot_markers=args.enrich_foot_markers,
        marker_weight_profile=args.marker_weights,
        anthropometric_prior=args.anthropometric_prior,
        outer_iters=args.outer_iters,
        ik_iters=args.ik_iters,
        chunk_size=args.chunk_size,
        device=args.device,
        treadmill_to_overground=args.treadmill_to_overground,
        travel_direction=(tuple(args.travel_direction) if args.travel_direction else None),
        ground_register=args.ground_register,
        measured_contacts=args.measured_contacts,
        bone_meshes=args.bone_meshes,
        fz_threshold=args.fz_threshold,
        contact_height_m=args.contact_height_mm * 1.0e-3,
        contact_load_fraction=args.contact_load_fraction,
        contact_load_floor_n=args.fz_threshold,
        right_plate_x_sign=args.right_plate_x_sign,
        dynamics_diagnostics=args.dynamics_diagnostics,
        quality=QualityThresholds(
            max_marker_rms_m=args.max_marker_rms_mm * 1.0e-3,
            max_anatomical_marker_rms_m=args.max_anatomical_rms_mm * 1.0e-3,
            max_delivered_marker_rms_m=args.max_delivered_marker_rms_mm * 1.0e-3,
            max_delivered_anatomical_marker_rms_m=(
                args.max_delivered_anatomical_rms_mm * 1.0e-3
            ),
            min_visible_markers=args.min_visible_markers,
        ),
        force=args.force,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_c3d_to_protomotions(config_from_args(args))
    except PipelineQualityError as exc:
        print(
            json.dumps(
                {"status": "quality_failed", "failures": exc.failures}, indent=2
            ),
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2), file=sys.stderr)
        return 2
    summary = {
        "bundle": str(result.bundle_dir),
        "manifest": str(result.manifest_path),
        "motion": str(result.motion_path),
        "fingerprint": result.fingerprint,
        "reused": result.reused,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
