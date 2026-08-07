# SPDX-License-Identifier: MIT

"""Repeatable C3D-to-ProtoMotions biomechanics pipeline.

This module connects the validated capture, marker fitting, treadmill conversion,
subject-specific model export, ground registration, measured-contact labeling, and
ProtoMotions validation paths. Dynamics fitting is deliberately diagnostic-only until
the measured free-moment convention and inertial-parameter persistence are validated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


PIPELINE_SCHEMA = "biomech.c3d_to_protomotions"
PIPELINE_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FINGERPRINT_SOURCE_ROOTS = (
    "projects/biomech/contact",
    "projects/biomech/export",
    "projects/biomech/fitting",
    "projects/biomech/io",
    "projects/biomech/osim",
    "projects/biomech/skeleton",
    "protomotions/components",
    "protomotions/envs",
    "protomotions/robot_configs",
    "protomotions/simulator",
)
_FINGERPRINT_SOURCE_FILES = (
    "projects/biomech/__init__.py",
    "projects/biomech/frames.py",
    "projects/biomech/pipeline.py",
    "projects/biomech/session.py",
)
_DEFAULT_OSIM = (
    Path(__file__).resolve().parent
    / "models"
    / "rajagopal_data"
    / "Rajagopal2015.osim"
)


@dataclass
class QualityThresholds:
    """Fail-closed reconstruction and export quality gates."""

    max_marker_rms_m: float = 0.025
    max_anatomical_marker_rms_m: float = 0.012
    max_delivered_marker_rms_m: float = 0.035
    max_delivered_anatomical_marker_rms_m: float = 0.022
    max_marker_degradation_m: float = 0.010
    max_marker_degradation_fraction: float = 0.40
    max_stance_foot_speed_mps: float = 0.10
    max_stance_foot_speed_p95_mps: float = 0.20
    min_scale: float = 0.5
    max_scale: float = 1.6
    scale_bound_margin: float = 1.0e-4
    min_visible_markers: int = 8
    joint_limit_tolerance: float = 1.0e-6
    quaternion_norm_tolerance: float = 1.0e-4
    subject_mass_tolerance_kg: float = 1.0e-3


@dataclass
class PipelineConfig:
    """Configuration for :func:`run_c3d_to_protomotions`."""

    trial_c3d: Path
    static_c3d: Path
    output_root: Path
    osim_path: Path = _DEFAULT_OSIM
    left_belt: Optional[Path] = None
    right_belt: Optional[Path] = None
    speedchange: Optional[Path] = None
    subject_mp: Optional[Path] = None
    subject_id: Optional[str] = None
    subject_mass_kg: Optional[float] = None
    allow_model_mass: bool = False
    belt_rate_hz: Optional[float] = None
    filter_cutoff_hz: Optional[float] = 20.0
    filter_order: int = 4
    marker_profile: str = "s001-pig"
    phase: str = "walk"
    window: Optional[tuple[int, int]] = None
    max_frames: Optional[int] = None
    calibration_phase: str = "walk"
    calibration_window: Optional[tuple[int, int]] = None
    calibration_frames: int = 60
    placement_frames: int = 60
    bilateral_scale_symmetry: bool = True
    collapse_clusters: bool = True
    enrich_foot_markers: bool = True
    clean_markers: bool = True
    marker_weight_profile: str = "robust-anatomical"
    anthropometric_prior: bool = False
    placement_outer_iters: int = 6
    outer_iters: int = 15
    ik_iters: int = 80
    chunk_size: int = 1000
    device: str = "cpu"
    coupled_knee: str = "coupled"
    treadmill_to_overground: bool = True
    travel_direction: Optional[tuple[float, float, float]] = None
    ground_register: bool = True
    measured_contacts: bool = True
    collision_schemes: tuple[str, ...] = ("boxes", "spheres")
    bone_meshes: bool = False
    fz_threshold: float = 50.0
    right_plate_x_sign: Optional[int] = None
    contact_height_m: float = 0.02
    contact_load_fraction: float = 0.0
    contact_load_floor_n: float = 50.0
    dynamics_diagnostics: bool = False
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    force: bool = False


@dataclass
class PipelineResult:
    bundle_dir: Path
    manifest_path: Path
    motion_path: Path
    asset_paths: dict[str, Path]
    fingerprint: str
    reused: bool = False


class PipelineQualityError(RuntimeError):
    """Raised when a measured-motion quality gate fails."""

    def __init__(self, failures: Sequence[dict[str, Any]]):
        self.failures = list(failures)
        details = "; ".join(
            f"{f['gate']}={f.get('value')!r} ({f['reason']})" for f in self.failures
        )
        super().__init__(f"biomech pipeline quality gates failed: {details}")


def canonical_json(data: Any) -> str:
    """Portable canonical JSON used for fingerprints and manifests."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    """SHA-256 a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def robust_marker_weights(
    marker_names: Sequence[str], mapping, anatomical: np.ndarray, profile: str
) -> np.ndarray:
    """Validated S001 marker weights, including collapsed lower-body centroids."""
    from biomech.fitting.cluster_collapse import LOWER_BODY_CLUSTERS
    from biomech.fitting.marker_map import LOWER_BODY_MARKERS

    names = list(marker_names)
    mapped = set(mapping.model_to_capture)
    lower_tracking = set(LOWER_BODY_MARKERS) | set(LOWER_BODY_CLUSTERS)
    anatomical = np.asarray(anatomical, dtype=bool)
    if profile == "uniform":
        return np.array([1.0 if n in mapped else 0.0 for n in names], dtype=np.float64)
    if profile == "robust-anatomical":
        lower_weight, upper_weight = 0.35, 0.15
    elif profile == "balanced-anatomical":
        lower_weight, upper_weight = 0.5, 0.25
    else:
        raise ValueError(
            "marker_weight_profile must be uniform, robust-anatomical, or "
            f"balanced-anatomical, got {profile!r}"
        )
    weights = np.zeros(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        if name not in mapped:
            continue
        if anatomical[i]:
            weights[i] = 4.0
        elif name in lower_tracking:
            weights[i] = lower_weight
        else:
            weights[i] = upper_weight
    return weights


def apply_bilateral_scale_symmetry(spec) -> list[list[str]]:
    """Tie homologous left/right segment scales to one subject parameter.

    The Rajagopal parser starts with one scale group per body, while the initializer and
    fitter natively support shared groups. For a single static calibration trial,
    independent left/right anisotropic widths are weakly observable and can absorb marker
    placement noise. Sharing homologous groups is the standard symmetric anthropometric
    prior while retaining measured per-side joint motion and marker offsets.
    """
    body_names = [body.name for body in spec.bodies]
    body_set = set(body_names)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for name in body_names:
        if name in seen:
            continue
        counterpart = None
        if name.endswith("_r"):
            counterpart = name[:-2] + "_l"
        elif name.endswith("_l"):
            counterpart = name[:-2] + "_r"
        group = [name]
        if counterpart in body_set and counterpart not in seen:
            right = name if name.endswith("_r") else counterpart
            left = counterpart if name.endswith("_r") else name
            group = [right, left]
        groups.append(group)
        seen.update(group)
    spec.scale_groups = groups
    return groups


def evaluate_quality(
    *,
    scales: np.ndarray,
    poses: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    marker_rms_m: float,
    anatomical_marker_rms_m: float,
    min_visible_markers: int,
    thresholds: QualityThresholds,
    check_joint_limits: bool = True,
) -> list[dict[str, Any]]:
    """Return named quality failures; an empty list means the fit can be exported."""
    failures: list[dict[str, Any]] = []

    def fail(gate: str, value: Any, reason: str) -> None:
        failures.append({"gate": gate, "value": value, "reason": reason})

    scales = np.asarray(scales, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    if scales.size == 0 or not np.isfinite(scales).all():
        fail("finite_scales", None, "scales must all be finite")
    else:
        if float(scales.min()) < thresholds.min_scale:
            fail("min_scale", float(scales.min()), f"must be >= {thresholds.min_scale}")
        if float(scales.max()) > thresholds.max_scale:
            fail("max_scale", float(scales.max()), f"must be <= {thresholds.max_scale}")
        if np.any(scales <= thresholds.min_scale + thresholds.scale_bound_margin):
            fail(
                "scale_lower_bound",
                float(scales.min()),
                "a scale hit the optimizer lower bound; the subject geometry is "
                "underconstrained",
            )
        if np.any(scales >= thresholds.max_scale - thresholds.scale_bound_margin):
            fail(
                "scale_upper_bound",
                float(scales.max()),
                "a scale hit the optimizer upper bound; the subject geometry is "
                "underconstrained",
            )
    if poses.ndim != 2 or poses.shape[0] < 2 or not np.isfinite(poses).all():
        fail("finite_poses", list(poses.shape), "need at least two finite pose frames")
    elif check_joint_limits and poses.shape[1] == np.asarray(lower).size:
        lo_violation = np.max(np.where(np.isfinite(lower), lower - poses, -np.inf))
        hi_violation = np.max(np.where(np.isfinite(upper), poses - upper, -np.inf))
        violation = float(max(lo_violation, hi_violation, 0.0))
        if violation > thresholds.joint_limit_tolerance:
            fail(
                "joint_limits",
                violation,
                f"maximum violation must be <= {thresholds.joint_limit_tolerance}",
            )
    if not np.isfinite(marker_rms_m) or marker_rms_m > thresholds.max_marker_rms_m:
        fail(
            "marker_rms_m",
            marker_rms_m if np.isfinite(marker_rms_m) else None,
            f"must be <= {thresholds.max_marker_rms_m}",
        )
    if (
        not np.isfinite(anatomical_marker_rms_m)
        or anatomical_marker_rms_m > thresholds.max_anatomical_marker_rms_m
    ):
        fail(
            "anatomical_marker_rms_m",
            anatomical_marker_rms_m if np.isfinite(anatomical_marker_rms_m) else None,
            f"must be <= {thresholds.max_anatomical_marker_rms_m}",
        )
    if min_visible_markers < thresholds.min_visible_markers:
        fail(
            "visible_markers",
            int(min_visible_markers),
            f"every frame needs at least {thresholds.min_visible_markers}",
        )
    return failures


def evaluate_delivered_quality(
    *,
    raw_marker_rms_m: float,
    delivered_marker_rms_m: float,
    delivered_anatomical_marker_rms_m: float,
    poses: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    thresholds: QualityThresholds,
) -> list[dict[str, Any]]:
    """Quality gates on the exact corrected poses written to the motion file."""
    failures: list[dict[str, Any]] = []

    def fail(gate: str, value: Any, reason: str) -> None:
        failures.append({"gate": gate, "value": value, "reason": reason})

    poses = np.asarray(poses, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not np.isfinite(poses).all():
        fail("delivered_finite_poses", None, "delivered poses contain NaN/Inf")
    else:
        lo_violation = np.max(np.where(np.isfinite(lower), lower - poses, -np.inf))
        hi_violation = np.max(np.where(np.isfinite(upper), poses - upper, -np.inf))
        violation = float(max(lo_violation, hi_violation, 0.0))
        if violation > thresholds.joint_limit_tolerance:
            fail(
                "delivered_joint_limits",
                violation,
                f"maximum violation must be <= {thresholds.joint_limit_tolerance}",
            )
    if (
        not np.isfinite(delivered_marker_rms_m)
        or delivered_marker_rms_m > thresholds.max_delivered_marker_rms_m
    ):
        fail(
            "delivered_marker_rms_m",
            delivered_marker_rms_m if np.isfinite(delivered_marker_rms_m) else None,
            f"must be <= {thresholds.max_delivered_marker_rms_m}",
        )
    if (
        not np.isfinite(delivered_anatomical_marker_rms_m)
        or delivered_anatomical_marker_rms_m
        > thresholds.max_delivered_anatomical_marker_rms_m
    ):
        fail(
            "delivered_anatomical_marker_rms_m",
            (
                delivered_anatomical_marker_rms_m
                if np.isfinite(delivered_anatomical_marker_rms_m)
                else None
            ),
            f"must be <= {thresholds.max_delivered_anatomical_marker_rms_m}",
        )
    degradation = delivered_marker_rms_m - raw_marker_rms_m
    fraction = degradation / max(raw_marker_rms_m, 1.0e-12)
    if (
        degradation > thresholds.max_marker_degradation_m
        or fraction > thresholds.max_marker_degradation_fraction
    ):
        fail(
            "delivered_marker_degradation",
            {"absolute_m": float(degradation), "fraction": float(fraction)},
            "correction degrades marker fit beyond the allowed absolute/fractional limit",
        )
    return failures


def _config_settings(config: PipelineConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in (
        "trial_c3d",
        "static_c3d",
        "output_root",
        "osim_path",
        "left_belt",
        "right_belt",
        "speedchange",
        "subject_mp",
        "force",
    ):
        data.pop(key, None)
    return data


def _input_records(config: PipelineConfig) -> tuple[dict[str, dict], dict[str, dict]]:
    paths = {
        "trial_c3d": config.trial_c3d,
        "static_c3d": config.static_c3d,
        "osim": config.osim_path,
        "left_belt": config.left_belt,
        "right_belt": config.right_belt,
        "speedchange": config.speedchange,
        "subject_mp": config.subject_mp,
    }
    required = {"trial_c3d", "static_c3d", "osim"}
    portable: dict[str, dict] = {}
    audit: dict[str, dict] = {}
    for role, value in paths.items():
        if value is None:
            if role in required:
                raise FileNotFoundError(f"required input {role} was not provided")
            portable[role] = {"sha256": None, "size_bytes": 0}
            audit[role] = {"path": None, **portable[role]}
            continue
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")
        record = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        portable[role] = record
        audit[role] = {"path": str(path.resolve()), **record}
    return portable, audit


def _slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return out or "subject"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, data: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    # Legacy pickle serialization is deterministic for the tensor-only motion schema;
    # the zip writer embeds the temporary file stem and changes bytes across rebuilds.
    torch.save(data, str(tmp), _use_new_zipfile_serialization=False)
    os.replace(tmp, path)


def _output_record(path: Path, bundle_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(bundle_dir).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_bundle_path(bundle_dir: Path, relative_path: str) -> Path:
    """Resolve a manifest path and reject absolute/traversing references."""
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"bundle output path must be relative: {relative_path!r}")
    root = bundle_dir.resolve()
    resolved = (bundle_dir / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"bundle output path escapes bundle: {relative_path!r}") from exc
    return resolved


def _try_reuse(
    manifest_path: Path, bundle_dir: Path, fingerprint: str
) -> Optional[PipelineResult]:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("fingerprint") != fingerprint:
            return None
        outputs = manifest["outputs"]
        required = {"motion", "fit", "asset_none"}
        required.update(
            f"asset_{scheme}" for scheme in manifest["settings"]["collision_schemes"]
        )
        if not required.issubset(outputs):
            return None
        for record in outputs.values():
            if not isinstance(record, dict) or not {"path", "sha256", "size_bytes"}.issubset(record):
                return None
            path = _safe_bundle_path(bundle_dir, record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                return None
        assets = {
            key.removeprefix("asset_"): _safe_bundle_path(bundle_dir, value["path"])
            for key, value in manifest["outputs"].items()
            if key.startswith("asset_")
        }
        mesh_record = manifest.get("export", {}).get("bone_meshes")
        if mesh_record:
            mesh_dir = _safe_bundle_path(bundle_dir, mesh_record["path"])
            if not mesh_dir.is_dir():
                return None
            mesh_files = sorted(mesh_dir.glob("*.stl"))
            mesh_hash = fingerprint_payload(
                [{"name": p.name, "sha256": sha256_file(p)} for p in mesh_files]
            )
            if (
                len(mesh_files) != mesh_record.get("files")
                or mesh_hash != mesh_record.get("sha256")
            ):
                return None
        return PipelineResult(
            bundle_dir=bundle_dir,
            manifest_path=manifest_path,
            motion_path=_safe_bundle_path(
                bundle_dir, manifest["outputs"]["motion"]["path"]
            ),
            asset_paths=assets,
            fingerprint=fingerprint,
            reused=True,
        )
    except (KeyError, OSError, ValueError, TypeError):
        return None


def _window(
    session,
    observations: np.ndarray,
    present: np.ndarray,
    phase: str,
    explicit: Optional[tuple[int, int]],
    max_frames: Optional[int],
) -> tuple[int, int]:
    from biomech.contact.pipeline import pick_visible_window

    if explicit is not None:
        lo, hi = (int(explicit[0]), int(explicit[1]))
    elif phase == "all":
        lo, hi = 0, session.n_frames
    else:
        lo, hi = session.phase_window(phase)
    if lo < 0 or hi > session.n_frames or hi <= lo:
        raise ValueError(f"invalid frame window {(lo, hi)} for {session.n_frames} frames")
    if max_frames is not None and max_frames < hi - lo:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        start = lo + pick_visible_window(observations[lo:hi], present, max_frames)
        lo, hi = start, start + max_frames
    return lo, hi


def _calibration_window(
    session,
    observations: np.ndarray,
    present: np.ndarray,
    config: PipelineConfig,
    clip_window: tuple[int, int],
) -> tuple[int, int]:
    from biomech.contact.pipeline import pick_visible_window

    if config.calibration_window is not None:
        lo, hi = map(int, config.calibration_window)
    elif config.calibration_phase == "all":
        lo, hi = 0, session.n_frames
    elif session.protocol is not None:
        lo, hi = session.phase_window(config.calibration_phase)
    else:
        lo, hi = clip_window
    if lo < 0 or hi > session.n_frames or hi <= lo:
        raise ValueError(f"invalid calibration search window {(lo, hi)}")
    n = min(config.calibration_frames, hi - lo)
    if n < 2:
        raise ValueError("calibration window must contain at least two frames")
    start = lo + pick_visible_window(observations[lo:hi], present, n)
    return start, start + n


def _marker_report_dict(report, anatomical: np.ndarray) -> dict[str, Any]:
    anatomical = np.asarray(anatomical, dtype=bool)
    anat_values = report.per_marker_rms[
        anatomical & np.isfinite(report.per_marker_rms)
    ]
    return {
        "overall_euclidean_rms_mm": float(report.rms * 1000.0),
        "mean_euclidean_error_mm": float(report.mean * 1000.0),
        "max_euclidean_error_mm": float(report.max * 1000.0),
        "median_frame_euclidean_rms_mm": float(
            np.nanmedian(report.per_frame_rms) * 1000.0
        ),
        "median_anatomical_per_marker_rms_mm": (
            float(np.median(anat_values) * 1000.0) if anat_values.size else None
        ),
        "visible_marker_samples": int(report.num_visible),
        "worst_markers_rms_mm": [
            {"name": name, "rms_mm": float(rms * 1000.0)}
            for name, rms in report.worst_markers(10)
        ],
    }


def _subject_mass(config: PipelineConfig, trial) -> Optional[float]:
    mass = config.subject_mass_kg
    if mass is None:
        mass = trial.subject_meta.get("mass_kg")
    if mass is not None and (not np.isfinite(mass) or mass <= 0.0):
        raise ValueError(f"subject mass must be positive and finite, got {mass!r}")
    if mass is None and not config.allow_model_mass:
        raise ValueError(
            "measured subject mass is required; pass --subject-mp/--subject-mass or "
            "explicitly opt into the generic model mass"
        )
    return None if mass is None else float(mass)


def _regularize_unobservable_scales(
    spec, scales: np.ndarray, marker_offsets: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Reset scale axes with no marker or joint-offset sensitivity to neutral.

    Such axes are gauge variables: changing them cannot alter marker IK or body
    kinematics, but a free optimizer can still drive them to a bound. Neutralizing them
    removes degenerate body widths without changing the reconstructed motion.
    """
    from biomech.skeleton import spatial as S

    scales = np.asarray(scales, dtype=np.float64).copy()
    marker_offsets = np.asarray(marker_offsets, dtype=np.float64)
    body_to_group = {
        body: group_index
        for group_index, group in enumerate(spec.scale_groups)
        for body in group
    }
    constrained = np.zeros_like(scales, dtype=bool)
    tol = 1.0e-8
    for marker, offset in zip(spec.markers, marker_offsets):
        group_index = body_to_group[marker.body]
        constrained[3 * group_index : 3 * group_index + 3] |= np.abs(offset) > tol
    for joint in spec.joints:
        if joint.parent_body is not None:
            group_index = body_to_group[joint.parent_body]
            parent_anchor = S.se3_inverse_np(joint.T_parent)[:3, 3]
            constrained[3 * group_index : 3 * group_index + 3] |= (
                np.abs(parent_anchor) > tol
            )
        group_index = body_to_group[joint.child_body]
        child_anchor = S.se3_inverse_np(joint.T_child)[:3, 3]
        constrained[3 * group_index : 3 * group_index + 3] |= np.abs(child_anchor) > tol

    changes: list[dict[str, Any]] = []
    for index in np.flatnonzero(~constrained):
        if abs(scales[index] - 1.0) <= tol:
            continue
        group_index, axis = divmod(int(index), 3)
        changes.append(
            {
                "bodies": list(spec.scale_groups[group_index]),
                "axis": axis,
                "fitted": float(scales[index]),
                "delivered": 1.0,
                "reason": "no marker or joint-offset sensitivity",
            }
        )
        scales[index] = 1.0
    return scales, changes


def _copy_bone_meshes(asset_root: Path) -> tuple[Optional[dict], Optional[dict]]:
    from biomech.export.bone_geometry import (
        bone_mesh_dir,
        bone_meshes_available,
        default_bone_geometry,
    )

    source_root = _REPO_ROOT / "protomotions" / "data" / "assets"
    if not bone_meshes_available(source_root):
        raise FileNotFoundError(
            "bone meshes requested but converted STL files are missing; run "
            "projects/biomech/tools/convert_bone_meshes.py"
        )
    source = bone_mesh_dir(source_root)
    destination = bone_mesh_dir(asset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    files = sorted(destination.glob("*.stl"))
    tree_hash = fingerprint_payload(
        [{"name": p.name, "sha256": sha256_file(p)} for p in files]
    )
    return default_bone_geometry(), {
        "path": destination.relative_to(asset_root.parent).as_posix(),
        "files": len(files),
        "sha256": tree_hash,
    }


def _motion_validation(
    clip,
    base_export,
    base_asset_path: Path,
    variant_exports: dict[str, Any],
    subject_mass: Optional[float],
    thresholds: QualityThresholds,
) -> dict[str, Any]:
    import mujoco
    import torch

    required = {
        "rigid_body_pos": 3,
        "rigid_body_rot": 4,
        "rigid_body_vel": 3,
        "rigid_body_ang_vel": 3,
    }
    frames = None
    for key, width in required.items():
        value = clip.data.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != width:
            raise ValueError(f"invalid motion field {key}: {getattr(value, 'shape', None)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"motion field {key} contains NaN/Inf")
        frames = value.shape[0] if frames is None else frames
        if value.shape[0] != frames:
            raise ValueError("motion fields have inconsistent frame counts")
    for key in ("dof_pos", "dof_vel"):
        value = clip.data.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != frames:
            raise ValueError(f"invalid motion field {key}: {getattr(value, 'shape', None)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"motion field {key} contains NaN/Inf")
    if frames is None or frames < 2 or not np.isfinite(clip.fps) or clip.fps <= 0:
        raise ValueError("motion must contain at least two frames at a positive FPS")

    quat = clip.data["rigid_body_rot"]
    quat_error = float((torch.linalg.norm(quat, dim=-1) - 1.0).abs().max())
    if quat_error > thresholds.quaternion_norm_tolerance:
        raise ValueError(f"motion quaternion norm error {quat_error} exceeds tolerance")
    if clip.body_names != base_export.body_names:
        raise ValueError("motion body order does not match the exact exported MJCF")
    if clip.dof_names != base_export.joint_names:
        raise ValueError("motion DOF order does not match the exact exported MJCF")
    if clip.data["dof_pos"].shape[1] != len(clip.dof_names):
        raise ValueError("motion DOF tensor width does not match DOF names")
    for scheme, export in variant_exports.items():
        if export.body_names != base_export.body_names or export.joint_names != base_export.joint_names:
            raise ValueError(f"{scheme} collision asset changes body/DOF topology")

    from protomotions.components.pose_lib import extract_kinematic_info
    from protomotions.simulator.base_simulator.simulator_state import (
        RobotState,
        StateConversion,
    )

    kinematic = extract_kinematic_info(str(base_asset_path))
    if kinematic.body_names != clip.body_names or kinematic.dof_names != clip.dof_names:
        raise ValueError("ProtoMotions kinematic extraction disagrees with motion ordering")
    state = RobotState.from_dict(clip.data, state_conversion=StateConversion.COMMON)
    if state.num_bodies != len(clip.body_names) or state.motion_num_frames != frames:
        raise ValueError("ProtoMotions RobotState rejected the exported motion schema")

    model = mujoco.MjModel.from_xml_path(str(base_asset_path))
    actual_mass = float(model.body_mass[1:].sum())
    mass_error = None
    if subject_mass is not None:
        mass_error = abs(actual_mass - subject_mass)
        if mass_error > thresholds.subject_mass_tolerance_kg:
            raise ValueError(
                f"exported mass {actual_mass:.6f} kg does not match measured "
                f"{subject_mass:.6f} kg"
            )
    contacts = clip.data.get("rigid_body_contacts")
    if contacts is not None:
        if contacts.shape != (frames, len(clip.body_names)):
            raise ValueError("reference contacts do not match motion frames/body order")
        if not bool(contacts.any()):
            raise ValueError("all measured reference contacts are false")
    return {
        "frames": int(frames),
        "bodies": len(clip.body_names),
        "dofs": len(clip.dof_names),
        "qpos_dim": int(base_export.qpos_dim),
        "quaternion_max_norm_error": quat_error,
        "actual_mass_kg": actual_mass,
        "subject_mass_error_kg": mass_error,
        "protomotions_body_order": "pass",
        "protomotions_dof_order": "pass",
        "contact_labels": "pass" if contacts is not None else "not_requested",
    }


def _verify_saved_motion(path: Path, clip) -> None:
    """Load the final serialized file through ProtoMotions' raw-motion path."""
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    library = MotionLib(MotionLibConfig(motion_file=str(path)), device="cpu")
    if library.num_motions() != 1:
        raise ValueError("ProtoMotions MotionLib did not load exactly one motion")
    if int(library.motion_num_frames[0]) != int(
        clip.data["rigid_body_pos"].shape[0]
    ):
        raise ValueError("MotionLib frame count differs from the serialized motion")
    if library.gts.shape[1] != len(clip.body_names) or library.dps.shape[1] != len(
        clip.dof_names
    ):
        raise ValueError("MotionLib body/DOF dimensions differ from the exact MJCF")
    if "rigid_body_contacts" in clip.data and library.contacts is None:
        raise ValueError("MotionLib discarded requested measured contact labels")


def _verify_equivalent_bundle(existing_dir: Path, staged_dir: Path, outputs: dict) -> None:
    """Verify a forced rebuild is scientifically equivalent to an immutable bundle."""
    import torch

    for key, record in outputs.items():
        existing_path = _safe_bundle_path(existing_dir, record["path"])
        staged_path = _safe_bundle_path(staged_dir, record["path"])
        if key.startswith("asset_"):
            if existing_path.read_bytes() != staged_path.read_bytes():
                raise RuntimeError(f"forced rebuild changed deterministic XML artifact {key}")
        elif key == "motion":
            existing = torch.load(existing_path, map_location="cpu", weights_only=True)
            staged = torch.load(staged_path, map_location="cpu", weights_only=True)
            if set(existing) != set(staged):
                raise RuntimeError("forced rebuild changed motion fields")
            for field in existing:
                left, right = existing[field], staged[field]
                if isinstance(left, torch.Tensor):
                    if left.dtype != right.dtype or left.shape != right.shape:
                        raise RuntimeError(f"forced rebuild changed motion schema for {field}")
                    equal = (
                        torch.equal(left, right)
                        if not left.is_floating_point()
                        else torch.allclose(left, right, rtol=0.0, atol=1.0e-7)
                    )
                    if not bool(equal):
                        raise RuntimeError(f"forced rebuild changed motion values for {field}")
                elif left != right:
                    raise RuntimeError(f"forced rebuild changed motion metadata {field}")
        elif key == "fit":
            with np.load(existing_path) as existing, np.load(staged_path) as staged:
                if set(existing.files) != set(staged.files):
                    raise RuntimeError("forced rebuild changed fit fields")
                for field in existing.files:
                    left, right = existing[field], staged[field]
                    equal = (
                        np.array_equal(left, right)
                        if not np.issubdtype(left.dtype, np.floating)
                        else np.allclose(left, right, rtol=0.0, atol=1.0e-10, equal_nan=True)
                    )
                    if not bool(equal):
                        raise RuntimeError(f"forced rebuild changed fit values for {field}")


def _dynamics_report(
    spec,
    poses: np.ndarray,
    scales: np.ndarray,
    fps: float,
    mjcf_xml: str,
    grf_by_side: dict[str, np.ndarray],
    cop_by_side: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Corrected-frame inverse-dynamics shadow report; never changes delivered poses."""
    if poses.shape[0] < 5:
        return {"status": "skipped", "reason": "need at least five frames"}
    from biomech.contact.tracking import mjcf_qpos_zup
    from biomech.fitting.dynamics_fitter import (
        ResidualHelper,
        accelerations_from_velocities,
        contacts_from_grf,
        merge_contacts,
        velocities_from_positions,
    )

    helper = ResidualHelper(mjcf_xml)
    qpos = np.stack(
        [mjcf_qpos_zup(spec, q, scales, "coupled") for q in poses], axis=0
    )
    qvel = velocities_from_positions(helper, qpos, 1.0 / fps)
    qacc = accelerations_from_velocities(qvel, 1.0 / fps)
    contacts = merge_contacts(
        contacts_from_grf(grf_by_side["R"], cop_by_side["R"], "calcn_r"),
        contacts_from_grf(grf_by_side["L"], cop_by_side["L"], "calcn_l"),
    )
    keep = slice(2, -2)
    report = helper.residual_report(qpos[keep], qvel[keep], qacc[keep], contacts[keep])
    summary = report.summary()
    return {
        "status": "diagnostic_only",
        "linear_residual": {
            "mean_N": summary["mean_force_residual_N"],
            "max_N": summary["max_force_residual_N"],
        },
        "angular_residual_valid": False,
        "notes": [
            "This stage never modifies the delivered motion or inertial parameters.",
            "Free moment is omitted until the force-plate action/reaction sign is validated.",
            "Point-rate force and COP are block-averaged separately, so angular wrench "
            "residuals are not authoritative.",
            "The whole-foot wrench is applied to calcn; distal MTP moments are not reported.",
        ],
    }


def _stance_patch_speed_metrics(
    clip,
    spec,
    static_session,
    group_scales: np.ndarray,
    grf_by_side: dict[str, np.ndarray],
    fz_threshold: float,
) -> dict[str, dict[str, float]]:
    """Horizontal plantar-patch speed during measured stance."""
    from biomech.contact.elastic_foundation import _quat_rotate_np
    from biomech.contact.foot_geometry import subject_sole_from_session
    from biomech.contact.kinematics import foot_trajectory_from_motion

    metrics: dict[str, dict[str, float]] = {}
    for side, body in (("R", "calcn_r"), ("L", "calcn_l")):
        if body not in clip.body_names or side not in grf_by_side:
            continue
        fz = np.asarray(grf_by_side[side])[:, 2]
        peak = float(np.nanmax(fz)) if fz.size else 0.0
        stance = fz >= max(fz_threshold, 0.5 * peak)
        if not stance.any():
            continue
        sole = subject_sole_from_session(
            static_session, spec, side, group_scales=group_scales
        )
        pos, quat, linvel, angvel = foot_trajectory_from_motion(clip, body)
        local = np.broadcast_to(sole.points, (pos.shape[0], sole.n, 3))
        lever = _quat_rotate_np(quat[:, None, :], local)
        world = pos[:, None, :] + lever
        velocity = linvel[:, None, :] + np.cross(angvel[:, None, :], lever)
        stance_world = world[stance]
        near_ground = stance_world[:, :, 2] <= (
            stance_world[:, :, 2].min(axis=1, keepdims=True) + 0.01
        )
        horizontal_all = np.linalg.norm(velocity[stance, :, :2], axis=2)
        horizontal = horizontal_all[near_ground]
        metrics[side] = {
            "median_mps": float(np.median(horizontal)),
            "p95_mps": float(np.percentile(horizontal, 95.0)),
            "frames": int(stance.sum()),
            "patch_samples": int(horizontal.size),
        }
    return metrics


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _runtime_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    for package in ("scipy", "torch", "warp", "mujoco", "newton"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001
            versions[package] = None
    return versions


def _implementation_record() -> dict[str, Any]:
    source_files: list[Path] = []
    for root in _FINGERPRINT_SOURCE_ROOTS:
        source_files.extend((_REPO_ROOT / root).rglob("*.py"))
    source_files.extend(_REPO_ROOT / path for path in _FINGERPRINT_SOURCE_FILES)
    source_hashes = {
        path.relative_to(_REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(set(source_files))
        if "__pycache__" not in path.parts and "tests" not in path.parts
    }
    return {
        "git_revision": _git_revision(),
        "runtime_versions": _runtime_versions(),
        "source_sha256": source_hashes,
        "source_tree_sha256": fingerprint_payload(source_hashes),
    }


def run_c3d_to_protomotions(config: PipelineConfig) -> PipelineResult:
    """Run the complete measured-human C3D-to-ProtoMotions export."""
    if config.marker_profile != "s001-pig":
        raise ValueError(
            "only marker_profile='s001-pig' is currently validated; add an explicit "
            "MarkerMap before using another capture protocol"
        )
    if config.coupled_knee != "coupled":
        raise ValueError("the fidelity pipeline requires coupled_knee='coupled'")
    if config.chunk_size < 1 or config.calibration_frames < 2:
        raise ValueError("chunk_size must be positive and calibration_frames >= 2")
    if any(s not in ("boxes", "spheres") for s in config.collision_schemes):
        raise ValueError("collision_schemes may contain only boxes and spheres")
    if config.measured_contacts and "boxes" not in config.collision_schemes:
        raise ValueError("measured contact labels require the boxes collision asset")
    if config.treadmill_to_overground and not (config.left_belt or config.right_belt):
        raise ValueError("treadmill-to-overground conversion requires a belt-speed log")

    portable_inputs, audit_inputs = _input_records(config)
    implementation = _implementation_record()
    fingerprint_implementation = {
        "runtime_versions": implementation["runtime_versions"],
        "source_tree_sha256": implementation["source_tree_sha256"],
    }
    fingerprint = fingerprint_payload(
        {
            "schema": PIPELINE_SCHEMA,
            "version": PIPELINE_VERSION,
            "implementation": fingerprint_implementation,
            "inputs": portable_inputs,
            "settings": _config_settings(config),
        }
    )
    subject_hint = config.subject_id or Path(config.trial_c3d).parent.name
    asset_stem = f"biomech_{_slug(subject_hint)}_{fingerprint[:12]}"
    bundle_dir = Path(config.output_root) / asset_stem
    manifest_path = bundle_dir / "manifest.json"
    if not config.force:
        reused = _try_reuse(manifest_path, bundle_dir, fingerprint)
        if reused is not None:
            return reused

    from biomech.contact.pipeline import (
        detect_right_plate_x_sign,
        measured_belt_grf,
    )
    from biomech.export.foot_collision import foot_collision_geoms
    from biomech.export.mjcf import export_mjcf
    from biomech.export.protomotions_robot import (
        build_simbody_motion,
        foot_contacts_from_clip,
        register_clip_to_ground,
    )
    from biomech.export.tm2og import cumulative_belt_displacement, tm2og_motion
    from biomech.fitting.cluster_collapse import collapse_clusters
    from biomech.fitting.ik import MarkerIKConfig, position_limits, solve_marker_ik
    from biomech.fitting.ik_initializer import IKInitializer
    from biomech.fitting.marker_fitter import MarkerFitConfig, MarkerFitter
    from biomech.fitting.marker_fixer import MarkerFixConfig, fix_markers
    from biomech.fitting.marker_map import (
        anatomical_mask,
        observations_from_session,
        s001_marker_map,
    )
    from biomech.fitting.marker_placement import (
        apply_foot_flat_to_poses,
        place_foot_markers,
    )
    from biomech.fitting.report import marker_errors
    from biomech.osim import parse_osim
    from biomech.session import load_session
    from biomech.skeleton.skeleton import WarpSkeleton

    trial = load_session(
        config.trial_c3d,
        left_belt_path=config.left_belt,
        right_belt_path=config.right_belt,
        belt_rate_hz=config.belt_rate_hz,
        subject_mp_path=config.subject_mp,
        speedchange_path=config.speedchange,
        subject_id=config.subject_id,
        fz_threshold=config.fz_threshold,
        filter_cutoff_hz=config.filter_cutoff_hz,
        filter_order=config.filter_order,
    )
    static = load_session(
        config.static_c3d,
        subject_id=config.subject_id,
        filter_cutoff_hz=None,
    )
    subject_mass = _subject_mass(config, trial)
    spec = parse_osim(str(config.osim_path))
    if config.bilateral_scale_symmetry:
        apply_bilateral_scale_symmetry(spec)
    mapping = s001_marker_map()
    centroid_names: list[str] = []
    if config.collapse_clusters:
        mapping, centroid_names = collapse_clusters(spec, mapping)

    placement = None
    if config.enrich_foot_markers:
        placement = place_foot_markers(
            spec,
            static,
            mapping=mapping,
            marker_config=MarkerFitConfig(outer_iters=config.placement_outer_iters),
            device=config.device,
            frame_range=(0, min(config.placement_frames, static.n_frames)),
        )

    skeleton = WarpSkeleton(spec, device=config.device)
    marker_names = skeleton.marker_names()
    observations, present = observations_from_session(
        trial, marker_names, mapping
    )
    anatomical = anatomical_mask(marker_names, mapping)
    weights = robust_marker_weights(
        marker_names, mapping, anatomical, config.marker_weight_profile
    )
    if not np.any(weights > 0):
        raise ValueError("marker map produced no weighted observations")

    fix_report = None
    if config.clean_markers:
        observations, fix_report = fix_markers(
            observations,
            body_of_marker=np.asarray(skeleton.topo.m_body, dtype=int),
            marker_names=marker_names,
            config=MarkerFixConfig(),
        )

    clip_window = _window(
        trial,
        observations,
        present,
        config.phase,
        config.window,
        config.max_frames,
    )
    calibration_window = _calibration_window(
        trial, observations, present, config, clip_window
    )
    clo, chi = calibration_window
    lo, hi = clip_window
    obs_cal = np.ascontiguousarray(observations[clo:chi])
    obs_clip = np.ascontiguousarray(observations[lo:hi])

    initializer_obs = obs_cal.copy()
    initializer_obs[:, weights <= 0.0, :] = np.nan
    initializer = IKInitializer(skeleton, initializer_obs, anatomical=anatomical)
    seed = initializer.run(MarkerIKConfig(max_iters=min(config.ik_iters, 40)))
    fit_config = MarkerFitConfig(
        outer_iters=config.outer_iters,
        inner_first=MarkerIKConfig(max_iters=max(config.ik_iters, 80)),
        inner=MarkerIKConfig(max_iters=min(config.ik_iters, 50)),
        final_inner=MarkerIKConfig(max_iters=config.ik_iters),
        offset_prior_weight=2.0,
        anatomical_prior_factor=(40.0 if config.marker_weight_profile == "robust-anatomical" else 35.0),
        offset_max_delta=0.035,
        scale_bounds=(config.quality.min_scale, config.quality.max_scale),
    )
    anthropometric_diagnostics = None
    if config.anthropometric_prior:
        if config.subject_mp is None:
            raise ValueError("anthropometric_prior requires subject_mp")
        from biomech.fitting.priors import Anthropometrics

        anthropometric_diagnostics = Anthropometrics(
            mp_path=config.subject_mp, include_upper_body=False
        ).apply_to_config(spec, fit_config)
    fitter = MarkerFitter(
        skeleton, obs_cal, weights=weights, anatomical=anatomical
    )
    fit = fitter.fit(
        init_scales=seed.group_scales,
        q_init=seed.poses,
        config=fit_config,
    )
    scales = np.asarray(fit.group_scales, dtype=np.float64)
    marker_offsets = np.asarray(fit.marker_offsets, dtype=np.float64)
    for marker, offset in zip(spec.markers, marker_offsets):
        marker.offset = np.asarray(offset, dtype=np.float64).copy()
    scales, scale_regularization = _regularize_unobservable_scales(
        spec, scales, marker_offsets
    )
    skeleton.set_marker_offsets(marker_offsets)

    frame_count = hi - lo
    raw_poses = np.empty((frame_count, skeleton.topo.num_dofs), dtype=np.float64)
    solver_rms = np.empty(frame_count, dtype=np.float64)
    seed_row = np.nanmean(fit.poses, axis=0)
    ik_config = MarkerIKConfig(max_iters=config.ik_iters)
    for c0 in range(0, frame_count, config.chunk_size):
        c1 = min(c0 + config.chunk_size, frame_count)
        q_init = np.repeat(seed_row[None, :], c1 - c0, axis=0)
        solved = solve_marker_ik(
            skeleton,
            obs_clip[c0:c1],
            q_init,
            group_scales=scales,
            weights=weights,
            config=ik_config,
        )
        raw_poses[c0:c1] = solved.q
        solver_rms[c0:c1] = solved.marker_rms

    raw_errors = marker_errors(
        skeleton,
        obs_clip,
        raw_poses,
        group_scales=scales,
        marker_offsets=marker_offsets,
    )
    foot_flat = placement.foot_flat if placement is not None else {}
    delivered_poses = apply_foot_flat_to_poses(spec, raw_poses, foot_flat)
    delivered_errors = marker_errors(
        skeleton,
        obs_clip,
        delivered_poses,
        group_scales=scales,
        marker_offsets=marker_offsets,
    )
    raw_metrics = _marker_report_dict(raw_errors, anatomical)
    delivered_metrics = _marker_report_dict(delivered_errors, anatomical)
    visible_per_frame = (
        np.isfinite(obs_clip).all(axis=2) & (weights[None, :] > 0)
    ).sum(axis=1)
    lower, upper = position_limits(spec)
    anatomical_values = raw_errors.per_marker_rms[
        anatomical & np.isfinite(raw_errors.per_marker_rms)
    ]
    anatomical_rms = (
        float(np.median(anatomical_values)) if anatomical_values.size else float("nan")
    )
    failures = evaluate_quality(
        scales=scales,
        poses=delivered_poses,
        lower=lower,
        upper=upper,
        marker_rms_m=float(raw_errors.rms),
        anatomical_marker_rms_m=anatomical_rms,
        min_visible_markers=int(visible_per_frame.min()),
        thresholds=config.quality,
        check_joint_limits=True,
    )
    delivered_anatomical_values = delivered_errors.per_marker_rms[
        anatomical & np.isfinite(delivered_errors.per_marker_rms)
    ]
    delivered_anatomical_rms = (
        float(np.median(delivered_anatomical_values))
        if delivered_anatomical_values.size
        else float("nan")
    )
    failures.extend(
        evaluate_delivered_quality(
            raw_marker_rms_m=float(raw_errors.rms),
            delivered_marker_rms_m=float(delivered_errors.rms),
            delivered_anatomical_marker_rms_m=delivered_anatomical_rms,
            poses=delivered_poses,
            lower=lower,
            upper=upper,
            thresholds=config.quality,
        )
    )
    if failures:
        raise PipelineQualityError(failures)

    publish_dir = bundle_dir
    scratch_context = tempfile.TemporaryDirectory(prefix=f"{asset_stem}-")
    scratch_dir = Path(scratch_context.name)
    mesh_map = None
    mesh_record = None
    asset_root = scratch_dir / "assets"
    if config.bone_meshes:
        mesh_map, mesh_record = _copy_bone_meshes(asset_root)
    base_export = export_mjcf(
        spec,
        group_scales=scales,
        coupled_knee=config.coupled_knee,
        visual_geoms=True,
        subject_mass=subject_mass,
        bone_meshes=mesh_map,
    )
    clip = build_simbody_motion(
        spec,
        delivered_poses,
        fps=trial.point_rate,
        group_scales=scales,
        coupled_knee=config.coupled_knee,
        mjcf_xml=base_export.xml,
    )

    belt_speed = None
    travel_direction = None
    treadmill_metrics: dict[str, Any] = {"applied": False}
    if config.treadmill_to_overground:
        traces = [
            np.asarray(trial.belt_speed_point[side], dtype=np.float64)
            for side in ("left", "right")
            if side in trial.belt_speed_point
        ]
        if not traces:
            raise ValueError("no belt-speed traces were loaded")
        belt_speed = np.abs(np.nanmean(np.stack(traces, axis=0), axis=0))[lo:hi]
        if belt_speed.shape != (frame_count,) or not np.isfinite(belt_speed).all():
            raise ValueError("selected belt-speed trace is missing or non-finite")
        requested_direction = config.travel_direction
        if requested_direction is not None:
            direction = np.asarray(requested_direction, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction).all() or norm <= 0.0:
                raise ValueError("travel_direction must be a finite nonzero vector")
            requested_direction = tuple((direction / norm).tolist())
        travel_direction = tm2og_motion(
            clip.data,
            belt_speed,
            trial.point_rate,
            clip.body_names,
            travel_dir=requested_direction,
        )
        displacement = cumulative_belt_displacement(belt_speed, trial.point_rate)
        treadmill_metrics = {
            "applied": True,
            "belt_rate_hz": float(trial.treadmill.rate_hz),
            "belt_rate_inferred": bool(trial.treadmill.rate_inferred),
            "belt_speed_min_mps": float(belt_speed.min()),
            "belt_speed_mean_mps": float(belt_speed.mean()),
            "belt_speed_max_mps": float(belt_speed.max()),
            "belt_displacement_m": float(displacement[-1]),
            "travel_direction": travel_direction.tolist(),
        }
        if len(traces) == 2:
            treadmill_metrics["split_belt_mean_abs_difference_mps"] = float(
                np.mean(np.abs(traces[0][lo:hi] - traces[1][lo:hi]))
            )

    grf_by_side: dict[str, np.ndarray] = {}
    cop_by_side: dict[str, np.ndarray] = {}
    right_sign = config.right_plate_x_sign
    needs_force = config.ground_register or config.measured_contacts or config.dynamics_diagnostics
    if needs_force:
        if len(trial.force_plates) < 2:
            raise ValueError("ground/contact/dynamics stages require two force plates")
        if right_sign is None:
            right_sign = detect_right_plate_x_sign(
                trial,
                static,
                spec,
                clip,
                clip_window,
                group_scales=scales,
                fz_threshold=config.fz_threshold,
            )
        measured = measured_belt_grf(trial, right_sign)
        for side in ("R", "L"):
            if side not in measured:
                raise ValueError(f"force-plate assignment did not produce side {side}")
            grf_by_side[side] = np.asarray(measured[side][0][lo:hi], dtype=np.float64)
            cop_by_side[side] = np.asarray(measured[side][1][lo:hi], dtype=np.float64)
            if grf_by_side[side].shape != (frame_count, 3):
                raise ValueError(f"measured GRF for {side} does not match clip length")

    ground_shift = 0.0
    if config.ground_register:
        ground_shift = register_clip_to_ground(
            clip,
            spec,
            static,
            trial,
            clip_window,
            group_scales=scales,
            fz_threshold=config.fz_threshold,
            right_plate_x_sign=right_sign,
        )

    variant_exports: dict[str, Any] = {}
    collision_counts: dict[str, dict[str, int]] = {}
    for scheme in config.collision_schemes:
        geoms = foot_collision_geoms(spec, scales, scheme, static)
        variant_exports[scheme] = export_mjcf(
            spec,
            group_scales=scales,
            coupled_knee=config.coupled_knee,
            visual_geoms=True,
            subject_mass=subject_mass,
            bone_meshes=mesh_map,
            collision_geoms=geoms,
        )
        collision_counts[scheme] = {
            "boxes": sum(g.kind == "box" for g in geoms),
            "spheres": sum(g.kind == "sphere" for g in geoms),
        }

    mjcf_dir = asset_root / "mjcf"
    base_asset_path = mjcf_dir / f"{asset_stem}.xml"
    asset_paths = {"none": base_asset_path}
    _atomic_write_text(base_asset_path, base_export.xml)
    for scheme, export in variant_exports.items():
        path = mjcf_dir / f"{asset_stem}_{scheme}.xml"
        _atomic_write_text(path, export.xml)
        asset_paths[scheme] = path

    if config.measured_contacts:
        clip.data["rigid_body_contacts"] = foot_contacts_from_clip(
            clip,
            str(asset_paths["boxes"]),
            grf_by_side,
            height_thresh=config.contact_height_m,
            load_frac=config.contact_load_fraction,
            load_floor=config.contact_load_floor_n,
        )

    validation = _motion_validation(
        clip,
        base_export,
        base_asset_path,
        variant_exports,
        subject_mass,
        config.quality,
    )

    force_metrics: dict[str, Any] = {"available": bool(grf_by_side)}
    if grf_by_side:
        total_fz = grf_by_side["R"][:, 2] + grf_by_side["L"][:, 2]
        loaded = total_fz > config.fz_threshold
        force_metrics.update(
            {
                "right_plate_x_sign": int(right_sign),
                "mean_loaded_total_fz_N": float(total_fz[loaded].mean()) if loaded.any() else 0.0,
                "peak_total_fz_N": float(total_fz.max()),
                "measured_bodyweight_ratio": (
                    float(total_fz[loaded].mean() / (subject_mass * 9.81))
                    if loaded.any() and subject_mass is not None
                    else None
                ),
            }
        )
        foot_slip = _stance_patch_speed_metrics(
            clip,
            spec,
            static,
            scales,
            grf_by_side,
            config.fz_threshold,
        )
        force_metrics["stance_plantar_patch_speed"] = foot_slip
        slip_failures = []
        for side, values in foot_slip.items():
            if values["median_mps"] > config.quality.max_stance_foot_speed_mps:
                slip_failures.append(
                    {
                        "gate": f"{side}_stance_patch_speed_median",
                        "value": values["median_mps"],
                        "reason": (
                            f"must be <= {config.quality.max_stance_foot_speed_mps} m/s"
                        ),
                    }
                )
            if values["p95_mps"] > config.quality.max_stance_foot_speed_p95_mps:
                slip_failures.append(
                    {
                        "gate": f"{side}_stance_patch_speed_p95",
                        "value": values["p95_mps"],
                        "reason": (
                            "must be <= "
                            f"{config.quality.max_stance_foot_speed_p95_mps} m/s"
                        ),
                    }
                )
        if slip_failures:
            raise PipelineQualityError(slip_failures)

    dynamics = {"status": "not_requested"}
    if config.dynamics_diagnostics:
        dynamics = _dynamics_report(
            spec,
            delivered_poses,
            scales,
            trial.point_rate,
            base_export.xml,
            grf_by_side,
            cop_by_side,
        )

    Path(config.output_root).mkdir(parents=True, exist_ok=True)
    stage_context = tempfile.TemporaryDirectory(
        prefix=f".{asset_stem}.staging-", dir=config.output_root
    )
    stage_dir = Path(stage_context.name)
    shutil.copytree(asset_root, stage_dir / "assets", dirs_exist_ok=True)
    staged_asset_paths = {
        scheme: stage_dir / "assets" / path.relative_to(asset_root)
        for scheme, path in asset_paths.items()
    }
    motion_path = stage_dir / "motions" / f"{asset_stem}.motion"
    fit_path = stage_dir / "fit" / "reconstruction.npz"
    arrays: dict[str, np.ndarray] = {
        "raw_poses": raw_poses,
        "delivered_poses": delivered_poses,
        "group_scales": scales,
        "marker_offsets": marker_offsets,
        "marker_weights": weights,
        "marker_names": np.asarray(marker_names),
        "anatomical": anatomical,
        "solver_marker_rms": solver_rms,
        "per_frame_marker_rms": raw_errors.per_frame_rms,
        "per_marker_rms": raw_errors.per_marker_rms,
        "delivered_per_frame_marker_rms": delivered_errors.per_frame_rms,
        "delivered_per_marker_rms": delivered_errors.per_marker_rms,
        "clip_window": np.asarray(clip_window, dtype=np.int64),
        "calibration_window": np.asarray(calibration_window, dtype=np.int64),
    }
    if belt_speed is not None:
        arrays["belt_speed"] = belt_speed
        arrays["travel_direction"] = np.asarray(travel_direction)
    for side in grf_by_side:
        arrays[f"grf_{side}"] = grf_by_side[side]
        arrays[f"cop_{side}"] = cop_by_side[side]
    if foot_flat:
        arrays["foot_flat_names"] = np.asarray(list(foot_flat))
        arrays["foot_flat_values"] = np.asarray(list(foot_flat.values()), dtype=np.float64)
    if placement is not None and placement.ankle_neutral:
        arrays["ankle_neutral_names"] = np.asarray(list(placement.ankle_neutral))
        arrays["ankle_neutral_values"] = np.asarray(
            list(placement.ankle_neutral.values()), dtype=np.float64
        )
    _atomic_save_npz(fit_path, arrays)
    _atomic_torch_save(motion_path, clip.data)
    _verify_saved_motion(motion_path, clip)
    validation["protomotions_motionlib_load"] = "pass"

    outputs: dict[str, Any] = {
        "motion": _output_record(motion_path, stage_dir),
        "fit": _output_record(fit_path, stage_dir),
        "asset_none": _output_record(staged_asset_paths["none"], stage_dir),
    }
    for scheme in variant_exports:
        outputs[f"asset_{scheme}"] = _output_record(staged_asset_paths[scheme], stage_dir)

    contacts = clip.data.get("rigid_body_contacts")
    contact_counts = {}
    if contacts is not None:
        contact_counts = {
            clip.body_names[i]: int(contacts[:, i].sum())
            for i in range(contacts.shape[1])
            if int(contacts[:, i].sum()) > 0
        }
        contact_validation_failures = []
        for side in ("R", "L"):
            body_indices = [
                clip.body_names.index(name)
                for name in (f"calcn_{side.lower()}", f"toes_{side.lower()}")
                if name in clip.body_names
            ]
            predicted = contacts[:, body_indices].any(dim=1).cpu().numpy()
            measured = (
                grf_by_side[side][:, 2]
                > max(
                    config.contact_load_floor_n,
                    config.contact_load_fraction
                    * float(np.nanmax(grf_by_side[side][:, 2])),
                )
            )
            recall = float(np.mean(predicted[measured])) if measured.any() else 0.0
            precision = float(np.mean(measured[predicted])) if predicted.any() else 0.0
            force_metrics.setdefault("contact_label_agreement", {})[side] = {
                "recall": recall,
                "precision": precision,
            }
            if recall < 0.80 or precision < 0.95:
                contact_validation_failures.append(
                    {
                        "gate": f"{side}_contact_label_agreement",
                        "value": {"recall": recall, "precision": precision},
                        "reason": "requires recall >= 0.80 and precision >= 0.95",
                    }
                )
        if contact_validation_failures:
            raise PipelineQualityError(contact_validation_failures)
    contact_thresholds = {
        side: max(
            config.contact_load_floor_n,
            config.contact_load_fraction
            * float(np.nanmax(np.asarray(grf_by_side[side])[:, 2])),
        )
        for side in grf_by_side
    }
    manifest = {
        "schema": PIPELINE_SCHEMA,
        "version": PIPELINE_VERSION,
        "status": "complete",
        "fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "implementation": implementation,
        "subject": {
            "id": trial.subject_id,
            "mass_kg": subject_mass,
            "metadata": trial.subject_meta,
        },
        "inputs": audit_inputs,
        "settings": _config_settings(config),
        "capture": {
            "point_rate_hz": float(trial.point_rate),
            "frames": int(trial.n_frames),
            "markers": len(trial.marker_labels),
            "analog_rate_hz": float(trial.analog_rate),
            "force_plates": len(trial.force_plates),
            "warnings": trial.warnings,
        },
        "reconstruction": {
            "clip_window": list(clip_window),
            "calibration_window": list(calibration_window),
            "frames": frame_count,
            "marker_profile": config.marker_profile,
            "weight_profile": config.marker_weight_profile,
            "anthropometric_prior": anthropometric_diagnostics,
            "mapped_markers": int(present.sum()),
            "minimum_visible_weighted_markers": int(visible_per_frame.min()),
            "collapsed_centroids": centroid_names,
            "marker_cleanup": (
                {
                    "rigid_rejected": fix_report.n_rigid_rejected,
                    "spikes_rejected": fix_report.n_spikes_rejected,
                    "samples_filled": fix_report.n_filled,
                }
                if fix_report is not None
                else {"enabled": False}
            ),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "scale_groups": [list(group) for group in spec.scale_groups],
            "unobservable_scale_regularization": scale_regularization,
            "raw_marker_optimum": raw_metrics,
            "delivered_foot_corrected": delivered_metrics,
            "foot_flat_deg": {
                key: float(np.degrees(value)) for key, value in (foot_flat or {}).items()
            },
            "ankle_neutral_deg": {
                key: float(np.degrees(value))
                for key, value in (
                    (placement.ankle_neutral if placement is not None else None) or {}
                ).items()
            },
        },
        "treadmill": treadmill_metrics,
        "force": force_metrics,
        "dynamics": dynamics,
        "export": {
            "asset_stem": asset_stem,
            "ground_shift_m": float(ground_shift),
            "body_names": clip.body_names,
            "dof_names": clip.dof_names,
            "coupled_knee_report": base_export.coupled_report,
            "inertia_report": base_export.inertia_report,
            "collision_counts": collision_counts,
            "contact_body_frame_counts": contact_counts,
            "contact_vertical_force_threshold_N": contact_thresholds,
            "bone_meshes": mesh_record,
        },
        "validation": validation,
        "outputs": outputs,
        "training": {
            "environment": {
                "BIOMECH_ASSET_ROOT": str((publish_dir / "assets").resolve()),
                "BIOMECH_ASSET_STEM": asset_stem,
                "BIOMECH_FOOT_COLLISION": "boxes",
            },
            "command": [
                "python",
                "protomotions/train_agent.py",
                "--robot-name",
                "biomech",
                "--simulator",
                "newton",
                "--experiment-path",
                "projects/biomech/experiments/mimic_newton.py",
                "--motion-file",
                str((publish_dir / "motions" / motion_path.name).resolve()),
                "--experiment-name",
                f"{asset_stem}_mimic",
                "--num-envs",
                "1024",
                "--batch-size",
                "16384",
                "--ngpu",
                "1",
            ],
        },
    }
    staged_manifest = stage_dir / "manifest.json"
    _atomic_write_text(
        staged_manifest, json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    try:
        if publish_dir.exists():
            existing = _try_reuse(manifest_path, publish_dir, fingerprint)
            if existing is None:
                raise FileExistsError(
                    f"immutable bundle path already exists but does not verify: {publish_dir}"
                )
            _verify_equivalent_bundle(publish_dir, stage_dir, outputs)
            stage_context.cleanup()
        else:
            os.replace(stage_dir, publish_dir)
            stage_context.cleanup()
        scratch_context.cleanup()
    except Exception:
        stage_context.cleanup()
        scratch_context.cleanup()
        raise
    manifest_path = publish_dir / "manifest.json"
    motion_path = publish_dir / "motions" / motion_path.name
    asset_paths = {
        scheme: publish_dir / path.relative_to(stage_dir)
        for scheme, path in staged_asset_paths.items()
    }
    return PipelineResult(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        motion_path=motion_path,
        asset_paths=asset_paths,
        fingerprint=fingerprint,
        reused=False,
    )
