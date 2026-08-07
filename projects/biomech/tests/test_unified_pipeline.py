# SPDX-License-Identifier: MIT

"""Fast contracts for the repeatable C3D-to-ProtoMotions pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from biomech.pipeline import (
    QualityThresholds,
    _regularize_unobservable_scales,
    _safe_bundle_path,
    _try_reuse,
    apply_bilateral_scale_symmetry,
    canonical_json,
    evaluate_delivered_quality,
    evaluate_quality,
    fingerprint_payload,
    robust_marker_weights,
    sha256_file,
)


def test_pipeline_fingerprint_is_canonical_and_content_sensitive():
    a = {"settings": {"b": 2, "a": 1}, "inputs": ["x", "y"]}
    b = {"inputs": ["x", "y"], "settings": {"a": 1, "b": 2}}
    assert canonical_json(a) == canonical_json(b)
    assert fingerprint_payload(a) == fingerprint_payload(b)
    assert fingerprint_payload(a) != fingerprint_payload({**a, "version": 2})
    try:
        canonical_json({"bad": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("canonical JSON must reject NaN")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.bin"
        path.write_bytes(b"abc")
        assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_motion_serialization_round_trips():
    import torch

    from biomech.pipeline import _atomic_torch_save

    data = {"x": torch.tensor([[1.0, 2.0]]), "fps": 100.0}
    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "first.motion"
        second = Path(tmp) / "second.motion"
        _atomic_torch_save(first, data)
        _atomic_torch_save(second, data)
        first_data = torch.load(first, weights_only=False)
        second_data = torch.load(second, weights_only=False)
        assert torch.equal(first_data["x"], second_data["x"])
        assert first_data["fps"] == second_data["fps"]


def test_reuse_rejects_missing_or_changed_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        motion = bundle / "motion.motion"
        asset = bundle / "asset.xml"
        fit = bundle / "fit.npz"
        motion.parent.mkdir(parents=True)
        motion.write_bytes(b"motion")
        asset.write_bytes(b"asset")
        fit.write_bytes(b"fit")
        fingerprint = "abc"
        manifest = {
            "status": "complete",
            "fingerprint": fingerprint,
            "outputs": {
                "motion": {
                    "path": "motion.motion",
                    "sha256": sha256_file(motion),
                    "size_bytes": motion.stat().st_size,
                },
                "asset_none": {
                    "path": "asset.xml",
                    "sha256": sha256_file(asset),
                    "size_bytes": asset.stat().st_size,
                },
                "fit": {
                    "path": "fit.npz",
                    "sha256": sha256_file(fit),
                    "size_bytes": fit.stat().st_size,
                },
            },
            "export": {"bone_meshes": None},
            "settings": {"collision_schemes": []},
        }
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        assert _try_reuse(manifest_path, bundle, fingerprint) is not None
        asset.write_bytes(b"changed")
        assert _try_reuse(manifest_path, bundle, fingerprint) is None


def test_bundle_paths_cannot_escape_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        assert _safe_bundle_path(bundle, "fit/data.npz") == (
            bundle / "fit" / "data.npz"
        ).resolve()
        for invalid in ("../outside", str((Path(tmp) / "absolute").resolve())):
            try:
                _safe_bundle_path(bundle, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted escaping bundle path {invalid!r}")


def test_pipeline_quality_gates_fail_closed():
    thresholds = QualityThresholds(
        max_marker_rms_m=0.02,
        max_anatomical_marker_rms_m=0.015,
        min_scale=0.5,
        max_scale=1.6,
        min_visible_markers=8,
    )
    base = dict(
        scales=np.ones(6),
        poses=np.zeros((3, 2)),
        lower=np.full(2, -1.0),
        upper=np.full(2, 1.0),
        marker_rms_m=0.02,
        anatomical_marker_rms_m=0.015,
        min_visible_markers=8,
        thresholds=thresholds,
    )
    assert evaluate_quality(**base) == []

    cases = [
        ("marker_rms_m", {"marker_rms_m": 0.02001}),
        ("anatomical_marker_rms_m", {"anatomical_marker_rms_m": np.nan}),
        ("finite_scales", {"scales": np.array([1.0, np.inf])}),
        ("scale_lower_bound", {"scales": np.array([0.5, 1.0])}),
        ("joint_limits", {"poses": np.array([[0.0, 1.1], [0.0, 0.0]])}),
        ("visible_markers", {"min_visible_markers": 7}),
    ]
    for expected, override in cases:
        args = {**base, **override}
        failures = evaluate_quality(**args)
        assert expected in {failure["gate"] for failure in failures}, failures


def test_delivered_quality_uses_corrected_errors_and_limits():
    thresholds = QualityThresholds()
    failures = evaluate_delivered_quality(
        raw_marker_rms_m=0.020,
        delivered_marker_rms_m=0.040,
        delivered_anatomical_marker_rms_m=0.030,
        poses=np.array([[0.0, 1.1], [0.0, 0.0]]),
        lower=np.full(2, -1.0),
        upper=np.full(2, 1.0),
        thresholds=thresholds,
    )
    gates = {failure["gate"] for failure in failures}
    assert "delivered_marker_rms_m" in gates
    assert "delivered_anatomical_marker_rms_m" in gates
    assert "delivered_marker_degradation" in gates
    assert "delivered_joint_limits" in gates


def test_robust_weights_include_collapsed_lower_body_centroids():
    class Mapping:
        model_to_capture = {"RASI": "RASI", "RTH_C": ("RTHI", "RTH2"), "RUA1": "RUPA"}

    names = ["RASI", "RTH_C", "RUA1", "unmapped"]
    anatomical = np.array([True, False, False, False])
    robust = robust_marker_weights(names, Mapping(), anatomical, "robust-anatomical")
    balanced = robust_marker_weights(names, Mapping(), anatomical, "balanced-anatomical")
    assert np.allclose(robust, [4.0, 0.35, 0.15, 0.0])
    assert np.allclose(balanced, [4.0, 0.5, 0.25, 0.0])


def test_bilateral_scale_symmetry_pairs_homologous_segments():
    from biomech.osim import parse_osim

    root = Path(__file__).resolve().parents[1]
    spec = parse_osim(str(root / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    groups = apply_bilateral_scale_symmetry(spec)
    assert ["femur_r", "femur_l"] in groups
    assert ["toes_r", "toes_l"] in groups
    assert ["pelvis"] in groups and ["torso"] in groups
    assert sorted(body for group in groups for body in group) == sorted(
        body.name for body in spec.bodies
    )


def test_unobservable_scale_axes_are_neutralized_without_fk_change():
    try:
        import warp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        from biomech.tests import SkipTest

        raise SkipTest(f"warp not available: {exc}")

    from biomech.fitting.marker_placement import unlock_mtp
    from biomech.osim import parse_osim
    from biomech.skeleton.skeleton import WarpSkeleton

    root = Path(__file__).resolve().parents[1]
    spec = parse_osim(str(root / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    unlock_mtp(spec)
    skeleton = WarpSkeleton(spec, device="cpu")
    scales = np.ones(3 * len(spec.scale_groups))
    toes_l = next(i for i, group in enumerate(spec.scale_groups) if "toes_l" in group)
    scales[3 * toes_l + 2] = 0.5
    offsets = skeleton.marker_offsets().copy()
    poses = np.zeros((2, spec.num_dofs))
    poses[:, spec.dof_index_map()["pelvis_ty"]] = 1.0
    world_before, markers_before = skeleton.forward(poses, scales)

    fixed_scales, changes = _regularize_unobservable_scales(spec, scales, offsets)
    world_after, markers_after = skeleton.forward(poses, fixed_scales)
    assert fixed_scales[3 * toes_l + 2] == 1.0
    assert changes and changes[0]["reason"] == "no marker or joint-offset sensitivity"
    assert np.allclose(world_before, world_after, atol=1e-10)
    assert np.allclose(markers_before, markers_after, atol=1e-10)


def test_fitted_marker_offsets_can_be_persisted_in_spec():
    from biomech.osim import parse_osim
    from biomech.skeleton.skeleton import WarpSkeleton

    root = Path(__file__).resolve().parents[1]
    spec = parse_osim(str(root / "models" / "rajagopal_data" / "Rajagopal2015.osim"))
    skeleton = WarpSkeleton(spec, device="cpu")
    offsets = skeleton.marker_offsets().copy()
    offsets[0] += np.array([0.001, -0.002, 0.003])
    for marker, offset in zip(spec.markers, offsets):
        marker.offset = offset.copy()
    rebuilt = WarpSkeleton(spec, device="cpu")
    assert np.allclose(rebuilt.marker_offsets(), offsets)


def test_unified_cli_defaults_to_fidelity_path():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "c3d_to_protomotions.py"
    spec = importlib.util.spec_from_file_location("biomech_c3d_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args([])
    config = module.config_from_args(args)
    assert config.device == "cpu"
    assert config.clean_markers
    assert config.bilateral_scale_symmetry
    assert config.collapse_clusters
    assert config.enrich_foot_markers
    assert config.treadmill_to_overground
    assert config.ground_register
    assert config.measured_contacts
    assert config.contact_load_fraction == 0.0
    assert config.contact_load_floor_n == config.fz_threshold
    assert config.marker_weight_profile == "robust-anatomical"
    json.dumps(config.quality.__dict__, allow_nan=False)


def test_unified_cli_reports_input_errors_without_traceback():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "c3d_to_protomotions.py"
    spec = importlib.util.spec_from_file_location("biomech_c3d_cli_error", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.c3d"
        code = module.main(
            [
                "--trial",
                str(missing),
                "--static",
                str(missing),
                "--output-root",
                str(Path(tmp) / "out"),
            ]
        )
    assert code == 2
