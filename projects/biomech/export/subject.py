# SPDX-License-Identifier: MIT
#
# Windows-native analog of Nimble's ``SubjectOnDisk`` / ``.b3d`` serialized output
# (``dart/biomechanics/SubjectOnDisk.*``). Nimble writes a fitted subject (skeleton
# scaling, marker offsets, per-frame poses, and dynamics results) to a single portable
# file so downstream tools (and AddBiomechanics) can reload a finished fit without
# re-running the pipeline. Our fit is Windows-native and NumPy-based, so instead of
# Nimble's protobuf ``.b3d`` we serialize to a single ``.npz`` bundle (numeric arrays)
# plus an embedded JSON header (names / scalars / metadata). It reloads to a
# :class:`FittedSubject` that can regenerate the M3 MJCF and the ProtoMotions motion clip.

"""Serialized fitted-subject bundle (Windows-native ``SubjectOnDisk``/``.b3d`` analog)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_FORMAT = "biomech.FittedSubject"
_VERSION = 1


@dataclass
class FittedSubject:
    """A complete kinematic (and optionally inertial) fit of one subject.

    Core kinematic fit
    ------------------
    group_scales : (3G,)   anisotropic per-scale-group scale
    marker_offsets : (M, 3) fitted marker local offsets (model frame)
    marker_names : list[str] marker order for ``marker_offsets``
    poses : (F, ndof)      fitted gold-standard DART/OpenSim generalized coordinates q(t)
    fps : float            sample rate of ``poses``

    Optional
    --------
    marker_rms : (F,)      per-frame weighted marker RMS (m)
    anatomical : (M,) bool anatomical-landmark flags for the markers
    body_names / dof_names : anatomical body / sim-DOF ordering (for the motion clip)
    inertial_params : (nb, 10) fitted [m, m*c, I_origin] per body (M2e)
    inertial_body_names : list[str] body order for ``inertial_params``
    mjcf_xml : str         exported MJCF (so the sim model reloads without re-export)
    coupled_knee : str     knee-coupling export mode used ("coupled"/"hinge")
    osim_path : str        source .osim (provenance; used by :meth:`spec` to re-parse)
    metadata : dict        free-form provenance (subject id, trial, phase, calibration)
    """

    group_scales: np.ndarray
    marker_offsets: np.ndarray
    marker_names: list[str]
    poses: np.ndarray
    fps: float
    marker_rms: Optional[np.ndarray] = None
    anatomical: Optional[np.ndarray] = None
    body_names: list[str] = field(default_factory=list)
    dof_names: list[str] = field(default_factory=list)
    inertial_params: Optional[np.ndarray] = None
    inertial_body_names: list[str] = field(default_factory=list)
    mjcf_xml: Optional[str] = None
    coupled_knee: str = "coupled"
    osim_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_marker_fit(
        cls,
        spec,
        result,
        fps: float,
        *,
        osim_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> "FittedSubject":
        """Build from a :class:`biomech.fitting.marker_fitter.MarkerFitResult`."""
        return cls(
            group_scales=np.asarray(result.group_scales, dtype=np.float64),
            marker_offsets=np.asarray(result.marker_offsets, dtype=np.float64),
            marker_names=[m.name for m in spec.markers],
            poses=np.asarray(result.poses, dtype=np.float64),
            fps=float(fps),
            marker_rms=np.asarray(result.marker_rms, dtype=np.float64),
            anatomical=np.array([m.anatomical for m in spec.markers], dtype=bool),
            osim_path=osim_path,
            metadata=dict(metadata or {}),
        )

    # -- derived artifacts -------------------------------------------------
    def spec(self):
        """Re-parse the source ``SkeletonSpec`` (requires ``osim_path``)."""
        if not self.osim_path:
            raise ValueError("osim_path not stored; cannot re-parse the SkeletonSpec")
        from biomech.osim import parse_osim

        return parse_osim(self.osim_path)

    def to_mjcf(self, spec=None, bone_meshes: bool = False):
        """Export the fitted skeleton to MJCF (uses stored ``mjcf_xml`` if present).

        ``bone_meshes=True`` renders each body with its OpenSim bone mesh(es); the
        converted STL meshes must exist (see ``tools/convert_bone_meshes.py``).
        """
        if self.mjcf_xml is not None and spec is None and not bone_meshes:
            return self.mjcf_xml
        from biomech.export.mjcf import export_mjcf

        mesh_map = None
        if bone_meshes:
            from biomech.export.bone_geometry import default_bone_geometry

            mesh_map = default_bone_geometry()

        spec = spec or self.spec()
        return export_mjcf(
            spec,
            group_scales=self.group_scales,
            coupled_knee=self.coupled_knee,
            bone_meshes=mesh_map,
        ).xml

    def to_motion(self, spec=None, device: str = "cpu"):
        """Build the ProtoMotions motion clip from the fitted poses."""
        from biomech.export.motion import build_motion

        spec = spec or self.spec()
        return build_motion(
            spec,
            self.poses,
            self.fps,
            group_scales=self.group_scales,
            coupled_knee=self.coupled_knee,
            device=device,
        )

    # -- I/O ---------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        return save_subject(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FittedSubject":
        return load_subject(path)


def save_subject(subject: FittedSubject, path: str | Path) -> Path:
    """Serialize a :class:`FittedSubject` to a single ``.npz`` bundle."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    arrays: dict[str, np.ndarray] = {
        "group_scales": np.asarray(subject.group_scales, dtype=np.float64),
        "marker_offsets": np.asarray(subject.marker_offsets, dtype=np.float64),
        "poses": np.asarray(subject.poses, dtype=np.float64),
    }
    if subject.marker_rms is not None:
        arrays["marker_rms"] = np.asarray(subject.marker_rms, dtype=np.float64)
    if subject.anatomical is not None:
        arrays["anatomical"] = np.asarray(subject.anatomical, dtype=bool)
    if subject.inertial_params is not None:
        arrays["inertial_params"] = np.asarray(subject.inertial_params, dtype=np.float64)

    header = {
        "format": _FORMAT,
        "version": _VERSION,
        "fps": float(subject.fps),
        "marker_names": list(subject.marker_names),
        "body_names": list(subject.body_names),
        "dof_names": list(subject.dof_names),
        "inertial_body_names": list(subject.inertial_body_names),
        "coupled_knee": subject.coupled_knee,
        "osim_path": subject.osim_path,
        "mjcf_xml": subject.mjcf_xml,
        "metadata": subject.metadata,
        "array_keys": sorted(arrays.keys()),
    }
    # JSON header stored as a 0-d unicode array (portable inside .npz)
    header_arr = np.array(json.dumps(header), dtype=object)
    np.savez_compressed(path, __header__=header_arr, **arrays)
    return path


def load_subject(path: str | Path) -> FittedSubject:
    """Load a :class:`FittedSubject` written by :func:`save_subject`."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        header = json.loads(str(npz["__header__"].item()))
        if header.get("format") != _FORMAT:
            raise ValueError(f"not a FittedSubject bundle: {path}")

        def get(key):
            return npz[key] if key in header["array_keys"] else None

        marker_rms = get("marker_rms")
        anatomical = get("anatomical")
        inertial = get("inertial_params")
        return FittedSubject(
            group_scales=np.array(npz["group_scales"], dtype=np.float64),
            marker_offsets=np.array(npz["marker_offsets"], dtype=np.float64),
            marker_names=list(header["marker_names"]),
            poses=np.array(npz["poses"], dtype=np.float64),
            fps=float(header["fps"]),
            marker_rms=None if marker_rms is None else np.array(marker_rms, dtype=np.float64),
            anatomical=None if anatomical is None else np.array(anatomical, dtype=bool),
            body_names=list(header["body_names"]),
            dof_names=list(header["dof_names"]),
            inertial_params=None if inertial is None else np.array(inertial, dtype=np.float64),
            inertial_body_names=list(header["inertial_body_names"]),
            mjcf_xml=header.get("mjcf_xml"),
            coupled_knee=header.get("coupled_knee", "coupled"),
            osim_path=header.get("osim_path"),
            metadata=dict(header.get("metadata", {})),
        )
