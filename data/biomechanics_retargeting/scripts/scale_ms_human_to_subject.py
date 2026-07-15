# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scale the simplified MS-Human lower-body MJCF to one subject from C3D markers.

This is the first implementation milestone for subject-specific digital twins:
use an initial static pose window in a Visual3D/Vicon C3D file to estimate the
subject's pelvis, thigh, shank, and foot dimensions, then scale the corresponding
MS-Human body offsets in the generated MJCF while preserving the original joint
axes and anatomical topology.

Only kinematic geometry is scaled here.  Segment masses and inertias are left as
in the base model for now, by design.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from utils.c3d_io import load_c3d, marker_index, read_metadata  # noqa: E402


BASE_XML = Path("protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-Simple.xml")
DEFAULT_OUTPUT_XML = Path("protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003.xml")
DEFAULT_REPORT = Path("data/biomechanics_retargeting/retargeted/S003_scaling_report.json")
DEFAULT_OUTPUT_XML_TEMPLATE = "MS-Human-700-Locomotion-{subject_id}.xml"
DEFAULT_REPORT_TEMPLATE = "{subject_id}_scaling_report.json"


BODY_SCALE_GROUPS = {
    "hip_offset_r": ["femur_r"],
    "hip_offset_l": ["femur_l"],
    "thigh_r": ["tibia_r", "patella_r"],
    "thigh_l": ["tibia_l", "patella_l"],
    "shank_r": ["talus_r"],
    "shank_l": ["talus_l"],
    "foot_r": ["calcn_r", "toes_r"],
    "foot_l": ["calcn_l", "toes_l"],
    "torso": [
        "sacrum",
        "Abdomen",
        "lumbar5",
        "lumbar4",
        "lumbar3",
        "lumbar2",
        "lumbar1",
        "thoracic12",
        "thoracic11",
        "thoracic10",
        "thoracic9",
        "thoracic8",
        "thoracic7",
        "thoracic6",
        "thoracic5",
        "thoracic4",
        "thoracic3",
        "thoracic2",
        "thoracic1",
        "head_neck",
        "sternum",
    ],
}


# Bodies whose attached geometry (skin capsules, muscle wrap surfaces) and
# muscle sites represent each scaled segment.  BODY_SCALE_GROUPS only moves the
# child joint offsets (segment lengths); without also scaling the geometry on
# the segment body the visual / collision shapes keep their original size and
# leave a visible gap between segments (most notably between the shank capsule
# and the foot when the shank is lengthened).  Each body's geometry is scaled
# about the body origin so it tracks the new joint locations.
GEOM_SCALE_GROUPS = {
    "thigh_r": ["femur_r"],
    "thigh_l": ["femur_l"],
    "shank_r": ["tibia_r"],
    "shank_l": ["tibia_l"],
    "foot_r": ["talus_r", "calcn_r", "toes_r"],
    "foot_l": ["talus_l", "calcn_l", "toes_l"],
}


# Marker sites attached directly to pelvis need lateral subject scaling too, but
# applying pelvis_width to all pelvis geoms would distort the pelvis collision
#/visual geometry.  Keep this limited to generated mocap marker sites.
MARKER_SITE_SCALE_GROUPS = {
    "pelvis_width": ["pelvis"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("c3d", type=Path, help="Subject C3D file with an initial static pose window.")
    parser.add_argument("--base-xml", type=Path, default=BASE_XML, help="Base simplified MS-Human MJCF.")
    parser.add_argument(
        "--marker-config",
        type=Path,
        help="JSON config defining marker-derived points and segment lengths used for scaling.",
    )
    parser.add_argument(
        "--subject-id",
        help="Subject identifier used in default output/report names; overrides marker_config subject_id.",
    )
    parser.add_argument("--output-xml", type=Path, help="Scaled subject MJCF.")
    parser.add_argument("--report", type=Path, help="JSON scaling report path.")
    parser.add_argument("--static-start", type=int, help="First 1-based C3D static frame.")
    parser.add_argument("--static-end", type=int, help="Last 1-based C3D static frame, inclusive.")
    parser.add_argument(
        "--auto-static",
        action="store_true",
        help="Detect a low-marker-velocity static window near the beginning instead of using --static-end.",
    )
    parser.add_argument(
        "--auto-window",
        type=int,
        default=100,
        help="Window length in frames for --auto-static.",
    )
    parser.add_argument(
        "--auto-search-frames",
        type=int,
        default=1000,
        help="Number of initial frames to search for --auto-static.",
    )
    parser.add_argument(
        "--copy-base-asset-dir",
        action="store_true",
        help="Copy the whole base MJCF directory before writing output if output lives elsewhere.",
    )
    parser.add_argument(
        "--add-marker-sites",
        action="store_true",
        help="Bake marker_config marker_sites from the static C3D window into the output MJCF as mocap_* sites.",
    )
    parser.add_argument("--site-prefix", default="mocap_", help="Prefix for generated marker site names.")
    parser.add_argument("--site-size", default="0.008", help="Generated marker site size attribute.")
    parser.add_argument("--site-rgba", default="0.1 0.45 1 1", help="Generated marker site rgba attribute.")
    parser.add_argument(
        "--strict-marker-sites",
        action="store_true",
        help="Fail when any configured marker site cannot be resolved instead of skipping it.",
    )
    parser.add_argument(
        "--list-markers",
        action="store_true",
        help="Print C3D marker labels and exit without writing a scaled MJCF.",
    )
    return parser.parse_args()


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-8, None)


def _mean_valid(points: np.ndarray) -> np.ndarray:
    return np.nanmean(points, axis=0)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _marker(data, name: str) -> np.ndarray:
    return data.markers[:, marker_index(data.marker_labels, name)]


def _optional_mean_marker(data, name: str) -> np.ndarray | None:
    try:
        return _mean_valid(_marker(data, name))
    except KeyError:
        return None


def load_marker_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _marker_mean_m(data, name: str, unit_scale: float) -> np.ndarray:
    return _mean_valid(_marker(data, name)) * unit_scale


def _marker_mean_many_m(data, names: list[str], unit_scale: float) -> np.ndarray:
    if not names:
        raise ValueError("Point config mean/midpoint requires at least one marker")
    points = np.stack([_marker_mean_m(data, name, unit_scale) for name in names], axis=0)
    return _mean_valid(points)


def _pelvis_hip_center_from_markers(
    data,
    spec: Mapping[str, Any],
    unit_scale: float,
) -> np.ndarray:
    side = str(spec.get("side", "right")).lower()
    if side not in {"right", "left"}:
        raise ValueError(f"pelvis_hip_center side must be 'right' or 'left', got {side!r}")

    rasi = _marker_mean_m(data, str(spec.get("rasi", "RASI")), unit_scale)
    lasi = _marker_mean_m(data, str(spec.get("lasi", "LASI")), unit_scale)
    rpsi = _marker_mean_m(data, str(spec.get("rpsi", "RPSI")), unit_scale)
    lpsi = _marker_mean_m(data, str(spec.get("lpsi", "LPSI")), unit_scale)

    asis_mid = 0.5 * (rasi + lasi)
    psis_mid = 0.5 * (rpsi + lpsi)
    asis_distance = _distance(rasi, lasi)

    ml_axis = _norm((rasi - lasi)[None])[0]
    ap_axis = _norm((asis_mid - psis_mid)[None])[0]
    axial_axis = _norm(np.cross(ml_axis, ap_axis)[None])[0]

    ml_coeff = float(spec.get("ml_coeff", 0.36))
    ap_coeff = float(spec.get("ap_coeff", -0.19))
    axial_coeff = float(spec.get("axial_coeff", -0.30))
    ml_sign = 1.0 if side == "right" else -1.0
    return (
        asis_mid
        + ml_sign * ml_coeff * asis_distance * ml_axis
        + ap_coeff * asis_distance * ap_axis
        + axial_coeff * asis_distance * axial_axis
    )


def _resolve_point(
    data,
    name: str,
    point_specs: Mapping[str, Any],
    resolved: dict[str, np.ndarray],
    unit_scale: float,
) -> np.ndarray:
    if name in resolved:
        return resolved[name]
    if name not in point_specs:
        raise KeyError(f"Point {name!r} is referenced by marker config but is not defined")

    spec = point_specs[name]
    if isinstance(spec, str):
        point = _marker_mean_m(data, spec, unit_scale)
    elif isinstance(spec, list):
        point = _marker_mean_many_m(data, [str(marker) for marker in spec], unit_scale)
    elif isinstance(spec, Mapping):
        if "marker" in spec:
            point = _marker_mean_m(data, str(spec["marker"]), unit_scale)
        elif "mean" in spec:
            point = _marker_mean_many_m(
                data,
                [str(marker) for marker in spec["mean"]],
                unit_scale,
            )
        elif "midpoint" in spec:
            point = _marker_mean_many_m(
                data,
                [str(marker) for marker in spec["midpoint"]],
                unit_scale,
            )
        elif "pelvis_hip_center" in spec:
            point = _pelvis_hip_center_from_markers(data, spec["pelvis_hip_center"], unit_scale)
        elif "from_point" in spec and "to_point" in spec and "fraction" in spec:
            start = _resolve_point(data, str(spec["from_point"]), point_specs, resolved, unit_scale)
            end = _resolve_point(data, str(spec["to_point"]), point_specs, resolved, unit_scale)
            point = start + float(spec["fraction"]) * (end - start)
        else:
            raise ValueError(f"Unsupported point config for {name!r}: {spec}")
    else:
        raise TypeError(f"Unsupported point config type for {name!r}: {type(spec).__name__}")

    resolved[name] = point
    return point


def compute_subject_lengths_from_config(
    c3d: Path,
    static_start: int,
    static_end: int,
    marker_config: Mapping[str, Any],
) -> dict[str, float]:
    data = load_c3d(c3d, start_frame=static_start, end_frame=static_end)
    unit_scale = 0.001 if data.point_units.lower() in {"mm", "millimeter", "millimeters"} else 1.0
    point_specs = marker_config.get("points", {})
    length_specs = marker_config.get("lengths", {})
    if not isinstance(point_specs, Mapping) or not isinstance(length_specs, Mapping):
        raise TypeError("marker config must contain object-valued 'points' and 'lengths' sections")

    resolved: dict[str, np.ndarray] = {}
    subject_lengths: dict[str, float] = {}
    for length_name, spec in length_specs.items():
        if not isinstance(spec, Mapping):
            raise TypeError(f"Length config for {length_name!r} must be an object")
        if "markers" in spec:
            markers = [str(marker) for marker in spec["markers"]]
            if len(markers) != 2:
                raise ValueError(
                    f"Length config {length_name!r} markers field must contain exactly two markers"
                )
            start = _marker_mean_m(data, markers[0], unit_scale)
            end = _marker_mean_m(data, markers[1], unit_scale)
        else:
            start = _resolve_point(data, str(spec["from"]), point_specs, resolved, unit_scale)
            end = _resolve_point(data, str(spec["to"]), point_specs, resolved, unit_scale)
        subject_lengths[str(length_name)] = _distance(start, end)
    return subject_lengths


def detect_static_window(
    c3d: Path,
    start_frame: int,
    search_frames: int,
    window: int,
    labels: list[str] | None = None,
) -> tuple[int, int]:
    metadata = read_metadata(c3d)
    end = min(metadata.header.last_frame, start_frame + search_frames - 1)
    data = load_c3d(c3d, start_frame=start_frame, end_frame=end)
    labels = labels or [
        "RASI",
        "LASI",
        "RPSI",
        "LPSI",
        "RKNE",
        "LKNE",
        "RANK",
        "LANK",
        "RTOE",
        "LTOE",
    ]
    marker_ids = [marker_index(data.marker_labels, label) for label in labels]
    pts = data.markers[:, marker_ids]
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    speed = np.nanmedian(speed, axis=1)
    if speed.shape[0] < window:
        return start_frame, start_frame + data.markers.shape[0] - 1
    scores = np.array([np.nanmedian(speed[i : i + window]) for i in range(speed.shape[0] - window + 1)])
    best = int(np.nanargmin(scores))
    return data.start_frame + best, data.start_frame + best + window - 1


def compute_subject_lengths(c3d: Path, static_start: int, static_end: int) -> dict[str, float]:
    data = load_c3d(c3d, start_frame=static_start, end_frame=static_end)
    unit_scale = 0.001 if data.point_units.lower() in {"mm", "millimeter", "millimeters"} else 1.0

    rasi = _mean_valid(_marker(data, "RASI")) * unit_scale
    lasi = _mean_valid(_marker(data, "LASI")) * unit_scale
    rpsi = _mean_valid(_marker(data, "RPSI")) * unit_scale
    lpsi = _mean_valid(_marker(data, "LPSI")) * unit_scale

    asis_mid = 0.5 * (rasi + lasi)
    psis_mid = 0.5 * (rpsi + lpsi)
    asis_distance = _distance(rasi, lasi)

    ml_axis = _norm((rasi - lasi)[None])[0]
    ap_axis = _norm((asis_mid - psis_mid)[None])[0]
    axial_axis = _norm(np.cross(ml_axis, ap_axis)[None])[0]

    right_hip_modelled = _optional_mean_marker(data, "RIGHT_HIP")
    left_hip_modelled = _optional_mean_marker(data, "LEFT_HIP")
    if right_hip_modelled is not None and left_hip_modelled is not None:
        right_hip = right_hip_modelled * unit_scale
        left_hip = left_hip_modelled * unit_scale
    else:
        # Visual3D/CODA pelvis hip-center formula from the MDH:
        # MCS_ML = +/-0.36*ASIS, MCS_AP = -0.19*ASIS, MCS_AXIAL = -0.30*ASIS.
        right_hip = (
            asis_mid
            + 0.36 * asis_distance * ml_axis
            - 0.19 * asis_distance * ap_axis
            - 0.30 * asis_distance * axial_axis
        )
        left_hip = (
            asis_mid
            - 0.36 * asis_distance * ml_axis
            - 0.19 * asis_distance * ap_axis
            - 0.30 * asis_distance * axial_axis
        )

    rkne = _mean_valid(_marker(data, "RKNE")) * unit_scale
    lkne = _mean_valid(_marker(data, "LKNE")) * unit_scale
    rank = _mean_valid(_marker(data, "RANK")) * unit_scale
    lank = _mean_valid(_marker(data, "LANK")) * unit_scale
    rtoe = _mean_valid(_marker(data, "RTOE")) * unit_scale
    ltoe = _mean_valid(_marker(data, "LTOE")) * unit_scale
    rhee = _mean_valid(_marker(data, "RHEE")) * unit_scale
    lhee = _mean_valid(_marker(data, "LHEE")) * unit_scale

    rknem = _optional_mean_marker(data, "RKNEM")
    lknem = _optional_mean_marker(data, "LKNEM")
    rankm = _optional_mean_marker(data, "RANKM")
    lankm = _optional_mean_marker(data, "LANKM")
    rknem = rknem * unit_scale if rknem is not None else rkne
    lknem = lknem * unit_scale if lknem is not None else lkne
    rankm = rankm * unit_scale if rankm is not None else rank
    lankm = lankm * unit_scale if lankm is not None else lank

    r_knee_center = 0.5 * (rkne + rknem)
    l_knee_center = 0.5 * (lkne + lknem)
    r_ankle_center = 0.5 * (rank + rankm)
    l_ankle_center = 0.5 * (lank + lankm)

    torso_length = math.nan
    try:
        c7 = _mean_valid(_marker(data, "C7")) * unit_scale
        torso_length = _distance(asis_mid, c7)
    except KeyError:
        pass

    return {
        "pelvis_width": asis_distance,
        "hip_offset_r": _distance(asis_mid, right_hip),
        "hip_offset_l": _distance(asis_mid, left_hip),
        "thigh_r": _distance(right_hip, r_knee_center),
        "thigh_l": _distance(left_hip, l_knee_center),
        "shank_r": _distance(r_knee_center, r_ankle_center),
        "shank_l": _distance(l_knee_center, l_ankle_center),
        "foot_r": _distance(rhee, rtoe),
        "foot_l": _distance(lhee, ltoe),
        "torso": torso_length,
    }


def _body_by_name(root: ET.Element) -> dict[str, ET.Element]:
    return {body.get("name"): body for body in root.iter("body") if body.get("name")}


def _parse_vec(text: str | None) -> np.ndarray:
    if text is None or not text.strip():
        return np.zeros(3, dtype=np.float64)
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def _format_vec(vec: np.ndarray) -> str:
    return " ".join(f"{x:.10g}" for x in vec.tolist())


def compute_base_lengths(root: ET.Element) -> dict[str, float]:
    bodies = _body_by_name(root)
    foot_r = np.linalg.norm(_parse_vec(bodies["calcn_r"].get("pos"))) + np.linalg.norm(
        _parse_vec(bodies["toes_r"].get("pos"))
    )
    foot_l = np.linalg.norm(_parse_vec(bodies["calcn_l"].get("pos"))) + np.linalg.norm(
        _parse_vec(bodies["toes_l"].get("pos"))
    )
    torso_names = [
        "sacrum",
        "lumbar5",
        "lumbar4",
        "lumbar3",
        "lumbar2",
        "lumbar1",
        "thoracic12",
        "thoracic11",
        "thoracic10",
        "thoracic9",
        "thoracic8",
        "thoracic7",
        "thoracic6",
        "thoracic5",
        "thoracic4",
        "thoracic3",
        "thoracic2",
        "thoracic1",
        "head_neck",
    ]
    torso = sum(np.linalg.norm(_parse_vec(bodies[name].get("pos"))) for name in torso_names if name in bodies)
    base_lengths = {
        "hip_offset_r": float(np.linalg.norm(_parse_vec(bodies["femur_r"].get("pos")))),
        "hip_offset_l": float(np.linalg.norm(_parse_vec(bodies["femur_l"].get("pos")))),
        "thigh_r": float(np.linalg.norm(_parse_vec(bodies["tibia_r"].get("pos")))),
        "thigh_l": float(np.linalg.norm(_parse_vec(bodies["tibia_l"].get("pos")))),
        "shank_r": float(np.linalg.norm(_parse_vec(bodies["talus_r"].get("pos")))),
        "shank_l": float(np.linalg.norm(_parse_vec(bodies["talus_l"].get("pos")))),
        "foot_r": float(foot_r),
        "foot_l": float(foot_l),
        "torso": float(torso),
    }
    rasi = bodies["pelvis"].find("site[@name='mocap_RASI']")
    lasi = bodies["pelvis"].find("site[@name='mocap_LASI']")
    if rasi is not None and lasi is not None:
        base_lengths["pelvis_width"] = float(np.linalg.norm(_parse_vec(rasi.get("pos")) - _parse_vec(lasi.get("pos"))))
    return base_lengths


def apply_scales(root: ET.Element, scales: dict[str, float]) -> dict[str, dict[str, list[float]]]:
    bodies = _body_by_name(root)
    changed: dict[str, dict[str, list[float]]] = {}
    for scale_name, body_names in BODY_SCALE_GROUPS.items():
        scale = scales.get(scale_name, 1.0)
        if not np.isfinite(scale) or scale <= 0:
            continue
        for body_name in body_names:
            body = bodies.get(body_name)
            if body is None:
                continue
            old = _parse_vec(body.get("pos"))
            new = old * scale
            body.set("pos", _format_vec(new))
            changed[body_name] = {"old_pos": old.tolist(), "new_pos": new.tolist(), "scale": float(scale)}
    return changed


def _scale_attr(elem: ET.Element, attr: str, scale: float) -> bool:
    text = elem.get(attr)
    if text is None or not text.strip():
        return False
    elem.set(attr, _format_vec(_parse_vec(text) * scale))
    return True


def apply_geometry_scales(root: ET.Element, scales: dict[str, float]) -> dict[str, dict[str, float]]:
    """Scale each segment body's own geoms and sites about the body origin.

    Only the body's direct ``geom``/``site`` children are scaled (not those of
    nested child bodies), so each segment is scaled by exactly its own factor.
    Geom ``pos``/``size``/``fromto`` and site ``pos`` are multiplied by the
    segment scale, lengthening the skin capsules and wrap surfaces to match the
    new joint offsets produced by ``apply_scales``.
    """
    bodies = _body_by_name(root)
    changed: dict[str, dict[str, float]] = {}
    for scale_name, body_names in GEOM_SCALE_GROUPS.items():
        scale = scales.get(scale_name, 1.0)
        if not np.isfinite(scale) or scale <= 0 or abs(scale - 1.0) < 1e-9:
            continue
        for body_name in body_names:
            body = bodies.get(body_name)
            if body is None:
                continue
            count = 0
            for geom in body.findall("geom"):
                count += _scale_attr(geom, "pos", scale)
                count += _scale_attr(geom, "size", scale)
                count += _scale_attr(geom, "fromto", scale)
            for site in body.findall("site"):
                count += _scale_attr(site, "pos", scale)
            changed[body_name] = {"scale": float(scale), "scaled_attrs": int(count)}
    return changed


def _find_or_create_asset(root: ET.Element) -> ET.Element:
    asset = root.find("asset")
    if asset is not None:
        return asset
    asset = ET.Element("asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        root.append(asset)
    else:
        root.insert(list(root).index(worldbody), asset)
    return asset


def _mesh_file_for_name(base_xml_dir: Path, mesh_name: str) -> str | None:
    for suffix in (".stl", ".obj"):
        candidate = Path("Geometry") / f"{mesh_name}{suffix}"
        if (base_xml_dir / candidate).is_file():
            return candidate.as_posix()
    return None


def apply_mesh_asset_scales(
    root: ET.Element,
    scales: dict[str, float],
    base_xml_dir: Path,
) -> dict[str, dict[str, object]]:
    """Define scaled mesh assets for segment bone meshes.

    MuJoCo's implicit mesh loading keeps referenced STL/OBJ vertices at their
    original size.  Scaling body offsets and geom/site positions is not enough
    for ``geom type="mesh"`` bone visuals, because those geoms have no size to
    multiply.  Add explicit mesh assets with a per-segment scale so the visible
    bone mesh follows the scaled joints and marker sites.
    """
    bodies = _body_by_name(root)
    asset = _find_or_create_asset(root)
    changed: dict[str, dict[str, object]] = {}
    for scale_name, body_names in GEOM_SCALE_GROUPS.items():
        scale = scales.get(scale_name, 1.0)
        if not np.isfinite(scale) or scale <= 0 or abs(scale - 1.0) < 1e-9:
            continue
        for body_name in body_names:
            body = bodies.get(body_name)
            if body is None:
                continue
            mesh_names: list[str] = []
            for geom in body.findall("geom"):
                if geom.get("type") != "mesh" or geom.get("mesh") is None:
                    continue
                mesh_name = str(geom.get("mesh"))
                mesh_file = _mesh_file_for_name(base_xml_dir, mesh_name)
                if mesh_file is None:
                    continue
                scaled_mesh_name = f"{mesh_name}_{scale_name}_scaled"
                mesh_elem = asset.find(f"mesh[@name='{scaled_mesh_name}']")
                if mesh_elem is None:
                    ET.SubElement(
                        asset,
                        "mesh",
                        {
                            "name": scaled_mesh_name,
                            "file": mesh_file,
                            "scale": f"{scale:.9g} {scale:.9g} {scale:.9g}",
                        },
                    )
                else:
                    mesh_elem.set("file", mesh_elem.get("file") or mesh_file)
                    mesh_elem.set("scale", f"{scale:.9g} {scale:.9g} {scale:.9g}")
                geom.set("mesh", scaled_mesh_name)
                mesh_names.append(f"{mesh_name}->{scaled_mesh_name}")
            if mesh_names:
                changed[body_name] = {"scale": float(scale), "meshes": mesh_names}
    if not list(asset):
        root.remove(asset)
    return changed


def apply_marker_site_scales(
    root: ET.Element,
    scales: dict[str, float],
    site_prefix: str = "mocap_",
) -> dict[str, dict[str, float]]:
    """Scale generated marker sites that are not covered by geometry scaling."""
    bodies = _body_by_name(root)
    changed: dict[str, dict[str, float]] = {}
    for scale_name, body_names in MARKER_SITE_SCALE_GROUPS.items():
        scale = scales.get(scale_name, 1.0)
        if not np.isfinite(scale) or scale <= 0 or abs(scale - 1.0) < 1e-9:
            continue
        for body_name in body_names:
            body = bodies.get(body_name)
            if body is None:
                continue
            count = 0
            for site in body.findall("site"):
                if (site.get("name") or "").startswith(site_prefix):
                    count += _scale_attr(site, "pos", scale)
            if count:
                changed[body_name] = {"scale": float(scale), "scaled_attrs": int(count)}
    return changed


def _unit_scale(data) -> float:
    return 0.001 if data.point_units.lower() in {"mm", "millimeter", "millimeters"} else 1.0


def _v3d_points_to_model(points: np.ndarray) -> np.ndarray:
    result = np.empty_like(points, dtype=np.float32)
    result[..., 0] = -points[..., 1]
    result[..., 1] = points[..., 0]
    result[..., 2] = points[..., 2]
    return result


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _root_pose_from_static_pelvis(data) -> tuple[np.ndarray, np.ndarray]:
    pelvis_markers = ("RASI", "LASI", "RPSI", "LPSI")
    scale = _unit_scale(data)
    pelvis = np.stack([_marker_mean_m(data, name, scale) for name in pelvis_markers], axis=0)
    pelvis = _v3d_points_to_model(pelvis.astype(np.float32))
    root_pos = np.nanmean(pelvis, axis=0).astype(np.float64)

    rasi, lasi, rpsi, lpsi = pelvis.astype(np.float64)
    x_axis_hint = 0.5 * (rasi + lasi) - 0.5 * (rpsi + lpsi)
    y_axis = lasi - rasi
    x_axis_hint /= np.clip(np.linalg.norm(x_axis_hint), 1e-8, None)
    y_axis /= np.clip(np.linalg.norm(y_axis), 1e-8, None)
    z_axis = np.cross(x_axis_hint, y_axis)
    z_axis /= np.clip(np.linalg.norm(z_axis), 1e-8, None)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.clip(np.linalg.norm(x_axis), 1e-8, None)
    root_rot = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return root_pos, _matrix_to_quat_wxyz(root_rot)


def _resolve_marker_site_position(data, marker_config: Mapping[str, Any], label: str, spec) -> np.ndarray:
    scale = _unit_scale(data)
    if isinstance(spec, str):
        point_v3d = _marker_mean_m(data, label, scale)
    elif isinstance(spec, Mapping):
        if "point" in spec:
            point_specs = marker_config.get("points", {})
            if not isinstance(point_specs, Mapping):
                raise TypeError("marker config 'points' must be an object when marker_sites uses point references")
            point_v3d = _resolve_point(data, str(spec["point"]), point_specs, {}, scale)
        elif "marker" in spec:
            point_v3d = _marker_mean_m(data, str(spec["marker"]), scale)
        else:
            point_v3d = _marker_mean_m(data, label, scale)
    else:
        raise TypeError(f"Unsupported marker site spec for {label!r}: {type(spec).__name__}")
    return _v3d_points_to_model(np.asarray(point_v3d, dtype=np.float32))


def _body_name_from_marker_site_spec(label: str, spec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping) and "body" in spec:
        return str(spec["body"])
    raise ValueError(f"marker_sites entry for {label!r} must be a body name or object with a body field")


def _remove_existing_sites_with_prefix(root: ET.Element, site_prefix: str) -> int:
    removed = 0
    if not site_prefix:
        return removed
    for body in root.iter("body"):
        for child in list(body):
            if child.tag == "site" and (child.get("name") or "").startswith(site_prefix):
                body.remove(child)
                removed += 1
    return removed


def _format_vec(vec: np.ndarray) -> str:
    return " ".join(f"{float(x):.9f}" for x in vec.tolist())


def add_marker_sites_from_static_c3d(
    mjcf: Path,
    c3d: Path,
    marker_config: Mapping[str, Any],
    static_start: int,
    static_end: int,
    *,
    site_prefix: str = "mocap_",
    site_size: str = "0.008",
    site_rgba: str = "0.1 0.45 1 1",
    strict: bool = False,
) -> dict[str, Any]:
    """Bake static C3D marker locations into an already-written MJCF."""
    marker_sites = marker_config.get("marker_sites", {})
    if not isinstance(marker_sites, Mapping):
        raise TypeError("marker config must contain an object-valued marker_sites section")

    import mujoco

    data = load_c3d(c3d, start_frame=static_start, end_frame=static_end)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    mj_data = mujoco.MjData(model)
    root_pos, root_quat = _root_pose_from_static_pelvis(data)
    if model.nq >= 7:
        mj_data.qpos[:3] = root_pos
        mj_data.qpos[3:7] = root_quat
    mujoco.mj_forward(model, mj_data)

    tree = ET.parse(mjcf)
    root = tree.getroot()
    bodies = _body_by_name(root)
    removed_existing = _remove_existing_sites_with_prefix(root, site_prefix)
    written: list[str] = []
    skipped: list[str] = []

    for label, spec in marker_sites.items():
        label = str(label)
        site_name = f"{site_prefix}{label}"
        body_name = _body_name_from_marker_site_spec(label, spec)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0 or body_name not in bodies:
            message = f"Body {body_name!r} for marker {label!r} not found in {mjcf}"
            if strict:
                raise ValueError(message)
            skipped.append(message)
            continue
        try:
            marker_world = _resolve_marker_site_position(data, marker_config, label, spec).astype(np.float64)
        except (KeyError, ValueError, TypeError) as exc:
            message = f"Could not resolve marker {label!r}: {exc}"
            if strict:
                raise ValueError(message) from exc
            skipped.append(message)
            continue
        body_pos = mj_data.xpos[body_id].copy()
        body_rot = mj_data.xmat[body_id].reshape(3, 3).copy()
        local_pos = body_rot.T @ (marker_world - body_pos)
        ET.SubElement(
            bodies[body_name],
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "group": "0",
                "size": str(site_size),
                "rgba": site_rgba,
                "pos": _format_vec(local_pos),
            },
        )
        written.append(f"{site_name}->{body_name}")

    ET.indent(tree, space="  ")
    tree.write(mjcf, encoding="unicode", xml_declaration=False)
    mjcf.write_text(mjcf.read_text() + "\n")
    return {"removed_existing": removed_existing, "written": written, "skipped": skipped}


def maybe_copy_asset_dir(base_xml: Path, output_xml: Path, copy_base_asset_dir: bool) -> None:
    if not copy_base_asset_dir:
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        return
    if base_xml.parent == output_xml.parent:
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        return
    if output_xml.parent.exists():
        shutil.rmtree(output_xml.parent)
    shutil.copytree(base_xml.parent, output_xml.parent)


def _default_output_paths(
    base_xml: Path,
    subject_id: str | None,
    output_xml: Path | None,
    report: Path | None,
) -> tuple[Path, Path]:
    if subject_id:
        generated_output_xml = base_xml.parent / DEFAULT_OUTPUT_XML_TEMPLATE.format(subject_id=subject_id)
        generated_report = DEFAULT_REPORT.parent / DEFAULT_REPORT_TEMPLATE.format(subject_id=subject_id)
    else:
        generated_output_xml = DEFAULT_OUTPUT_XML
        generated_report = DEFAULT_REPORT
    return output_xml or generated_output_xml, report or generated_report


def print_c3d_markers(c3d: Path) -> None:
    metadata = read_metadata(c3d)
    labels = metadata.parameters.get("POINT.LABELS", [])
    print(f"C3D: {c3d}")
    print(
        f"Frames: {metadata.header.first_frame}-{metadata.header.last_frame} "
        f"rate={metadata.header.point_rate:g} Hz points={metadata.header.point_count}"
    )
    for idx, label in enumerate(labels, start=1):
        print(f"{idx:03d}: {label}")


def main() -> None:
    args = parse_args()
    if args.list_markers:
        print_c3d_markers(args.c3d)
        return

    marker_config = load_marker_config(args.marker_config)
    subject_id = args.subject_id or marker_config.get("subject_id")
    subject_id = str(subject_id) if subject_id is not None else None
    output_xml, report_path = _default_output_paths(args.base_xml, subject_id, args.output_xml, args.report)

    static_config = marker_config.get("static", {}) if isinstance(marker_config.get("static", {}), Mapping) else {}
    static_start_arg = args.static_start or int(static_config.get("start", 1))
    static_end_arg = args.static_end or int(static_config.get("end", 200))
    if args.auto_static:
        static_detection_markers = marker_config.get("static_detection_markers")
        if static_detection_markers is not None:
            static_detection_markers = [str(marker) for marker in static_detection_markers]
        static_start, static_end = detect_static_window(
            args.c3d,
            static_start_arg,
            args.auto_search_frames,
            args.auto_window,
            static_detection_markers,
        )
    else:
        static_start, static_end = static_start_arg, static_end_arg

    if marker_config:
        subject_lengths = compute_subject_lengths_from_config(args.c3d, static_start, static_end, marker_config)
    else:
        subject_lengths = compute_subject_lengths(args.c3d, static_start, static_end)

    tree = ET.parse(args.base_xml)
    root = tree.getroot()
    base_lengths = compute_base_lengths(root)
    scales = {
        name: subject_lengths[name] / base_length
        for name, base_length in base_lengths.items()
        if base_length > 0 and np.isfinite(subject_lengths.get(name, math.nan))
    }
    changed = apply_scales(root, scales)
    geom_changed = apply_geometry_scales(root, scales)
    mesh_asset_changed = apply_mesh_asset_scales(root, scales, args.base_xml.parent)
    marker_site_changed = apply_marker_site_scales(root, scales)

    maybe_copy_asset_dir(args.base_xml, output_xml, args.copy_base_asset_dir)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)

    marker_site_bake: dict[str, Any] | None = None
    if args.add_marker_sites:
        marker_site_bake = add_marker_sites_from_static_c3d(
            output_xml,
            args.c3d,
            marker_config,
            static_start,
            static_end,
            site_prefix=args.site_prefix,
            site_size=args.site_size,
            site_rgba=args.site_rgba,
            strict=args.strict_marker_sites,
        )

    report = {
        "c3d": str(args.c3d),
        "base_xml": str(args.base_xml),
        "output_xml": str(output_xml),
        "marker_config": str(args.marker_config) if args.marker_config else None,
        "subject_id": subject_id,
        "static_window": [static_start, static_end],
        "subject_lengths_m": subject_lengths,
        "base_lengths_m": base_lengths,
        "scales": scales,
        "changed_bodies": changed,
        "changed_geometry": geom_changed,
        "changed_mesh_assets": mesh_asset_changed,
        "changed_marker_sites": marker_site_changed,
        "static_marker_site_bake": marker_site_bake,
        "notes": [
            "Kinematic body offsets were scaled; masses/inertias were intentionally left unchanged.",
            "Segment geometry (skin capsules, wrap surfaces, muscle sites) was scaled about each body "
            "origin to track the new joint offsets.",
            "Bone mesh assets were given explicit mesh scale attributes so visual meshes track the scaled segments.",
            "Hip centers use the Visual3D/CODA pelvis formulas from the MDH file.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Static window: {static_start}-{static_end}")
    print(f"Wrote scaled MJCF: {output_xml}")
    if marker_site_bake is not None:
        print(
            f"Baked {len(marker_site_bake['written'])} static marker sites into {output_xml} "
            f"(removed {marker_site_bake['removed_existing']} existing {args.site_prefix}* sites)"
        )
        if marker_site_bake["skipped"]:
            print(f"Skipped {len(marker_site_bake['skipped'])} marker sites")
    print(f"Wrote scaling report: {report_path}")
    for name, scale in scales.items():
        print(f"  {name}: subject={subject_lengths[name]:.4f} m base={base_lengths[name]:.4f} m scale={scale:.3f}")


if __name__ == "__main__":
    main()
