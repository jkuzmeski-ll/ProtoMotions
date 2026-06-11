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
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from protomotions.utils.c3d_io import load_c3d, marker_index, read_metadata


BASE_XML = Path("protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-Simple.xml")
DEFAULT_OUTPUT_XML = Path("protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003.xml")
DEFAULT_REPORT = Path("data/ms-human-lower-retargeted/S003_scaling_report.json")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("c3d", type=Path, help="Subject C3D file with an initial static pose window.")
    parser.add_argument("--base-xml", type=Path, default=BASE_XML, help="Base simplified MS-Human MJCF.")
    parser.add_argument("--output-xml", type=Path, default=DEFAULT_OUTPUT_XML, help="Scaled subject MJCF.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON scaling report path.")
    parser.add_argument("--static-start", type=int, default=1, help="First 1-based C3D static frame.")
    parser.add_argument("--static-end", type=int, default=200, help="Last 1-based C3D static frame, inclusive.")
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


def detect_static_window(c3d: Path, start_frame: int, search_frames: int, window: int) -> tuple[int, int]:
    metadata = read_metadata(c3d)
    end = min(metadata.header.last_frame, start_frame + search_frames - 1)
    data = load_c3d(c3d, start_frame=start_frame, end_frame=end)
    labels = ["RASI", "LASI", "RPSI", "LPSI", "RKNE", "LKNE", "RANK", "LANK", "RTOE", "LTOE"]
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
        right_hip = asis_mid + 0.36 * asis_distance * ml_axis - 0.19 * asis_distance * ap_axis - 0.30 * asis_distance * axial_axis
        left_hip = asis_mid - 0.36 * asis_distance * ml_axis - 0.19 * asis_distance * ap_axis - 0.30 * asis_distance * axial_axis

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
    foot_r = np.linalg.norm(_parse_vec(bodies["calcn_r"].get("pos"))) + np.linalg.norm(_parse_vec(bodies["toes_r"].get("pos")))
    foot_l = np.linalg.norm(_parse_vec(bodies["calcn_l"].get("pos"))) + np.linalg.norm(_parse_vec(bodies["toes_l"].get("pos")))
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
    return {
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


def main() -> None:
    args = parse_args()
    if args.auto_static:
        static_start, static_end = detect_static_window(
            args.c3d, args.static_start, args.auto_search_frames, args.auto_window
        )
    else:
        static_start, static_end = args.static_start, args.static_end

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

    maybe_copy_asset_dir(args.base_xml, args.output_xml, args.copy_base_asset_dir)
    tree.write(args.output_xml, encoding="utf-8", xml_declaration=False)

    report = {
        "c3d": str(args.c3d),
        "base_xml": str(args.base_xml),
        "output_xml": str(args.output_xml),
        "static_window": [static_start, static_end],
        "subject_lengths_m": subject_lengths,
        "base_lengths_m": base_lengths,
        "scales": scales,
        "changed_bodies": changed,
        "changed_geometry": geom_changed,
        "notes": [
            "Kinematic body offsets were scaled; masses/inertias were intentionally left unchanged.",
            "Segment geometry (skin capsules, wrap surfaces, muscle sites) was scaled about each body origin to track the new joint offsets.",
            "Hip centers use the Visual3D/CODA pelvis formulas from the MDH file.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Static window: {static_start}-{static_end}")
    print(f"Wrote scaled MJCF: {args.output_xml}")
    print(f"Wrote scaling report: {args.report}")
    for name, scale in scales.items():
        print(f"  {name}: subject={subject_lengths[name]:.4f} m base={base_lengths[name]:.4f} m scale={scale:.3f}")


if __name__ == "__main__":
    main()
