# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the lower-body ellipsoid-foot MJCF from a static C3D trial.

This script is intentionally dependency-light: it includes a small C3D point reader so
static calibration can run in environments where ezc3d/c3d/btk are not installed.

Marker assumptions are tailored to the MITLL Plug-in-Gait style marker set in
``projects/data/S001/Cal 101.v3d.c3d``:

- Pelvis frame: LASI/RASI/LPSI/RPSI. LIC/RIC are used only for pelvis geom sizing.
- Hip centers: CODA/Bell-style regression from ASIS width.
- Knee centers: midpoint of lateral/medial knee markers: KNE/MKNE.
- Ankle centers: midpoint of lateral/medial ankle markers: ANK/MANK.
- Foot contacts: HEE/HEE2/HEE3 for heel, MTH1/MTH5 for ball, HLX for toe tip.
- Segment cluster markers such as THI/TH2/TH3/TH4 and TIB/TIB2/TIB3/TIB4 are
  reported but not used to define static joint centers.

Output coordinates follow the MuJoCo/Newton convention used by the target XML:
``+x`` forward, ``+y`` subject-left, ``+z`` up, meters.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

DEFAULT_INPUT_XML = Path(
    "projects/humanoid_model/smpl_lower_humanoid_elipsoid_foot_contact.xml"
)
DEFAULT_STATIC_C3D = Path("projects/data/S001/Cal 101.v3d.c3d")
DEFAULT_OUTPUT_XML = Path(
    "projects/humanoid_model/smpl_lower_humanoid_elipsoid_foot_contact_calibrated.xml"
)
DEFAULT_REPORT_JSON = Path(
    "projects/humanoid_model/smpl_lower_humanoid_elipsoid_foot_contact_calibrated_report.json"
)

# Bell/CODA-style hip joint center regression coefficients, expressed in the
# pelvis frame from the ASIS midpoint:
#   lateral:  +/- 0.36 * ASIS width
#   posterior: -0.19 * ASIS width
#   inferior:  -0.30 * ASIS width
CODA_LATERAL_SCALE = 0.36
CODA_POSTERIOR_SCALE = 0.19
CODA_INFERIOR_SCALE = 0.30

REQUIRED_MARKERS = [
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LIC",
    "RIC",
    "LKNE",
    "LMKNE",
    "RKNE",
    "RMKNE",
    "LANK",
    "LMANK",
    "RANK",
    "RMANK",
    "LHEE",
    "LHEE2",
    "LHEE3",
    "LTOE",
    "LHLX",
    "LMTH1",
    "LMTH5",
    "RHEE",
    "RHEE2",
    "RHEE3",
    "RTOE",
    "RHLX",
    "RMTH1",
    "RMTH5",
]

TRACKING_ONLY_MARKERS = [
    "LTHI",
    "LTH2",
    "LTH3",
    "LTH4",
    "RTHI",
    "RTH2",
    "RTH3",
    "RTH4",
    "LTIB",
    "LTIB2",
    "LTIB3",
    "LTIB4",
    "RTIB",
    "RTIB2",
    "RTIB3",
    "RTIB4",
]


@dataclass
class C3DPoints:
    labels: List[str]
    points: Dict[str, np.ndarray]  # stripped marker name -> [valid_frames, 3]
    prefixed_labels: Dict[str, str]
    units: str
    frame_rate: float
    frames: int


@dataclass
class MarkerStats:
    mean: np.ndarray
    std: np.ndarray
    valid_frames: int


@dataclass
class Calibration:
    markers: Dict[str, np.ndarray]
    marker_stats: Dict[str, MarkerStats]
    ground: Dict[str, Any]
    pelvis: Dict[str, Any]
    joints: Dict[str, np.ndarray]
    feet: Dict[str, Dict[str, Any]]
    warnings: List[str]


def parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def fmt_vec(values: Iterable[float], decimals: int = 5) -> str:
    return " ".join(f"{float(v):.{decimals}f}" for v in values)


def norm(vec: np.ndarray, *, eps: float = 1e-9) -> float:
    return float(np.linalg.norm(vec))


def unit(vec: np.ndarray, *, name: str = "vector", eps: float = 1e-9) -> np.ndarray:
    length = norm(vec)
    if length < eps:
        raise ValueError(f"Cannot normalize near-zero {name}: {vec}")
    return vec / length


def strip_prefix(label: str) -> str:
    return label.split(":", 1)[-1].strip()


def read_c3d_points(path: Path) -> C3DPoints:
    """Read point labels and 3-D point samples from a C3D file.

    This is a compact reader for static calibration files. It supports DEC/VAX-style
    little-endian parameter blocks used by this data and standard integer/float point
    storage. Analog channels are skipped after each point frame.
    """

    data = path.read_bytes()
    if len(data) < 512:
        raise ValueError(f"{path} is too small to be a C3D file")

    parameter_block = data[0]
    parameter_offset = (parameter_block - 1) * 512
    processor_type = data[parameter_offset + 3]
    endian = "<" if processor_type in (84, 86) else ">"

    groups: Dict[int, str] = {}
    params: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pos = parameter_offset + 4

    while pos < len(data):
        name_len_raw = struct.unpack("b", data[pos : pos + 1])[0]
        if name_len_raw == 0:
            break
        group_id = struct.unpack("b", data[pos + 1 : pos + 2])[0]
        pos += 2

        name_len = abs(name_len_raw)
        name = data[pos : pos + name_len].decode("latin1").strip()
        pos += name_len

        offset_pos = pos
        next_offset = struct.unpack(endian + "h", data[pos : pos + 2])[0]
        pos += 2
        if next_offset == 0:
            break
        next_pos = offset_pos + next_offset

        if group_id < 0:
            desc_len = data[pos]
            groups[-group_id] = name
            pos += 1 + desc_len
        else:
            param_type = struct.unpack("b", data[pos : pos + 1])[0]
            pos += 1
            dim_count = data[pos]
            pos += 1
            dims = list(data[pos : pos + dim_count])
            pos += dim_count
            value_count = int(np.prod(dims)) if dims else 1

            if param_type == -1:  # char
                raw = data[pos : pos + value_count]
                if len(dims) == 2:
                    width, count = dims
                    values: Any = [
                        raw[i * width : (i + 1) * width]
                        .decode("latin1", errors="ignore")
                        .strip()
                        for i in range(count)
                    ]
                else:
                    values = raw.decode("latin1", errors="ignore").strip()
                pos += value_count
            elif param_type == 1:  # byte
                values = list(data[pos : pos + value_count])
                pos += value_count
            elif param_type == 2:  # int16
                values = list(
                    struct.unpack(
                        endian + str(value_count) + "h",
                        data[pos : pos + 2 * value_count],
                    )
                )
                pos += 2 * value_count
            elif param_type == 4:  # float32
                values = list(
                    struct.unpack(
                        endian + str(value_count) + "f",
                        data[pos : pos + 4 * value_count],
                    )
                )
                pos += 4 * value_count
            else:
                values = None

            desc_len = data[pos]
            pos += 1 + desc_len
            group_name = groups.get(group_id, str(group_id))
            params[(group_name, name)] = {
                "type": param_type,
                "dims": dims,
                "values": values,
            }

        pos = next_pos

    def param(group: str, name: str, default: Any = None) -> Any:
        item = params.get((group, name))
        if item is None:
            return default
        values = item["values"]
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        return values

    labels = list(param("POINT", "LABELS", []))
    used = int(param("POINT", "USED", len(labels)))
    frames = int(param("POINT", "FRAMES", 0))
    data_start = int(param("POINT", "DATA_START", 0))
    scale = float(param("POINT", "SCALE", 1.0))
    frame_rate = float(param("POINT", "RATE", 0.0))
    units = str(param("POINT", "UNITS", "mm")).strip() or "mm"
    analog_used = int(param("ANALOG", "USED", 0) or 0)
    analog_rate = float(param("ANALOG", "RATE", 0.0) or 0.0)
    analog_per_frame = int(round(analog_rate / frame_rate)) if frame_rate else 0

    if used <= 0 or frames <= 0 or data_start <= 0:
        raise ValueError(
            f"Could not read point metadata from {path}: "
            f"used={used}, frames={frames}, data_start={data_start}"
        )

    label_arrays: Dict[str, List[np.ndarray]] = {
        strip_prefix(label): [] for label in labels[:used]
    }
    prefixed_labels = {strip_prefix(label): label for label in labels[:used]}
    point_pos = (data_start - 1) * 512
    float_points = scale < 0

    for _frame in range(frames):
        for label in labels[:used]:
            clean_label = strip_prefix(label)
            if float_points:
                x, y, z, residual = struct.unpack(
                    endian + "ffff", data[point_pos : point_pos + 16]
                )
                point_pos += 16
                valid = (
                    all(math.isfinite(value) for value in (x, y, z)) and residual >= 0
                )
            else:
                xi, yi, zi, residual_i = struct.unpack(
                    endian + "hhhh", data[point_pos : point_pos + 8]
                )
                point_pos += 8
                x, y, z = xi * scale, yi * scale, zi * scale
                valid = residual_i >= 0

            if valid and not (x == 0.0 and y == 0.0 and z == 0.0):
                label_arrays[clean_label].append(np.array([x, y, z], dtype=np.float64))

        analog_sample_count = analog_used * analog_per_frame
        point_pos += analog_sample_count * (4 if float_points else 2)

    unit_scale = {"mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "m": 1.0}.get(
        units.lower(),
        1e-3,
    )
    points = {
        label: np.vstack(samples) * unit_scale
        for label, samples in label_arrays.items()
        if len(samples) > 0
    }

    return C3DPoints(
        labels=labels[:used],
        points=points,
        prefixed_labels=prefixed_labels,
        units=units,
        frame_rate=frame_rate,
        frames=frames,
    )


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    distances = (points - centroid) @ normal
    rms = float(np.sqrt(np.mean(distances**2)))
    return centroid, unit(normal, name="ground normal"), rms


def transform_to_mujoco_newton(
    marker_means_raw_m: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    platform_points = np.array(
        [
            value
            for name, value in marker_means_raw_m.items()
            if name.startswith("Platform:")
            or name in {"FLeft", "FRight", "ORight", "BLeft", "BRight"}
        ],
        dtype=np.float64,
    )
    if len(platform_points) < 3:
        raise ValueError(
            "Need at least three Platform:* markers to fit the ground plane"
        )

    ground_origin, up_raw, ground_rms = fit_plane(platform_points)

    approximate_forward_raw = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    forward_raw = approximate_forward_raw - up_raw * float(
        approximate_forward_raw @ up_raw
    )
    forward_raw = unit(forward_raw, name="floor-projected forward axis")
    left_raw = unit(np.cross(up_raw, forward_raw), name="left axis")
    # Re-orthogonalize forward to make x/y/z exactly right-handed.
    forward_raw = unit(np.cross(left_raw, up_raw), name="forward axis")

    transformed = {}
    for name, point in marker_means_raw_m.items():
        rel = point - ground_origin
        transformed[name] = np.array(
            [rel @ forward_raw, rel @ left_raw, rel @ up_raw], dtype=np.float64
        )

    ground_report = {
        "origin_c3d_m": ground_origin.tolist(),
        "normal_c3d": up_raw.tolist(),
        "rms_fit_m": ground_rms,
        "model_axes_in_c3d": {
            "x_forward": forward_raw.tolist(),
            "y_left": left_raw.tolist(),
            "z_up": up_raw.tolist(),
        },
    }
    return transformed, ground_report


def compute_marker_stats(
    c3d: C3DPoints,
) -> Tuple[Dict[str, np.ndarray], Dict[str, MarkerStats]]:
    raw_means: Dict[str, np.ndarray] = {}
    stats: Dict[str, MarkerStats] = {}

    for label, samples in c3d.points.items():
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        raw_means[label] = mean
        stats[label] = MarkerStats(mean=mean, std=std, valid_frames=len(samples))

    # Preserve platform names after prefix stripping by C3D reader. Some C3D files
    # expose platform labels as FLeft instead of Platform:FLeft after stripping.
    for label in c3d.labels:
        clean = strip_prefix(label)
        if label.startswith("Platform:") and clean in raw_means:
            raw_means[label] = raw_means[clean]
            stats[label] = stats[clean]

    return raw_means, stats


def check_required_markers(markers: Mapping[str, np.ndarray]) -> None:
    missing = [name for name in REQUIRED_MARKERS if name not in markers]
    platform_count = sum(
        1
        for name in markers
        if name.startswith("Platform:")
        or name in {"FLeft", "FRight", "ORight", "BLeft", "BRight"}
    )
    if platform_count < 3:
        missing.append("at least three Platform:* markers")
    if missing:
        raise ValueError("Missing required calibration markers: " + ", ".join(missing))


def coda_hip_centers(
    markers: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    lasi = markers["LASI"]
    rasi = markers["RASI"]
    lpsi = markers["LPSI"]
    rpsi = markers["RPSI"]

    asis_mid = 0.5 * (lasi + rasi)
    psis_mid = 0.5 * (lpsi + rpsi)
    pelvis_left = unit(lasi - rasi, name="pelvis left axis")
    pelvis_forward = asis_mid - psis_mid
    pelvis_forward = pelvis_forward - pelvis_left * float(pelvis_forward @ pelvis_left)
    pelvis_forward = unit(pelvis_forward, name="pelvis forward axis")
    pelvis_up = unit(np.cross(pelvis_forward, pelvis_left), name="pelvis up axis")
    pelvis_forward = unit(np.cross(pelvis_left, pelvis_up), name="pelvis forward axis")

    asis_width = norm(lasi - rasi)
    posterior = -CODA_POSTERIOR_SCALE * asis_width * pelvis_forward
    inferior = -CODA_INFERIOR_SCALE * asis_width * pelvis_up
    lateral = CODA_LATERAL_SCALE * asis_width * pelvis_left

    left_hip = asis_mid + lateral + posterior + inferior
    right_hip = asis_mid - lateral + posterior + inferior

    report = {
        "origin": asis_mid.tolist(),
        "asis_width_m": asis_width,
        "axes": {
            "forward": pelvis_forward.tolist(),
            "left": pelvis_left.tolist(),
            "up": pelvis_up.tolist(),
        },
        "coda_scales": {
            "lateral": CODA_LATERAL_SCALE,
            "posterior": CODA_POSTERIOR_SCALE,
            "inferior": CODA_INFERIOR_SCALE,
        },
    }
    return left_hip, right_hip, report


def foot_calibration(markers: Mapping[str, np.ndarray], side: str) -> Dict[str, Any]:
    heel = markers[f"{side}HEE"]
    heel_2 = markers[f"{side}HEE2"]
    heel_3 = markers[f"{side}HEE3"]
    mth1 = markers[f"{side}MTH1"]
    mth5 = markers[f"{side}MTH5"]
    hlx = markers[f"{side}HLX"]
    toe_marker = markers[f"{side}TOE"]

    heel_center = np.mean(np.array([heel, heel_2, heel_3]), axis=0)
    ball_center = 0.5 * (mth1 + mth5)
    toe_tip = hlx  # confirmed: HLX is always farther anterior than TOE.

    forward = toe_tip - heel_center
    forward[2] = 0.0
    forward = unit(forward, name=f"{side} foot forward")
    lateral = mth5 - mth1
    lateral[2] = 0.0
    lateral = unit(lateral, name=f"{side} foot lateral")
    up = unit(np.cross(forward, lateral), name=f"{side} foot up")
    if up[2] < 0:
        lateral = -lateral
        up = -up
    forward = unit(np.cross(lateral, up), name=f"{side} foot forward")

    heel_width = norm(heel_3 - heel_2)
    forefoot_width = norm(mth5 - mth1)
    foot_length = float((toe_tip - heel_center) @ forward)
    heel_to_ball = float((ball_center - heel_center) @ forward)
    toe_length = max(0.03, float((toe_tip - ball_center) @ forward))

    return {
        "heel_center": heel_center,
        "ball_center": ball_center,
        "toe_tip": toe_tip,
        "toe_marker": toe_marker,
        "forward": forward,
        "lateral": lateral,
        "up": up,
        "heel_width_m": heel_width,
        "forefoot_width_m": forefoot_width,
        "foot_length_m": foot_length,
        "heel_to_ball_m": heel_to_ball,
        "toe_length_m": toe_length,
    }


def build_calibration(c3d: C3DPoints, max_static_std_m: float) -> Calibration:
    raw_means, raw_stats = compute_marker_stats(c3d)
    transformed_means, ground_report = transform_to_mujoco_newton(raw_means)

    # Recompute stats in the transformed frame by applying the same orientation is not
    # necessary for movement magnitude; std norm is invariant under rotation.
    marker_stats = raw_stats
    check_required_markers(transformed_means)

    warnings: List[str] = []
    for marker in REQUIRED_MARKERS + TRACKING_ONLY_MARKERS:
        if marker not in marker_stats:
            continue
        std_norm_m = norm(marker_stats[marker].std)
        if std_norm_m > max_static_std_m:
            warnings.append(
                f"Marker {marker} moved {std_norm_m * 1000.0:.2f} mm std, "
                f"above threshold {max_static_std_m * 1000.0:.2f} mm"
            )

    low_valid_required = [
        marker
        for marker in REQUIRED_MARKERS
        if marker in marker_stats
        and marker_stats[marker].valid_frames < 0.8 * c3d.frames
    ]
    if low_valid_required:
        warnings.append(
            "Required markers with <80% valid frames: " + ", ".join(low_valid_required)
        )

    left_hip, right_hip, pelvis_report = coda_hip_centers(transformed_means)
    joints = {
        "Pelvis": np.array(pelvis_report["origin"], dtype=np.float64),
        "L_Hip": left_hip,
        "R_Hip": right_hip,
        "L_Knee": 0.5 * (transformed_means["LKNE"] + transformed_means["LMKNE"]),
        "R_Knee": 0.5 * (transformed_means["RKNE"] + transformed_means["RMKNE"]),
        "L_Ankle": 0.5 * (transformed_means["LANK"] + transformed_means["LMANK"]),
        "R_Ankle": 0.5 * (transformed_means["RANK"] + transformed_means["RMANK"]),
    }

    feet = {
        "L": foot_calibration(transformed_means, "L"),
        "R": foot_calibration(transformed_means, "R"),
    }
    joints["L_Toe"] = feet["L"]["ball_center"]
    joints["R_Toe"] = feet["R"]["ball_center"]

    pelvis_markers = np.array(
        [
            transformed_means[name]
            for name in ["LASI", "RASI", "LPSI", "RPSI", "LIC", "RIC"]
        ],
        dtype=np.float64,
    )
    pelvis_report["marker_bounds_min"] = pelvis_markers.min(axis=0).tolist()
    pelvis_report["marker_bounds_max"] = pelvis_markers.max(axis=0).tolist()
    pelvis_report["iliac_width_m"] = norm(
        transformed_means["LIC"] - transformed_means["RIC"]
    )
    pelvis_report["asis_to_psis_depth_m"] = norm(
        0.5 * (transformed_means["LASI"] + transformed_means["RASI"])
        - 0.5 * (transformed_means["LPSI"] + transformed_means["RPSI"])
    )

    return Calibration(
        markers=transformed_means,
        marker_stats=marker_stats,
        ground=ground_report,
        pelvis=pelvis_report,
        joints=joints,
        feet=feet,
        warnings=warnings,
    )


def find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    raise KeyError(f"Could not find <{tag} name='{name}'> in XML")


def first_child(
    element: ET.Element, tag: str, name: Optional[str] = None
) -> ET.Element:
    for child in element:
        if child.tag != tag:
            continue
        if name is None or child.get("name") == name:
            return child
    raise KeyError(
        f"Could not find child <{tag}{' name=' + name if name else ''}> under {element.get('name')}"
    )


def set_vec_attr(
    element: ET.Element, attr: str, values: Iterable[float], decimals: int = 5
) -> None:
    element.set(attr, fmt_vec(values, decimals=decimals))


def update_capsule_between_joints(
    body: ET.Element,
    parent_joint: np.ndarray,
    child_joint: np.ndarray,
    start_fraction: float = 0.20,
    end_fraction: float = 0.82,
) -> None:
    geom = first_child(body, "geom")
    local_child = child_joint - parent_joint
    start = local_child * start_fraction
    end = local_child * end_fraction
    set_vec_attr(geom, "fromto", np.concatenate([start, end]))


def scale_box_geom_to_points(
    geom: ET.Element,
    points: np.ndarray,
    origin: np.ndarray,
    padding: np.ndarray,
    min_size: np.ndarray,
) -> None:
    local = points - origin
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_size = np.maximum(0.5 * (maxs - mins) + padding, min_size)
    set_vec_attr(geom, "pos", center)
    set_vec_attr(geom, "size", half_size)


def update_foot_contact(
    root: ET.Element,
    side: str,
    cal: Calibration,
    vertical_radius: float = 0.030,
) -> None:
    foot = cal.feet[side]
    ankle = cal.joints[f"{side}_Ankle"]
    toe_joint = cal.joints[f"{side}_Toe"]
    floor_z_local = -ankle[2]

    foot_segment = find_named(root, "body", f"{side}_elipsoid_foot_segment")
    toe_segment = find_named(root, "body", f"{side}_elipsoid_toe_segment")
    smpl_toe = find_named(root, "body", f"{side}_Toe")

    # Keep the helper foot segment colocated with the anatomical ankle. This makes
    # all contact positions easy to interpret in ankle-local coordinates.
    set_vec_attr(foot_segment, "pos", [0.0, 0.0, 0.0])

    heel_local = foot["heel_center"] - ankle
    ball_local = foot["ball_center"] - ankle
    ball_from_ankle = foot["ball_center"] - ankle

    heel_size = np.array(
        [
            max(0.025, min(0.055, 0.18 * foot["heel_to_ball_m"])),
            max(0.030, 0.55 * foot["heel_width_m"]),
            vertical_radius,
        ],
        dtype=np.float64,
    )
    ball_size = np.array(
        [
            max(0.040, min(0.080, 0.35 * foot["toe_length_m"])),
            max(0.030, 0.35 * foot["forefoot_width_m"]),
            vertical_radius,
        ],
        dtype=np.float64,
    )

    heel_geom = find_named(root, "geom", f"{side}_heel_ellipsoid")
    ball_geom = find_named(root, "geom", f"{side}_ball_ellipsoid")
    set_vec_attr(
        heel_geom, "pos", [heel_local[0], heel_local[1], floor_z_local + heel_size[2]]
    )
    set_vec_attr(heel_geom, "size", heel_size, decimals=4)
    set_vec_attr(
        ball_geom, "pos", [ball_local[0], ball_local[1], floor_z_local + ball_size[2]]
    )
    set_vec_attr(ball_geom, "size", ball_size, decimals=4)

    heel_touch = find_named(root, "site", f"{side}_heel_touch")
    ball_touch = find_named(root, "site", f"{side}_ball_touch")
    set_vec_attr(
        heel_touch, "pos", [heel_local[0], heel_local[1], floor_z_local + 0.004]
    )
    set_vec_attr(
        heel_touch,
        "size",
        [max(0.045, heel_size[0] * 1.5), max(0.040, heel_size[1] * 1.3), 0.004],
        decimals=4,
    )
    set_vec_attr(
        ball_touch, "pos", [ball_local[0], ball_local[1], floor_z_local + 0.004]
    )
    set_vec_attr(
        ball_touch,
        "size",
        [
            max(0.055, ball_size[0] * 1.4),
            max(0.045, 0.55 * foot["forefoot_width_m"]),
            0.004,
        ],
        decimals=4,
    )

    set_vec_attr(toe_segment, "pos", ball_from_ankle)
    toe_vec = foot["toe_tip"] - foot["ball_center"]
    toe_vec[2] = 0.0
    if norm(toe_vec) < 1e-6:
        toe_vec = foot["forward"] * foot["toe_length_m"]
    toe_vec = unit(toe_vec, name=f"{side} toe vector") * foot["toe_length_m"]
    toe_width = max(0.035, 0.55 * foot["forefoot_width_m"])
    toe_radius = 0.016
    toe_z = floor_z_local - ball_from_ankle[2] + toe_radius
    start = 0.20 * toe_vec + 0.45 * toe_width * foot["lateral"]
    end = 0.92 * toe_vec - 0.45 * toe_width * foot["lateral"]
    start[2] = toe_z
    end[2] = toe_z

    toe_capsule = find_named(root, "geom", f"{side}_toe_capsule")
    set_vec_attr(toe_capsule, "fromto", np.concatenate([start, end]))
    toe_capsule.set("size", f"{toe_radius:.4f}")

    toe_touch = find_named(root, "site", f"{side}_toe_touch")
    toe_touch_center = 0.55 * toe_vec
    toe_touch_center[2] = floor_z_local - ball_from_ankle[2] + 0.004
    set_vec_attr(toe_touch, "pos", toe_touch_center)
    set_vec_attr(
        toe_touch,
        "size",
        [max(0.035, 0.5 * foot["toe_length_m"]), max(0.035, 0.45 * toe_width), 0.004],
        decimals=4,
    )

    foot_visual = find_named(root, "geom", f"{side}_foot_inertial_visual")
    foot_points = np.array(
        [foot["heel_center"], foot["ball_center"], foot["toe_marker"]], dtype=np.float64
    )
    scale_box_geom_to_points(
        foot_visual,
        foot_points,
        ankle,
        padding=np.array([0.025, 0.025, 0.030], dtype=np.float64),
        min_size=np.array([0.060, 0.035, 0.035], dtype=np.float64),
    )

    # The separate SMPL toe body should sit at the same metatarsal-head center as
    # the passive ellipsoid toe hinge.
    set_vec_attr(smpl_toe, "pos", toe_joint - ankle)
    toe_visual = find_named(root, "geom", f"{side}_toe_inertial_visual")
    scale_box_geom_to_points(
        toe_visual,
        np.array(
            [foot["ball_center"], foot["toe_tip"], foot["toe_marker"]], dtype=np.float64
        ),
        toe_joint,
        padding=np.array([0.015, 0.018, 0.018], dtype=np.float64),
        min_size=np.array([0.035, 0.025, 0.018], dtype=np.float64),
    )


def update_xml(
    input_xml: Path, output_xml: Path, cal: Calibration, root_position: str
) -> None:
    tree = ET.parse(input_xml)
    root = tree.getroot()
    root.set("model", output_xml.stem)

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("angle", "radian")
        compiler.set("coordinate", "local")

    pelvis_body = find_named(root, "body", "Pelvis")
    pelvis_origin = cal.joints["Pelvis"]
    if root_position == "static":
        set_vec_attr(pelvis_body, "pos", [0.0, 0.0, pelvis_origin[2]])
    elif root_position == "zero":
        set_vec_attr(pelvis_body, "pos", [0.0, 0.0, 0.0])
    elif root_position == "preserve":
        pass
    else:
        raise ValueError(f"Unknown root_position mode: {root_position}")

    # Joint body positions.
    parent_of = {
        "L_Hip": "Pelvis",
        "R_Hip": "Pelvis",
        "L_Knee": "L_Hip",
        "R_Knee": "R_Hip",
        "L_Ankle": "L_Knee",
        "R_Ankle": "R_Knee",
    }
    for body_name, parent_name in parent_of.items():
        body = find_named(root, "body", body_name)
        local_pos = cal.joints[body_name] - cal.joints[parent_name]
        set_vec_attr(body, "pos", local_pos)

    # Segment capsules, preserving their existing radii/densities/contact settings.
    update_capsule_between_joints(
        find_named(root, "body", "L_Hip"), cal.joints["L_Hip"], cal.joints["L_Knee"]
    )
    update_capsule_between_joints(
        find_named(root, "body", "R_Hip"), cal.joints["R_Hip"], cal.joints["R_Knee"]
    )
    update_capsule_between_joints(
        find_named(root, "body", "L_Knee"), cal.joints["L_Knee"], cal.joints["L_Ankle"]
    )
    update_capsule_between_joints(
        find_named(root, "body", "R_Knee"), cal.joints["R_Knee"], cal.joints["R_Ankle"]
    )

    pelvis_geom = first_child(pelvis_body, "geom")
    pelvis_points = np.array(
        [cal.markers[name] for name in ["LASI", "RASI", "LPSI", "RPSI", "LIC", "RIC"]]
        + [cal.joints["L_Hip"], cal.joints["R_Hip"]],
        dtype=np.float64,
    )
    scale_box_geom_to_points(
        pelvis_geom,
        pelvis_points,
        pelvis_origin,
        padding=np.array([0.030, 0.030, 0.035], dtype=np.float64),
        min_size=np.array([0.070, 0.090, 0.065], dtype=np.float64),
    )

    update_foot_contact(root, "L", cal)
    update_foot_contact(root, "R", cal)

    ET.indent(tree, space="  ")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)


def calibration_report(c3d: C3DPoints, cal: Calibration) -> Dict[str, Any]:
    def arr(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: arr(item) for key, item in value.items()}
        if isinstance(value, list):
            return [arr(item) for item in value]
        return value

    segment_lengths = {
        "L_thigh_m": norm(cal.joints["L_Knee"] - cal.joints["L_Hip"]),
        "R_thigh_m": norm(cal.joints["R_Knee"] - cal.joints["R_Hip"]),
        "L_shank_m": norm(cal.joints["L_Ankle"] - cal.joints["L_Knee"]),
        "R_shank_m": norm(cal.joints["R_Ankle"] - cal.joints["R_Knee"]),
        "L_ankle_to_toe_m": norm(cal.joints["L_Toe"] - cal.joints["L_Ankle"]),
        "R_ankle_to_toe_m": norm(cal.joints["R_Toe"] - cal.joints["R_Ankle"]),
    }

    marker_report = {}
    for label in sorted(cal.marker_stats):
        stats = cal.marker_stats[label]
        marker_report[label] = {
            "valid_frames": stats.valid_frames,
            "mean_input_units_m": stats.mean.tolist(),
            "std_m": stats.std.tolist(),
            "std_norm_mm": norm(stats.std) * 1000.0,
        }

    feet = {
        side: {
            key: arr(value)
            for key, value in foot.items()
            if key not in {"forward", "lateral", "up"}
        }
        | {
            "axes": {
                "forward": foot["forward"].tolist(),
                "lateral": foot["lateral"].tolist(),
                "up": foot["up"].tolist(),
            }
        }
        for side, foot in cal.feet.items()
    }

    return {
        "source": {
            "frames": c3d.frames,
            "frame_rate_hz": c3d.frame_rate,
            "units": c3d.units,
            "labels": c3d.labels,
        },
        "coordinate_convention": {
            "description": "+x forward, +y subject-left, +z up, meters",
            "target": "MuJoCo/Newton",
        },
        "warnings": cal.warnings,
        "ground": cal.ground,
        "pelvis": cal.pelvis,
        "joints": {name: value.tolist() for name, value in cal.joints.items()},
        "segment_lengths": segment_lengths,
        "feet": feet,
        "marker_stats": marker_report,
        "tracking_only_markers": TRACKING_ONLY_MARKERS,
        "ignored_upper_body_markers": [
            "LFHD",
            "RFHD",
            "LBHD",
            "RBHD",
            "C7",
            "CLAV",
            "STRN",
            "RBAK",
            "T10",
            "LSHO",
            "LUPA",
            "LELB",
            "LFRM",
            "LWRA",
            "LWRB",
            "LFIN",
            "RSHO",
            "RUPA",
            "RELB",
            "RFRM",
            "RWRA",
            "RWRB",
            "RFIN",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize smpl_lower_humanoid_elipsoid_foot_contact.xml from a static C3D calibration trial."
    )
    parser.add_argument("--static-c3d", type=Path, default=DEFAULT_STATIC_C3D)
    parser.add_argument("--input-xml", type=Path, default=DEFAULT_INPUT_XML)
    parser.add_argument("--output-xml", type=Path, default=DEFAULT_OUTPUT_XML)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument(
        "--max-static-std-mm",
        type=float,
        default=10.0,
        help="Warn when any required/tracking marker has larger 3-D std during the static trial.",
    )
    parser.add_argument(
        "--root-position",
        choices=["static", "zero", "preserve"],
        default="static",
        help=(
            "How to set the root Pelvis body pos. 'static' centers x/y at zero and uses "
            "the static pelvis height above the fitted floor; 'zero' writes 0 0 0; "
            "'preserve' keeps the template root pos."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    c3d = read_c3d_points(args.static_c3d)
    cal = build_calibration(c3d, max_static_std_m=args.max_static_std_mm * 1e-3)
    update_xml(args.input_xml, args.output_xml, cal, root_position=args.root_position)

    report = calibration_report(c3d, cal)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote calibrated XML: {args.output_xml}")
    print(f"Wrote calibration report: {args.report_json}")
    print("Segment lengths:")
    for key, value in report["segment_lengths"].items():
        print(f"  {key}: {value:.4f} m")
    if cal.warnings:
        print("Warnings:")
        for warning in cal.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
