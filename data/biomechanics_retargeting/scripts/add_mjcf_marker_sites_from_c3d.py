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
"""Attach C3D calibration marker sites to an MJCF model.

The scaler changes segment lengths, but it does not by itself add anatomical
marker offsets.  This utility uses a static C3D window and a marker-site mapping
from the marker config to create ``mocap_*`` MJCF ``site`` elements on the
corresponding model bodies.  The resulting sites can be edited with
``data/scripts/edit_mjcf_marker_sites.py`` and used by the IK retargeter with
``--marker-offset-source=site`` or ``site-or-calibrated``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from protomotions.utils.c3d_io import load_c3d, marker_index

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from scale_ms_human_to_subject import _resolve_point, load_marker_config  # noqa: E402

DEFAULT_MJCF = Path(
    "protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S081-LowerOnly.xml"
)
DEFAULT_CONFIG = Path("data/yaml_files/ms_human_700_cal101_marker_scaling_config.json")
PELVIS_MARKERS = ("RASI", "LASI", "RPSI", "LPSI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("c3d", type=Path, help="Static calibration C3D file.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MJCF to annotate with marker sites.")
    parser.add_argument(
        "--output-mjcf",
        type=Path,
        help="Output MJCF path. Defaults to updating --mjcf in place.",
    )
    parser.add_argument(
        "--marker-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON config containing static window, points, and marker_sites mapping.",
    )
    parser.add_argument("--static-start", type=int, help="First 1-based static frame.")
    parser.add_argument("--static-end", type=int, help="Last 1-based static frame, inclusive.")
    parser.add_argument("--site-prefix", default="mocap_", help="Prefix for generated site names.")
    parser.add_argument("--site-size", default="0.008", help="Generated MJCF site size attribute.")
    parser.add_argument("--site-rgba", default="0.1 0.45 1 1", help="Generated MJCF site rgba attribute.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing markers/bodies instead of skipping incompatible marker-set entries.",
    )
    return parser.parse_args()


def _unit_scale(data) -> float:
    return 0.001 if data.point_units.lower() in {"mm", "millimeter", "millimeters"} else 1.0


def _v3d_points_to_model(points: np.ndarray) -> np.ndarray:
    result = np.empty_like(points, dtype=np.float32)
    result[..., 0] = -points[..., 1]
    result[..., 1] = points[..., 0]
    result[..., 2] = points[..., 2]
    return result


def _marker_mean(data, label: str, scale: float) -> np.ndarray:
    return np.nanmean(data.markers[:, marker_index(data.marker_labels, label)], axis=0) * scale


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
    scale = _unit_scale(data)
    pelvis = np.stack([_marker_mean(data, name, scale) for name in PELVIS_MARKERS], axis=0)
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


def _static_window(marker_config: Mapping[str, object], args: argparse.Namespace) -> tuple[int, int]:
    static_cfg = marker_config.get("static", {})
    if not isinstance(static_cfg, Mapping):
        static_cfg = {}
    start = args.static_start or int(static_cfg.get("start", 1))
    end = args.static_end or int(static_cfg.get("end", 200))
    return start, end


def _resolve_marker_site_position(data, marker_config: Mapping[str, object], label: str, spec) -> np.ndarray:
    scale = _unit_scale(data)
    if isinstance(spec, str):
        point_v3d = _marker_mean(data, label, scale)
    elif isinstance(spec, Mapping):
        if "point" in spec:
            point_specs = marker_config.get("points", {})
            if not isinstance(point_specs, Mapping):
                raise TypeError("marker config 'points' must be an object when marker_sites uses point references")
            point_v3d = _resolve_point(data, str(spec["point"]), point_specs, {}, scale)
        elif "marker" in spec:
            point_v3d = _marker_mean(data, str(spec["marker"]), scale)
        else:
            point_v3d = _marker_mean(data, label, scale)
    else:
        raise TypeError(f"Unsupported marker site spec for {label!r}: {type(spec).__name__}")
    return _v3d_points_to_model(np.asarray(point_v3d, dtype=np.float32))


def _warn_or_raise(message: str, *, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    print(f"[WARN] {message}; skipping")


def _body_name_from_spec(label: str, spec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping) and "body" in spec:
        return str(spec["body"])
    raise ValueError(f"marker_sites entry for {label!r} must be a body name or object with a body field")


def _body_elements(root: ET.Element) -> dict[str, ET.Element]:
    return {body.get("name"): body for body in root.iter("body") if body.get("name")}


def _remove_existing_site(root: ET.Element, site_name: str) -> None:
    for body in root.iter("body"):
        for child in list(body):
            if child.tag == "site" and child.get("name") == site_name:
                body.remove(child)


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


def main() -> None:
    args = parse_args()
    marker_config = load_marker_config(args.marker_config)
    marker_sites = marker_config.get("marker_sites", {})
    if not isinstance(marker_sites, Mapping):
        raise TypeError("marker config must contain an object-valued marker_sites section")

    static_start, static_end = _static_window(marker_config, args)
    data = load_c3d(args.c3d, start_frame=static_start, end_frame=static_end)

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    mj_data = mujoco.MjData(model)
    root_pos, root_quat = _root_pose_from_static_pelvis(data)
    if model.nq >= 7:
        mj_data.qpos[:3] = root_pos
        mj_data.qpos[3:7] = root_quat
    mujoco.mj_forward(model, mj_data)

    tree = ET.parse(args.mjcf)
    root = tree.getroot()
    bodies = _body_elements(root)
    removed_existing = _remove_existing_sites_with_prefix(root, args.site_prefix)

    written = []
    skipped = []
    for label, spec in marker_sites.items():
        label = str(label)
        site_name = f"{args.site_prefix}{label}"
        body_name = _body_name_from_spec(label, spec)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            message = f"Body {body_name!r} for marker {label!r} not found in {args.mjcf}"
            _warn_or_raise(message, strict=args.strict)
            skipped.append(f"{label}: {message}")
            continue
        if body_name not in bodies:
            message = f"Body {body_name!r} for marker {label!r} not found in XML tree"
            _warn_or_raise(message, strict=args.strict)
            skipped.append(f"{label}: {message}")
            continue

        try:
            marker_world = _resolve_marker_site_position(data, marker_config, label, spec).astype(np.float64)
        except (KeyError, ValueError, TypeError) as exc:
            message = f"Could not resolve marker {label!r} for this C3D/config marker set: {exc}"
            _warn_or_raise(message, strict=args.strict)
            skipped.append(f"{label}: {message}")
            continue
        body_pos = mj_data.xpos[body_id].copy()
        body_rot = mj_data.xmat[body_id].reshape(3, 3).copy()
        local_pos = body_rot.T @ (marker_world - body_pos)

        _remove_existing_site(root, site_name)
        ET.SubElement(
            bodies[body_name],
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "group": "0",
                "size": str(args.site_size),
                "rgba": args.site_rgba,
                "pos": _format_vec(local_pos),
            },
        )
        written.append(f"{site_name}->{body_name}")

    output_mjcf = args.output_mjcf or args.mjcf
    ET.indent(tree, space="  ")
    output_mjcf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_mjcf, encoding="unicode", xml_declaration=False)
    output_mjcf.write_text(output_mjcf.read_text() + "\n")

    print(f"Static window: {static_start}-{static_end}")
    if removed_existing:
        print(f"Removed {removed_existing} existing '{args.site_prefix}*' marker sites before writing subset")
    print(f"Wrote {len(written)} marker sites to {output_mjcf}")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible marker-site entries")
    print("Marker sites:")
    for item in written:
        print(f"  {item}")


if __name__ == "__main__":
    main()
