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
"""Bake a canonical base-model marker-site template from a scaled subject MJCF.

This is for the workflow where a subject-specific scaled model has data-grounded
``mocap_*`` sites, and those sites should be converted back into the base model's
proportions.  Future subject scaling can then scale the base marker template to
new subject sizes instead of fitting human data to an unscaled model.

Typical flow:
  1. Scale the base model to a calibration subject.
  2. Calibrate/export ``mocap_*`` sites on that scaled subject model.
  3. Run this script to inverse-scale those sites back onto the base MJCF.
  4. Use the updated base MJCF as the canonical marker template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from scale_ms_human_to_subject import GEOM_SCALE_GROUPS, MARKER_SITE_SCALE_GROUPS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--base-mjcf",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-Simple.xml"),
        help="Base MJCF whose marker-site template should be updated.",
    )
    parser.add_argument(
        "--scaled-mjcf",
        type=Path,
        required=True,
        help="Scaled subject MJCF containing calibrated marker sites.",
    )
    parser.add_argument(
        "--scaling-report",
        type=Path,
        required=True,
        help="JSON report written by scale_ms_human_to_subject.py for the scaled subject.",
    )
    parser.add_argument(
        "--output-mjcf",
        type=Path,
        help="Output base-template MJCF. Defaults to updating --base-mjcf in place.",
    )
    parser.add_argument("--site-prefix", default="mocap_", help="Marker site prefix to bake.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing base marker sites not present in --scaled-mjcf.",
    )
    return parser.parse_args()


def _parse_vec(text: str | None) -> np.ndarray:
    if text is None or not text.strip():
        return np.zeros(3, dtype=np.float64)
    values = np.array([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {text!r}")
    return values


def _format_vec(vec: np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in vec.tolist())


def _body_elements(root: ET.Element) -> dict[str, ET.Element]:
    return {body.get("name"): body for body in root.iter("body") if body.get("name")}


def _body_scale_map(scales: dict[str, float]) -> dict[str, float]:
    body_scales: dict[str, float] = {}
    for scale_name, body_names in GEOM_SCALE_GROUPS.items():
        scale = float(scales.get(scale_name, 1.0))
        for body_name in body_names:
            body_scales[body_name] = scale
    for scale_name, body_names in MARKER_SITE_SCALE_GROUPS.items():
        scale = float(scales.get(scale_name, 1.0))
        for body_name in body_names:
            body_scales[body_name] = scale
    return body_scales


def _iter_prefixed_sites(root: ET.Element, site_prefix: str):
    for body in root.iter("body"):
        body_name = body.get("name")
        if body_name is None:
            continue
        for site in body.findall("site"):
            site_name = site.get("name") or ""
            if site_name.startswith(site_prefix):
                yield body_name, site


def _remove_prefixed_sites(root: ET.Element, site_prefix: str) -> int:
    removed = 0
    for body in root.iter("body"):
        for child in list(body):
            if child.tag == "site" and (child.get("name") or "").startswith(site_prefix):
                body.remove(child)
                removed += 1
    return removed


def _copy_site_with_base_pos(target_body: ET.Element, source_site: ET.Element, base_pos: np.ndarray) -> None:
    attrs = dict(source_site.attrib)
    attrs["pos"] = _format_vec(base_pos)
    ET.SubElement(target_body, "site", attrs)


def main() -> None:
    args = parse_args()
    report = json.loads(args.scaling_report.read_text(encoding="utf-8"))
    scales = report.get("scales", {})
    if not isinstance(scales, dict):
        raise TypeError("Scaling report must contain an object-valued 'scales' field")
    body_scales = _body_scale_map(scales)

    base_tree = ET.parse(args.base_mjcf)
    base_root = base_tree.getroot()
    scaled_root = ET.parse(args.scaled_mjcf).getroot()
    base_bodies = _body_elements(base_root)

    if not args.keep_existing:
        removed = _remove_prefixed_sites(base_root, args.site_prefix)
        if removed:
            print(f"Removed {removed} existing '{args.site_prefix}*' sites from base template")

    written: list[str] = []
    for body_name, source_site in _iter_prefixed_sites(scaled_root, args.site_prefix):
        target_body = base_bodies.get(body_name)
        if target_body is None:
            raise ValueError(f"Body {body_name!r} from {args.scaled_mjcf} is missing in {args.base_mjcf}")
        scale = body_scales.get(body_name, 1.0)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid scale {scale} for body {body_name!r}")
        base_pos = _parse_vec(source_site.get("pos")) / scale
        _copy_site_with_base_pos(target_body, source_site, base_pos)
        written.append(f"{source_site.get('name')}->{body_name} scale={scale:.6g}")

    output_mjcf = args.output_mjcf or args.base_mjcf
    ET.indent(base_tree, space="  ")
    output_mjcf.parent.mkdir(parents=True, exist_ok=True)
    base_tree.write(output_mjcf, encoding="unicode", xml_declaration=False)
    output_mjcf.write_text(output_mjcf.read_text() + "\n")

    print(f"Wrote {len(written)} base-template marker sites to {output_mjcf}")
    for item in written:
        print(f"  {item}")


if __name__ == "__main__":
    main()
