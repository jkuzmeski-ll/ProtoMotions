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
"""Convert MS-Human-700 locomotion MJCF into a ProtoMotions-friendly asset.

The upstream MS-Human-700 locomotion model uses muscles, tendons, coupled knee
coordinates, and a six-slider/hinge root. ProtoMotions expects a floating root
and each non-root body to have either one or three hinge DOFs. This script keeps
only the skeleton and rigid-body geoms, replaces the root with a free joint,
removes muscles/tendons/equality constraints/keyframes, and simplifies each knee
to a single hinge joint.

The MS-Human-700 repository does not currently ship a license file. This script
therefore does not vendor any third-party assets into ProtoMotions; it copies
files only when the user explicitly provides a local MS-Human-700 checkout.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REQUIRED_DIRS = ("Asset", "Body_Locomotion", "Contact", "Geometry")
TOP_LEVEL_XML = "MS-Human-700-Locomotion.xml"
OUTPUT_XML = "MS-Human-700-Locomotion-Simple.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Path to a local clone of https://github.com/LNSGroup/MS-Human-700.git",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/ms_human_700"),
        help="Directory where the converted MJCF package will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print helper compatible with older Python versions."""
    space = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = space + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = space
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = space


def remove_children(parent: ET.Element, predicate) -> None:
    for child in list(parent):
        if predicate(child):
            parent.remove(child)


def copy_required_files(source: Path, output_dir: Path, overwrite: bool) -> None:
    if not (source / TOP_LEVEL_XML).is_file():
        raise FileNotFoundError(f"{source / TOP_LEVEL_XML} does not exist")

    missing = [name for name in REQUIRED_DIRS if not (source / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing required directories in {source}: {missing}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    for name in REQUIRED_DIRS:
        shutil.copytree(source / name, output_dir / name)

    # dm_control resolves mesh paths in included asset files relative to the
    # include file location. Upstream asset XML uses file="Geometry/...", so keep
    # a Geometry mirror under Asset/ as well as the original top-level directory.
    shutil.copytree(source / "Geometry", output_dir / "Asset" / "Geometry")
    (output_dir / "Asset" / "Asset").mkdir()
    for asset_file in (source / "Asset").iterdir():
        if asset_file.is_file():
            shutil.copy2(asset_file, output_dir / "Asset" / "Asset" / asset_file.name)


def simplify_leg_file(path: Path, side: str) -> list[str]:
    """Remove coupled knee/patella coordinates and return remaining joint names."""
    tree = ET.parse(path)
    root = tree.getroot()
    kept_joint_names: list[str] = []

    tibia_name = f"tibia_{side}"
    patella_name = f"patella_{side}"
    knee_name = f"knee_angle_{side}"

    for body in root.iter("body"):
        body_name = body.get("name")
        if body_name == tibia_name:
            remove_children(
                body,
                lambda child: child.tag == "joint" and child.get("name") != knee_name,
            )
        elif body_name == patella_name:
            remove_children(body, lambda child: child.tag == "joint")

        for joint in body.findall("joint"):
            name = joint.get("name")
            if name:
                kept_joint_names.append(name)

    indent(root)
    tree.write(path, encoding="utf-8", xml_declaration=False)
    return kept_joint_names


def collect_joint_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [joint.get("name") for joint in root.iter("joint") if joint.get("name")]


def simplify_torso_file(path: Path) -> None:
    """Drop simplified arm includes so the generated robot is lower-body only."""
    tree = ET.parse(path)
    root = tree.getroot()
    for body in root.iter("body"):
        remove_children(
            body,
            lambda child: child.tag == "include"
            and child.get("file", "").startswith("Body_Locomotion/Body_Arm_"),
        )
    indent(root)
    tree.write(path, encoding="utf-8", xml_declaration=False)


def inline_body_includes(body: ET.Element, output_dir: Path) -> None:
    """Replace <include> tags inside a body with the included fragment children."""
    for child in list(body):
        if child.tag == "include":
            include_file = child.get("file")
            if include_file is None:
                raise ValueError("Found body-level <include> without a file attribute")
            fragment_root = ET.parse(output_dir / include_file).getroot()
            insert_at = list(body).index(child)
            body.remove(child)
            for fragment_child in list(fragment_root):
                if fragment_child.tag == "body":
                    inline_body_includes(fragment_child, output_dir)
                body.insert(insert_at, fragment_child)
                insert_at += 1
        elif child.tag == "body":
            inline_body_includes(child, output_dir)


def convert_top_level(
    source_xml: Path, output_xml: Path, output_dir: Path, joint_names: list[str]
) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()

    # Remove upstream pieces that are not used by the simplified joint-control model.
    remove_children(root, lambda child: child.tag == "equality")
    remove_children(root, lambda child: child.tag == "tendon")
    remove_children(root, lambda child: child.tag == "actuator")
    remove_children(root, lambda child: child.tag == "keyframe")
    remove_children(
        root,
        lambda child: child.tag == "include"
        and child.get("file", "").startswith(("Tendon/", "Muscle/", "Equality/")),
    )

    default = root.find("default")
    if default is not None:
        remove_children(default, lambda child: child.tag == "tendon")
        remove_children(
            default,
            lambda child: child.tag == "default" and child.get("class") == "muscle",
        )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Top-level MJCF is missing <worldbody>")

    remove_children(worldbody, lambda child: child.tag == "geom" and child.get("name") == "floor")

    pelvis = next(
        (child for child in list(worldbody) if child.tag == "body" and child.get("name") == "pelvis"),
        None,
    )
    if pelvis is None:
        raise ValueError("Top-level MJCF is missing root body named 'pelvis'")

    # ProtoMotions requires a single free root with identity local quaternion.
    inline_body_includes(pelvis, output_dir)
    remove_children(pelvis, lambda child: child.tag == "joint")
    pelvis.set("pos", "0 0 0")

    root_body = ET.Element("body", {"name": "ms_human_root", "pos": "0 0 0"})
    ET.SubElement(root_body, "freejoint", {"name": "ms_human_root"})
    worldbody.remove(pelvis)
    root_body.append(pelvis)
    worldbody.append(root_body)

    actuator = ET.SubElement(root, "actuator")
    for name in joint_names:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"{name}_motor",
                "joint": name,
                "gear": "1",
            },
        )

    indent(root)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.resolve()

    copy_required_files(source, output_dir, args.overwrite)

    joint_names: list[str] = []
    joint_names.extend(simplify_leg_file(output_dir / "Body_Locomotion/Body_Leg_Foot_r.xml", "r"))
    joint_names.extend(simplify_leg_file(output_dir / "Body_Locomotion/Body_Leg_Foot_l.xml", "l"))
    simplify_torso_file(output_dir / "Body_Locomotion/Body_Torso_Simple.xml")
    joint_names.extend(collect_joint_names(output_dir / "Body_Locomotion/Body_Torso_Simple.xml"))

    convert_top_level(source / TOP_LEVEL_XML, output_dir / OUTPUT_XML, output_dir, joint_names)

    print(f"Wrote {output_dir / OUTPUT_XML}")
    print(f"Actuated non-root DOFs: {len(joint_names)}")


if __name__ == "__main__":
    main()
