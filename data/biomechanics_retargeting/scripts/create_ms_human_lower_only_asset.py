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
"""Create a pelvis-and-legs-only MS-Human S003 MJCF asset.

The subject-scaled lower-body asset still carries the fixed spine, thorax, ribs,
and neck/head chain from the original MS-Human model.  This utility removes the
upper-body branch rooted at ``sacrum`` while preserving the pelvis, both legs,
and all lower-limb actuators.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_INPUT_XML = Path(
    "protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003.xml"
)
DEFAULT_OUTPUT_XML = Path(
    "protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003-LowerOnly.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--input-xml",
        type=Path,
        default=DEFAULT_INPUT_XML,
        help="Subject-scaled S003 MJCF input.",
    )
    parser.add_argument(
        "--output-xml",
        type=Path,
        default=DEFAULT_OUTPUT_XML,
        help="Lower-only MJCF output.",
    )
    return parser.parse_args()


def remove_body(root: ET.Element, body_name: str) -> None:
    for parent in root.iter("body"):
        for child in list(parent):
            if child.tag == "body" and child.get("name") == body_name:
                parent.remove(child)
                return
    raise ValueError(f"Could not find body '{body_name}' in MJCF.")


def main() -> None:
    args = parse_args()
    tree = ET.parse(args.input_xml)
    root = tree.getroot()
    remove_body(root, "sacrum")
    ET.indent(tree, space="  ")
    args.output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_xml, encoding="unicode", xml_declaration=False)
    args.output_xml.write_text(args.output_xml.read_text() + "\n")
    print(f"Wrote lower-only MJCF: {args.output_xml}")


if __name__ == "__main__":
    main()
