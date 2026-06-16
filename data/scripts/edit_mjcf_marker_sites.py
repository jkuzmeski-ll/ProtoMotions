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
"""Interactively adjust MJCF marker sites on a still model.

This editor opens MuJoCo's passive viewer, highlights one marker site at a time,
and writes edited local site positions back into the MJCF XML without reformatting
unrelated XML content.

Hotkeys:
    n / p       Select next / previous marker site
    x / a       Move selected site along local +X / -X
    y / b       Move selected site along local +Y / -Y
    z / c       Move selected site along local +Z / -Z
    = / -       Increase / decrease edit step
    s           Save selected marker positions back to XML
    h           Print help

Example:
    python data/scripts/edit_mjcf_marker_sites.py \
        --mjcf protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003-LowerOnly.xml
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class SiteEditorState:
    site_ids: list[int]
    selected_idx: int
    step: float
    dirty: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-S003-LowerOnly.xml"),
        help="MJCF file containing marker <site> definitions to edit.",
    )
    parser.add_argument(
        "--site-prefix",
        default="mocap_",
        help="Only sites whose names start with this prefix are editable.",
    )
    parser.add_argument("--step", type=float, default=0.005, help="Initial local-position edit step in meters.")
    parser.add_argument("--site-size", type=float, default=0.008, help="Unselected marker sphere radius in meters.")
    return parser.parse_args()


def _site_name(model: mujoco.MjModel, site_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)


def _body_name(model: mujoco.MjModel, site_id: int) -> str:
    body_id = int(model.site_bodyid[site_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _find_editable_sites(model: mujoco.MjModel, site_prefix: str) -> list[int]:
    site_ids = []
    for site_id in range(model.nsite):
        site_name = _site_name(model, site_id)
        if site_name and site_name.startswith(site_prefix):
            site_ids.append(site_id)
    return site_ids


def _print_help() -> None:
    print(
        "\nMarker-site editor hotkeys:\n"
        "  n / p  : next / previous marker\n"
        "  x / a  : local +X / -X\n"
        "  y / b  : local +Y / -Y\n"
        "  z / c  : local +Z / -Z\n"
        "  = / -  : increase / decrease step\n"
        "  s      : save MJCF site positions\n"
        "  h      : print this help\n"
    )


def _print_selected(model: mujoco.MjModel, state: SiteEditorState) -> None:
    site_id = state.site_ids[state.selected_idx]
    site_name = _site_name(model, site_id)
    body_name = _body_name(model, site_id)
    pos = model.site_pos[site_id]
    dirty = "*" if state.dirty else ""
    print(
        f"{dirty} selected {state.selected_idx + 1}/{len(state.site_ids)} "
        f"{site_name} on {body_name}: pos={pos[0]: .6f} {pos[1]: .6f} {pos[2]: .6f} step={state.step:.4f} m"
    )


def _highlight_sites(model: mujoco.MjModel, state: SiteEditorState, site_size: float) -> None:
    for idx, site_id in enumerate(state.site_ids):
        model.site_group[site_id] = 0
        model.site_size[site_id, 0] = site_size
        if idx == state.selected_idx:
            model.site_rgba[site_id] = np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32)
        else:
            model.site_rgba[site_id] = np.array([0.1, 0.45, 1.0, 1.0], dtype=np.float32)


def _format_pos(pos: np.ndarray) -> str:
    return f"{pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}"


def _replace_site_pos(xml_text: str, site_name: str, new_pos: str) -> tuple[str, bool]:
    pattern = re.compile(r'(<site\b(?=[^>]*\bname="' + re.escape(site_name) + r'")[^>]*\bpos=")([^"]*)(")')
    xml_text, count = pattern.subn(r"\g<1>" + new_pos + r"\3", xml_text, count=1)
    return xml_text, count == 1


def _save_site_positions(xml_path: Path, model: mujoco.MjModel, site_ids: list[int]) -> None:
    xml_text = xml_path.read_text()
    missing = []
    for site_id in site_ids:
        site_name = _site_name(model, site_id)
        xml_text, replaced = _replace_site_pos(xml_text, site_name, _format_pos(model.site_pos[site_id]))
        if not replaced:
            missing.append(site_name)
    if missing:
        raise ValueError("Could not update site positions for: " + ", ".join(missing))
    xml_path.write_text(xml_text)
    print(f"Saved {len(site_ids)} marker site positions to {xml_path}")


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    site_ids = _find_editable_sites(model, args.site_prefix)
    if not site_ids:
        raise ValueError(f"No editable sites found with prefix '{args.site_prefix}' in {args.mjcf}")

    state = SiteEditorState(site_ids=site_ids, selected_idx=0, step=float(args.step))
    _highlight_sites(model, state, args.site_size)
    _print_help()
    _print_selected(model, state)

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode)
        except ValueError:
            return

        selected_site_id = state.site_ids[state.selected_idx]
        moved = False
        key_upper = key.upper()

        if key_upper == "N":
            state.selected_idx = (state.selected_idx + 1) % len(state.site_ids)
        elif key_upper == "P":
            state.selected_idx = (state.selected_idx - 1) % len(state.site_ids)
        elif key_upper in {"X", "A", "Y", "B", "Z", "C"}:
            axis = {"X": 0, "A": 0, "Y": 1, "B": 1, "Z": 2, "C": 2}[key_upper]
            sign = 1.0 if key_upper in {"X", "Y", "Z"} else -1.0
            model.site_pos[selected_site_id, axis] += sign * state.step
            state.dirty = True
            moved = True
        elif key == "=":
            state.step *= 2.0
        elif key == "-":
            state.step = max(state.step * 0.5, 1e-5)
        elif key == "s":
            _save_site_positions(args.mjcf, model, state.site_ids)
            state.dirty = False
        elif key == "h":
            _print_help()
        else:
            return

        _highlight_sites(model, state, args.site_size)
        mujoco.mj_forward(model, data)
        if moved or key_upper in {"N", "P"} or key in {"=", "-", "s"}:
            _print_selected(model, state)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.opt.sitegroup[:] = 1
        print("Marker sites are blue spheres; the selected site is red.")
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)

    if state.dirty:
        print("Unsaved marker-site edits remain. Press 's' before closing the viewer to write them to XML.")


if __name__ == "__main__":
    main()
