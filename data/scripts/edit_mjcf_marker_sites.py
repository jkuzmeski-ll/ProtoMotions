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
    m           Mirror selected R/L marker to its pair by flipping local Z
    n / p       Save pending edits before marker switch when --save-on-switch is set
    s           Save selected marker positions back to XML
    --autosave  Save positions after every edit, so viewer crashes do not lose work
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
    parser.add_argument(
        "--site-name-regex",
        help="Optional regex applied after --site-prefix filtering, e.g. 'mocap_[RL](THI|TIB)$'.",
    )
    parser.add_argument(
        "--tracking-markers-only",
        action="store_true",
        help="Only edit rigid-segment tracking markers: mocap_RTHI, mocap_LTHI, mocap_RTIB, mocap_LTIB.",
    )
    parser.add_argument("--step", type=float, default=0.005, help="Initial local-position edit step in meters.")
    parser.add_argument("--site-size", type=float, default=0.008, help="Unselected marker sphere radius in meters.")
    parser.add_argument(
        "--start-site",
        help="Optional site name to select first, e.g. mocap_RASI.",
    )
    parser.add_argument(
        "--mirror-on-edit",
        action="store_true",
        help="After every movement, update the paired R/L marker by copying local X/Y and flipping local Z.",
    )
    parser.add_argument(
        "--save-on-switch",
        action="store_true",
        help="Save pending marker positions when changing selected marker. More responsive than --autosave.",
    )
    parser.add_argument(
        "--autosave",
        action="store_true",
        help="Write MJCF site positions after every edit. Useful because MuJoCo's passive viewer can segfault on close/input.",
    )
    return parser.parse_args()


def _site_name(model: mujoco.MjModel, site_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)


def _body_name(model: mujoco.MjModel, site_id: int) -> str:
    body_id = int(model.site_bodyid[site_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _find_editable_sites(model: mujoco.MjModel, site_prefix: str, site_name_regex: str | None) -> list[int]:
    site_pattern = re.compile(site_name_regex) if site_name_regex else None
    site_ids = []
    for site_id in range(model.nsite):
        site_name = _site_name(model, site_id)
        if site_name and site_name.startswith(site_prefix) and (site_pattern is None or site_pattern.search(site_name)):
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
        "  m      : mirror selected R/L marker to its pair by flipping local Z\n"
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


def _paired_site_name(site_name: str) -> str | None:
    if "RIGHT" in site_name:
        return site_name.replace("RIGHT", "LEFT", 1)
    if "LEFT" in site_name:
        return site_name.replace("LEFT", "RIGHT", 1)

    marker_name = site_name.rsplit("_", 1)[-1]
    marker_start = len(site_name) - len(marker_name)
    if marker_name.startswith("R") and len(marker_name) > 1:
        return site_name[:marker_start] + "L" + marker_name[1:]
    if marker_name.startswith("L") and len(marker_name) > 1:
        return site_name[:marker_start] + "R" + marker_name[1:]
    return None


def _site_id_by_name(model: mujoco.MjModel, site_name: str) -> int | None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return int(site_id) if site_id >= 0 else None


def _mirror_site_to_pair(model: mujoco.MjModel, site_id: int) -> bool:
    site_name = _site_name(model, site_id)
    paired_name = _paired_site_name(site_name)
    if paired_name is None:
        print(f"No R/L paired marker name inferred for {site_name}")
        return False
    paired_site_id = _site_id_by_name(model, paired_name)
    if paired_site_id is None:
        print(f"Paired marker site {paired_name} not found for {site_name}")
        return False
    model.site_pos[paired_site_id] = model.site_pos[site_id]
    model.site_pos[paired_site_id, 2] *= -1.0
    print(f"Mirrored {site_name} -> {paired_name}: pos={_format_pos(model.site_pos[paired_site_id])}")
    return True


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


def _autosave_site_positions(xml_path: Path, model: mujoco.MjModel, site_ids: list[int], state: SiteEditorState) -> None:
    _save_site_positions(xml_path, model, site_ids)
    state.dirty = False


def main() -> None:
    args = parse_args()
    site_name_regex = args.site_name_regex
    if args.tracking_markers_only:
        tracking_regex = rf"^{re.escape(args.site_prefix)}[RL](THI|TIB)$"
        if site_name_regex is not None and site_name_regex != tracking_regex:
            raise ValueError("Use either --tracking-markers-only or --site-name-regex, not both.")
        site_name_regex = tracking_regex
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    site_ids = _find_editable_sites(model, args.site_prefix, site_name_regex)
    if not site_ids:
        raise ValueError(f"No editable sites found with prefix '{args.site_prefix}' in {args.mjcf}")

    state = SiteEditorState(site_ids=site_ids, selected_idx=0, step=float(args.step))
    if args.start_site:
        start_site_id = _site_id_by_name(model, args.start_site)
        if start_site_id is None or start_site_id not in site_ids:
            raise ValueError(f"Start site {args.start_site!r} is not an editable site in {args.mjcf}")
        state.selected_idx = site_ids.index(start_site_id)
    _highlight_sites(model, state, args.site_size)
    _print_help()
    _print_selected(model, state)

    def save_if_needed() -> None:
        if state.dirty:
            _save_site_positions(args.mjcf, model, state.site_ids)
            state.dirty = False

    def key_callback(keycode: int) -> None:
        try:
            key = chr(keycode)
        except ValueError:
            return

        selected_site_id = state.site_ids[state.selected_idx]
        moved = False
        key_upper = key.upper()

        if key_upper == "N":
            if args.save_on_switch:
                save_if_needed()
            state.selected_idx = (state.selected_idx + 1) % len(state.site_ids)
        elif key_upper == "P":
            if args.save_on_switch:
                save_if_needed()
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
        elif key_upper == "M":
            state.dirty = _mirror_site_to_pair(model, selected_site_id) or state.dirty
            moved = state.dirty
        elif key == "s":
            _save_site_positions(args.mjcf, model, state.site_ids)
            state.dirty = False
        elif key == "h":
            _print_help()
        else:
            return

        _highlight_sites(model, state, args.site_size)
        mujoco.mj_forward(model, data)
        if args.mirror_on_edit and moved and key_upper != "M":
            _mirror_site_to_pair(model, selected_site_id)
        if args.autosave and moved:
            _autosave_site_positions(args.mjcf, model, state.site_ids, state)
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
