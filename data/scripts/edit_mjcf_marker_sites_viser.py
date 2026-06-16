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
"""Browser-based MJCF marker-site editor using Viser transform controls.

This is intended for building a responsive base-model marker-site template.  It
loads an MJCF with MuJoCo for kinematics, renders simple geometry in Viser, and
lets the selected ``mocap_*`` site be dragged in world space.  The edited world
position is converted back into the site's local body frame and written to MJCF
when switching markers or pressing Save.

Example:
    python data/scripts/edit_mjcf_marker_sites_viser.py \
        --mjcf protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-Simple.xml \
        --site-prefix mocap_
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh


@dataclass
class SiteRecord:
    site_id: int
    site_name: str
    body_id: int
    body_name: str


@dataclass
class EditorState:
    selected_idx: int = 0
    dirty: bool = False
    suppress_transform_callback: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/ms_human_700/MS-Human-700-Locomotion-Simple.xml"),
        help="MJCF file containing marker <site> definitions to edit.",
    )
    parser.add_argument("--site-prefix", default="mocap_", help="Only edit sites whose names start with this prefix.")
    parser.add_argument(
        "--site-name-regex",
        help="Optional regex applied after --site-prefix filtering, e.g. 'mocap_[RL](THI|TIB)$'.",
    )
    parser.add_argument(
        "--tracking-markers-only",
        action="store_true",
        help="Only edit rigid-segment tracking markers: mocap_RTHI, mocap_LTHI, mocap_RTIB, mocap_LTIB.",
    )
    parser.add_argument("--start-site", help="Optional site name to select first, e.g. mocap_RASI.")
    parser.add_argument("--port", type=int, default=8080, help="Viser web-server port.")
    parser.add_argument("--site-radius", type=float, default=0.012, help="Marker sphere radius in meters.")
    parser.add_argument("--control-scale", type=float, default=0.12, help="Transform-control gizmo scale.")
    parser.add_argument(
        "--save-on-switch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save pending MJCF edits whenever the selected marker changes.",
    )
    parser.add_argument(
        "--mirror-on-edit",
        action="store_true",
        help="Keep the paired R/L site symmetric by copying local X/Y and flipping local Z after each drag.",
    )
    parser.add_argument(
        "--hide-geoms",
        action="store_true",
        help="Only render marker sites, not simple MJCF geometry.",
    )
    return parser.parse_args()


def _site_name(model: mujoco.MjModel, site_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _find_sites(model: mujoco.MjModel, site_prefix: str, site_name_regex: str | None) -> list[SiteRecord]:
    site_pattern = re.compile(site_name_regex) if site_name_regex else None
    records = []
    for site_id in range(model.nsite):
        site_name = _site_name(model, site_id)
        if not site_name or not site_name.startswith(site_prefix):
            continue
        if site_pattern is not None and not site_pattern.search(site_name):
            continue
        body_id = int(model.site_bodyid[site_id])
        records.append(
            SiteRecord(
                site_id=site_id,
                site_name=site_name,
                body_id=body_id,
                body_name=_body_name(model, body_id),
            )
        )
    return records


def _format_pos(pos: np.ndarray) -> str:
    return f"{pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}"


def _replace_site_pos(xml_text: str, site_name: str, new_pos: str) -> tuple[str, bool]:
    pattern = re.compile(r'(<site\b(?=[^>]*\bname="' + re.escape(site_name) + r'")[^>]*\bpos=")([^"]*)(")')
    xml_text, count = pattern.subn(r"\g<1>" + new_pos + r"\3", xml_text, count=1)
    return xml_text, count == 1


def _save_site_positions(xml_path: Path, model: mujoco.MjModel, records: list[SiteRecord]) -> None:
    xml_text = xml_path.read_text()
    missing = []
    for record in records:
        xml_text, replaced = _replace_site_pos(xml_text, record.site_name, _format_pos(model.site_pos[record.site_id]))
        if not replaced:
            missing.append(record.site_name)
    if missing:
        raise ValueError("Could not update site positions for: " + ", ".join(missing))
    xml_path.write_text(xml_text)
    print(f"Saved {len(records)} marker site positions to {xml_path}")


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


def _mirror_site_to_pair(model: mujoco.MjModel, site_id: int) -> int | None:
    site_name = _site_name(model, site_id)
    paired_name = _paired_site_name(site_name)
    if paired_name is None:
        return None
    paired_site_id = _site_id_by_name(model, paired_name)
    if paired_site_id is None:
        return None
    model.site_pos[paired_site_id] = model.site_pos[site_id]
    model.site_pos[paired_site_id, 2] *= -1.0
    return paired_site_id


def _rotation_matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
    return quat


def _body_xmat(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3)


def _site_world_position(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    body_id = int(model.site_bodyid[site_id])
    return data.xpos[body_id] + _body_xmat(data, body_id) @ model.site_pos[site_id]


def _set_site_from_world_position(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, world_pos: np.ndarray) -> None:
    body_id = int(model.site_bodyid[site_id])
    body_rot = _body_xmat(data, body_id)
    model.site_pos[site_id] = body_rot.T @ (world_pos - data.xpos[body_id])
    mujoco.mj_forward(model, data)


def _marker_arrays(model: mujoco.MjModel, data: mujoco.MjData, records: list[SiteRecord], selected_idx: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.stack([_site_world_position(model, data, record.site_id) for record in records], axis=0)
    colors = np.full((len(records), 3), np.array([40, 120, 255], dtype=np.uint8), dtype=np.uint8)
    colors[selected_idx] = np.array([255, 30, 30], dtype=np.uint8)
    return points, colors


def _make_geom_mesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size[:3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2.0 * size[1]), count=[16, 16])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2.0 * size[1]), sections=24)
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.vertices *= size[:3]
    else:
        return None

    rgba = model.geom_rgba[geom_id].copy()
    if rgba[3] <= 0.0:
        rgba = np.array([0.7, 0.7, 0.7, 0.35])
    rgba[3] = min(float(rgba[3]), 0.55)
    mesh.visual.vertex_colors = np.tile((rgba * 255).astype(np.uint8), (len(mesh.vertices), 1))
    return mesh


def _add_model_geoms(server: Any, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for geom_id in range(model.ngeom):
        mesh = _make_geom_mesh(model, geom_id)
        if mesh is None:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        position = data.geom_xpos[geom_id]
        wxyz = _rotation_matrix_to_wxyz(data.geom_xmat[geom_id].reshape(3, 3))
        server.scene.add_mesh_trimesh(
            f"/model/{geom_name}",
            mesh=mesh,
            position=position,
            wxyz=wxyz,
        )


def main() -> None:
    try:
        import viser
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("viser is required. Install it with `pip install viser`.") from exc

    args = parse_args()
    site_name_regex = args.site_name_regex
    if args.tracking_markers_only:
        tracking_regex = rf"^{re.escape(args.site_prefix)}[RL](THI|TIB)$"
        if site_name_regex is not None and site_name_regex != tracking_regex:
            raise ValueError("Use either --tracking-markers-only or --site-name-regex, not both.")
        site_name_regex = tracking_regex
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    records = _find_sites(model, args.site_prefix, site_name_regex)
    if not records:
        raise ValueError(f"No editable sites found with prefix '{args.site_prefix}' in {args.mjcf}")

    state = EditorState()
    if args.start_site:
        names = [record.site_name for record in records]
        if args.start_site not in names:
            raise ValueError(f"Start site {args.start_site!r} is not an editable site in {args.mjcf}")
        state.selected_idx = names.index(args.start_site)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    if not args.hide_geoms:
        _add_model_geoms(server, model, data)

    marker_points, marker_colors = _marker_arrays(model, data, records, state.selected_idx)
    marker_cloud = server.scene.add_point_cloud(
        "/marker_sites",
        points=marker_points,
        colors=marker_colors,
        point_size=args.site_radius,
        point_shape="circle",
    )

    selected_record = records[state.selected_idx]
    selected_pos = _site_world_position(model, data, selected_record.site_id)
    transform = server.scene.add_transform_controls(
        "/selected_marker_control",
        position=selected_pos,
        scale=args.control_scale,
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    site_dropdown = server.gui.add_dropdown(
        "selected marker",
        options=[record.site_name for record in records],
        initial_value=selected_record.site_name,
    )
    info_text = server.gui.add_text("selected info", "")
    save_button = server.gui.add_button("Save MJCF now")
    prev_button = server.gui.add_button("Previous marker")
    next_button = server.gui.add_button("Next marker")
    mirror_button = server.gui.add_button("Mirror selected to R/L pair")

    def refresh_scene(update_control: bool = True) -> None:
        marker_points, marker_colors = _marker_arrays(model, data, records, state.selected_idx)
        marker_cloud.points = marker_points
        marker_cloud.colors = marker_colors
        record = records[state.selected_idx]
        pos = model.site_pos[record.site_id]
        info_text.value = (
            f"{state.selected_idx + 1}/{len(records)} {record.site_name} on {record.body_name}: "
            f"local=({pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}) dirty={state.dirty}"
        )
        if update_control:
            state.suppress_transform_callback = True
            transform.position = _site_world_position(model, data, record.site_id)
            state.suppress_transform_callback = False

    def save_if_needed() -> None:
        if state.dirty:
            _save_site_positions(args.mjcf, model, records)
            state.dirty = False
            refresh_scene(update_control=False)

    def select_idx(new_idx: int) -> None:
        if args.save_on_switch:
            save_if_needed()
        state.selected_idx = new_idx % len(records)
        site_dropdown.value = records[state.selected_idx].site_name
        refresh_scene(update_control=True)

    @transform.on_update
    def _(_: Any) -> None:
        if state.suppress_transform_callback:
            return
        record = records[state.selected_idx]
        _set_site_from_world_position(model, data, record.site_id, np.asarray(transform.position, dtype=np.float64))
        if args.mirror_on_edit:
            _mirror_site_to_pair(model, record.site_id)
        state.dirty = True
        refresh_scene(update_control=False)

    @site_dropdown.on_update
    def _(_: Any) -> None:
        names = [record.site_name for record in records]
        select_idx(names.index(site_dropdown.value))

    @save_button.on_click
    def _(_: Any) -> None:
        _save_site_positions(args.mjcf, model, records)
        state.dirty = False
        refresh_scene(update_control=False)

    @prev_button.on_click
    def _(_: Any) -> None:
        select_idx(state.selected_idx - 1)

    @next_button.on_click
    def _(_: Any) -> None:
        select_idx(state.selected_idx + 1)

    @mirror_button.on_click
    def _(_: Any) -> None:
        record = records[state.selected_idx]
        paired_site_id = _mirror_site_to_pair(model, record.site_id)
        if paired_site_id is not None:
            state.dirty = True
            refresh_scene(update_control=False)

    refresh_scene(update_control=True)
    print(f"Viser marker-site editor is running on port {args.port}.")
    print("Drag the selected transform control in the browser; edits save on marker switch by default.")
    print("Press Ctrl+C here to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        if state.dirty:
            print("Saving pending marker-site edits before exit...")
            _save_site_positions(args.mjcf, model, records)


if __name__ == "__main__":
    main()
