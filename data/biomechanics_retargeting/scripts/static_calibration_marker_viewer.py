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
"""View a scaled robot with static C3D calibration markers overlaid.

This is a quick calibration-check viewer: it loads a ProtoMotions robot in a
simulator, places the robot root at the static C3D pelvis, and overlays the mean
static marker positions so marker placement can be checked against the model
meshes and collision/contact bodies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("c3d", type=Path, help="Static calibration C3D file.")
    parser.add_argument(
        "--marker-config",
        type=Path,
        default=Path("data/yaml_files/ms_human_700_cal101_marker_scaling_config.json"),
        help="Marker scaling config whose derived points should also be visualized.",
    )
    parser.add_argument(
        "--robot-name",
        default="ms_human_lower_s081",
        help="ProtoMotions robot config to load.",
    )
    parser.add_argument(
        "--simulator",
        choices=["newton", "isaacgym", "isaaclab", "genesis", "mujoco"],
        default="newton",
        help="Simulator backend to use for the viewer.",
    )
    parser.add_argument("--static-start", type=int, help="First 1-based static frame.")
    parser.add_argument("--static-end", type=int, help="Last 1-based static frame, inclusive.")
    parser.add_argument(
        "--include-platform",
        action="store_true",
        help="Also visualize markers with Platform: prefixes.",
    )
    parser.add_argument(
        "--no-derived-points",
        action="store_true",
        help="Do not show derived points from the marker config.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening a viewer.")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU tensors instead of CUDA.")
    parser.add_argument(
        "--enable-gravity",
        action="store_true",
        help="Leave simulator gravity enabled. By default this calibration viewer sets gravity to zero.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Advance physics. By default this viewer freezes the static pose for calibration inspection.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many simulation steps. 0 means run until interrupted.",
    )
    return parser.parse_args()


args = parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402

from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.base_simulator.config import (  # noqa: E402
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.utils.c3d_io import load_c3d, marker_index  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.rotations import matrix_to_quaternion  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from scale_ms_human_to_subject import load_marker_config, _resolve_point  # noqa: E402

PELVIS_MARKERS = ("RASI", "LASI", "RPSI", "LPSI")


def _parse_vec(text: str | None) -> np.ndarray:
    if text is None or not text.strip():
        return np.zeros(3, dtype=np.float64)
    return np.array([float(value) for value in text.split()], dtype=np.float64)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64)
    quat = quat / np.clip(np.linalg.norm(quat), 1e-8, None)
    w, x, y, z = quat.tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pelvis_marker_centroid_in_root_frame(robot_cfg) -> np.ndarray | None:
    """Return the local root-to-pelvis-marker-centroid offset from MJCF sites."""
    xml_path = Path(robot_cfg.asset.asset_root) / robot_cfg.asset.asset_file_name
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        print(f"[WARN] Could not read MJCF marker sites from {xml_path}: {exc}; using raw pelvis marker centroid")
        return None

    pelvis_body = next((body for body in root.iter("body") if body.get("name") == "pelvis"), None)
    if pelvis_body is None:
        return None
    site_positions = []
    for label in PELVIS_MARKERS:
        site = pelvis_body.find(f"site[@name='mocap_{label}']")
        if site is None:
            site = pelvis_body.find(f"site[@name='{label}']")
        if site is None:
            return None
        site_positions.append(_parse_vec(site.get("pos")))
    pelvis_local_centroid = np.nanmean(np.stack(site_positions, axis=0), axis=0)
    pelvis_pos = _parse_vec(pelvis_body.get("pos"))
    pelvis_quat = _parse_vec(pelvis_body.get("quat"))
    if pelvis_quat.shape[0] != 4 or not np.any(pelvis_quat):
        pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pelvis_rot = _quat_wxyz_to_matrix(pelvis_quat)
    return (pelvis_pos + pelvis_rot @ pelvis_local_centroid).astype(np.float32)


def _unit_scale(data) -> float:
    return 0.001 if data.point_units.lower() in {"mm", "millimeter", "millimeters"} else 1.0


def _v3d_points_to_model(points: np.ndarray) -> np.ndarray:
    """Map Visual3D lab coordinates to ProtoMotions/MS-Human coordinates."""
    result = np.empty_like(points, dtype=np.float32)
    result[..., 0] = -points[..., 1]
    result[..., 1] = points[..., 0]
    result[..., 2] = points[..., 2]
    return result


def _short_label(label: str) -> str:
    return label.rsplit(":", 1)[-1]


def _marker_mean(data, label: str, scale: float) -> np.ndarray:
    return np.nanmean(data.markers[:, marker_index(data.marker_labels, label)], axis=0) * scale


def _load_static_window(c3d: Path, marker_config: dict[str, Any]) -> tuple[Any, int, int]:
    static_cfg = marker_config.get("static", {}) if isinstance(marker_config.get("static", {}), dict) else {}
    static_start = args.static_start or int(static_cfg.get("start", 1))
    static_end = args.static_end or int(static_cfg.get("end", 200))
    data = load_c3d(c3d, start_frame=static_start, end_frame=static_end)
    return data, static_start, static_end


def _marker_labels(data) -> list[str]:
    labels = []
    for label in data.marker_labels:
        if not args.include_platform and label.startswith("Platform:"):
            continue
        labels.append(label)
    return labels


def _static_marker_positions(data, labels: list[str]) -> np.ndarray:
    scale = _unit_scale(data)
    positions_v3d = np.stack([_marker_mean(data, label, scale) for label in labels], axis=0)
    return _v3d_points_to_model(positions_v3d)


def _derived_point_positions(data, marker_config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    if args.no_derived_points or not marker_config:
        return [], np.zeros((0, 3), dtype=np.float32)
    point_specs = marker_config.get("points", {})
    if not isinstance(point_specs, dict):
        return [], np.zeros((0, 3), dtype=np.float32)

    scale = _unit_scale(data)
    resolved: dict[str, np.ndarray] = {}
    names = []
    points = []
    for name in point_specs:
        try:
            point = _resolve_point(data, name, point_specs, resolved, scale)
        except (KeyError, ValueError, TypeError) as exc:
            print(f"[WARN] Could not resolve derived point {name!r} for this marker set: {exc}; skipping")
            continue
        names.append(str(name))
        points.append(point)
    if not points:
        return [], np.zeros((0, 3), dtype=np.float32)
    return names, _v3d_points_to_model(np.stack(points, axis=0))


def _root_pose_from_pelvis(
    data,
    device: torch.device,
    root_local_marker_centroid: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _unit_scale(data)
    pelvis = np.stack([_marker_mean(data, name, scale) for name in PELVIS_MARKERS], axis=0)
    pelvis = _v3d_points_to_model(pelvis.astype(np.float32))
    marker_centroid = torch.tensor(np.nanmean(pelvis, axis=0), device=device, dtype=torch.float32)

    rasi, lasi, rpsi, lpsi = [torch.tensor(p, device=device, dtype=torch.float32) for p in pelvis]
    x_axis_hint = 0.5 * (rasi + lasi) - 0.5 * (rpsi + lpsi)
    y_axis = lasi - rasi
    x_axis_hint = x_axis_hint / torch.linalg.norm(x_axis_hint).clamp_min(1e-8)
    y_axis = y_axis / torch.linalg.norm(y_axis).clamp_min(1e-8)
    z_axis = torch.cross(x_axis_hint, y_axis, dim=0)
    z_axis = z_axis / torch.linalg.norm(z_axis).clamp_min(1e-8)
    x_axis = torch.cross(y_axis, z_axis, dim=0)
    x_axis = x_axis / torch.linalg.norm(x_axis).clamp_min(1e-8)
    rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1).unsqueeze(0)
    root_rot = matrix_to_quaternion(rot_mat, w_last=True)[0]
    root_pos = marker_centroid
    if root_local_marker_centroid is not None:
        local_offset = torch.tensor(root_local_marker_centroid, device=device, dtype=torch.float32)
        root_pos = marker_centroid - rot_mat[0] @ local_offset
    return root_pos, root_rot


def _identity_quat(count: int, device: torch.device) -> torch.Tensor:
    quat = torch.zeros(1, count, 4, device=device, dtype=torch.float32)
    quat[..., 3] = 1.0
    return quat


class StaticCalibrationMarkerViewer:
    def __init__(self) -> None:
        self.device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda:0")
        self.marker_config = load_marker_config(args.marker_config) if args.marker_config else {}
        self.data, self.static_start, self.static_end = _load_static_window(args.c3d, self.marker_config)
        self.marker_labels = _marker_labels(self.data)
        self.marker_positions = torch.tensor(
            _static_marker_positions(self.data, self.marker_labels),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.derived_names, derived_positions = _derived_point_positions(self.data, self.marker_config)
        self.derived_positions = torch.tensor(derived_positions, device=self.device, dtype=torch.float32).unsqueeze(0)

        self.robot_cfg = robot_config(args.robot_name)
        self.robot_cfg.asset.disable_gravity = True
        self.robot_cfg.asset.fix_base_link = False
        self.robot_cfg.asset.self_collisions = False
        self.root_local_marker_centroid = _pelvis_marker_centroid_in_root_frame(self.robot_cfg)

        self.simulator_cfg = simulator_config(
            args.simulator,
            self.robot_cfg,
            headless=args.headless,
            num_envs=1,
            experiment_name="static_calibration_marker_viewer",
        )

        extra_simulator_params = {}
        if args.simulator == "isaaclab":
            app_launcher = AppLauncher({"headless": args.headless, "device": str(self.device)})
            extra_simulator_params["simulation_app"] = app_launcher.app

        simulator_class = get_class(self.simulator_cfg._target_)
        scene_lib = SceneLib.empty(num_envs=self.simulator_cfg.num_envs, device=self.device)
        self.simulator = simulator_class(
            config=self.simulator_cfg,
            robot_config=self.robot_cfg,
            terrain=None,
            device=self.device,
            scene_lib=scene_lib,
            **extra_simulator_params,
        )
        self.simulator._initialize_with_markers(self._marker_configs())
        self._configure_gravity()
        self._reset_robot_to_static_pelvis()
        self._print_summary()

    def _configure_gravity(self) -> None:
        gravity = (0.0, 0.0, -9.81) if args.enable_gravity else (0.0, 0.0, 0.0)
        model = getattr(self.simulator, "model", None)
        if model is not None and hasattr(model, "set_gravity"):
            model.set_gravity(gravity)
        print(f"Simulator gravity: {gravity}")

    def _marker_configs(self) -> dict[str, VisualizationMarkerConfig]:
        configs = {
            "c3d_static_markers": VisualizationMarkerConfig(
                type="sphere",
                color=(0.1, 0.35, 1.0),
                markers=[MarkerConfig(size="small") for _ in self.marker_labels],
            ),
            "contact_body_centers": VisualizationMarkerConfig(
                type="sphere",
                color=(1.0, 0.05, 0.05),
                markers=[MarkerConfig(size="regular") for _ in self.robot_cfg.contact_bodies],
            ),
        }
        if self.derived_names:
            configs["derived_scaling_points"] = VisualizationMarkerConfig(
                type="sphere",
                color=(1.0, 0.85, 0.05),
                markers=[MarkerConfig(size="regular") for _ in self.derived_names],
            )
        return configs

    def _reset_robot_to_static_pelvis(self) -> None:
        root_pos, root_rot = _root_pose_from_pelvis(self.data, self.device, self.root_local_marker_centroid)
        current_state = self.simulator.get_robot_state()
        current_state.dof_pos = torch.zeros_like(current_state.dof_pos)
        current_state.dof_vel = torch.zeros_like(current_state.dof_vel)
        current_state.rigid_body_pos[:, 0, :] = root_pos.unsqueeze(0)
        current_state.rigid_body_rot[:, 0, :] = root_rot.unsqueeze(0)
        current_state.rigid_body_vel[:, 0, :] = 0.0
        current_state.rigid_body_ang_vel[:, 0, :] = 0.0
        env_ids = torch.arange(1, device=self.device)
        self.simulator.reset_envs(current_state, new_object_states=None, env_ids=env_ids)

    def _contact_body_positions(self) -> torch.Tensor:
        current_state = self.simulator.get_bodies_state()
        body_indices = [self.simulator._body_names.index(name) for name in self.robot_cfg.contact_bodies]
        return current_state.rigid_body_pos[:, body_indices, :].detach().clone()

    def _markers_state(self) -> dict[str, MarkerState]:
        states = {
            "c3d_static_markers": MarkerState(
                translation=self.marker_positions,
                orientation=_identity_quat(self.marker_positions.shape[1], self.device),
            ),
            "contact_body_centers": MarkerState(
                translation=self._contact_body_positions(),
                orientation=_identity_quat(len(self.robot_cfg.contact_bodies), self.device),
            ),
        }
        if self.derived_names:
            states["derived_scaling_points"] = MarkerState(
                translation=self.derived_positions,
                orientation=_identity_quat(self.derived_positions.shape[1], self.device),
            )
        return states

    def _print_summary(self) -> None:
        print("\n=== Static Calibration Marker Viewer ===")
        print(f"C3D: {args.c3d}")
        print(f"Static frames: {self.static_start}-{self.static_end}")
        print(f"Robot: {args.robot_name}")
        print(f"Asset: {self.robot_cfg.asset.asset_file_name}")
        if self.root_local_marker_centroid is not None:
            print(
                "Root anchor: MJCF pelvis marker-site centroid "
                f"{self.root_local_marker_centroid.tolist()} in root frame"
            )
        else:
            print("Root anchor: raw static pelvis marker centroid")
        print(f"Markers shown: {len(self.marker_labels)} blue C3D markers")
        print(f"Contact bodies shown: {', '.join(self.robot_cfg.contact_bodies)} red spheres")
        if self.derived_names:
            print(f"Derived scaling points shown: {', '.join(self.derived_names)} yellow spheres")
        print(f"Physics stepping: {'enabled' if args.simulate else 'disabled/static freeze'}")
        print("Blue marker order:")
        for idx, label in enumerate(self.marker_labels, start=1):
            print(f"  {idx:02d}: {label} ({_short_label(label)})")
        print("\nClose the Newton viewer window or press Ctrl+C in the terminal to stop.\n")

    def run(self) -> None:
        actions = torch.zeros(1, self.robot_cfg.number_of_actions, device=self.device)
        step_count = 0
        while args.max_steps <= 0 or step_count < args.max_steps:
            if args.simulate:
                self.simulator.step(actions, markers_callback=self._markers_state)
            else:
                self._reset_robot_to_static_pelvis()
                self.simulator._update_markers(self._markers_state())
                self.simulator.render()
                time.sleep(1.0 / 60.0)
            step_count += 1

    def close(self) -> None:
        self.simulator.close()


def main() -> None:
    viewer = StaticCalibrationMarkerViewer()
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nShutting down static calibration marker viewer...")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
