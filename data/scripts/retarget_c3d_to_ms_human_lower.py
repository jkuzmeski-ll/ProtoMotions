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
"""Create an initial MS-Human lower-body motion from a Visual3D C3D file.

This script is an implementation scaffold for the marker-IK retargeter.  It uses
Visual3D-modeled joint-angle point channels as a strong initialization/baseline,
then writes a ProtoMotions ``.motion`` for the subject-specific scaled
``ms_human_lower_s003`` model.  The later marker-IK step can use this output as
warm start and compare against its marker-RMS optimization result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.base_simulator.simulator_state import StateConversion
from protomotions.utils.c3d_io import load_c3d, marker_index, read_metadata


ANGLE_LABELS = {
    "r_hip": "RHipAngles",
    "r_knee": "RKneeAngles",
    "r_ankle": "RAnkleAngles",
    "l_hip": "LHipAngles",
    "l_knee": "LKneeAngles",
    "l_ankle": "LAnkleAngles",
}


DOF_FROM_V3D = {
    "hip_flexion_r": ("r_hip", 0, 1.0),
    "hip_adduction_r": ("r_hip", 1, 1.0),
    "hip_rotation_r": ("r_hip", 2, 1.0),
    "knee_angle_r": ("r_knee", 0, 1.0),
    "ankle_angle_r": ("r_ankle", 0, 1.0),
    "subtalar_angle_r": (None, 0, 0.0),
    "mtp_angle_r": (None, 0, 0.0),
    "hip_flexion_l": ("l_hip", 0, 1.0),
    "hip_adduction_l": ("l_hip", 1, 1.0),
    "hip_rotation_l": ("l_hip", 2, 1.0),
    "knee_angle_l": ("l_knee", 0, 1.0),
    "ankle_angle_l": ("l_ankle", 0, 1.0),
    "subtalar_angle_l": (None, 0, 0.0),
    "mtp_angle_l": (None, 0, 0.0),
}


PELVIS_MARKERS = ["RASI", "LASI", "RPSI", "LPSI"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("c3d", type=Path, help="Visual3D/Vicon C3D file.")
    parser.add_argument("--robot-name", default="ms_human_lower_s003", help="Scaled MS-Human robot config.")
    parser.add_argument("--output", type=Path, default=Path("data/ms-human-lower-retargeted/proto/S003_v3d_angles.motion"))
    parser.add_argument("--start-frame", type=int, default=1, help="First 1-based C3D frame to export.")
    parser.add_argument("--end-frame", type=int, default=None, help="Last 1-based C3D frame to export, inclusive.")
    parser.add_argument(
        "--output-fps",
        type=int,
        default=None,
        help="Optional target FPS. Defaults to the native C3D point rate; lower values decimate frames.",
    )
    parser.add_argument("--max-frames", type=int, default=2000, help="Optional frame cap after optional FPS decimation; <=0 disables.")
    parser.add_argument("--height-offset", type=float, default=0.04)
    parser.add_argument("--force-threshold", type=float, default=50.0, help="GRF magnitude threshold for contact labels.")
    return parser.parse_args()


def _point(data, name: str) -> np.ndarray:
    return data.markers[:, marker_index(data.marker_labels, name)]


def _unit_scale(data) -> float:
    return 0.001 if data.point_units.lower() in {"mm", "millimeter", "millimeters"} else 1.0


def _load_window(c3d: Path, start_frame: int, end_frame: int | None, output_fps: int | None, max_frames: int):
    metadata = read_metadata(c3d)
    source_fps = float(metadata.parameters.get("POINT.RATE", metadata.header.point_rate))
    if output_fps is not None and output_fps <= 0:
        raise ValueError(f"--output-fps must be positive when set; got {output_fps}")
    factor = 1 if output_fps is None else max(1, round(source_fps / output_fps))
    if end_frame is None:
        if max_frames > 0:
            end_frame = min(metadata.header.last_frame, start_frame + factor * max_frames - 1)
        else:
            end_frame = metadata.header.last_frame
    data = load_c3d(c3d, start_frame=start_frame, end_frame=end_frame)
    if factor > 1:
        data.markers = data.markers[::factor]
    loaded_fps = float(source_fps / factor)
    data.point_rate = loaded_fps
    return data, loaded_fps


def extract_joint_angles(data, dof_names: list[str]) -> torch.Tensor:
    angle_points = {key: _point(data, label) for key, label in ANGLE_LABELS.items()}
    target = np.zeros((data.markers.shape[0], len(dof_names)), dtype=np.float32)
    for dof_idx, dof_name in enumerate(dof_names):
        source_key, axis, sign = DOF_FROM_V3D.get(dof_name, (None, 0, 0.0))
        if source_key is None:
            continue
        # Visual3D stores angle point channels in degrees.
        angle_deg = angle_points[source_key][:, axis].astype(np.float32) * sign
        finite = np.isfinite(angle_deg)
        if finite.any() and np.nanmedian(np.abs(angle_deg[finite])) > 720.0:
            # Some Visual3D C3D exports store generated angle points in
            # centidegrees while POINT.UNITS still says mm.
            angle_deg = angle_deg * 0.01
        target[:, dof_idx] = np.deg2rad(angle_deg)
    return torch.from_numpy(target)


def extract_root_positions(data) -> torch.Tensor:
    scale = _unit_scale(data)
    pelvis = np.stack([_point(data, name) for name in PELVIS_MARKERS], axis=0)
    root = np.nanmean(pelvis, axis=0) * scale
    root = root.astype(np.float32)
    root[:, :2] -= root[0:1, :2]
    return torch.from_numpy(root)


def extract_contacts(data, cfg, threshold: float) -> torch.Tensor:
    contacts = torch.zeros(data.markers.shape[0], cfg.kinematic_info.num_bodies, dtype=torch.bool)
    body_index = {name: idx for idx, name in enumerate(cfg.kinematic_info.body_names)}
    for label, bodies in [
        ("LGroundReactionForce", ["calcn_l", "toes_l"]),
        ("RGroundReactionForce", ["calcn_r", "toes_r"]),
    ]:
        try:
            force = _point(data, label)
        except KeyError:
            continue
        active = torch.from_numpy((np.linalg.norm(force, axis=-1) > threshold).astype(bool))
        for body in bodies:
            if body in body_index:
                contacts[:, body_index[body]] = active
    return contacts


def main() -> None:
    args = parse_args()
    data, output_fps = _load_window(args.c3d, args.start_frame, args.end_frame, args.output_fps, args.max_frames)
    cfg = robot_config(args.robot_name)
    ki = cfg.kinematic_info

    dof_pos = extract_joint_angles(data, ki.dof_names).to(torch.float32)
    dof_pos = torch.max(torch.min(dof_pos, ki.dof_limits_upper), ki.dof_limits_lower)
    root_pos = extract_root_positions(data).to(torch.float32)
    root_rot_wxyz = torch.zeros(root_pos.shape[0], 4, dtype=torch.float32)
    root_rot_wxyz[:, 0] = 1.0

    qpos = torch.cat([root_pos, root_rot_wxyz, dof_pos], dim=-1)
    fk_root_pos, joint_rot_mats = extract_transforms_from_qpos(ki, qpos)
    motion = fk_from_transforms_with_velocities(
        kinematic_info=ki,
        root_pos=fk_root_pos,
        joint_rot_mats=joint_rot_mats,
        fps=output_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    dof_vel = compute_cartesian_velocity(dof_pos.unsqueeze(1), fps=output_fps).squeeze(1)
    motion.dof_pos = dof_pos
    motion.dof_vel = dof_vel
    motion.rigid_body_contacts = extract_contacts(data, cfg, args.force_threshold)
    motion.local_rigid_body_rot = None
    motion.state_conversion = StateConversion.COMMON
    motion.fix_height_per_frame(height_offset=args.height_offset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), args.output)
    print(f"Wrote {args.output}")
    print(f"  frames={motion.rigid_body_pos.shape[0]} fps={output_fps} bodies={motion.rigid_body_pos.shape[1]} dofs={motion.dof_pos.shape[1]}")
    print("  Note: this is a Visual3D-angle baseline/warm start; marker-IK refinement is the next step.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
