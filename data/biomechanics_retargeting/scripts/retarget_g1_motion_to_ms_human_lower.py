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
"""Simple G1-to-MS-Human-lower retargeting demo.

This is intentionally lightweight: it copies a G1 root trajectory and maps the
major lower-body joint coordinates to the simplified MS-Human-700 lower-body
skeleton. It is not an optimizer and does not claim anatomical accuracy; it is a
quick smoke-test path for proving the new skeleton can load, run FK, save a
ProtoMotions motion, and be consumed by MotionLib/simulators.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.base_simulator.simulator_state import StateConversion
from protomotions.utils.rotations import xyzw_to_wxyz

from contact_detection import compute_contact_labels_from_pos_and_vel


G1_TO_MS_HUMAN_DOF_MAP = {
    "hip_flexion_r": ("right_hip_pitch_joint", 1.0),
    "hip_adduction_r": ("right_hip_roll_joint", 1.0),
    "hip_rotation_r": ("right_hip_yaw_joint", 1.0),
    "knee_angle_r": ("right_knee_joint", 1.0),
    "ankle_angle_r": ("right_ankle_pitch_joint", 1.0),
    "subtalar_angle_r": ("right_ankle_roll_joint", 1.0),
    "mtp_angle_r": (None, 0.0),
    "hip_flexion_l": ("left_hip_pitch_joint", 1.0),
    "hip_adduction_l": ("left_hip_roll_joint", 1.0),
    "hip_rotation_l": ("left_hip_yaw_joint", 1.0),
    "knee_angle_l": ("left_knee_joint", 1.0),
    "ankle_angle_l": ("left_ankle_pitch_joint", 1.0),
    "subtalar_angle_l": ("left_ankle_roll_joint", 1.0),
    "mtp_angle_l": (None, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="G1 .motion file or directory containing .motion files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/biomechanics_retargeting/retargeted/proto"),
        help="Directory where MS-Human lower-body .motion files will be saved.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite outputs.")
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0.04,
        help="Minimum ground clearance after per-frame height correction.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick demos.",
    )
    return parser.parse_args()


def find_motion_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".motion":
            raise ValueError(f"Input file must end in .motion: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    motion_files = sorted(path.rglob("*.motion"))
    if not motion_files:
        raise FileNotFoundError(f"No .motion files found under {path}")
    return motion_files


def retarget_motion(source_motion_path: Path, output_path: Path, max_frames: int | None, height_offset: float) -> None:
    source = torch.load(source_motion_path, map_location="cpu", weights_only=False)

    source_fps = source.get("fps", 30)
    source_dof_pos = source["dof_pos"].to(torch.float32)
    source_root_pos = source["rigid_body_pos"][:, 0].to(torch.float32)
    source_root_rot_xyzw = source["rigid_body_rot"][:, 0].to(torch.float32)
    source_contacts = source.get("rigid_body_contacts")

    if max_frames is not None:
        source_dof_pos = source_dof_pos[:max_frames]
        source_root_pos = source_root_pos[:max_frames]
        source_root_rot_xyzw = source_root_rot_xyzw[:max_frames]
        if source_contacts is not None:
            source_contacts = source_contacts[:max_frames]

    g1_cfg = robot_config("g1")
    ms_cfg = robot_config("ms_human_lower")
    g1_dof_index = {name: idx for idx, name in enumerate(g1_cfg.kinematic_info.dof_names)}

    target_dof_pos = torch.zeros(
        source_dof_pos.shape[0],
        ms_cfg.kinematic_info.num_dofs,
        dtype=torch.float32,
    )

    for target_idx, target_name in enumerate(ms_cfg.kinematic_info.dof_names):
        source_name, scale = G1_TO_MS_HUMAN_DOF_MAP[target_name]
        if source_name is None:
            continue
        target_dof_pos[:, target_idx] = source_dof_pos[:, g1_dof_index[source_name]] * scale

    lower = ms_cfg.kinematic_info.dof_limits_lower
    upper = ms_cfg.kinematic_info.dof_limits_upper
    target_dof_pos = torch.max(torch.min(target_dof_pos, upper), lower)

    root_rot_wxyz = xyzw_to_wxyz(source_root_rot_xyzw)
    qpos = torch.cat([source_root_pos, root_rot_wxyz, target_dof_pos], dim=-1)

    root_pos, joint_rot_mats = extract_transforms_from_qpos(ms_cfg.kinematic_info, qpos)
    motion = fk_from_transforms_with_velocities(
        kinematic_info=ms_cfg.kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=joint_rot_mats,
        fps=source_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    target_dof_vel = compute_cartesian_velocity(
        batched_robot_pos=target_dof_pos.unsqueeze(1),
        fps=source_fps,
    ).squeeze(1)
    motion.dof_pos = target_dof_pos
    motion.dof_vel = target_dof_vel

    translation_vecs = motion.fix_height_per_frame(height_offset=height_offset)
    if motion.rigid_body_vel is not None:
        vel_delta = torch.zeros(
            translation_vecs.shape[0],
            1,
            3,
            device=motion.rigid_body_vel.device,
            dtype=motion.rigid_body_vel.dtype,
        )
        vel_delta[:-1] = (translation_vecs[1:] - translation_vecs[:-1]).unsqueeze(1) / motion.motion_dt
        motion.rigid_body_vel = motion.rigid_body_vel + vel_delta

    motion.rigid_body_contacts = retarget_contacts(
        source_contacts=source_contacts,
        g1_cfg=g1_cfg,
        ms_cfg=ms_cfg,
        num_frames=target_dof_pos.shape[0],
    )
    if not motion.rigid_body_contacts.any():
        motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
            positions=motion.rigid_body_pos,
            velocity=motion.rigid_body_vel,
            vel_thres=0.15,
            height_thresh=0.1,
        ).to(torch.bool)
    motion.local_rigid_body_rot = None
    motion.state_conversion = StateConversion.COMMON

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), output_path)
    print(f"Wrote {output_path}")
    print(f"  frames: {motion.rigid_body_pos.shape[0]}, bodies: {motion.rigid_body_pos.shape[1]}, dofs: {motion.dof_pos.shape[1]}")


def retarget_contacts(source_contacts, g1_cfg, ms_cfg, num_frames: int) -> torch.Tensor:
    target_contacts = torch.zeros(
        num_frames,
        ms_cfg.kinematic_info.num_bodies,
        dtype=torch.bool,
    )
    if source_contacts is None:
        return target_contacts

    g1_body_index = {name: idx for idx, name in enumerate(g1_cfg.kinematic_info.body_names)}
    ms_body_index = {name: idx for idx, name in enumerate(ms_cfg.kinematic_info.body_names)}
    body_pairs = [
        ("left_ankle_roll_link", ["calcn_l", "toes_l"]),
        ("right_ankle_roll_link", ["calcn_r", "toes_r"]),
    ]
    for source_body, target_bodies in body_pairs:
        if source_body not in g1_body_index:
            continue
        source_label = source_contacts[:, g1_body_index[source_body]].to(torch.bool)
        for target_body in target_bodies:
            if target_body in ms_body_index:
                target_contacts[:, ms_body_index[target_body]] = source_label
    return target_contacts


def main() -> None:
    args = parse_args()
    motion_files = find_motion_files(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for motion_file in motion_files:
        rel_name = motion_file.name
        output_path = args.output_dir / rel_name
        if output_path.exists() and not args.force:
            print(f"Skipping existing {output_path}")
            continue
        retarget_motion(motion_file, output_path, args.max_frames, args.height_offset)


if __name__ == "__main__":
    with torch.no_grad():
        main()
