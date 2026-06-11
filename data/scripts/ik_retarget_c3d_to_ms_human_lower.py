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
"""Marker-IK retarget a C3D trial onto the scaled MS-Human lower-body model.

Newton's batched Warp IK solver is the only optimizer backend.  PyTorch is used
only for C3D preprocessing, warm-start construction, reporting, and export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    compute_forward_kinematics_from_transforms,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.base_simulator.simulator_state import StateConversion
from protomotions.utils.c3d_io import marker_index
from protomotions.utils.rotations import matrix_to_quaternion

from retarget_c3d_to_ms_human_lower import (  # noqa: E402
    PELVIS_MARKERS,
    _load_window,
    _point,
    _unit_scale,
    extract_contacts,
    extract_joint_angles,
)


MARKER_BODY_MAP: dict[str, str] = {
    # Pelvis landmarks and Visual3D-modeled hip centers.
    "RASI": "pelvis",
    "LASI": "pelvis",
    "RPSI": "pelvis",
    "LPSI": "pelvis",
    "RIGHT_HIP": "pelvis",
    "LEFT_HIP": "pelvis",
    # Lateral lower-limb markers.  S003 does not have medial knee/ankle markers.
    "RKNE": "femur_r",
    "LKNE": "femur_l",
    "RTHI": "femur_r",
    "LTHI": "femur_l",
    "RANK": "tibia_r",
    "LANK": "tibia_l",
    "RTIB": "tibia_r",
    "LTIB": "tibia_l",
    "RHEE": "calcn_r",
    "LHEE": "calcn_l",
    "RTOE": "toes_r",
    "LTOE": "toes_l",
}

BODY_SEGMENT_FRAME_MAP: dict[str, str] = {
    "pelvis": "PEL",
    "femur_r": "RFE",
    "femur_l": "LFE",
    "tibia_r": "RTI",
    "tibia_l": "LTI",
    "calcn_r": "RFO",
    "calcn_l": "LFO",
    "toes_r": "RTO",
    "toes_l": "LTO",
}


def _v3d_points_to_model(points: np.ndarray) -> np.ndarray:
    """Map Visual3D points to the ProtoMotions/MS-Human world frame.

    Visual3D stores this trial as X=mediolateral, Y=anteroposterior,
    Z=vertical.  The MS-Human MJCF rest pose used by ProtoMotions is Z-up with
    X=anterior/forward and Y=left.  In the S003 lab coordinates anterior is the
    negative Visual3D Y direction, so the world-frame mapping is:

        model_x = -v3d_y, model_y = v3d_x, model_z = v3d_z.
    """
    result = np.empty_like(points, dtype=np.float32)
    result[..., 0] = -points[..., 1]
    result[..., 1] = points[..., 0]
    result[..., 2] = points[..., 2]
    return result


def _v3d_tensor_to_model(points: torch.Tensor) -> torch.Tensor:
    return torch.stack([-points[..., 1], points[..., 0], points[..., 2]], dim=-1)


@dataclass
class MarkerSet:
    labels: list[str]
    body_indices: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor


@dataclass
class RMSStats:
    overall_mm: float
    per_marker_mm: dict[str, float]
    num_valid_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("c3d", type=Path, help="Visual3D/Vicon C3D file.")
    parser.add_argument("--robot-name", default="ms_human_lower_s003", help="Scaled MS-Human robot config.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ms-human-lower-retargeted/proto/S003_marker_ik.motion"),
        help="Output ProtoMotions .motion path.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON marker-RMS report path.")
    parser.add_argument("--start-frame", type=int, default=1, help="First 1-based C3D frame to export.")
    parser.add_argument("--end-frame", type=int, default=None, help="Last 1-based C3D frame to export, inclusive.")
    parser.add_argument("--output-fps", type=int, default=50, help="Downsampled output FPS.")
    parser.add_argument("--max-frames", type=int, default=2000, help="Optional frame cap after downsampling; <=0 disables.")
    parser.add_argument("--calibration-frame", type=int, default=0, help="0-based downsampled frame used for virtual marker offsets.")
    parser.add_argument(
        "--offset-calibration",
        choices=["frame", "mean", "median"],
        default="mean",
        help="How to estimate rigid local marker offsets before IK. Whole-window mean/median is much more robust than one frame.",
    )
    parser.add_argument(
        "--marker-offset-source",
        choices=["qpos", "v3d-segment"],
        default="qpos",
        help="Calibrate marker link offsets from the model warm start or from Visual3D segment frames.",
    )
    parser.add_argument(
        "--root-orientation",
        choices=["pelvis", "identity"],
        default="pelvis",
        help="Warm-start floating-root orientation. Pelvis uses RASI/LASI/RPSI/LPSI marker axes.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--backend",
        choices=["newton"],
        default="newton",
        help="IK backend. Newton is the only supported solver/optimizer backend.",
    )
    parser.add_argument("--rerun-output", type=Path, default=None, help="Optional .rrd recording path for Newton skeleton + marker playback.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Frames per FK chunk during each optimizer step.")
    parser.add_argument("--newton-iterations", type=int, default=60, help="Newton IK iterations when --backend newton.")
    parser.add_argument("--newton-optimizer", choices=["lm", "lbfgs"], default="lm")
    parser.add_argument("--newton-jacobian", choices=["analytic", "autodiff", "mixed"], default="analytic")
    parser.add_argument("--newton-lambda", type=float, default=1e-2, help="Initial LM damping for Newton IK.")
    parser.add_argument("--newton-step-size", type=float, default=1.0)
    parser.add_argument("--marker-weight", type=float, default=1.0)
    parser.add_argument(
        "--offset-refine-passes",
        type=int,
        default=0,
        help="After the first Newton solve, recalibrate fixed marker offsets from solved qpos and solve this many extra passes.",
    )
    parser.add_argument(
        "--warmstart-angle-mode",
        choices=["raw", "closest-initial", "zero"],
        default="zero",
        help=(
            "Visual3D angle handling for the Newton warm start. 'closest-initial' keeps each hinge on the equivalent "
            "2*pi branch closest to frame 0 without using anatomical joint limits; 'zero' keeps only the measured root."
        ),
    )
    parser.add_argument(
        "--warmstart-clamp-limits",
        action="store_true",
        help="Clamp only the warm-start DOFs to robot limits before Newton IK. Disabled by default for diagnostics.",
    )
    parser.add_argument("--segment-origin-weight", type=float, default=0.0, help="Optional Newton IK weight for Visual3D segment-origin position targets.")
    parser.add_argument("--segment-rotation-weight", type=float, default=0.0, help="Optional Newton IK weight for Visual3D segment-frame rotation targets.")
    parser.add_argument("--height-offset", type=float, default=0.04)
    parser.add_argument("--force-threshold", type=float, default=50.0, help="GRF magnitude threshold for contact labels.")
    return parser.parse_args()


def _normalized_marker_positions(data) -> np.ndarray:
    """Return markers in meters in the model frame, origin-normalized in XY."""
    scale = _unit_scale(data)
    markers = _v3d_points_to_model(data.markers.astype(np.float32) * scale)
    pelvis = _v3d_points_to_model(np.stack([_point(data, name) for name in PELVIS_MARKERS], axis=0).astype(np.float32) * scale)
    root = np.nanmean(pelvis, axis=0)
    markers[:, :, :2] -= root[0:1, None, :2]
    return markers


def _build_marker_set(data, body_names: list[str], device: torch.device) -> MarkerSet:
    body_index = {name: idx for idx, name in enumerate(body_names)}
    marker_positions = _normalized_marker_positions(data)

    labels: list[str] = []
    bodies: list[int] = []
    targets: list[np.ndarray] = []
    skipped: list[str] = []
    for marker_label, body_name in MARKER_BODY_MAP.items():
        if body_name not in body_index:
            skipped.append(f"{marker_label}->{body_name} (missing body)")
            continue
        try:
            marker_idx = marker_index(data.marker_labels, marker_label)
        except KeyError:
            skipped.append(f"{marker_label}->{body_name} (missing marker)")
            continue
        labels.append(marker_label)
        bodies.append(body_index[body_name])
        targets.append(marker_positions[:, marker_idx])

    if not labels:
        raise ValueError("No configured IK markers were found in the C3D data and robot body list.")

    target_tensor = torch.from_numpy(np.stack(targets, axis=1)).to(device=device, dtype=torch.float32)
    valid = torch.isfinite(target_tensor).all(dim=-1)
    target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Using {len(labels)} marker constraints: {', '.join(labels)}")
    if skipped:
        print(f"Skipped {len(skipped)} configured marker constraints: {', '.join(skipped)}")

    return MarkerSet(
        labels=labels,
        body_indices=torch.tensor(bodies, device=device, dtype=torch.long),
        targets=target_tensor,
        valid=valid,
    )


def _extract_root_rotations(data, device: torch.device, mode: str) -> torch.Tensor:
    root_rot_wxyz = torch.zeros(data.markers.shape[0], 4, device=device, dtype=torch.float32)
    root_rot_wxyz[:, 0] = 1.0
    if mode == "identity":
        return root_rot_wxyz

    scale = _unit_scale(data)
    rasi = _v3d_tensor_to_model(torch.from_numpy(_point(data, "RASI").astype(np.float32) * scale).to(device))
    lasi = _v3d_tensor_to_model(torch.from_numpy(_point(data, "LASI").astype(np.float32) * scale).to(device))
    rpsi = _v3d_tensor_to_model(torch.from_numpy(_point(data, "RPSI").astype(np.float32) * scale).to(device))
    lpsi = _v3d_tensor_to_model(torch.from_numpy(_point(data, "LPSI").astype(np.float32) * scale).to(device))

    # In the model frame, +X is anterior, +Y is left, and +Z is vertical.
    x_axis_hint = 0.5 * (rasi + lasi) - 0.5 * (rpsi + lpsi)
    y_axis = lasi - rasi
    x_axis_hint = x_axis_hint / torch.linalg.norm(x_axis_hint, dim=-1, keepdim=True).clamp_min(1e-8)
    y_axis = y_axis / torch.linalg.norm(y_axis, dim=-1, keepdim=True).clamp_min(1e-8)
    z_axis = torch.cross(x_axis_hint, y_axis, dim=-1)
    z_axis = z_axis / torch.linalg.norm(z_axis, dim=-1, keepdim=True).clamp_min(1e-8)
    x_axis = torch.cross(y_axis, z_axis, dim=-1)
    x_axis = x_axis / torch.linalg.norm(x_axis, dim=-1, keepdim=True).clamp_min(1e-8)
    root_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return matrix_to_quaternion(root_rot_mat, w_last=False)


def _extract_model_root_positions(data) -> torch.Tensor:
    scale = _unit_scale(data)
    pelvis = _v3d_points_to_model(np.stack([_point(data, name) for name in PELVIS_MARKERS], axis=0).astype(np.float32) * scale)
    root = np.nanmean(pelvis, axis=0).astype(np.float32)
    root[:, :2] -= root[0:1, :2]
    return torch.from_numpy(root)


def _canonicalize_dof_angles(dof_pos: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(dof_pos)
    if mode == "raw":
        return dof_pos
    if mode == "closest-initial":
        initial = dof_pos[0:1]
        return dof_pos - torch.round((dof_pos - initial) / (2.0 * math.pi)) * (2.0 * math.pi)
    raise ValueError(f"Unsupported warm-start angle mode: {mode}")


def _initial_qpos(data, ki, device: torch.device, root_orientation: str, angle_mode: str, clamp_limits: bool) -> torch.Tensor:
    dof_pos = extract_joint_angles(data, ki.dof_names).to(device=device, dtype=torch.float32)
    dof_pos = _canonicalize_dof_angles(dof_pos, angle_mode)
    if clamp_limits:
        limits_lower = ki.dof_limits_lower.to(device=device, dtype=torch.float32)
        limits_upper = ki.dof_limits_upper.to(device=device, dtype=torch.float32)
        dof_pos = torch.max(torch.min(dof_pos, limits_upper), limits_lower)

    root_pos = _extract_model_root_positions(data).to(device=device, dtype=torch.float32)
    root_rot_wxyz = _extract_root_rotations(data, device, root_orientation)
    return torch.cat([root_pos, root_rot_wxyz, dof_pos], dim=-1)


def _calibrate_local_offsets(ki, qpos: torch.Tensor, marker_set: MarkerSet, calibration_frame: int, mode: str = "frame") -> torch.Tensor:
    calibration_frame = max(0, min(calibration_frame, qpos.shape[0] - 1))
    with torch.no_grad():
        root_pos, joint_rot_mats = extract_transforms_from_qpos(ki, qpos)
        world_pos, world_rot = compute_forward_kinematics_from_transforms(ki, root_pos, joint_rot_mats)
        body_pos = world_pos[:, marker_set.body_indices]
        body_rot = world_rot[:, marker_set.body_indices]
        local_samples = torch.matmul(body_rot.transpose(-1, -2), (marker_set.targets - body_pos).unsqueeze(-1)).squeeze(-1)

        if mode == "frame":
            valid = marker_set.valid[calibration_frame]
            if not bool(valid.all()):
                missing = [label for label, is_valid in zip(marker_set.labels, valid.tolist()) if not is_valid]
                raise ValueError(
                    "Calibration frame has invalid marker samples for: "
                    + ", ".join(missing)
                    + ". Choose another --calibration-frame."
                )
            return local_samples[calibration_frame]

        offsets = torch.zeros_like(local_samples[0])
        for marker_idx in range(len(marker_set.labels)):
            samples = local_samples[marker_set.valid[:, marker_idx], marker_idx]
            if samples.numel() == 0:
                raise ValueError(f"No finite samples available while calibrating marker {marker_set.labels[marker_idx]}.")
            if mode == "median":
                offsets[marker_idx] = samples.median(dim=0).values
            elif mode == "mean":
                offsets[marker_idx] = samples.mean(dim=0)
            else:
                raise ValueError(f"Unsupported offset calibration mode: {mode}")
        return offsets


def _calibrate_local_offsets_from_v3d_segments(
    data,
    ki,
    qpos: torch.Tensor,
    marker_set: MarkerSet,
    calibration_frame: int,
    mode: str,
) -> torch.Tensor:
    calibration_frame = max(0, min(calibration_frame, qpos.shape[0] - 1))
    with torch.no_grad():
        root_pos, joint_rot_mats = extract_transforms_from_qpos(ki, qpos)
        _, body_rot = compute_forward_kinematics_from_transforms(ki, root_pos, joint_rot_mats)

        offsets = torch.empty((len(marker_set.labels), 3), device=qpos.device, dtype=qpos.dtype)
        for marker_idx, marker_label in enumerate(marker_set.labels):
            body_name = MARKER_BODY_MAP[marker_label]
            body_idx = int(marker_set.body_indices[marker_idx].item())
            origin, segment_rot = _v3d_segment_frames(data, BODY_SEGMENT_FRAME_MAP[body_name], qpos.device)
            local_samples = torch.matmul(
                segment_rot.transpose(-1, -2),
                (marker_set.targets[:, marker_idx] - origin).unsqueeze(-1),
            ).squeeze(-1)

            valid = marker_set.valid[:, marker_idx]
            if mode == "frame":
                if not bool(valid[calibration_frame]):
                    raise ValueError(f"Calibration frame has an invalid marker sample for {marker_label}.")
                segment_offset = local_samples[calibration_frame]
            else:
                samples = local_samples[valid]
                if samples.numel() == 0:
                    raise ValueError(f"No finite samples available while calibrating marker {marker_label}.")
                if mode == "median":
                    segment_offset = samples.median(dim=0).values
                elif mode == "mean":
                    segment_offset = samples.mean(dim=0)
                else:
                    raise ValueError(f"Unsupported offset calibration mode: {mode}")

            link_to_segment = torch.matmul(body_rot[calibration_frame, body_idx].transpose(-1, -2), segment_rot[calibration_frame])
            offsets[marker_idx] = torch.matmul(link_to_segment, segment_offset[:, None]).squeeze(-1)

        return offsets


def _compute_fk_functional(ki, root_pos: torch.Tensor, joint_rot_mats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional FK variant that avoids in-place tensor writes in autograd graphs."""
    device = root_pos.device
    dtype = root_pos.dtype
    local_pos = ki.local_pos.to(device=device, dtype=dtype)
    local_rot_ref_mat = ki.local_rot_ref_mat.to(device=device, dtype=dtype)
    local_joint_pos = getattr(ki, "local_joint_pos", torch.zeros_like(ki.local_pos)).to(device=device, dtype=dtype)

    world_pos: list[torch.Tensor] = []
    world_rot: list[torch.Tensor] = []
    for body_idx, parent_idx in enumerate(ki.parent_indices):
        if parent_idx == -1:
            world_pos.append(root_pos)
            world_rot.append(joint_rot_mats[:, 0])
            continue

        parent_pos = world_pos[parent_idx]
        parent_rot = world_rot[parent_idx]
        effective_local_rot = torch.matmul(local_rot_ref_mat[body_idx], joint_rot_mats[:, body_idx])
        body_rot = torch.matmul(parent_rot, effective_local_rot)
        pivot = local_joint_pos[body_idx]
        local_offset = (
            local_pos[body_idx]
            + torch.matmul(local_rot_ref_mat[body_idx], pivot.view(3, 1)).squeeze(-1)
            - torch.matmul(effective_local_rot, pivot.view(1, 3, 1)).squeeze(-1)
        )
        body_pos = parent_pos + torch.matmul(parent_rot, local_offset[:, :, None]).squeeze(-1)
        world_pos.append(body_pos)
        world_rot.append(body_rot)

    return torch.stack(world_pos, dim=1), torch.stack(world_rot, dim=1)


def _predict_markers(ki, qpos_chunk: torch.Tensor, marker_set: MarkerSet, local_offsets: torch.Tensor) -> torch.Tensor:
    root_pos, joint_rot_mats = extract_transforms_from_qpos(ki, qpos_chunk)
    world_pos, world_rot = _compute_fk_functional(ki, root_pos, joint_rot_mats)
    body_pos = world_pos[:, marker_set.body_indices]
    body_rot = world_rot[:, marker_set.body_indices]
    offset_world = torch.matmul(body_rot, local_offsets[None, :, :, None]).squeeze(-1)
    return body_pos + offset_world


def _compute_rms_stats(ki, qpos: torch.Tensor, marker_set: MarkerSet, local_offsets: torch.Tensor, chunk_size: int) -> RMSStats:
    total_sq = 0.0
    total_count = 0
    per_marker_sq = torch.zeros(len(marker_set.labels), device=qpos.device)
    per_marker_count = torch.zeros(len(marker_set.labels), device=qpos.device)
    with torch.no_grad():
        for start in range(0, qpos.shape[0], chunk_size):
            end = min(start + chunk_size, qpos.shape[0])
            pred = _predict_markers(ki, qpos[start:end], marker_set, local_offsets)
            err_sq = ((pred - marker_set.targets[start:end]) ** 2).sum(dim=-1)
            valid = marker_set.valid[start:end]
            err_sq = torch.where(valid, err_sq, torch.zeros_like(err_sq))
            total_sq += float(err_sq.sum().item())
            total_count += int(valid.sum().item())
            per_marker_sq += err_sq.sum(dim=0)
            per_marker_count += valid.sum(dim=0)

    per_marker_mm = {
        label: float(torch.sqrt(per_marker_sq[idx] / torch.clamp(per_marker_count[idx], min=1)).item() * 1000.0)
        for idx, label in enumerate(marker_set.labels)
    }
    overall_mm = float(np.sqrt(total_sq / max(total_count, 1)) * 1000.0)
    return RMSStats(overall_mm=overall_mm, per_marker_mm=per_marker_mm, num_valid_samples=total_count)


def _normalized_point(data, label: str, device: torch.device) -> torch.Tensor:
    scale = _unit_scale(data)
    point = torch.from_numpy(_v3d_points_to_model(_point(data, label).astype(np.float32) * scale)).to(device=device, dtype=torch.float32)
    pelvis = _v3d_points_to_model(np.stack([_point(data, name) for name in PELVIS_MARKERS], axis=0).astype(np.float32) * scale)
    root0_xy = torch.from_numpy(np.nanmean(pelvis, axis=0)[0, :2]).to(device=device, dtype=torch.float32)
    point[:, :2] -= root0_xy
    return point


def _v3d_segment_frames(data, prefix: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    origin = _normalized_point(data, prefix + "O", device)
    # Visual3D segment-frame point labels: L=mediolateral, A=anteroposterior,
    # P=vertical.  After mapping into the model frame, A is +X/anterior and L is
    # +Y/left.
    x_hint = _normalized_point(data, prefix + "A", device) - origin
    y_hint = _normalized_point(data, prefix + "L", device) - origin
    z_hint = _normalized_point(data, prefix + "P", device) - origin
    x_axis = x_hint / torch.linalg.norm(x_hint, dim=-1, keepdim=True).clamp_min(1e-8)
    y_axis = y_hint - (y_hint * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_axis = y_axis / torch.linalg.norm(y_axis, dim=-1, keepdim=True).clamp_min(1e-8)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    z_axis = torch.where((z_axis * z_hint).sum(dim=-1, keepdim=True) < 0.0, -z_axis, z_axis)
    z_axis = z_axis / torch.linalg.norm(z_axis, dim=-1, keepdim=True).clamp_min(1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / torch.linalg.norm(y_axis, dim=-1, keepdim=True).clamp_min(1e-8)
    return origin, torch.stack([x_axis, y_axis, z_axis], dim=-1)


def _sliding_mean_offsets(local_samples: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return local_samples
    radius = window // 2
    offsets = torch.empty_like(local_samples)
    for frame_idx in range(local_samples.shape[0]):
        start = max(0, frame_idx - radius)
        end = min(local_samples.shape[0], frame_idx + radius + 1)
        sample_window = local_samples[start:end]
        valid_window = valid[start:end].to(local_samples.dtype)[..., None]
        denom = valid_window.sum(dim=0).clamp_min(1.0)
        offsets[frame_idx] = (sample_window * valid_window).sum(dim=0) / denom
    return offsets


def _predict_markers_v3d_segments(data, marker_set: MarkerSet, device: torch.device, window: int) -> torch.Tensor:
    targets = marker_set.targets.to(device)
    origins: list[torch.Tensor] = []
    rotations: list[torch.Tensor] = []
    for marker_label in marker_set.labels:
        body_name = MARKER_BODY_MAP[marker_label]
        prefix = BODY_SEGMENT_FRAME_MAP[body_name]
        origin, rotation = _v3d_segment_frames(data, prefix, device)
        origins.append(origin)
        rotations.append(rotation)
    origin_tensor = torch.stack(origins, dim=1)
    rotation_tensor = torch.stack(rotations, dim=1)
    local_samples = torch.matmul(rotation_tensor.transpose(-1, -2), (targets - origin_tensor).unsqueeze(-1)).squeeze(-1)
    local_offsets = _sliding_mean_offsets(local_samples, marker_set.valid.to(device), window)
    return origin_tensor + torch.matmul(rotation_tensor, local_offsets.unsqueeze(-1)).squeeze(-1)


def _compute_v3d_segment_rms_stats(data, marker_set: MarkerSet, device: torch.device, window: int) -> tuple[RMSStats, torch.Tensor]:
    with torch.no_grad():
        pred = _predict_markers_v3d_segments(data, marker_set, device, window)
        targets = marker_set.targets.to(device)
        valid = marker_set.valid.to(device)
        err_sq = ((pred - targets) ** 2).sum(dim=-1)
        err_sq = torch.where(valid, err_sq, torch.zeros_like(err_sq))
        per_marker_sq = err_sq.sum(dim=0)
        per_marker_count = valid.sum(dim=0)
        per_marker_mm = {
            label: float(torch.sqrt(per_marker_sq[idx] / torch.clamp(per_marker_count[idx], min=1)).item() * 1000.0)
            for idx, label in enumerate(marker_set.labels)
        }
        overall_mm = float(torch.sqrt(err_sq.sum() / torch.clamp(valid.sum(), min=1)).item() * 1000.0)
        return RMSStats(overall_mm=overall_mm, per_marker_mm=per_marker_mm, num_valid_samples=int(valid.sum().item())), pred


def _quat_inverse_xyzw(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., :3] = -result[..., :3]
    return result / torch.sum(quat * quat, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_mul_xyzw(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_xyz = lhs[..., :3]
    rhs_xyz = rhs[..., :3]
    lhs_w = lhs[..., 3:4]
    rhs_w = rhs[..., 3:4]
    xyz = lhs_w * rhs_xyz + rhs_w * lhs_xyz + torch.cross(lhs_xyz, rhs_xyz, dim=-1)
    w = lhs_w * rhs_w - torch.sum(lhs_xyz * rhs_xyz, dim=-1, keepdim=True)
    return torch.cat([xyz, w], dim=-1)


def _qpos_wxyz_to_newton_xyzw(qpos: torch.Tensor) -> torch.Tensor:
    joint_q = qpos.clone()
    joint_q[:, 3:7] = torch.stack([qpos[:, 4], qpos[:, 5], qpos[:, 6], qpos[:, 3]], dim=-1)
    return joint_q


def _newton_xyzw_to_qpos_wxyz(joint_q: torch.Tensor) -> torch.Tensor:
    qpos = joint_q.clone()
    qpos[:, 3:7] = torch.stack([joint_q[:, 6], joint_q[:, 3], joint_q[:, 4], joint_q[:, 5]], dim=-1)
    return qpos


def _build_newton_model(cfg, device: torch.device):
    import newton

    asset_path = os.path.join(cfg.asset.asset_root, cfg.asset.asset_file_name)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig()
    builder.default_shape_cfg.mu = 1.0
    builder.add_mjcf(
        asset_path,
        ignore_names=["floor", "ground"],
        ignore_classes=["wrap"],
        collapse_fixed_joints=False,
        floating=not cfg.asset.fix_base_link,
        enable_self_collisions=cfg.asset.self_collisions,
    )
    builder.articulation_label = ["robot"]
    return builder.finalize(device=str(device), requires_grad=True)


def _optimize_qpos_newton(args, cfg, data, qpos_init: torch.Tensor, marker_set: MarkerSet, local_offsets: torch.Tensor) -> torch.Tensor:
    import newton.ik as ik
    import warp as wp

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = _build_newton_model(cfg, device)
    if model.joint_coord_count != qpos_init.shape[1]:
        raise ValueError(f"Newton joint_coord_count={model.joint_coord_count}, expected qpos width {qpos_init.shape[1]}")

    joint_q_torch = _qpos_wxyz_to_newton_xyzw(qpos_init).contiguous()
    joint_q = wp.from_torch(joint_q_torch, dtype=wp.float32, requires_grad=args.newton_jacobian in {"autodiff", "mixed"})
    joint_q_out = wp.empty_like(joint_q)

    objectives = []
    target_cpu = marker_set.targets.detach().cpu().numpy().astype(np.float32)
    offsets_cpu = local_offsets.detach().cpu().numpy().astype(np.float32)
    body_indices_cpu = marker_set.body_indices.detach().cpu().numpy().astype(np.int32)
    weight = float(args.marker_weight)
    for marker_idx, label in enumerate(marker_set.labels):
        targets = wp.array(target_cpu[:, marker_idx], dtype=wp.vec3, device=str(device))
        offset = wp.vec3(*offsets_cpu[marker_idx].tolist())
        objectives.append(
            ik.IKObjectivePosition(
                link_index=int(body_indices_cpu[marker_idx]),
                link_offset=offset,
                target_positions=targets,
                weight=weight,
            )
        )

    body_index = {name: idx for idx, name in enumerate(cfg.kinematic_info.body_names)}
    segment_bodies = [body for body in BODY_SEGMENT_FRAME_MAP if body in body_index]
    if args.segment_origin_weight > 0.0 or args.segment_rotation_weight > 0.0:
        root_pos, joint_rot_mats = extract_transforms_from_qpos(cfg.kinematic_info, qpos_init)
        _, initial_body_rot = compute_forward_kinematics_from_transforms(cfg.kinematic_info, root_pos, joint_rot_mats)

    for body_name in segment_bodies:
        body_idx = body_index[body_name]
        origin, rotation = _v3d_segment_frames(data, BODY_SEGMENT_FRAME_MAP[body_name], qpos_init.device)
        if args.segment_origin_weight > 0.0:
            objectives.append(
                ik.IKObjectivePosition(
                    link_index=body_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array(origin.detach().cpu().numpy().astype(np.float32), dtype=wp.vec3, device=str(device)),
                    weight=float(args.segment_origin_weight),
                )
            )
        if args.segment_rotation_weight > 0.0:
            target_rot_xyzw = matrix_to_quaternion(rotation, w_last=True)
            initial_rot_xyzw = matrix_to_quaternion(initial_body_rot[:, body_idx], w_last=True)
            offset_xyzw = _quat_mul_xyzw(_quat_inverse_xyzw(initial_rot_xyzw[args.calibration_frame]), target_rot_xyzw[args.calibration_frame])
            objectives.append(
                ik.IKObjectiveRotation(
                    link_index=body_idx,
                    link_offset_rotation=wp.quat(*offset_xyzw.detach().cpu().numpy().astype(np.float32).tolist()),
                    target_rotations=wp.array(target_rot_xyzw.detach().cpu().numpy().astype(np.float32), dtype=wp.vec4, device=str(device)),
                    weight=float(args.segment_rotation_weight),
                )
            )

    solver = ik.IKSolver(
        model,
        qpos_init.shape[0],
        objectives,
        optimizer=args.newton_optimizer,
        jacobian_mode=args.newton_jacobian,
        lambda_initial=args.newton_lambda,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    solver.step(joint_q, joint_q_out, iterations=args.newton_iterations, step_size=args.newton_step_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    sps = qpos_init.shape[0] * args.newton_iterations / max(elapsed, 1e-12)
    fps_equiv = qpos_init.shape[0] / max(elapsed, 1e-12)
    print(
        f"Newton IK timing: elapsed={elapsed:.3f}s frames={qpos_init.shape[0]} "
        f"iterations={args.newton_iterations} sps={sps:,.0f} frame_sps={fps_equiv:,.0f}"
    )
    timing = {
        "elapsed_seconds": elapsed,
        "sample_iterations_per_second": sps,
        "frames_per_second": fps_equiv,
        "timed_iterations": args.newton_iterations,
    }
    args._timing = timing
    if not hasattr(args, "_timing_history"):
        args._timing_history = []
    args._timing_history.append(timing)

    joint_q_result = wp.to_torch(joint_q_out).detach().clone().to(device=qpos_init.device)
    qpos = _newton_xyzw_to_qpos_wxyz(joint_q_result)
    qpos[:, 3:7] /= torch.linalg.norm(qpos[:, 3:7], dim=-1, keepdim=True).clamp_min(1e-8)
    return qpos


def _write_report(report_path: Path, args, stats: RMSStats, marker_set: MarkerSet) -> None:
    report = {
        "c3d": str(args.c3d),
        "robot_name": args.robot_name,
        "output": str(args.output),
        "overall_rms_mm": stats.overall_mm,
        "num_valid_marker_samples": stats.num_valid_samples,
        "markers": marker_set.labels,
        "per_marker_rms_mm": stats.per_marker_mm,
        "backend": args.backend,
        "newton_iterations": args.newton_iterations,
        "newton_optimizer": args.newton_optimizer,
        "newton_jacobian": args.newton_jacobian,
        "chunk_size": args.chunk_size,
        "offset_calibration": args.offset_calibration,
        "offset_refine_passes": args.offset_refine_passes,
        "marker_offset_source": args.marker_offset_source,
        "root_orientation": args.root_orientation,
        "segment_origin_weight": args.segment_origin_weight,
        "segment_rotation_weight": args.segment_rotation_weight,
    }
    if hasattr(args, "_timing"):
        report["timing"] = args._timing
    if hasattr(args, "_timing_history"):
        report["timing_history"] = args._timing_history
        report["total_ik_elapsed_seconds"] = sum(item["elapsed_seconds"] for item in args._timing_history)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def _write_rerun_recording(output_path: Path, motion, ki, marker_set: MarkerSet, marker_pred: torch.Tensor | None, marker_targets: torch.Tensor) -> None:
    try:
        import rerun as rr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("rerun-sdk is required for --rerun-output. Install with `pip install rerun-sdk`.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("ms_human_lower_marker_ik", spawn=False)
    rr.save(str(output_path))

    parent_indices = ki.parent_indices
    body_pos = motion.rigid_body_pos.detach().cpu().numpy()
    targets = marker_targets.detach().cpu().numpy()
    pred = None if marker_pred is None else marker_pred.detach().cpu().numpy()
    for frame_idx in range(body_pos.shape[0]):
        rr.set_time_sequence("frame", frame_idx)
        rr.log("newton/body_points", rr.Points3D(body_pos[frame_idx], radii=0.012))
        lines = []
        for body_idx, parent_idx in enumerate(parent_indices):
            if parent_idx >= 0:
                lines.append(np.stack([body_pos[frame_idx, parent_idx], body_pos[frame_idx, body_idx]], axis=0))
        if lines:
            rr.log("newton/skeleton", rr.LineStrips3D(lines, radii=0.006))
        rr.log("markers/target", rr.Points3D(targets[frame_idx], radii=0.018, labels=marker_set.labels))
        if pred is not None:
            rr.log("markers/reconstructed", rr.Points3D(pred[frame_idx], radii=0.014, labels=marker_set.labels))
    print(f"Wrote Rerun recording: {output_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    data, output_fps = _load_window(args.c3d, args.start_frame, args.end_frame, args.output_fps, args.max_frames)
    cfg = robot_config(args.robot_name)
    ki = cfg.kinematic_info

    qpos_init = _initial_qpos(data, ki, device, args.root_orientation, args.warmstart_angle_mode, args.warmstart_clamp_limits)
    marker_set = _build_marker_set(data, ki.body_names, device)
    if args.marker_offset_source == "qpos":
        local_offsets = _calibrate_local_offsets(ki, qpos_init, marker_set, args.calibration_frame, args.offset_calibration)
    elif args.marker_offset_source == "v3d-segment":
        local_offsets = _calibrate_local_offsets_from_v3d_segments(
            data,
            ki,
            qpos_init,
            marker_set,
            args.calibration_frame,
            args.offset_calibration,
        )
    else:
        raise ValueError(f"Unsupported marker offset source: {args.marker_offset_source}")
    print(
        f"Optimizing {qpos_init.shape[0]} frames at {output_fps} fps on {device} with {args.backend}; "
        f"nq={qpos_init.shape[1]} bodies={ki.num_bodies} markers={len(marker_set.labels)}"
    )

    marker_pred_for_rerun = None
    qpos = _optimize_qpos_newton(args, cfg, data, qpos_init, marker_set, local_offsets)
    for refine_pass in range(max(0, args.offset_refine_passes)):
        local_offsets = _calibrate_local_offsets(ki, qpos, marker_set, args.calibration_frame, args.offset_calibration)
        print(f"Refining marker offsets from solved qpos: pass {refine_pass + 1}/{args.offset_refine_passes}")
        qpos = _optimize_qpos_newton(args, cfg, data, qpos, marker_set, local_offsets)
    stats = _compute_rms_stats(ki, qpos, marker_set, local_offsets, max(1, args.chunk_size))
    print(f"Final marker RMS: {stats.overall_mm:.3f} mm over {stats.num_valid_samples} marker samples")
    for label, rms_mm in sorted(stats.per_marker_mm.items()):
        print(f"  {label:>10s}: {rms_mm:8.3f} mm")

    root_pos, joint_rot_mats = extract_transforms_from_qpos(ki, qpos)
    motion = fk_from_transforms_with_velocities(
        kinematic_info=ki,
        root_pos=root_pos,
        joint_rot_mats=joint_rot_mats,
        fps=output_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    dof_pos = qpos[:, 7:].detach().cpu()
    motion.dof_pos = dof_pos
    motion.dof_vel = compute_cartesian_velocity(dof_pos.unsqueeze(1), fps=output_fps).squeeze(1)
    motion.rigid_body_contacts = extract_contacts(data, cfg, args.force_threshold)
    motion.local_rigid_body_rot = None
    motion.state_conversion = StateConversion.COMMON
    motion.fix_height_per_frame(height_offset=args.height_offset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), args.output)
    report_path = args.report or args.output.with_suffix(".rms.json")
    _write_report(report_path, args, stats, marker_set)
    print(f"Wrote motion: {args.output}")
    print(f"Wrote RMS report: {report_path}")
    if args.rerun_output is not None:
        _write_rerun_recording(args.rerun_output, motion, ki, marker_set, marker_pred_for_rerun, marker_set.targets)


if __name__ == "__main__":
    main()
