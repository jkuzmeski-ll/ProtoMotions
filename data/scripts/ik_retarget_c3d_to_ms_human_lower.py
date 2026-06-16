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
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp
from newton._src.sim.ik.ik_common import IKJacobianType
from newton._src.sim.ik.ik_objectives import IKObjective

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    compute_forward_kinematics_from_transforms,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.base_simulator.simulator_state import StateConversion
from protomotions.utils.c3d_io import events_from_metadata, marker_index, read_metadata
from protomotions.utils.rotations import matrix_to_quaternion
from protomotions.utils.treadmill_overground import (
    apply_virtual_origin_mapping,
    apply_virtual_origin_mapping_with_speed_profile,
    c3d_contact_mask_from_events,
    c3d_speed_profile_from_events,
    estimate_speed_profile_from_stance_points,
)

from retarget_c3d_to_ms_human_lower import (  # noqa: E402
    PELVIS_MARKERS,
    _load_window,
    _point,
    _unit_scale,
    extract_contacts,
    extract_joint_angles,
)


MARKER_BODY_MAP: dict[str, str] = {
    # Pelvis landmarks.
    "RASI": "pelvis",
    "LASI": "pelvis",
    "RPSI": "pelvis",
    "LPSI": "pelvis",
    # Visual3D-modeled hip centers at the proximal femur/hip joint.
    "RIGHT_HIP": "femur_r",
    "LEFT_HIP": "femur_l",
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

FOOT_MARKERS = {"RHEE", "LHEE", "RTOE", "LTOE"}


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


def _model_points_to_v3d(points: np.ndarray) -> np.ndarray:
    result = np.empty_like(points, dtype=np.float32)
    result[..., 0] = points[..., 1]
    result[..., 1] = -points[..., 0]
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


@dataclass(frozen=True)
class MarkerSite:
    site_name: str
    body_name: str
    local_offset: tuple[float, float, float]


@dataclass
class RMSStats:
    overall_mm: float
    per_marker_mm: dict[str, float]
    num_valid_samples: int


@dataclass
class NewtonMarkerIKContext:
    solver: object
    joint_q_out: object
    local_offsets: torch.Tensor
    target_positions: object
    valid: object
    link_indices: object
    offset_sums: object
    offset_counts: object
    marker_pred: object
    device: torch.device


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
    parser.add_argument(
        "--output-fps",
        type=int,
        default=None,
        help="Optional target FPS. Defaults to the native C3D point rate; lower values decimate frames.",
    )
    parser.add_argument("--max-frames", type=int, default=2000, help="Optional frame cap after optional FPS decimation; <=0 disables.")
    parser.add_argument("--calibration-frame", type=int, default=0, help="0-based loaded frame used for virtual marker offsets.")
    parser.add_argument(
        "--offset-calibration",
        choices=["frame", "mean", "median"],
        default="mean",
        help="How to estimate rigid local marker offsets before IK. Whole-window mean/median is much more robust than one frame.",
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
    parser.add_argument(
        "--joint-limit-weight",
        type=float,
        default=0.0,
        help="Newton IK joint-limit objective weight. Set >0 to discourage foot/ankle twists outside MJCF joint ranges.",
    )
    parser.add_argument("--marker-weight", type=float, default=1.0)
    parser.add_argument(
        "--foot-marker-weight",
        type=float,
        default=1.0,
        help="Newton IK weight for heel/toe markers. Higher values prioritize contact-body marker accuracy.",
    )
    parser.add_argument(
        "--marker-offset-source",
        choices=["calibrated", "site", "site-or-calibrated"],
        default="calibrated",
        help=(
            "How to choose each marker's local offset on its body. 'calibrated' learns offsets from the C3D trial; "
            "'site' requires an MJCF <site> for every used marker; 'site-or-calibrated' uses MJCF sites where present "
            "and calibrates the rest."
        ),
    )
    parser.add_argument(
        "--marker-site-prefix",
        default="mocap_",
        help="MJCF marker site prefix. With the default, RHEE first looks for site name 'mocap_RHEE', then 'RHEE'.",
    )
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
    parser.add_argument("--height-offset", type=float, default=0.04)
    parser.add_argument(
        "--joint-angle-plot-dir",
        type=Path,
        default=None,
        help=(
            "Directory for final joint-angle plots. Defaults to a sibling 'plots' directory next to the output proto "
            "directory, or '<output parent>/plots' otherwise."
        ),
    )
    parser.add_argument(
        "--no-joint-angle-plots",
        action="store_true",
        help="Disable writing final joint-angle plots.",
    )
    parser.add_argument("--force-threshold", type=float, default=50.0, help="GRF magnitude threshold for contact labels.")
    parser.add_argument(
        "--treadmill-overground",
        action="store_true",
        help="Apply treadmill-to-overground virtual-origin mapping to marker/segment point trajectories before IK.",
    )
    parser.add_argument(
        "--treadmill-speed-mps",
        type=float,
        default=None,
        help="Positive belt speed in m/s. If omitted with --treadmill-overground, estimate it from stance foot markers.",
    )
    return parser.parse_args()


def _treadmill_position_labels(data) -> list[str]:
    labels = set(PELVIS_MARKERS)
    labels.update(MARKER_BODY_MAP.keys())

    present = []
    for label in sorted(labels):
        try:
            marker_index(data.marker_labels, label)
        except KeyError:
            continue
        present.append(label)
    return present


def _apply_treadmill_overground_if_requested(data, output_fps: int, args) -> None:
    if not args.treadmill_overground:
        return

    scale = _unit_scale(data)
    metadata = read_metadata(args.c3d)
    events = events_from_metadata(metadata)
    source_fps = float(metadata.parameters.get("POINT.RATE", metadata.header.point_rate))
    speed_mps = args.treadmill_speed_mps
    speed_profile = None
    speed_source = "constant"
    valid_stance_samples = 0
    if speed_mps is None:
        speed_profile = c3d_speed_profile_from_events(
            events,
            data.markers.shape[0],
            output_fps,
            source_fps,
            data.first_frame,
            data.start_frame,
        )
        speed_source = "c3d-speed-events" if speed_profile is not None else "stance-estimate"

    if speed_mps is None and speed_profile is None:
        foot_labels = ["RHEE", "RTOE", "LHEE", "LTOE"]
        foot_points = []
        foot_sides = []
        for label in foot_labels:
            try:
                foot_points.append(_point(data, label).astype(np.float32) * scale)
            except KeyError:
                continue
            foot_sides.append(label[0])
        if not foot_points:
            raise ValueError("No foot markers found for treadmill speed estimation; pass --treadmill-speed-mps.")
        foot_points_model = _v3d_points_to_model(np.stack(foot_points, axis=1))
        event_stance_mask = c3d_contact_mask_from_events(
            events,
            data.markers.shape[0],
            output_fps,
            source_fps,
            data.first_frame,
            data.start_frame,
            foot_sides,
        )
        speed_profile, valid_stance_samples = estimate_speed_profile_from_stance_points(
            foot_points_model,
            output_fps,
            event_stance_mask=event_stance_mask,
        )
        if event_stance_mask is not None:
            speed_source = "stance-estimate-with-c3d-contact-events"

    position_labels = _treadmill_position_labels(data)
    if not position_labels:
        raise ValueError("No positional marker/segment labels found for treadmill-overground mapping.")
    marker_indices = [marker_index(data.marker_labels, label) for label in position_labels]
    points_model = _v3d_points_to_model(data.markers[:, marker_indices].astype(np.float32) * scale)
    if speed_profile is None:
        mapped_model, report = apply_virtual_origin_mapping(points_model, output_fps, float(speed_mps))
    else:
        mapped_model, report = apply_virtual_origin_mapping_with_speed_profile(
            points_model,
            output_fps,
            speed_profile,
            estimated=speed_source != "c3d-speed-events",
            valid_stance_samples=valid_stance_samples,
            speed_source=speed_source,
        )
    data.markers[:, marker_indices] = _model_points_to_v3d(mapped_model) / scale
    args._treadmill_mapping = {
        "enabled": True,
        "speed_mps": report.speed_mps,
        "mean_speed_mps": report.speed_mps,
        "speed_source": report.speed_source,
        "speed_change_frames": list(report.speed_change_frames),
        "estimated_speed": report.estimated,
        "valid_stance_samples": valid_stance_samples,
        "total_displacement_m": report.total_displacement_m,
        "mapped_labels": position_labels,
    }
    print(
        "Applied treadmill-to-overground mapping: "
        f"mean_speed={report.speed_mps:.4f} m/s total_displacement={report.total_displacement_m:.3f} m "
        f"labels={len(position_labels)} source={report.speed_source} changes={len(report.speed_change_frames)}"
    )


def _normalized_marker_positions(data) -> np.ndarray:
    """Return markers in meters in the model frame, origin-normalized in XY."""
    scale = _unit_scale(data)
    markers = _v3d_points_to_model(data.markers.astype(np.float32) * scale)
    pelvis = _v3d_points_to_model(np.stack([_point(data, name) for name in PELVIS_MARKERS], axis=0).astype(np.float32) * scale)
    root = np.nanmean(pelvis, axis=0)
    markers[:, :, :2] -= root[0:1, None, :2]
    return markers


def _asset_xml_path(cfg) -> Path:
    return Path(cfg.asset.asset_root) / cfg.asset.asset_file_name


def _parse_site_pos(site: ET.Element) -> tuple[float, float, float]:
    pos_text = site.get("pos", "0 0 0")
    values = [float(value) for value in pos_text.split()]
    if len(values) != 3:
        raise ValueError(f"MJCF site '{site.get('name', '<unnamed>')}' has invalid pos='{pos_text}'.")
    return values[0], values[1], values[2]


def _marker_site_name_candidates(marker_label: str, site_prefix: str) -> list[str]:
    candidates: list[str] = []
    if site_prefix:
        candidates.append(f"{site_prefix}{marker_label}")
    candidates.append(marker_label)
    return candidates


def _load_marker_sites(cfg, site_prefix: str) -> dict[str, MarkerSite]:
    xml_path = _asset_xml_path(cfg)
    root = ET.parse(xml_path).getroot()
    candidate_to_label = {
        candidate: label
        for label in MARKER_BODY_MAP
        for candidate in _marker_site_name_candidates(label, site_prefix)
    }
    marker_sites: dict[str, MarkerSite] = {}
    for body in root.iter("body"):
        body_name = body.get("name")
        if body_name is None:
            continue
        for site in body.findall("site"):
            site_name = site.get("name")
            if site_name not in candidate_to_label:
                continue
            label = candidate_to_label[site_name]
            if label in marker_sites:
                previous = marker_sites[label]
                raise ValueError(
                    f"Marker {label} has duplicate MJCF marker sites: "
                    f"{previous.site_name} on {previous.body_name} and {site_name} on {body_name}."
                )
            marker_sites[label] = MarkerSite(
                site_name=site_name,
                body_name=body_name,
                local_offset=_parse_site_pos(site),
            )
    return marker_sites


def _build_marker_set(
    data,
    body_names: list[str],
    device: torch.device,
    marker_sites: dict[str, MarkerSite] | None = None,
    prefer_site_bodies: bool = False,
) -> MarkerSet:
    body_index = {name: idx for idx, name in enumerate(body_names)}
    marker_positions = _normalized_marker_positions(data)

    labels: list[str] = []
    bodies: list[int] = []
    targets: list[np.ndarray] = []
    skipped: list[str] = []
    for marker_label, mapped_body_name in MARKER_BODY_MAP.items():
        marker_site = marker_sites.get(marker_label) if marker_sites is not None else None
        body_name = marker_site.body_name if prefer_site_bodies and marker_site is not None else mapped_body_name
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


def _apply_marker_site_offsets(
    marker_set: MarkerSet,
    calibrated_offsets: torch.Tensor,
    marker_sites: dict[str, MarkerSite],
    offset_source: str,
) -> torch.Tensor:
    if offset_source == "calibrated":
        return calibrated_offsets

    local_offsets = calibrated_offsets.clone()
    missing: list[str] = []
    used: list[str] = []
    for marker_idx, label in enumerate(marker_set.labels):
        marker_site = marker_sites.get(label)
        if marker_site is None:
            missing.append(label)
            continue
        local_offsets[marker_idx] = torch.tensor(
            marker_site.local_offset,
            device=calibrated_offsets.device,
            dtype=calibrated_offsets.dtype,
        )
        used.append(f"{label}:{marker_site.site_name}->{marker_site.body_name}")

    if offset_source == "site" and missing:
        raise ValueError(
            "--marker-offset-source=site requires MJCF <site> definitions for every used marker. Missing: "
            + ", ".join(missing)
        )

    print(
        f"Marker offset source: {offset_source}; "
        f"using {len(used)} MJCF marker sites"
        + (f" and calibrating {len(missing)} missing markers" if missing else "")
    )
    if used:
        print(f"MJCF marker sites: {', '.join(used)}")
    if missing and offset_source == "site-or-calibrated":
        print(f"Calibrated marker offsets: {', '.join(missing)}")
    return local_offsets


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


def _quat_rotate_inverse_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz = -quat[..., :3]
    quat_w = quat[..., 3:4]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)


def _newton_context_body_transforms(args, marker_set: MarkerSet) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not hasattr(args, "_newton_marker_context"):
        return None
    context: NewtonMarkerIKContext = args._newton_marker_context
    body_q = wp.to_torch(context.solver._impl.body_q)
    body_indices = marker_set.body_indices.to(device=body_q.device, dtype=torch.long)
    body_pos = body_q[:, body_indices, :3]
    body_quat_xyzw = body_q[:, body_indices, 3:7]
    return body_pos, body_quat_xyzw


def _calibrate_local_offsets_from_newton_context(args, marker_set: MarkerSet, calibration_frame: int, mode: str) -> torch.Tensor | None:
    if not hasattr(args, "_newton_marker_context"):
        return None
    context: NewtonMarkerIKContext = args._newton_marker_context
    body_q = context.solver._impl.body_q
    calibration_frame = max(0, min(calibration_frame, body_q.shape[0] - 1))

    if mode == "frame":
        valid = marker_set.valid[calibration_frame]
        if not bool(valid.all()):
            missing = [label for label, is_valid in zip(marker_set.labels, valid.tolist()) if not is_valid]
            raise ValueError(
                "Calibration frame has invalid marker samples for: "
                + ", ".join(missing)
                + ". Choose another --calibration-frame."
            )
        wp.launch(
            _set_marker_local_offsets_from_frame,
            dim=len(marker_set.labels),
            inputs=[body_q, context.target_positions, context.link_indices, calibration_frame],
            outputs=[wp.from_torch(context.local_offsets, dtype=wp.vec3)],
            device=context.solver.device,
        )
        return context.local_offsets

    if mode == "mean":
        wp.launch(
            _zero_marker_offset_accumulators,
            dim=len(marker_set.labels),
            outputs=[context.offset_sums, context.offset_counts],
            device=context.solver.device,
        )
        wp.launch(
            _accumulate_marker_local_offsets,
            dim=[body_q.shape[0], len(marker_set.labels)],
            inputs=[body_q, context.target_positions, context.valid, context.link_indices],
            outputs=[context.offset_sums, context.offset_counts],
            device=context.solver.device,
        )
        wp.launch(
            _normalize_marker_local_offsets,
            dim=len(marker_set.labels),
            inputs=[context.offset_sums, context.offset_counts],
            outputs=[wp.from_torch(context.local_offsets, dtype=wp.vec3)],
            device=context.solver.device,
        )
        return context.local_offsets

    if mode == "median":
        transforms = _newton_context_body_transforms(args, marker_set)
        if transforms is None:
            return None
        body_pos, body_quat_xyzw = transforms
        local_samples = _quat_rotate_inverse_xyzw(body_quat_xyzw, marker_set.targets - body_pos)
        offsets = torch.zeros_like(local_samples[0])
        for marker_idx in range(len(marker_set.labels)):
            samples = local_samples[marker_set.valid[:, marker_idx], marker_idx]
            if samples.numel() == 0:
                raise ValueError(f"No finite samples available while calibrating marker {marker_set.labels[marker_idx]}.")
            offsets[marker_idx] = samples.median(dim=0).values
        return offsets.detach().clone()

    raise ValueError(f"Unsupported offset calibration mode: {mode}")


def _predict_markers_from_newton_context(args, marker_set: MarkerSet, local_offsets: torch.Tensor) -> torch.Tensor | None:
    if not hasattr(args, "_newton_marker_context"):
        return None
    context: NewtonMarkerIKContext = args._newton_marker_context
    if local_offsets.data_ptr() != context.local_offsets.data_ptr():
        context.local_offsets.copy_(local_offsets.detach().to(device=context.local_offsets.device, dtype=torch.float32))
    wp.launch(
        _predict_markers_from_body_transforms,
        dim=[context.solver._impl.body_q.shape[0], len(marker_set.labels)],
        inputs=[
            context.solver._impl.body_q,
            context.link_indices,
            wp.from_torch(context.local_offsets, dtype=wp.vec3),
        ],
        outputs=[context.marker_pred],
        device=context.solver.device,
    )
    return wp.to_torch(context.marker_pred)


def _compute_rms_stats_from_predictions(pred: torch.Tensor, marker_set: MarkerSet) -> RMSStats:
    with torch.no_grad():
        err_sq = ((pred - marker_set.targets) ** 2).sum(dim=-1)
        valid = marker_set.valid.to(device=err_sq.device)
        err_sq = torch.where(valid, err_sq, torch.zeros_like(err_sq))
        per_marker_sq = err_sq.sum(dim=0)
        per_marker_count = valid.sum(dim=0)
        per_marker_mm = {
            label: float(torch.sqrt(per_marker_sq[idx] / torch.clamp(per_marker_count[idx], min=1)).item() * 1000.0)
            for idx, label in enumerate(marker_set.labels)
        }
        overall_mm = float(torch.sqrt(err_sq.sum() / torch.clamp(valid.sum(), min=1)).item() * 1000.0)
        return RMSStats(overall_mm=overall_mm, per_marker_mm=per_marker_mm, num_valid_samples=int(valid.sum().item()))


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


@wp.kernel
def _batched_marker_pos_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    target_pos: wp.array2d(dtype=wp.vec3),
    link_indices: wp.array1d(dtype=wp.int32),
    link_offsets: wp.array1d(dtype=wp.vec3),
    weights: wp.array1d(dtype=wp.float32),
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row, marker_idx = wp.tid()
    base = problem_idx_map[row]
    body_tf = body_q[row, link_indices[marker_idx]]
    ee_pos = wp.transform_point(body_tf, link_offsets[marker_idx])
    error = target_pos[base, marker_idx] - ee_pos
    weight = weights[marker_idx]
    residual_idx = start_idx + marker_idx * 3
    residuals[row, residual_idx + 0] = weight * error[0]
    residuals[row, residual_idx + 1] = weight * error[1]
    residuals[row, residual_idx + 2] = weight * error[2]


@wp.kernel
def _batched_marker_pos_jac_analytic(
    link_indices: wp.array1d(dtype=wp.int32),
    link_offsets: wp.array1d(dtype=wp.vec3),
    weights: wp.array1d(dtype=wp.float32),
    affects_dof: wp.array2d(dtype=wp.uint8),
    body_q: wp.array2d(dtype=wp.transform),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    start_idx: int,
    n_dofs: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx, marker_idx, dof_idx = wp.tid()
    if affects_dof[marker_idx, dof_idx] == wp.uint8(0):
        return

    body_tf = body_q[problem_idx, link_indices[marker_idx]]
    rot_w = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    pos_w = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    ee_pos_world = pos_w + wp.quat_rotate(rot_w, link_offsets[marker_idx])

    S = joint_S_s[problem_idx, dof_idx]
    v_orig = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])
    v_ee = v_orig + wp.cross(omega, ee_pos_world)

    weight = weights[marker_idx]
    residual_idx = start_idx + marker_idx * 3
    jacobian[problem_idx, residual_idx + 0, dof_idx] = -weight * v_ee[0]
    jacobian[problem_idx, residual_idx + 1, dof_idx] = -weight * v_ee[1]
    jacobian[problem_idx, residual_idx + 2, dof_idx] = -weight * v_ee[2]


@wp.kernel
def _zero_marker_offset_accumulators(
    offset_sums: wp.array2d(dtype=wp.float32),
    offset_counts: wp.array1d(dtype=wp.float32),
):
    marker_idx = wp.tid()
    offset_sums[marker_idx, 0] = 0.0
    offset_sums[marker_idx, 1] = 0.0
    offset_sums[marker_idx, 2] = 0.0
    offset_counts[marker_idx] = 0.0


@wp.kernel
def _accumulate_marker_local_offsets(
    body_q: wp.array2d(dtype=wp.transform),
    target_pos: wp.array2d(dtype=wp.vec3),
    valid: wp.array2d(dtype=wp.bool),
    link_indices: wp.array1d(dtype=wp.int32),
    offset_sums: wp.array2d(dtype=wp.float32),
    offset_counts: wp.array1d(dtype=wp.float32),
):
    frame_idx, marker_idx = wp.tid()
    if not valid[frame_idx, marker_idx]:
        return

    body_tf = body_q[frame_idx, link_indices[marker_idx]]
    body_pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    local_offset = wp.quat_rotate_inv(body_rot, target_pos[frame_idx, marker_idx] - body_pos)

    wp.atomic_add(offset_sums, marker_idx, 0, local_offset[0])
    wp.atomic_add(offset_sums, marker_idx, 1, local_offset[1])
    wp.atomic_add(offset_sums, marker_idx, 2, local_offset[2])
    wp.atomic_add(offset_counts, marker_idx, 1.0)


@wp.kernel
def _normalize_marker_local_offsets(
    offset_sums: wp.array2d(dtype=wp.float32),
    offset_counts: wp.array1d(dtype=wp.float32),
    local_offsets: wp.array1d(dtype=wp.vec3),
):
    marker_idx = wp.tid()
    count = wp.max(offset_counts[marker_idx], 1.0)
    local_offsets[marker_idx] = wp.vec3(
        offset_sums[marker_idx, 0] / count,
        offset_sums[marker_idx, 1] / count,
        offset_sums[marker_idx, 2] / count,
    )


@wp.kernel
def _set_marker_local_offsets_from_frame(
    body_q: wp.array2d(dtype=wp.transform),
    target_pos: wp.array2d(dtype=wp.vec3),
    link_indices: wp.array1d(dtype=wp.int32),
    calibration_frame: int,
    local_offsets: wp.array1d(dtype=wp.vec3),
):
    marker_idx = wp.tid()
    body_tf = body_q[calibration_frame, link_indices[marker_idx]]
    body_pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    local_offsets[marker_idx] = wp.quat_rotate_inv(body_rot, target_pos[calibration_frame, marker_idx] - body_pos)


@wp.kernel
def _predict_markers_from_body_transforms(
    body_q: wp.array2d(dtype=wp.transform),
    link_indices: wp.array1d(dtype=wp.int32),
    local_offsets: wp.array1d(dtype=wp.vec3),
    marker_pred: wp.array2d(dtype=wp.vec3),
):
    frame_idx, marker_idx = wp.tid()
    body_tf = body_q[frame_idx, link_indices[marker_idx]]
    body_pos = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    marker_pred[frame_idx, marker_idx] = body_pos + wp.quat_rotate(body_rot, local_offsets[marker_idx])


class BatchedMarkerPositionObjective(IKObjective):
    """One Warp objective for all marker position residuals.

    Newton's built-in position objective launches one residual and one analytic
    Jacobian kernel per marker.  This objective fuses all marker work into one
    2D/3D launch so frames, markers, and DOFs are exposed to Warp together.
    """

    def __init__(self, link_indices, link_offsets, target_positions, weights):
        super().__init__()
        self.link_indices = link_indices
        self.link_offsets = link_offsets
        self.target_positions = target_positions
        self.weights = weights
        self.num_markers = int(link_indices.shape[0])
        self.affects_dof = None

    def residual_dim(self):
        return self.num_markers * 3

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode != IKJacobianType.ANALYTIC:
            return

        joint_qd_start_np = model.joint_qd_start.numpy()
        dof_to_joint_np = np.empty(joint_qd_start_np[-1], dtype=np.int32)
        for joint_idx in range(len(joint_qd_start_np) - 1):
            dof_to_joint_np[joint_qd_start_np[joint_idx] : joint_qd_start_np[joint_idx + 1]] = joint_idx

        joint_child_np = model.joint_child.numpy()
        body_to_joint_np = np.full(model.body_count, -1, np.int32)
        for joint_idx in range(model.joint_count):
            child = joint_child_np[joint_idx]
            if child != -1:
                body_to_joint_np[child] = joint_idx

        joint_q_start_np = model.joint_q_start.numpy()
        joint_parent_np = model.joint_parent.numpy()
        link_indices_np = self.link_indices.numpy()
        affects_dof_np = np.zeros((self.num_markers, model.joint_dof_count), dtype=np.uint8)
        for marker_idx, link_index in enumerate(link_indices_np):
            ancestors = np.zeros(len(joint_q_start_np) - 1, dtype=bool)
            body = int(link_index)
            while body != -1:
                joint_idx = body_to_joint_np[body]
                if joint_idx != -1:
                    ancestors[joint_idx] = True
                body = joint_parent_np[joint_idx] if joint_idx != -1 else -1
            affects_dof_np[marker_idx] = ancestors[dof_to_joint_np]
        self.affects_dof = wp.array(affects_dof_np, dtype=wp.uint8, device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _batched_marker_pos_residuals,
            dim=[body_q.shape[0], self.num_markers],
            inputs=[
                body_q,
                self.target_positions,
                self.link_indices,
                self.link_offsets,
                self.weights,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        raise NotImplementedError("BatchedMarkerPositionObjective supports analytic Jacobians only.")

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(
            _batched_marker_pos_jac_analytic,
            dim=[body_q.shape[0], self.num_markers, model.joint_dof_count],
            inputs=[
                self.link_indices,
                self.link_offsets,
                self.weights,
                self.affects_dof,
                body_q,
                joint_S_s,
                start_idx,
                model.joint_dof_count,
            ],
            outputs=[jacobian],
            device=self.device,
        )


def _can_use_batched_marker_objective(args) -> bool:
    return args.newton_jacobian == "analytic"


def _create_newton_marker_ik_context(args, cfg, qpos_init: torch.Tensor, marker_set: MarkerSet, local_offsets: torch.Tensor) -> NewtonMarkerIKContext:
    import newton.ik as ik

    device = torch.device("cpu") if args.device == "cpu" else torch.device(args.device)
    model = _build_newton_model(cfg, device)
    if model.joint_coord_count != qpos_init.shape[1]:
        raise ValueError(f"Newton joint_coord_count={model.joint_coord_count}, expected qpos width {qpos_init.shape[1]}")

    local_offsets_buffer = local_offsets.detach().to(device=qpos_init.device, dtype=torch.float32).contiguous().clone()
    link_offsets_wp = wp.from_torch(local_offsets_buffer, dtype=wp.vec3)
    targets_wp = wp.from_torch(marker_set.targets.contiguous(), dtype=wp.vec3)
    valid_wp = wp.from_torch(marker_set.valid.contiguous(), dtype=wp.bool)
    link_indices_torch = marker_set.body_indices.to(device=qpos_init.device, dtype=torch.int32).contiguous()
    link_indices_wp = wp.from_torch(link_indices_torch, dtype=wp.int32)
    weights_torch = torch.tensor(
        [float(args.foot_marker_weight if label in FOOT_MARKERS else args.marker_weight) for label in marker_set.labels],
        device=qpos_init.device,
        dtype=torch.float32,
    )
    weights_wp = wp.from_torch(weights_torch, dtype=wp.float32)
    objectives = [BatchedMarkerPositionObjective(link_indices_wp, link_offsets_wp, targets_wp, weights_wp)]
    if args.joint_limit_weight > 0.0:
        objectives.append(
            ik.IKObjectiveJointLimit(
                model.joint_limit_lower,
                model.joint_limit_upper,
                weight=float(args.joint_limit_weight),
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
    joint_q_out = wp.empty((qpos_init.shape[0], qpos_init.shape[1]), dtype=wp.float32, device=str(device))
    offset_sums = wp.zeros((len(marker_set.labels), 3), dtype=wp.float32, device=str(device))
    offset_counts = wp.zeros(len(marker_set.labels), dtype=wp.float32, device=str(device))
    marker_pred = wp.empty((qpos_init.shape[0], len(marker_set.labels)), dtype=wp.vec3, device=str(device))
    return NewtonMarkerIKContext(
        solver=solver,
        joint_q_out=joint_q_out,
        local_offsets=local_offsets_buffer,
        target_positions=targets_wp,
        valid=valid_wp,
        link_indices=link_indices_wp,
        offset_sums=offset_sums,
        offset_counts=offset_counts,
        marker_pred=marker_pred,
        device=device,
    )


def _optimize_qpos_newton(
    args,
    cfg,
    qpos_init: torch.Tensor,
    marker_set: MarkerSet,
    local_offsets: torch.Tensor,
    use_context_output_as_input: bool = False,
    return_qpos: bool = True,
) -> torch.Tensor | None:
    import newton.ik as ik

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if _can_use_batched_marker_objective(args):
        if not hasattr(args, "_newton_marker_context"):
            args._newton_marker_context = _create_newton_marker_ik_context(args, cfg, qpos_init, marker_set, local_offsets)
            args._newton_objective_mode = "batched_marker"
        context: NewtonMarkerIKContext = args._newton_marker_context
        if local_offsets.data_ptr() != context.local_offsets.data_ptr():
            context.local_offsets.copy_(local_offsets.detach().to(device=qpos_init.device, dtype=torch.float32))
        if use_context_output_as_input:
            joint_q = context.joint_q_out
        else:
            joint_q_torch = _qpos_wxyz_to_newton_xyzw(qpos_init).contiguous()
            joint_q = wp.from_torch(joint_q_torch, dtype=wp.float32, requires_grad=False)

        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        start_time = time.perf_counter()
        context.solver.step(joint_q, context.joint_q_out, iterations=args.newton_iterations, step_size=args.newton_step_size)
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        elapsed = time.perf_counter() - start_time
        sps = qpos_init.shape[0] * args.newton_iterations / max(elapsed, 1e-12)
        fps_equiv = qpos_init.shape[0] / max(elapsed, 1e-12)
        print(
            f"Newton IK timing: elapsed={elapsed:.3f}s frames={qpos_init.shape[0]} "
            f"iterations={args.newton_iterations} sps={sps:,.0f} frame_sps={fps_equiv:,.0f} mode=batched_marker"
        )
        timing = {
            "elapsed_seconds": elapsed,
            "sample_iterations_per_second": sps,
            "frames_per_second": fps_equiv,
            "timed_iterations": args.newton_iterations,
            "objective_mode": "batched_marker",
        }
        args._timing = timing
        if not hasattr(args, "_timing_history"):
            args._timing_history = []
        args._timing_history.append(timing)

        if not return_qpos:
            return None

        joint_q_result = wp.to_torch(context.joint_q_out).detach().clone().to(device=qpos_init.device)
        qpos = _newton_xyzw_to_qpos_wxyz(joint_q_result)
        qpos[:, 3:7] /= torch.linalg.norm(qpos[:, 3:7], dim=-1, keepdim=True).clamp_min(1e-8)
        return qpos

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
    for marker_idx, label in enumerate(marker_set.labels):
        targets = wp.array(target_cpu[:, marker_idx], dtype=wp.vec3, device=str(device))
        offset = wp.vec3(*offsets_cpu[marker_idx].tolist())
        weight = float(args.foot_marker_weight if label in FOOT_MARKERS else args.marker_weight)
        objectives.append(
            ik.IKObjectivePosition(
                link_index=int(body_indices_cpu[marker_idx]),
                link_offset=offset,
                target_positions=targets,
                weight=weight,
            )
        )
    if args.joint_limit_weight > 0.0:
        objectives.append(
            ik.IKObjectiveJointLimit(
                model.joint_limit_lower,
                model.joint_limit_upper,
                weight=float(args.joint_limit_weight),
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
        "newton_objective_mode": getattr(args, "_newton_objective_mode", "per_marker"),
        "newton_iterations": args.newton_iterations,
        "newton_optimizer": args.newton_optimizer,
        "newton_jacobian": args.newton_jacobian,
        "joint_limit_weight": args.joint_limit_weight,
        "chunk_size": args.chunk_size,
        "offset_calibration": args.offset_calibration,
        "offset_refine_passes": args.offset_refine_passes,
        "root_orientation": args.root_orientation,
        "marker_weight": args.marker_weight,
        "foot_marker_weight": args.foot_marker_weight,
    }
    if hasattr(args, "_treadmill_mapping"):
        report["treadmill_mapping"] = args._treadmill_mapping
    if hasattr(args, "_timing"):
        report["timing"] = args._timing
    if hasattr(args, "_timing_history"):
        report["timing_history"] = args._timing_history
        report["total_ik_elapsed_seconds"] = sum(item["elapsed_seconds"] for item in args._timing_history)
    if hasattr(args, "_joint_angle_plot_paths"):
        report["joint_angle_plots"] = [str(path) for path in args._joint_angle_plot_paths]
    report["marker_offset_source"] = args.marker_offset_source
    report["marker_site_prefix"] = args.marker_site_prefix
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def _default_joint_angle_plot_dir(output_path: Path) -> Path:
    if output_path.parent.name == "proto":
        return output_path.parent.parent / "plots"
    return output_path.parent / "plots"


def _write_joint_angle_plots(
    output_dir: Path,
    output_stem: str,
    dof_names: list[str],
    initial_dof_pos: torch.Tensor,
    solved_dof_pos: torch.Tensor,
    fps: int,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required for joint-angle plots. Install with `pip install matplotlib`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    initial_deg = torch.rad2deg(initial_dof_pos.detach().cpu()).numpy()
    solved_deg = torch.rad2deg(solved_dof_pos.detach().cpu()).numpy()
    time_s = np.arange(solved_deg.shape[0], dtype=np.float32) / float(fps)
    paths: list[Path] = []

    overview_path = output_dir / f"{output_stem}_joint_angles_overview.png"
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    for dof_idx, name in enumerate(dof_names):
        ax.plot(time_s, solved_deg[:, dof_idx], label=name, linewidth=0.8, alpha=0.85)
    ax.set_title("Solved joint angles")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="x-small", ncols=1)
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)
    paths.append(overview_path)

    comparison_path = output_dir / f"{output_stem}_joint_angles.pdf"
    with PdfPages(comparison_path) as pdf:
        for dof_idx, name in enumerate(dof_names):
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
            axes[0].plot(time_s, initial_deg[:, dof_idx], label="warm start", color="tab:orange", linewidth=1.0)
            axes[0].set_title("Warm-start angle")
            axes[0].set_ylabel("deg")
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(loc="upper right", fontsize="small")

            axes[1].plot(time_s, solved_deg[:, dof_idx], label="solved", color="tab:blue", linewidth=1.0)
            axes[1].set_title("Solved angle")
            axes[1].set_ylabel("deg")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(loc="upper right", fontsize="small")

            delta_deg = solved_deg[:, dof_idx] - initial_deg[:, dof_idx]
            axes[2].plot(time_s, delta_deg, label="solved - warm start", color="tab:green", linewidth=1.0)
            axes[2].set_title("IK correction")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("deg")
            axes[2].grid(True, alpha=0.25)
            axes[2].legend(loc="upper right", fontsize="small")

            fig.suptitle(f"Joint angle: {name} ({output_stem})", fontsize=13)
            pdf.savefig(fig)
            plt.close(fig)
    paths.append(comparison_path)

    return paths


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
    _apply_treadmill_overground_if_requested(data, output_fps, args)
    cfg = robot_config(args.robot_name)
    ki = cfg.kinematic_info

    qpos_init = _initial_qpos(data, ki, device, args.root_orientation, args.warmstart_angle_mode, args.warmstart_clamp_limits)
    marker_sites = _load_marker_sites(cfg, args.marker_site_prefix) if args.marker_offset_source != "calibrated" else {}
    marker_set = _build_marker_set(
        data,
        ki.body_names,
        device,
        marker_sites=marker_sites,
        prefer_site_bodies=args.marker_offset_source != "calibrated",
    )
    calibrated_offsets = _calibrate_local_offsets(ki, qpos_init, marker_set, args.calibration_frame, args.offset_calibration)
    local_offsets = _apply_marker_site_offsets(marker_set, calibrated_offsets, marker_sites, args.marker_offset_source)
    print(
        f"Optimizing {qpos_init.shape[0]} frames at {output_fps} fps on {device} with {args.backend}; "
        f"nq={qpos_init.shape[1]} bodies={ki.num_bodies} markers={len(marker_set.labels)}"
    )

    num_refine_passes = max(0, args.offset_refine_passes)
    if args.marker_offset_source == "site" and num_refine_passes > 0:
        print("Ignoring --offset-refine-passes because --marker-offset-source=site uses fixed MJCF marker offsets.")
        num_refine_passes = 0
    use_cached_context = _can_use_batched_marker_objective(args)
    qpos = _optimize_qpos_newton(
        args,
        cfg,
        qpos_init,
        marker_set,
        local_offsets,
        return_qpos=not use_cached_context or num_refine_passes == 0,
    )
    for refine_pass in range(num_refine_passes):
        context_offsets = _calibrate_local_offsets_from_newton_context(args, marker_set, args.calibration_frame, args.offset_calibration)
        if context_offsets is None:
            if qpos is None:
                raise RuntimeError("qpos is required for fallback offset calibration.")
            calibrated_offsets = _calibrate_local_offsets(ki, qpos, marker_set, args.calibration_frame, args.offset_calibration)
            use_context_output_as_input = False
        else:
            calibrated_offsets = context_offsets
            use_context_output_as_input = True
        local_offsets = _apply_marker_site_offsets(marker_set, calibrated_offsets, marker_sites, args.marker_offset_source)
        print(f"Refining marker offsets from solved qpos: pass {refine_pass + 1}/{args.offset_refine_passes}")
        qpos = _optimize_qpos_newton(
            args,
            cfg,
            qpos if qpos is not None else qpos_init,
            marker_set,
            local_offsets,
            use_context_output_as_input=use_context_output_as_input,
            return_qpos=not use_context_output_as_input or refine_pass == num_refine_passes - 1,
        )
    marker_pred_for_rerun = _predict_markers_from_newton_context(args, marker_set, local_offsets)
    if marker_pred_for_rerun is None:
        stats = _compute_rms_stats(ki, qpos, marker_set, local_offsets, max(1, args.chunk_size))
        marker_pred_for_rerun = _predict_markers(ki, qpos, marker_set, local_offsets).detach()
    else:
        marker_pred_for_rerun = marker_pred_for_rerun.detach()
        stats = _compute_rms_stats_from_predictions(marker_pred_for_rerun, marker_set)
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
    height_translation = motion.fix_height_per_frame(height_offset=args.height_offset)
    marker_targets_for_rerun = marker_set.targets + height_translation[:, None]
    marker_pred_for_rerun = marker_pred_for_rerun + height_translation[:, None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    motion_dict = motion.to_dict()
    motion_dict["marker_targets"] = marker_targets_for_rerun.detach().cpu()
    motion_dict["marker_reconstructed"] = marker_pred_for_rerun.detach().cpu()
    motion_dict["marker_labels"] = marker_set.labels
    if hasattr(args, "_treadmill_mapping"):
        motion_dict["treadmill_mapping"] = args._treadmill_mapping
    torch.save(motion_dict, args.output)
    if not args.no_joint_angle_plots:
        plot_dir = args.joint_angle_plot_dir or _default_joint_angle_plot_dir(args.output)
        args._joint_angle_plot_paths = _write_joint_angle_plots(
            plot_dir,
            args.output.stem,
            ki.dof_names,
            qpos_init[:, 7:],
            dof_pos,
            output_fps,
        )
        print("Wrote joint-angle plots:")
        for plot_path in args._joint_angle_plot_paths:
            print(f"  {plot_path}")
    report_path = args.report or args.output.with_suffix(".rms.json")
    _write_report(report_path, args, stats, marker_set)
    print(f"Wrote motion: {args.output}")
    print(f"Wrote RMS report: {report_path}")
    if args.rerun_output is not None:
        _write_rerun_recording(
            args.rerun_output,
            motion,
            ki,
            marker_set,
            marker_pred_for_rerun,
            marker_targets_for_rerun,
        )


if __name__ == "__main__":
    main()
