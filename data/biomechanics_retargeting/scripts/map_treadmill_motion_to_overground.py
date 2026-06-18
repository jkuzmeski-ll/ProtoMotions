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
"""Convert an in-place treadmill ProtoMotions motion to overground travel.

This applies the same virtual-origin mapping used for C3D marker IK: add the
integrated treadmill belt displacement along the model forward axis.  Rotations
and joint coordinates are unchanged; rigid-body positions and linear velocities
are translated into an overground trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protomotions.robot_configs.factory import robot_config
from protomotions.utils.c3d_io import events_from_metadata, read_metadata
from protomotions.utils.treadmill_overground import (
    c3d_speed_profile_from_events,
    displacement_from_speed,
    displacement_from_speed_profile,
    normalized_direction,
    speed_change_frames_from_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=Path, help="Input treadmill/in-place .motion file.")
    parser.add_argument("--output", type=Path, required=True, help="Output overground .motion file.")
    parser.add_argument("--speed-mps", type=float, default=None, help="Positive treadmill belt speed in m/s.")
    parser.add_argument(
        "--c3d-events",
        type=Path,
        default=None,
        help="Optional source C3D whose numeric events (10=1.0 m/s) define speed changes.",
    )
    parser.add_argument(
        "--motion-start-frame",
        type=int,
        default=1,
        help="1-based C3D source frame aligned with motion frame 0 when --c3d-events is used.",
    )
    parser.add_argument(
        "--robot-name",
        default="ms_human_lower_s003",
        help="Robot config used when estimating speed from contacts.",
    )
    parser.add_argument(
        "--direction",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Forward walking direction in ProtoMotions COMMON/model coordinates.",
    )
    parser.add_argument(
        "--estimate-from-contacts",
        action="store_true",
        help="Estimate belt speed from contacted foot-body forward velocities when --speed-mps is omitted.",
    )
    return parser.parse_args()


def _estimate_speed_from_contacts(motion: dict, robot_name: str, direction: torch.Tensor) -> tuple[float, int]:
    if "rigid_body_vel" not in motion or "rigid_body_contacts" not in motion:
        raise ValueError("Motion must contain rigid_body_vel and rigid_body_contacts to estimate treadmill speed.")
    cfg = robot_config(robot_name)
    body_index = {name: idx for idx, name in enumerate(cfg.kinematic_info.body_names)}
    foot_indices = [
        body_index[name]
        for name in ("calcn_r", "toes_r", "calcn_l", "toes_l")
        if name in body_index
    ]
    if not foot_indices:
        raise ValueError(f"No known foot bodies found in robot config {robot_name!r}.")

    vel = motion["rigid_body_vel"].detach().to(device="cpu", dtype=torch.float32)[:, foot_indices]
    contacts = motion["rigid_body_contacts"].detach().to(device="cpu")[:, foot_indices].bool()
    forward_vel = torch.sum(vel * direction.view(1, 1, 3), dim=-1)
    samples = -forward_vel[contacts & (-forward_vel > 0.05)]
    if samples.numel() == 0:
        raise ValueError("Could not estimate speed from contacts; pass --speed-mps explicitly.")
    return float(samples.median().item()), int(samples.numel())


def _speed_profile_from_c3d_events(c3d_path: Path, frame_count: int, fps: float, start_frame: int) -> torch.Tensor | None:
    metadata = read_metadata(c3d_path)
    events = events_from_metadata(metadata)
    source_fps = float(metadata.parameters.get("POINT.RATE", metadata.header.point_rate))
    speed_profile = c3d_speed_profile_from_events(
        events,
        frame_count,
        fps,
        source_fps,
        metadata.header.first_frame,
        start_frame,
    )
    if speed_profile is None:
        return None
    return torch.from_numpy(speed_profile).to(dtype=torch.float32)


def main() -> None:
    args = parse_args()
    motion = torch.load(args.input, map_location="cpu", weights_only=False)
    fps = float(motion["fps"])
    rigid_body_pos = motion["rigid_body_pos"].detach().to(device="cpu", dtype=torch.float32)
    frame_count = rigid_body_pos.shape[0]
    direction_np = normalized_direction(tuple(args.direction))
    direction = torch.tensor(direction_np, dtype=torch.float32)

    speed_mps = args.speed_mps
    speed_profile = None
    valid_samples = 0
    estimated = False
    speed_source = "constant"
    if speed_mps is None:
        if args.c3d_events is not None:
            speed_profile = _speed_profile_from_c3d_events(args.c3d_events, frame_count, fps, args.motion_start_frame)
            speed_source = "c3d-speed-events"
        if speed_profile is None:
            if not args.estimate_from_contacts:
                raise ValueError("Pass --speed-mps, --c3d-events, or enable --estimate-from-contacts.")
            speed_mps, valid_samples = _estimate_speed_from_contacts(motion, args.robot_name, direction)
            estimated = True
            speed_source = "contact-estimate"

    if speed_profile is None:
        displacement = torch.from_numpy(displacement_from_speed(frame_count, fps, speed_mps)).to(dtype=torch.float32)
        speed_changes = [(0, float(speed_mps))]
        mean_speed = float(speed_mps)
    else:
        displacement = torch.from_numpy(displacement_from_speed_profile(speed_profile.numpy(), fps)).to(dtype=torch.float32)
        speed_changes = list(speed_change_frames_from_profile(speed_profile.numpy()))
        mean_speed = float(speed_profile.mean().item())
    motion["rigid_body_pos"] = rigid_body_pos + displacement.view(-1, 1, 1) * direction.view(1, 1, 3)

    if "rigid_body_vel" in motion:
        rigid_body_vel = motion["rigid_body_vel"].detach().to(device="cpu", dtype=torch.float32)
        mapped_velocity = torch.empty_like(displacement)
        if frame_count <= 1:
            mapped_velocity.fill_(mean_speed)
        else:
            mapped_velocity[:-1] = torch.diff(displacement) * fps
            mapped_velocity[-1] = mapped_velocity[-2]
        motion["rigid_body_vel"] = rigid_body_vel + mapped_velocity.view(-1, 1, 1) * direction.view(1, 1, 3)

    motion["treadmill_mapping"] = {
        "enabled": True,
        "speed_mps": mean_speed,
        "mean_speed_mps": mean_speed,
        "speed_source": speed_source,
        "speed_change_frames": speed_changes,
        "estimated_speed": estimated,
        "valid_contact_samples": valid_samples,
        "total_displacement_m": float(displacement[-1]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion, args.output)
    print(f"Wrote overground motion: {args.output}")
    print(
        f"  mean_speed={mean_speed:.4f} m/s total_displacement={float(displacement[-1]):.3f} m "
        f"frames={frame_count} fps={fps:g} source={speed_source} changes={len(speed_changes)} contact_samples={valid_samples}"
    )


if __name__ == "__main__":
    main()
