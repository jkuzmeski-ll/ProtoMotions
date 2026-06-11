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
"""Treadmill-to-overground virtual-origin mapping helpers.

The mapping follows the virtual-origin idea from Jung and Lee, Sensors 21(3),
786 (2021): for a treadmill marker trajectory measured in a fixed capture
volume, add the belt travel displacement along the walking direction.  This is
equivalent to expressing the marker from a virtual origin moving backward with
the treadmill belt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TreadmillMappingReport:
    speed_mps: float
    total_displacement_m: float
    frame_count: int
    fps: float
    estimated: bool
    valid_stance_samples: int = 0
    speed_source: str = "constant"
    speed_change_frames: tuple[tuple[int, float], ...] = ()


def normalized_direction(direction: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    axis = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        raise ValueError(f"Invalid zero-length treadmill direction: {direction}")
    return axis / norm


def displacement_from_speed(frame_count: int, fps: float, speed_mps: float) -> np.ndarray:
    """Return per-frame forward belt travel, with frame 0 displacement at zero."""
    if frame_count <= 0:
        return np.zeros((0,), dtype=np.float32)
    frame_time = np.arange(frame_count, dtype=np.float32) / float(fps)
    return frame_time * float(speed_mps)


def displacement_from_speed_profile(speed_mps: np.ndarray, fps: float) -> np.ndarray:
    """Integrate a per-frame treadmill belt speed profile into displacement."""
    speed = np.asarray(speed_mps, dtype=np.float32).reshape(-1)
    if speed.size == 0:
        return np.zeros((0,), dtype=np.float32)
    displacement = np.zeros_like(speed, dtype=np.float32)
    if speed.size > 1:
        displacement[1:] = np.cumsum(speed[:-1], dtype=np.float32) / float(fps)
    return displacement


def speed_change_frames_from_profile(speed_mps: np.ndarray, atol: float = 1e-4) -> tuple[tuple[int, float], ...]:
    """Compress a per-frame speed profile into ``(frame_idx, speed)`` changes."""
    speed = np.asarray(speed_mps, dtype=np.float32).reshape(-1)
    if speed.size == 0:
        return ()
    changes: list[tuple[int, float]] = [(0, float(speed[0]))]
    for idx in range(1, speed.size):
        if not np.isclose(speed[idx], speed[idx - 1], atol=atol, rtol=0.0):
            changes.append((idx, float(speed[idx])))
    return tuple(changes)


def apply_virtual_origin_mapping_with_speed_profile(
    points: np.ndarray,
    fps: float,
    speed_mps: np.ndarray,
    direction: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    *,
    estimated: bool = False,
    valid_stance_samples: int = 0,
    speed_source: str = "profile",
) -> tuple[np.ndarray, TreadmillMappingReport]:
    """Map treadmill points using a per-frame belt speed profile."""
    axis = normalized_direction(direction)
    speed = np.asarray(speed_mps, dtype=np.float32).reshape(-1)
    if speed.shape[0] != points.shape[0]:
        raise ValueError(f"speed_mps must have one value per frame; got {speed.shape[0]} for {points.shape[0]} frames")
    displacement = displacement_from_speed_profile(speed, fps)
    mapped = points.copy()
    mapped += displacement.reshape((points.shape[0],) + (1,) * (points.ndim - 2) + (1,)) * axis
    report = TreadmillMappingReport(
        speed_mps=float(np.nanmean(speed)) if speed.size else 0.0,
        total_displacement_m=float(displacement[-1]) if displacement.size else 0.0,
        frame_count=int(points.shape[0]),
        fps=float(fps),
        estimated=estimated,
        valid_stance_samples=valid_stance_samples,
        speed_source=speed_source,
        speed_change_frames=speed_change_frames_from_profile(speed),
    )
    return mapped, report


def apply_virtual_origin_mapping(
    points: np.ndarray,
    fps: float,
    speed_mps: float,
    direction: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
) -> tuple[np.ndarray, TreadmillMappingReport]:
    """Map treadmill point trajectories to overground trajectories.

    Args:
        points: Array shaped ``(frames, ..., 3)`` in meters.
        fps: Point sampling rate after any downsampling.
        speed_mps: Positive treadmill belt speed in the subject's forward
            direction, in meters per second.
        direction: Forward walking direction in the coordinate frame of
            ``points``.
    """
    axis = normalized_direction(direction)
    displacement = displacement_from_speed(points.shape[0], fps, speed_mps)
    mapped = points.copy()
    mapped += displacement.reshape((points.shape[0],) + (1,) * (points.ndim - 2) + (1,)) * axis
    report = TreadmillMappingReport(
        speed_mps=float(speed_mps),
        total_displacement_m=float(displacement[-1]) if displacement.size else 0.0,
        frame_count=int(points.shape[0]),
        fps=float(fps),
        estimated=False,
        speed_change_frames=((0, float(speed_mps)),),
    )
    return mapped, report


def c3d_speed_profile_from_events(
    events: list[object],
    frame_count: int,
    fps: float,
    source_fps: float,
    first_frame: int,
    start_frame: int,
) -> np.ndarray | None:
    """Build a per-frame speed profile from numeric C3D events.

    Event labels such as ``10``, ``15``, ``30`` are interpreted as 1.0, 1.5,
    3.0 m/s.  The speed is zero before the first numeric event and after an
    ``END`` event when present.
    """
    changes: list[tuple[int, float]] = [(0, 0.0)]
    has_speed_events = False
    for event in events:
        label = str(getattr(event, "label", "")).strip()
        if label.isdigit():
            speed = float(int(label)) * 0.1
        elif label.upper() == "END":
            speed = 0.0
        else:
            continue
        has_speed_events = True
        event_time = float(getattr(event, "time_seconds"))
        start_time = (start_frame - first_frame) / float(source_fps)
        event_frame = int(round((event_time - start_time) * float(fps)))
        event_frame = max(0, min(frame_count, event_frame))
        if event_frame < frame_count:
            changes.append((event_frame, speed))

    if len(changes) == 1:
        return np.zeros(frame_count, dtype=np.float32) if has_speed_events else None

    speed_profile = np.zeros(frame_count, dtype=np.float32)
    changes = sorted(changes, key=lambda item: item[0])
    for change_idx, (start_idx, speed) in enumerate(changes):
        end_idx = changes[change_idx + 1][0] if change_idx + 1 < len(changes) else frame_count
        if end_idx > start_idx:
            speed_profile[start_idx:end_idx] = speed
    return speed_profile


def c3d_contact_mask_from_events(
    events: list[object],
    frame_count: int,
    fps: float,
    source_fps: float,
    first_frame: int,
    start_frame: int,
    marker_sides: list[str],
) -> np.ndarray | None:
    """Return per-marker stance mask from LON/LOFF/RON/ROFF C3D events."""
    side_event_frames: dict[str, list[tuple[int, bool]]] = {"L": [], "R": []}
    for event in events:
        label = str(getattr(event, "label", "")).strip().upper()
        side = label[0] if label else ""
        if side not in side_event_frames:
            continue
        if label.startswith(f"{side}ON"):
            is_on = True
        elif label.startswith(f"{side}OFF"):
            is_on = False
        else:
            continue
        event_time = float(getattr(event, "time_seconds"))
        start_time = (start_frame - first_frame) / float(source_fps)
        event_frame = int(round((event_time - start_time) * float(fps)))
        event_frame = max(0, min(frame_count, event_frame))
        side_event_frames[side].append((event_frame, is_on))

    if not any(side_event_frames.values()):
        return None

    side_masks: dict[str, np.ndarray] = {}
    for side, side_events in side_event_frames.items():
        mask = np.zeros(frame_count, dtype=bool)
        side_events = sorted(side_events, key=lambda item: item[0])
        if side_events and not side_events[0][1]:
            stance_start = 0
        else:
            stance_start = None
        for frame_idx, is_on in side_events:
            if is_on:
                stance_start = frame_idx
            elif stance_start is not None:
                mask[stance_start:frame_idx] = True
                stance_start = None
        if stance_start is not None:
            mask[stance_start:] = True
        side_masks[side] = mask

    marker_mask = np.zeros((frame_count, len(marker_sides)), dtype=bool)
    for marker_idx, side in enumerate(marker_sides):
        marker_mask[:, marker_idx] = side_masks.get(side.upper(), np.zeros(frame_count, dtype=bool))
    return marker_mask


def estimate_speed_from_stance_points(
    foot_points: np.ndarray,
    fps: float,
    direction: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    vertical_axis: int = 2,
    stance_height_percentile: float = 35.0,
    stance_height_margin: float = 0.08,
    max_vertical_speed: float = 0.30,
    min_abs_forward_speed: float = 0.05,
) -> tuple[float, int]:
    """Estimate treadmill belt speed from foot markers during stance.

    During treadmill stance, planted foot markers move backward in the lab at
    roughly the belt speed.  This returns the median of ``-v_forward`` over
    low, vertically quiet foot-marker samples.
    """
    if foot_points.ndim != 3 or foot_points.shape[-1] != 3:
        raise ValueError(f"foot_points must have shape (frames, markers, 3); got {foot_points.shape}")
    if foot_points.shape[0] < 2:
        raise ValueError("At least two frames are required to estimate treadmill speed.")

    axis = normalized_direction(direction)
    velocity = np.diff(foot_points, axis=0) * float(fps)
    midpoint = 0.5 * (foot_points[1:] + foot_points[:-1])
    finite = np.isfinite(velocity).all(axis=-1) & np.isfinite(midpoint).all(axis=-1)
    forward_velocity = np.einsum("fmc,c->fm", velocity, axis)
    vertical_velocity = velocity[..., vertical_axis]
    height = midpoint[..., vertical_axis]

    low_height = np.zeros_like(finite, dtype=bool)
    for marker_idx in range(foot_points.shape[1]):
        valid_height = height[finite[:, marker_idx], marker_idx]
        if valid_height.size == 0:
            continue
        threshold = np.nanpercentile(valid_height, stance_height_percentile) + stance_height_margin
        low_height[:, marker_idx] = height[:, marker_idx] <= threshold

    stance = (
        finite
        & low_height
        & (np.abs(vertical_velocity) <= max_vertical_speed)
        & (-forward_velocity >= min_abs_forward_speed)
    )
    samples = -forward_velocity[stance]
    if samples.size == 0:
        raise ValueError(
            "Could not estimate treadmill speed from stance markers. "
            "Pass an explicit --treadmill-speed-mps value."
        )
    return float(np.nanmedian(samples)), int(samples.size)


def estimate_speed_profile_from_stance_points(
    foot_points: np.ndarray,
    fps: float,
    direction: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
    vertical_axis: int = 2,
    event_stance_mask: np.ndarray | None = None,
    stance_height_percentile: float = 35.0,
    stance_height_margin: float = 0.08,
    max_vertical_speed: float = 0.30,
    min_abs_forward_speed: float = 0.05,
    smoothing_seconds: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Estimate a time-varying treadmill speed profile from stance markers.

    C3D contact events can be supplied as ``event_stance_mask``.  When absent,
    stance is detected from low marker height and low vertical velocity.
    """
    if foot_points.ndim != 3 or foot_points.shape[-1] != 3:
        raise ValueError(f"foot_points must have shape (frames, markers, 3); got {foot_points.shape}")
    frame_count = foot_points.shape[0]
    if frame_count < 2:
        raise ValueError("At least two frames are required to estimate treadmill speed.")

    axis = normalized_direction(direction)
    velocity = np.diff(foot_points, axis=0) * float(fps)
    midpoint = 0.5 * (foot_points[1:] + foot_points[:-1])
    finite = np.isfinite(velocity).all(axis=-1) & np.isfinite(midpoint).all(axis=-1)
    forward_velocity = np.einsum("fmc,c->fm", velocity, axis)
    vertical_velocity = velocity[..., vertical_axis]

    if event_stance_mask is not None:
        if event_stance_mask.shape != foot_points.shape[:2]:
            raise ValueError(f"event_stance_mask must have shape {foot_points.shape[:2]}; got {event_stance_mask.shape}")
        stance = finite & (event_stance_mask[:-1] | event_stance_mask[1:])
    else:
        height = midpoint[..., vertical_axis]
        low_height = np.zeros_like(finite, dtype=bool)
        for marker_idx in range(foot_points.shape[1]):
            valid_height = height[finite[:, marker_idx], marker_idx]
            if valid_height.size == 0:
                continue
            threshold = np.nanpercentile(valid_height, stance_height_percentile) + stance_height_margin
            low_height[:, marker_idx] = height[:, marker_idx] <= threshold
        stance = finite & low_height & (np.abs(vertical_velocity) <= max_vertical_speed)

    stance &= -forward_velocity >= min_abs_forward_speed
    interval_speed = np.full(frame_count - 1, np.nan, dtype=np.float32)
    for frame_idx in range(frame_count - 1):
        samples = -forward_velocity[frame_idx, stance[frame_idx]]
        if samples.size:
            interval_speed[frame_idx] = float(np.nanmedian(samples))

    valid = np.isfinite(interval_speed)
    valid_count = int(np.count_nonzero(stance))
    if not valid.any():
        raise ValueError("Could not estimate treadmill speed profile from stance markers.")

    valid_idx = np.flatnonzero(valid)
    filled = np.interp(
        np.arange(frame_count - 1, dtype=np.float32),
        valid_idx.astype(np.float32),
        interval_speed[valid_idx],
    ).astype(np.float32)
    filled[: valid_idx[0]] = 0.0
    if valid_idx[-1] < frame_count - 2:
        filled[valid_idx[-1] + 1 :] = 0.0

    window = max(1, int(round(float(smoothing_seconds) * float(fps))))
    if window > 1:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        filled = np.convolve(filled, kernel, mode="same").astype(np.float32)

    speed_profile = np.zeros(frame_count, dtype=np.float32)
    speed_profile[:-1] = np.maximum(filled, 0.0)
    speed_profile[-1] = speed_profile[-2]
    return speed_profile, valid_count
