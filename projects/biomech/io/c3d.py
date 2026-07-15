# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light C3D reader (points + analog + parameters).

Unlike the compact reader in
``data/scripts/calibrate_lower_body_elipsoid_from_static_c3d.py`` (which drops
invalid samples and only keeps 3-D points), this reader is built for
*time-aligned* ingestion:

- Points are returned as a dense ``[n_frames, n_points, 3]`` array in **meters**
  with ``NaN`` in gaps, so the sample index is the frame index.
- Analog channels are returned as a dense ``[n_analog, n_channels]`` array with
  C3D scaling applied, at the analog rate.
- The full parameter dictionary is preserved for force-plate / event parsing.

Supports both DEC/Intel float storage (``POINT:SCALE`` < 0) and scaled-integer
storage. No third-party dependency is required.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..frames import MM_TO_M

ParamKey = Tuple[str, str]


@dataclass
class C3DHeader:
    n_points: int
    analog_measurements_per_frame: int
    first_frame: int
    last_frame: int
    point_scale: float
    data_start_block: int
    analog_samples_per_frame: int
    point_rate: float


@dataclass
class C3DFile:
    """Fully parsed C3D capture with dense, SI point and analog arrays."""

    path: Path
    header: C3DHeader
    params: Dict[ParamKey, Dict[str, Any]]

    # Points -------------------------------------------------------------------
    point_labels: List[str]  # stripped of the "Subject:" prefix
    point_labels_raw: List[str]
    point_rate: float  # Hz
    point_units: str  # original units string (e.g. "mm")
    first_frame: int
    n_frames: int
    points: np.ndarray  # [n_frames, n_points, 3] meters, NaN in gaps
    residuals: np.ndarray  # [n_frames, n_points] raw residual (<0 == invalid)

    # Analog -------------------------------------------------------------------
    analog_labels: List[str]
    analog_units: List[str]
    analog_rate: float  # Hz
    analog_per_point_frame: int  # analog samples per point frame
    analog: np.ndarray  # [n_analog_samples, n_channels] scaled to recorded units

    # --- convenience ----------------------------------------------------------
    def point(self, label: str) -> np.ndarray:
        """Return ``[n_frames, 3]`` (meters, NaN gaps) for a stripped label."""

        try:
            idx = self.point_labels.index(label)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"marker '{label}' not in {self.point_labels}") from exc
        return self.points[:, idx, :]

    def param(self, group: str, name: str, default: Any = None) -> Any:
        item = self.params.get((group, name))
        if item is None:
            return default
        values = item["values"]
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        return values

    @property
    def n_analog_samples(self) -> int:
        return self.analog.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.point_rate if self.point_rate else 0.0


def _strip_prefix(label: str) -> str:
    return label.split(":", 1)[-1].strip()


def _parse_parameter_section(
    data: bytes,
) -> Tuple[str, Dict[ParamKey, Dict[str, Any]]]:
    parameter_block = data[0]
    parameter_offset = (parameter_block - 1) * 512
    processor_type = data[parameter_offset + 3]
    endian = "<" if processor_type in (84, 86) else ">"

    groups: Dict[int, str] = {}
    params: Dict[ParamKey, Dict[str, Any]] = {}
    pos = parameter_offset + 4

    while pos < len(data):
        name_len_raw = struct.unpack("b", data[pos : pos + 1])[0]
        if name_len_raw == 0:
            break
        group_id = struct.unpack("b", data[pos + 1 : pos + 2])[0]
        pos += 2

        name_len = abs(name_len_raw)
        name = data[pos : pos + name_len].decode("latin1").strip()
        pos += name_len

        offset_pos = pos
        next_offset = struct.unpack(endian + "h", data[pos : pos + 2])[0]
        pos += 2
        if next_offset == 0:
            break
        next_pos = offset_pos + next_offset

        if group_id < 0:
            desc_len = data[pos]
            groups[-group_id] = name
            pos += 1 + desc_len
        else:
            param_type = struct.unpack("b", data[pos : pos + 1])[0]
            pos += 1
            dim_count = data[pos]
            pos += 1
            dims = list(data[pos : pos + dim_count])
            pos += dim_count
            value_count = int(np.prod(dims)) if dims else 1

            if param_type == -1:  # char
                raw = data[pos : pos + value_count]
                if len(dims) == 2:
                    width, count = dims
                    values: Any = [
                        raw[i * width : (i + 1) * width]
                        .decode("latin1", errors="ignore")
                        .strip()
                        for i in range(count)
                    ]
                else:
                    values = raw.decode("latin1", errors="ignore").strip()
                pos += value_count
            elif param_type == 1:  # byte
                values = list(data[pos : pos + value_count])
                pos += value_count
            elif param_type == 2:  # int16
                values = list(
                    struct.unpack(
                        endian + str(value_count) + "h",
                        data[pos : pos + 2 * value_count],
                    )
                )
                pos += 2 * value_count
            elif param_type == 4:  # float32
                values = list(
                    struct.unpack(
                        endian + str(value_count) + "f",
                        data[pos : pos + 4 * value_count],
                    )
                )
                pos += 4 * value_count
            else:
                values = None

            desc_len = data[pos]
            pos += 1 + desc_len
            group_name = groups.get(group_id, str(group_id))
            params[(group_name, name)] = {
                "type": param_type,
                "dims": dims,
                "values": values,
            }

        pos = next_pos

    return endian, params


def _get(
    params: Dict[ParamKey, Dict[str, Any]],
    group: str,
    name: str,
    default: Any = None,
) -> Any:
    item = params.get((group, name))
    if item is None:
        return default
    values = item["values"]
    if isinstance(values, list) and len(values) == 1:
        return values[0]
    return values


def read_c3d(path: str | Path) -> C3DFile:
    """Read a C3D file into dense, SI, frame-indexed arrays."""

    path = Path(path)
    data = path.read_bytes()
    if len(data) < 512:
        raise ValueError(f"{path} is too small to be a C3D file")

    endian, params = _parse_parameter_section(data)

    n_points = int(_get(params, "POINT", "USED", 0))
    n_frames = int(_get(params, "POINT", "FRAMES", 0))
    data_start = int(_get(params, "POINT", "DATA_START", 0))
    point_scale = float(_get(params, "POINT", "SCALE", 1.0))
    point_rate = float(_get(params, "POINT", "RATE", 0.0))
    point_units = str(_get(params, "POINT", "UNITS", "mm")).strip() or "mm"
    start_field = _get(params, "TRIAL", "ACTUAL_START_FIELD", 1)
    if isinstance(start_field, list):
        start_field = start_field[0] if start_field else 1
    first_frame = int(start_field)

    point_labels_raw = list(_get(params, "POINT", "LABELS", []))[:n_points]
    point_labels = [_strip_prefix(lbl) for lbl in point_labels_raw]

    n_analog_ch = int(_get(params, "ANALOG", "USED", 0) or 0)
    analog_rate = float(_get(params, "ANALOG", "RATE", 0.0) or 0.0)
    analog_labels = list(_get(params, "ANALOG", "LABELS", []))[:n_analog_ch]
    analog_units = list(_get(params, "ANALOG", "UNITS", []))[:n_analog_ch]
    analog_per_frame = (
        int(round(analog_rate / point_rate)) if point_rate else 0
    )

    if n_points <= 0 or n_frames <= 0 or data_start <= 0:
        raise ValueError(
            f"Could not read C3D metadata from {path}: "
            f"points={n_points}, frames={n_frames}, data_start={data_start}"
        )

    header = C3DHeader(
        n_points=n_points,
        analog_measurements_per_frame=n_analog_ch * analog_per_frame,
        first_frame=first_frame,
        last_frame=first_frame + n_frames - 1,
        point_scale=point_scale,
        data_start_block=data_start,
        analog_samples_per_frame=analog_per_frame,
        point_rate=point_rate,
    )

    float_storage = point_scale < 0
    values_per_point = 4  # x, y, z, residual
    point_values_per_frame = n_points * values_per_point
    analog_values_per_frame = n_analog_ch * analog_per_frame
    record_values = point_values_per_frame + analog_values_per_frame

    data_offset = (data_start - 1) * 512
    dtype = np.dtype((endian + "f4") if float_storage else (endian + "i2"))
    needed = record_values * n_frames
    region = np.frombuffer(
        data, dtype=dtype, count=needed, offset=data_offset
    ).astype(np.float64)
    region = region.reshape(n_frames, record_values)

    # --- points ---------------------------------------------------------------
    point_block = region[:, :point_values_per_frame].reshape(
        n_frames, n_points, values_per_point
    )
    coords = point_block[:, :, :3].copy()
    residuals = point_block[:, :, 3].copy()

    if not float_storage:
        coords *= abs(point_scale)
        # In integer storage the residual is a signed camera-mask/residual word.

    # Invalid samples: negative residual, or exact-zero placeholder triples.
    invalid = (residuals < 0) | np.all(coords == 0.0, axis=2)
    coords[invalid] = np.nan

    unit_scale = {
        "mm": MM_TO_M,
        "millimeter": MM_TO_M,
        "millimeters": MM_TO_M,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
    }.get(point_units.lower(), MM_TO_M)
    coords *= unit_scale

    # --- analog ---------------------------------------------------------------
    if analog_values_per_frame > 0:
        analog_block = region[:, point_values_per_frame:].reshape(
            n_frames, analog_per_frame, n_analog_ch
        )
        analog = analog_block.reshape(n_frames * analog_per_frame, n_analog_ch)

        gen_scale = float(_get(params, "ANALOG", "GEN_SCALE", 1.0) or 1.0)
        scale = _get(params, "ANALOG", "SCALE", None)
        offset = _get(params, "ANALOG", "OFFSET", None)
        scale_arr = _as_channel_array(scale, n_analog_ch, 1.0)
        offset_arr = _as_channel_array(offset, n_analog_ch, 0.0)
        analog = (analog - offset_arr) * scale_arr * gen_scale
    else:
        analog = np.zeros((0, 0), dtype=np.float64)

    return C3DFile(
        path=path,
        header=header,
        params=params,
        point_labels=point_labels,
        point_labels_raw=point_labels_raw,
        point_rate=point_rate,
        point_units=point_units,
        first_frame=first_frame,
        n_frames=n_frames,
        points=coords,
        residuals=residuals,
        analog_labels=[str(x) for x in analog_labels],
        analog_units=[str(x) for x in analog_units],
        analog_rate=analog_rate,
        analog_per_point_frame=analog_per_frame,
        analog=analog,
    )


def _as_channel_array(
    value: Optional[Any], n_channels: int, default: float
) -> np.ndarray:
    if value is None:
        return np.full(n_channels, default, dtype=np.float64)
    if isinstance(value, (int, float)):
        return np.full(n_channels, float(value), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        return np.full(n_channels, float(arr[0]), dtype=np.float64)
    if arr.size < n_channels:
        pad = np.full(n_channels - arr.size, default, dtype=np.float64)
        arr = np.concatenate([arr, pad])
    return arr[:n_channels]
