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
"""Small C3D reader utilities for marker-based human retargeting.

The project only needs a narrow subset of C3D functionality at first: point
labels, marker trajectories, point units/rate, and analog labels.  This module
uses :mod:`ezc3d` when it is installed, but also includes a lightweight native
reader for Intel/IEEE-float C3D files exported by Vicon Nexus/Visual3D.

The native reader intentionally avoids loading the full 500+ MB file unless the
caller requests it.  Use ``start_frame`` and ``end_frame`` for calibration/static
windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np


@dataclass
class C3DHeader:
    parameter_block: int
    point_count: int
    analog_value_count: int
    first_frame: int
    last_frame: int
    scale: float
    data_block: int
    analog_samples_per_frame: int
    point_rate: float

    @property
    def frame_count(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def analog_channel_count(self) -> int:
        if self.analog_samples_per_frame == 0:
            return 0
        return self.analog_value_count // self.analog_samples_per_frame


@dataclass
class C3DData:
    markers: np.ndarray
    marker_labels: list[str]
    point_rate: float
    point_units: str
    first_frame: int
    start_frame: int
    analog: np.ndarray | None = None
    analog_labels: list[str] | None = None
    analog_rate: float | None = None
    analog_units: list[str] | None = None


@dataclass
class C3DMetadata:
    header: C3DHeader
    groups: dict[int, str]
    parameters: dict[str, Any]


def read_header(path: str | Path) -> C3DHeader:
    """Read the 512-byte C3D header."""
    with open(path, "rb") as f:
        header = f.read(512)
    if len(header) < 512:
        raise ValueError(f"{path} is too small to be a C3D file")
    if header[1] != 80:
        raise ValueError(f"{path} does not look like a C3D file (header byte 2 != 80)")

    return C3DHeader(
        parameter_block=header[0],
        point_count=struct.unpack_from("<h", header, 2)[0],
        analog_value_count=struct.unpack_from("<h", header, 4)[0],
        first_frame=struct.unpack_from("<h", header, 6)[0],
        last_frame=struct.unpack_from("<h", header, 8)[0],
        scale=struct.unpack_from("<f", header, 12)[0],
        data_block=struct.unpack_from("<h", header, 16)[0],
        analog_samples_per_frame=struct.unpack_from("<h", header, 18)[0],
        point_rate=struct.unpack_from("<f", header, 20)[0],
    )


def _decode_parameter(raw: bytes, data_type: int, dimensions: list[int]) -> Any:
    if not dimensions:
        item_count = 1
    else:
        item_count = int(np.prod(dimensions))

    if data_type == -1:
        if not dimensions:
            return raw.decode("latin1", errors="ignore").strip()
        width = dimensions[0]
        count = int(np.prod(dimensions[1:])) if len(dimensions) > 1 else 1
        return [
            raw[i * width : (i + 1) * width]
            .decode("latin1", errors="ignore")
            .replace("\x00", "")
            .strip()
            for i in range(count)
        ]

    dtype_map = {
        1: np.dtype("<i1"),
        2: np.dtype("<i2"),
        4: np.dtype("<f4"),
    }
    if data_type not in dtype_map:
        return raw

    values = np.frombuffer(raw, dtype=dtype_map[data_type], count=item_count).copy()
    if dimensions:
        values = values.reshape(tuple(reversed(dimensions))).T
    if values.size == 1:
        return values.reshape(-1)[0].item()
    return values


def read_metadata(path: str | Path) -> C3DMetadata:
    """Read C3D parameter groups and parameters.

    Parameter keys are stored as ``GROUP.PARAM`` in uppercase.  String-array
    parameters are returned as Python lists; numeric parameters as NumPy arrays
    or scalars.
    """
    path = Path(path)
    header = read_header(path)
    with open(path, "rb") as f:
        content = f.read((header.data_block - 1) * 512)

    start = (header.parameter_block - 1) * 512
    processor = content[start + 3]
    if processor != 84:
        raise NotImplementedError(
            f"Native C3D metadata reader only supports Intel processor files; got {processor}. "
            "Install ezc3d for broader C3D support."
        )

    pos = start + 4
    groups: dict[int, str] = {}
    pending_params: list[tuple[int, str, Any]] = []

    while pos + 4 <= len(content):
        name_len_raw = struct.unpack_from("<b", content, pos)[0]
        group_id = struct.unpack_from("<b", content, pos + 1)[0]
        pos += 2
        if name_len_raw == 0:
            break

        name_len = abs(name_len_raw)
        name = content[pos : pos + name_len].decode("latin1", errors="ignore").strip().upper()
        pos += name_len
        offset_pos = pos
        next_offset = struct.unpack_from("<h", content, pos)[0]
        pos += 2
        record_end = offset_pos + next_offset

        if group_id < 0:
            if pos < len(content):
                desc_len = content[pos]
                pos += 1 + desc_len
            groups[abs(group_id)] = name
        else:
            data_type = struct.unpack_from("<b", content, pos)[0]
            dim_count = content[pos + 1]
            pos += 2
            dimensions = list(content[pos : pos + dim_count])
            pos += dim_count
            type_size = abs(data_type)
            data_size = type_size * (int(np.prod(dimensions)) if dimensions else 1)
            raw = content[pos : pos + data_size]
            pos += data_size
            pending_params.append((group_id, name, _decode_parameter(raw, data_type, dimensions)))
            if pos < record_end:
                desc_len = content[pos]
                pos += 1 + desc_len

        pos = record_end

    parameters: dict[str, Any] = {}
    for group_id, name, value in pending_params:
        group = groups.get(group_id, f"GROUP_{group_id}")
        parameters[f"{group}.{name}"] = value

    return C3DMetadata(header=header, groups=groups, parameters=parameters)


def _labels_from_metadata(metadata: C3DMetadata, key: str, expected: int | None = None) -> list[str]:
    labels = metadata.parameters.get(key, [])
    if isinstance(labels, np.ndarray):
        labels = labels.reshape(-1).tolist()
    if isinstance(labels, str):
        labels = [labels]
    labels = [str(label).strip() for label in labels if str(label).strip()]
    if expected is not None and labels and len(labels) != expected:
        # Visual3D/Vicon files can store generated/modelled points in the same
        # label list.  Keep the metadata but do not fail on this; callers can
        # still inspect the labels.
        labels = labels[:expected]
    return labels


def _load_with_ezc3d(path: Path, start_frame: int | None, end_frame: int | None, load_analog: bool) -> C3DData:
    import ezc3d  # type: ignore[import-not-found]

    c3d = ezc3d.c3d(str(path), extract_forceplat_data=load_analog)
    params = c3d["parameters"]
    points = c3d["data"]["points"][:3].transpose(2, 1, 0)
    first_frame = int(params["POINT"].get("FIRST_FRAME", {"value": [1]})["value"][0])
    start = 0 if start_frame is None else max(0, start_frame - first_frame)
    stop = points.shape[0] if end_frame is None else max(start, end_frame - first_frame + 1)
    marker_labels = [str(label).strip() for label in params["POINT"]["LABELS"]["value"]]
    point_units = str(params["POINT"].get("UNITS", {"value": [""]})["value"][0]).strip()
    point_rate = float(params["POINT"]["RATE"]["value"][0])

    analog = None
    analog_labels = None
    analog_rate = None
    analog_units = None
    if load_analog and "analogs" in c3d["data"]:
        analog = c3d["data"]["analogs"].squeeze(0).T
        analog_labels = [str(label).strip() for label in params["ANALOG"]["LABELS"]["value"]]
        analog_rate = float(params["ANALOG"]["RATE"]["value"][0])
        analog_units = [str(unit).strip() for unit in params["ANALOG"].get("UNITS", {"value": []})["value"]]

    return C3DData(
        markers=points[start:stop].astype(np.float32, copy=False),
        marker_labels=marker_labels,
        point_rate=point_rate,
        point_units=point_units,
        first_frame=first_frame,
        start_frame=first_frame + start,
        analog=analog,
        analog_labels=analog_labels,
        analog_rate=analog_rate,
        analog_units=analog_units,
    )


def load_c3d(
    path: str | Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    load_analog: bool = False,
    prefer_ezc3d: bool = True,
) -> C3DData:
    """Load C3D marker trajectories.

    Frames are 1-based C3D frame numbers.  ``end_frame`` is inclusive.
    """
    path = Path(path)
    if prefer_ezc3d:
        try:
            return _load_with_ezc3d(path, start_frame, end_frame, load_analog)
        except ModuleNotFoundError:
            pass

    metadata = read_metadata(path)
    header = metadata.header
    if header.scale >= 0:
        raise NotImplementedError(
            "Native C3D reader currently supports IEEE-float point data (negative SCALE). "
            "Install ezc3d to read integer-scaled C3D files."
        )
    if load_analog:
        raise NotImplementedError("Native C3D reader does not load analog data yet; install ezc3d.")

    start = header.first_frame if start_frame is None else max(header.first_frame, start_frame)
    stop = header.last_frame if end_frame is None else min(header.last_frame, end_frame)
    if stop < start:
        raise ValueError(f"Invalid frame window: start={start}, end={stop}")

    frame_offset = start - header.first_frame
    frame_count = stop - start + 1
    record_len = header.point_count * 4 + header.analog_value_count
    data_offset = (header.data_block - 1) * 512 + frame_offset * record_len * 4
    raw = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=data_offset,
        shape=(frame_count, record_len),
    )
    points = np.asarray(raw[:, : header.point_count * 4]).reshape(frame_count, header.point_count, 4)
    markers = points[:, :, :3].astype(np.float32, copy=True)
    markers[np.abs(markers).sum(axis=-1) == 0.0] = np.nan

    marker_labels = _labels_from_metadata(metadata, "POINT.LABELS", header.point_count)
    analog_labels = _labels_from_metadata(metadata, "ANALOG.LABELS", header.analog_channel_count)
    point_units_value = metadata.parameters.get("POINT.UNITS", "")
    if isinstance(point_units_value, list):
        point_units = point_units_value[0] if point_units_value else ""
    else:
        point_units = str(point_units_value)
    analog_rate = metadata.parameters.get("ANALOG.RATE")
    analog_units_value = metadata.parameters.get("ANALOG.UNITS", [])
    analog_units = [str(unit).strip() for unit in analog_units_value] if isinstance(analog_units_value, list) else []

    return C3DData(
        markers=markers,
        marker_labels=marker_labels,
        point_rate=float(metadata.parameters.get("POINT.RATE", header.point_rate)),
        point_units=point_units.replace("\x00", "").strip(),
        first_frame=header.first_frame,
        start_frame=start,
        analog=None,
        analog_labels=analog_labels,
        analog_rate=float(analog_rate) if analog_rate is not None else None,
        analog_units=analog_units,
    )


def marker_index(labels: list[str], name: str) -> int:
    """Find a marker by suffix-aware Visual3D/Vicon label matching."""
    candidates = [name, f"*:{name}"]
    for candidate in candidates:
        if candidate in labels:
            return labels.index(candidate)
    suffix = f":{name}"
    for idx, label in enumerate(labels):
        if label == name or label.endswith(suffix):
            return idx
    raise KeyError(f"Marker {name!r} not found in C3D labels")


def markers_by_name(data: C3DData, names: list[str]) -> dict[str, np.ndarray]:
    return {name: data.markers[:, marker_index(data.marker_labels, name)] for name in names}
