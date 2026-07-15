# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrumented-treadmill belt-speed reader and resampler.

The S001 capture exports one belt-speed file per belt (``LeftBelt101.txt`` /
``RightBelt101.txt``) from Visual3D as a ``SPEEDCHANGE`` signal::

    <tab>D:\\...\\Trial 101.v3d.c3d<tab>
    <tab>SPEEDCHANGE<tab>
    <tab>METRIC<tab>
    <tab>PROCESSED<tab>
    ITEM<tab>0<tab>
    1.0<tab><tab>0.0
    2.0<tab><tab>0.0
    ...

Column 1 is a 1-based sample index, column 2 is the belt speed in m/s.

Important: the belt log and the C3D are logged by different clocks, and the belt
sample rate is *not* recorded in the file. It must therefore be supplied
explicitly (``rate_hz``) or inferred from a reference duration. Belt velocity is
expressed in the ``TREADMILL`` frame (same orientation as ``LAB``), pointing
along the lab forward axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..frames import CoordinateFrame

_HEADER_TOKENS = {"SPEEDCHANGE", "METRIC", "PROCESSED", "ITEM"}

# --- protocol events ---------------------------------------------------------
# The ``Speedchange<trial>.txt`` export lists protocol timestamps as a
# SPEEDCHANGE-style table of ``item_index<tab><tab>time_s``. For the S001
# treadmill protocol the named phase boundaries live at 1-based item rows
# 1, 3, 4, 5, 6, 7 (item 2 and items 8+ are ramp/settle offsets):
#
#     START = item 1, WALK_START = item 3, WALK_END = item 4,
#     RUN_START = item 5, RUN_END = item 6, END = item 7
#
# i.e. START + WALK_START + WALK_END + RUN_START + RUN_END + END == items
# 1 + 3 + 4 + 5 + 6 + 7 (per the capture operator's note).
PROTOCOL_ITEM_MAP: Dict[str, int] = {
    "START": 1,
    "WALK_START": 3,
    "WALK_END": 4,
    "RUN_START": 5,
    "RUN_END": 6,
    "END": 7,
}

# Named windows -> (start_event, end_event).
PROTOCOL_PHASES: Dict[str, Tuple[str, str]] = {
    "walk": ("WALK_START", "WALK_END"),
    "run": ("RUN_START", "RUN_END"),
    "all": ("START", "END"),
}


@dataclass
class BeltSignal:
    """A single belt's speed trace."""

    name: str  # e.g. "left" / "right"
    path: Optional[Path]
    sample_index: np.ndarray  # [n] 1-based indices as stored
    speed: np.ndarray  # [n] m/s
    rate_hz: float  # samples per second (may be inferred)
    rate_inferred: bool
    frame: CoordinateFrame = CoordinateFrame.TREADMILL

    @property
    def n_samples(self) -> int:
        return self.speed.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.rate_hz if self.rate_hz else 0.0

    def time_s(self) -> np.ndarray:
        """Belt sample timestamps in seconds (first sample at t=0)."""

        return np.arange(self.n_samples, dtype=np.float64) / self.rate_hz

    def resample_to(self, t_seconds: np.ndarray) -> np.ndarray:
        """Linearly resample belt speed onto an arbitrary time base (s).

        Values outside the belt time range are held at the nearest endpoint.
        """

        t_seconds = np.asarray(t_seconds, dtype=np.float64)
        if self.n_samples == 0:
            return np.full_like(t_seconds, np.nan)
        return np.interp(t_seconds, self.time_s(), self.speed)


def _parse_indexed_rows(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a Visual3D SPEEDCHANGE-style table into ``(indices, values)``.

    Skips the file-path line and the header tokens; every remaining row is
    ``index<tab...>value`` (column 1 is a 1-based index, the last column is the
    numeric payload -- belt speed for a belt file, time in seconds for a
    Speedchange file).
    """

    path = Path(path)
    indices: List[float] = []
    values: List[float] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p for p in line.split("\t") if p != ""]
        if not parts:
            continue
        first = parts[0].strip()
        if first in _HEADER_TOKENS or any(tok in line for tok in _HEADER_TOKENS):
            continue
        if first.lower().endswith(".c3d") or "\\" in first or "/" in first:
            continue
        try:
            idx = float(parts[0])
            val = float(parts[-1])
        except ValueError:
            continue
        indices.append(idx)
        values.append(val)
    return (
        np.asarray(indices, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def read_belt_file(
    path: str | Path,
    name: str,
    rate_hz: Optional[float] = None,
) -> BeltSignal:
    """Read a Visual3D ``SPEEDCHANGE`` belt-speed export.

    ``rate_hz`` may be ``None`` here; it is expected to be resolved (or inferred)
    by the caller. A placeholder rate of ``0.0`` is stored until then.
    """

    indices, speeds = _parse_indexed_rows(path)
    return BeltSignal(
        name=name,
        path=Path(path),
        sample_index=indices,
        speed=speeds,
        rate_hz=float(rate_hz) if rate_hz else 0.0,
        rate_inferred=rate_hz is None,
    )


@dataclass
class Treadmill:
    """A pair of belt signals sharing a (possibly inferred) sample rate."""

    left: Optional[BeltSignal]
    right: Optional[BeltSignal]
    rate_hz: float
    rate_inferred: bool
    frame: CoordinateFrame = CoordinateFrame.TREADMILL

    def _belt(self, side: str) -> Optional[BeltSignal]:
        return self.left if side == "left" else self.right

    def resample_to(self, t_seconds: np.ndarray) -> dict:
        """Return ``{"left": speed, "right": speed}`` on the given time base."""

        out = {}
        for side in ("left", "right"):
            belt = self._belt(side)
            out[side] = (
                belt.resample_to(t_seconds)
                if belt is not None
                else np.full_like(np.asarray(t_seconds, np.float64), np.nan)
            )
        return out


def load_treadmill(
    left_path: Optional[str | Path] = None,
    right_path: Optional[str | Path] = None,
    rate_hz: Optional[float] = None,
    reference_duration_s: Optional[float] = None,
) -> Optional[Treadmill]:
    """Load left/right belt files and resolve a common sample rate.

    Rate resolution priority:
      1. explicit ``rate_hz``;
      2. inferred as ``n_samples / reference_duration_s`` (e.g. the C3D point
         duration) if a reference duration is supplied;
      3. otherwise left as ``0.0`` with ``rate_inferred=True`` (caller must set).

    Returns ``None`` if neither belt file is provided.
    """

    if left_path is None and right_path is None:
        return None

    left = (
        read_belt_file(left_path, "left", rate_hz)
        if left_path is not None
        else None
    )
    right = (
        read_belt_file(right_path, "right", rate_hz)
        if right_path is not None
        else None
    )

    inferred = rate_hz is None
    resolved = float(rate_hz) if rate_hz else 0.0
    if inferred and reference_duration_s and reference_duration_s > 0:
        n = max(
            (b.n_samples for b in (left, right) if b is not None),
            default=0,
        )
        if n > 0:
            resolved = n / reference_duration_s

    for belt in (left, right):
        if belt is not None:
            belt.rate_hz = resolved
            belt.rate_inferred = inferred

    return Treadmill(
        left=left,
        right=right,
        rate_hz=resolved,
        rate_inferred=inferred,
    )


# ---------------------------------------------------------------------------
# Protocol events (Speedchange<trial>.txt)
# ---------------------------------------------------------------------------


@dataclass
class TreadmillProtocol:
    """Named phase boundaries from a ``Speedchange<trial>.txt`` export.

    Times are seconds on the *trial* timeline (the same clock as the C3D point
    stream, which for these captures starts at ``t=0``). ``item_times`` holds the
    full raw table in item order; ``events`` maps the named protocol boundaries
    (see :data:`PROTOCOL_ITEM_MAP`) to their times.
    """

    path: Optional[Path]
    item_times: np.ndarray  # (n,) seconds, in item order
    events: Dict[str, float] = field(default_factory=dict)

    def event_time(self, name: str) -> float:
        """Time (s) of a named event, e.g. ``"WALK_START"``."""
        key = name.upper()
        if key not in self.events:
            raise KeyError(
                f"unknown protocol event {name!r}; have {sorted(self.events)}"
            )
        return self.events[key]

    def phase_window_s(self, phase: str) -> Tuple[float, float]:
        """``(t_start, t_end)`` seconds for a named phase (``walk``/``run``/``all``)."""
        key = phase.lower()
        if key not in PROTOCOL_PHASES:
            raise KeyError(
                f"unknown phase {phase!r}; have {sorted(PROTOCOL_PHASES)}"
            )
        a, b = PROTOCOL_PHASES[key]
        return self.event_time(a), self.event_time(b)

    def phase_window_frames(
        self,
        phase: str,
        rate_hz: float,
        t_start: float = 0.0,
        n_frames: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Half-open frame window ``(lo, hi)`` for a phase on a point timeline.

        ``rate_hz`` is the point (mocap) rate; ``t_start`` is the time of frame 0
        (``t_point[0]``, usually 0). Result is clamped to ``[0, n_frames]`` when
        ``n_frames`` is given.
        """
        t0, t1 = self.phase_window_s(phase)
        lo = int(round((t0 - t_start) * rate_hz))
        hi = int(round((t1 - t_start) * rate_hz))
        if n_frames is not None:
            lo = max(0, min(lo, n_frames))
            hi = max(0, min(hi, n_frames))
        return lo, hi


def read_speedchange(
    path: str | Path,
    item_map: Optional[Dict[str, int]] = None,
) -> TreadmillProtocol:
    """Read a ``Speedchange<trial>.txt`` protocol-events export.

    The file is a SPEEDCHANGE-style table whose payload column is a *time in
    seconds* (not a speed). ``item_map`` maps event names to 1-based item rows;
    it defaults to :data:`PROTOCOL_ITEM_MAP`.
    """

    item_map = item_map or PROTOCOL_ITEM_MAP
    indices, times = _parse_indexed_rows(path)
    # Build a 1-based item -> time lookup (indices are the stored item numbers).
    by_item: Dict[int, float] = {
        int(round(i)): float(t) for i, t in zip(indices, times)
    }
    events: Dict[str, float] = {}
    for name, item in item_map.items():
        if item in by_item:
            events[name] = by_item[item]
    return TreadmillProtocol(
        path=Path(path),
        item_times=times,
        events=events,
    )
