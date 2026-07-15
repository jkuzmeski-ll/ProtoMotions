# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified, time-aligned capture session (Milestone 1 deliverable).

:func:`load_session` ingests a mocap C3D (points + analog force channels),
per-belt treadmill speed logs, gait events, and optional subject metadata into
one :class:`CaptureSession`. Everything is converted to SI at ingest and tagged
with an explicit :class:`~projects.biomech.frames.Frames` convention bundle.

Two time bases are exposed:
- ``t_point`` at the mocap rate (100 Hz for S001), the master timeline for
  markers.
- ``t_analog`` at the analog rate (2000 Hz for S001), the timeline for
  force-plate GRF/COP.

Belt speed lives on its own (possibly inferred-rate) timeline and is resampled
onto both. Force/COP can also be block-averaged down to the point timeline for
convenience.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .frames import Frames
from .io.c3d import C3DFile, read_c3d
from .io.force_plate import ForcePlate, compute_force_plates
from .io.treadmill import Treadmill, TreadmillProtocol, load_treadmill, read_speedchange


@dataclass
class GaitEvent:
    """A labelled event on the master (seconds) timeline."""

    context: str  # e.g. "Left", "Right", "General"
    label: str  # e.g. "Foot Strike", "Foot Off"
    time_s: float


@dataclass
class CaptureSession:
    """A fully time-aligned, explicitly-framed capture."""

    subject_id: str
    frames: Frames
    source: Dict[str, Optional[str]]

    # Marker / point stream (master timeline) ---------------------------------
    point_rate: float
    t_point: np.ndarray  # [n_frames] seconds
    marker_labels: List[str]
    markers: np.ndarray  # [n_frames, n_markers, 3] m, NaN gaps

    # Analog / force-plate stream ---------------------------------------------
    analog_rate: float
    t_analog: np.ndarray  # [n_analog] seconds
    analog_per_point_frame: int
    force_plates: List[ForcePlate]

    # Treadmill ----------------------------------------------------------------
    treadmill: Optional[Treadmill]
    belt_speed_point: Dict[str, np.ndarray]  # side -> [n_frames] m/s
    belt_speed_analog: Dict[str, np.ndarray]  # side -> [n_analog] m/s
    protocol: Optional[TreadmillProtocol] = None

    # Events & metadata --------------------------------------------------------
    events: List[GaitEvent] = field(default_factory=list)
    subject_meta: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # Preprocessing provenance (e.g. the low-pass filter applied at ingest), or None.
    filter_info: Optional[Dict[str, Any]] = None

    # raw handle for advanced use
    c3d: Optional[C3DFile] = None

    # --- convenience ----------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.t_point.shape[0]

    @property
    def n_analog(self) -> int:
        return self.t_analog.shape[0]

    def marker(self, label: str) -> np.ndarray:
        idx = self.marker_labels.index(label)
        return self.markers[:, idx, :]

    def phase_window(self, phase: str) -> tuple[int, int]:
        """Half-open point-frame window ``(lo, hi)`` for a protocol phase.

        ``phase`` is one of ``"walk"``, ``"run"``, ``"all"`` (see
        :data:`~biomech.io.treadmill.PROTOCOL_PHASES`). Requires the session to
        have been loaded with a ``speedchange_path``.
        """
        if self.protocol is None:
            raise ValueError(
                "no treadmill protocol loaded; pass speedchange_path to load_session"
            )
        t0 = float(self.t_point[0]) if self.n_frames else 0.0
        return self.protocol.phase_window_frames(
            phase, self.point_rate, t_start=t0, n_frames=self.n_frames
        )

    def forces_on_point_timeline(self) -> Dict[str, np.ndarray]:
        """Block-average GRF and COP from analog down to the point timeline.

        Returns a dict with per-plate ``grf`` ``[n_frames, 3]`` and ``cop``
        ``[n_frames, 3]`` (NaN-aware mean over each block of analog samples).
        """

        n = self.analog_per_point_frame
        out: Dict[str, np.ndarray] = {}
        for plate in self.force_plates:
            grf = plate.grf[: self.n_frames * n].reshape(self.n_frames, n, 3)
            cop = plate.cop_world[: self.n_frames * n].reshape(self.n_frames, n, 3)
            out[f"plate{plate.index}_grf"] = grf.mean(axis=1)
            with warnings.catch_warnings():
                # Swing blocks are all-NaN COP by design; ignore the mean warning.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[f"plate{plate.index}_cop"] = np.nanmean(cop, axis=1)
        return out

    def report(self) -> Dict[str, Any]:
        rep: Dict[str, Any] = {
            "subject_id": self.subject_id,
            "source": self.source,
            "frames": self.frames.as_dict(),
            "point": {
                "rate_hz": self.point_rate,
                "n_frames": self.n_frames,
                "duration_s": float(self.t_point[-1]) if self.n_frames else 0.0,
                "n_markers": len(self.marker_labels),
                "markers": self.marker_labels,
            },
            "analog": {
                "rate_hz": self.analog_rate,
                "n_samples": self.n_analog,
                "samples_per_point_frame": self.analog_per_point_frame,
            },
            "force_plates": [p.summary() for p in self.force_plates],
            "events": [
                {"context": e.context, "label": e.label, "time_s": e.time_s}
                for e in self.events
            ],
            "subject_meta": self.subject_meta,
            "warnings": self.warnings,
        }
        if self.filter_info is not None:
            rep["preprocessing"] = {"filter": self.filter_info}
        if self.treadmill is not None:
            rep["treadmill"] = {
                "rate_hz": self.treadmill.rate_hz,
                "rate_inferred": self.treadmill.rate_inferred,
                "belts": {
                    side: {
                        "n_samples": belt.n_samples,
                        "duration_s": belt.duration_s,
                        "max_speed_mps": float(np.nanmax(belt.speed))
                        if belt.n_samples
                        else 0.0,
                    }
                    for side, belt in (
                        ("left", self.treadmill.left),
                        ("right", self.treadmill.right),
                    )
                    if belt is not None
                },
            }
        if self.protocol is not None:
            rep["protocol"] = {
                "events": {k: round(v, 4) for k, v in self.protocol.events.items()},
                "n_items": int(self.protocol.item_times.size),
            }
        return rep


def _parse_events(c3d: C3DFile) -> List[GaitEvent]:
    used = int(c3d.param("EVENT", "USED", 0) or 0)
    if used <= 0:
        return []
    times = c3d.param("EVENT", "TIMES")  # [2, n]: (minutes, seconds)
    labels = c3d.param("EVENT", "LABELS")
    contexts = c3d.param("EVENT", "CONTEXTS")
    times_arr = np.asarray(times, dtype=np.float64).ravel()
    # Column-major [2, n]: even indices = minutes, odd = seconds.
    times_arr = times_arr.reshape(2, used, order="F")
    if not isinstance(labels, list):
        labels = [labels]
    if not isinstance(contexts, list):
        contexts = [contexts]

    events: List[GaitEvent] = []
    for i in range(used):
        t = float(times_arr[0, i]) * 60.0 + float(times_arr[1, i])
        events.append(
            GaitEvent(
                context=str(contexts[i]) if i < len(contexts) else "",
                label=str(labels[i]) if i < len(labels) else "",
                time_s=t,
            )
        )
    events.sort(key=lambda e: e.time_s)
    return events


_MP_LINE = re.compile(r"^\$(?P<key>\w+)\s*=\s*(?P<val>-?[\d.]+)")


def read_subject_mp(path: str | Path) -> Dict[str, float]:
    """Read a Visual3D ``.mp`` subject-metrics file into a flat dict.

    Length metrics are in millimeters as stored (not converted); mass in kg.
    Convenience SI keys ``mass_kg`` and ``height_m`` are added when available.
    """

    meta: Dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        m = _MP_LINE.match(line.strip())
        if m:
            try:
                meta[m.group("key")] = float(m.group("val"))
            except ValueError:
                continue
    if "Bodymass" in meta:
        meta["mass_kg"] = meta["Bodymass"]
    if "Height" in meta:
        meta["height_m"] = meta["Height"] * 1.0e-3
    return meta


def load_session(
    c3d_path: str | Path,
    left_belt_path: Optional[str | Path] = None,
    right_belt_path: Optional[str | Path] = None,
    belt_rate_hz: Optional[float] = None,
    subject_mp_path: Optional[str | Path] = None,
    speedchange_path: Optional[str | Path] = None,
    subject_id: Optional[str] = None,
    fz_threshold: float = 20.0,
    frames: Optional[Frames] = None,
    filter_cutoff_hz: Optional[float] = 20.0,
    filter_order: int = 4,
) -> CaptureSession:
    """Load a full capture session into a time-aligned :class:`CaptureSession`.

    Parameters
    ----------
    c3d_path
        Mocap C3D containing markers and analog force-plate channels.
    left_belt_path, right_belt_path
        Optional Visual3D ``SPEEDCHANGE`` belt-speed exports.
    belt_rate_hz
        Belt sample rate in Hz. If ``None`` it is inferred from the C3D point
        duration (``n_belt_samples / point_duration_s``) and a warning is
        recorded, because the belt log carries no rate and is on a separate
        clock from the mocap.
    subject_mp_path
        Optional Visual3D ``.mp`` subject-metrics file (mass/height, etc.).
    filter_cutoff_hz
        Zero-phase (``filtfilt``) Butterworth low-pass cutoff, applied at ingest to
        **both** the markers (kinematics, at the point rate) and the force-plate
        force/moment channels (kinetics, at the analog rate). The matched cutoff keeps
        kinematics and kinetics at the same bandwidth for consistent inverse dynamics
        (Kristianslund et al. 2012). Marker NaN gaps are preserved (each contiguous run
        is filtered independently). Default ``20.0`` Hz; pass ``None`` to disable.
    filter_order
        Butterworth order (default 4; the ``filtfilt`` pass doubles the effective order).
    """

    c3d_path = Path(c3d_path)
    frames = frames or Frames()
    warnings: List[str] = []

    c3d = read_c3d(c3d_path)
    n_frames = c3d.n_frames

    # Preprocessing: zero-phase Butterworth low-pass on kinematics + kinetics.
    filter_info: Optional[Dict[str, Any]] = None
    markers = c3d.points
    kinetics_cutoff: Optional[float] = None
    if filter_cutoff_hz is not None:
        point_nyq = 0.5 * c3d.point_rate if c3d.point_rate else 0.0
        if 0.0 < filter_cutoff_hz < point_nyq:
            from .io.filters import filter_markers

            markers = filter_markers(
                c3d.points, c3d.point_rate, filter_cutoff_hz, filter_order
            )
        else:
            warnings.append(
                f"marker low-pass skipped: cutoff {filter_cutoff_hz} Hz not below "
                f"point Nyquist ({point_nyq} Hz)"
            )
        analog_nyq = 0.5 * c3d.analog_rate if c3d.analog_rate else 0.0
        if 0.0 < filter_cutoff_hz < analog_nyq:
            kinetics_cutoff = filter_cutoff_hz
        filter_info = {
            "type": "butterworth_lowpass",
            "order": filter_order,
            "cutoff_hz": filter_cutoff_hz,
            "zero_phase": True,
            "markers_filtered": markers is not c3d.points,
            "kinetics_filtered": kinetics_cutoff is not None,
        }

    # Master timelines. Point sample k is at (first_frame - 1 + k) / rate so the
    # analog samples for a point frame align at the start of that frame's block.
    base = (c3d.first_frame - 1) / c3d.point_rate if c3d.point_rate else 0.0
    t_point = base + np.arange(n_frames, dtype=np.float64) / c3d.point_rate
    n_analog = c3d.n_analog_samples
    t_analog = (
        base + np.arange(n_analog, dtype=np.float64) / c3d.analog_rate
        if c3d.analog_rate
        else np.zeros(n_analog)
    )

    force_plates = compute_force_plates(
        c3d,
        fz_threshold=fz_threshold,
        filter_cutoff_hz=kinetics_cutoff,
        filter_order=filter_order,
    )
    for plate in force_plates:
        warnings.extend(plate.warnings)

    # Treadmill --------------------------------------------------------------
    treadmill: Optional[Treadmill] = None
    belt_point: Dict[str, np.ndarray] = {}
    belt_analog: Dict[str, np.ndarray] = {}
    if left_belt_path is not None or right_belt_path is not None:
        treadmill = load_treadmill(
            left_path=left_belt_path,
            right_path=right_belt_path,
            rate_hz=belt_rate_hz,
            reference_duration_s=c3d.duration_s,
        )
        if treadmill is not None:
            if treadmill.rate_inferred:
                warnings.append(
                    f"belt sample rate inferred as {treadmill.rate_hz:.3f} Hz "
                    f"from C3D point duration ({c3d.duration_s:.3f} s); the belt "
                    f"log carries no rate and uses a separate clock. Pass "
                    f"belt_rate_hz to override."
                )
            resampled_point = treadmill.resample_to(t_point)
            resampled_analog = treadmill.resample_to(t_analog)
            belt_point = resampled_point
            belt_analog = resampled_analog

    events = _parse_events(c3d)

    protocol: Optional[TreadmillProtocol] = None
    if speedchange_path is not None:
        protocol = read_speedchange(speedchange_path)

    subject_meta: Dict[str, float] = {}
    if subject_mp_path is not None:
        subject_meta = read_subject_mp(subject_mp_path)

    if subject_id is None:
        # Try the C3D "Subject:" prefix, else the file stem.
        prefix = ""
        if c3d.point_labels_raw and ":" in c3d.point_labels_raw[0]:
            prefix = c3d.point_labels_raw[0].split(":", 1)[0]
        subject_id = prefix or c3d_path.stem

    return CaptureSession(
        subject_id=subject_id,
        frames=frames,
        source={
            "c3d": str(c3d_path),
            "left_belt": str(left_belt_path) if left_belt_path else None,
            "right_belt": str(right_belt_path) if right_belt_path else None,
            "subject_mp": str(subject_mp_path) if subject_mp_path else None,
            "speedchange": str(speedchange_path) if speedchange_path else None,
        },
        point_rate=c3d.point_rate,
        t_point=t_point,
        marker_labels=c3d.point_labels,
        markers=markers,
        analog_rate=c3d.analog_rate,
        t_analog=t_analog,
        analog_per_point_frame=c3d.analog_per_point_frame,
        force_plates=force_plates,
        treadmill=treadmill,
        belt_speed_point=belt_point,
        belt_speed_analog=belt_analog,
        protocol=protocol,
        events=events,
        subject_meta=subject_meta,
        warnings=warnings,
        filter_info=filter_info,
        c3d=c3d,
    )
