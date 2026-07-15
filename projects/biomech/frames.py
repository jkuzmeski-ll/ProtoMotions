# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit coordinate-frame and unit conventions for the biomech pipeline.

The whole point of this module is that *no downstream code should ever guess* a
frame, unit, or up-axis. Every geometric quantity produced by the loader carries
an associated :class:`CoordinateFrame`, and everything is converted to SI
(meters, Newtons, Newton-meters, seconds) at ingest time.

Frames tracked by Milestone 1
-----------------------------
- ``LAB``: the mocap/lab frame as stored in the C3D (after mm -> m). For the
  MITLL / Motek split-belt capture (``projects/data/S001``) this frame is
  **Z-up, right-handed, meters** (verified from ``TRIAL:Z_DIRECTION`` and the
  ``POINT:X_SCREEN``/``Y_SCREEN`` parameters).
- ``FORCE_PLATE``: each treadmill plate's own coordinate frame. For this capture
  the plates are axis-aligned with the lab frame (corners are axis-aligned and
  share a constant z), so the plate->lab transform is a pure translation.
- ``TREADMILL``: the belt-fixed frame. Shares orientation with ``LAB``; the belt
  translates along the lab forward axis at the belt speed. Belt velocity lives
  here.
- ``WORLD``: the ProtoMotions / Newton simulation world. Z-up, meters. For this
  capture ``WORLD == LAB`` (identity), so :func:`lab_to_world_rotation` returns
  identity. Downstream OpenSim skeletons are Y-up and require a *separate*
  conversion (``R_OS2PM``); that conversion is intentionally NOT applied here,
  because the lab/measured data is already Z-up.

Sign convention for forces
--------------------------
Raw force-plate channels record the force applied *by the subject onto the
plate* (vertical channel is negative during stance). The biomechanics
"ground reaction force" (GRF) is the force applied *onto the subject*, i.e. the
negation. The loader exposes both: ``force_measured`` (as recorded) and ``grf``
(on the subject, +Z up during stance).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# --- Unit conversions to SI ---------------------------------------------------
MM_TO_M: float = 1.0e-3
NMM_TO_NM: float = 1.0e-3


class UpAxis(enum.IntEnum):
    """Index of the up axis in a right-handed (x, y, z) frame."""

    X = 0
    Y = 1
    Z = 2


class CoordinateFrame(enum.Enum):
    """Named coordinate frames tracked by the pipeline."""

    LAB = "lab"
    FORCE_PLATE = "force_plate"
    TREADMILL = "treadmill"
    WORLD = "world"


def lab_to_world_rotation() -> np.ndarray:
    """Rotation mapping lab-frame vectors into the ProtoMotions world frame.

    Identity for the S001 capture because the lab frame is already Z-up meters,
    matching the ProtoMotions/Newton world convention.
    """

    return np.eye(3, dtype=np.float64)


@dataclass(frozen=True)
class Frames:
    """Bundle of frame/unit metadata attached to a loaded capture.

    Attributes are intentionally explicit so that consumers can assert on them
    rather than assuming a convention.
    """

    lab_up_axis: UpAxis = UpAxis.Z
    world_up_axis: UpAxis = UpAxis.Z
    length_unit: str = "m"
    force_unit: str = "N"
    moment_unit: str = "N*m"
    time_unit: str = "s"
    # Rotation lab -> world. Identity for Z-up lab captures.
    lab_to_world: np.ndarray = field(default_factory=lab_to_world_rotation)
    # Human-readable notes recording where each convention came from.
    notes: Dict[str, str] = field(
        default_factory=lambda: {
            "lab": "Z-up, right-handed, meters (TRIAL:Z_DIRECTION=+Z).",
            "world": "ProtoMotions/Newton world == LAB (identity).",
            "grf_sign": "grf = -force_measured (force ON subject, +Z up in stance).",
            "opensim": "OpenSim skeleton is Y-up; converted downstream, not here.",
        }
    )

    def as_dict(self) -> Dict[str, object]:
        return {
            "lab_up_axis": self.lab_up_axis.name,
            "world_up_axis": self.world_up_axis.name,
            "length_unit": self.length_unit,
            "force_unit": self.force_unit,
            "moment_unit": self.moment_unit,
            "time_unit": self.time_unit,
            "lab_to_world": self.lab_to_world.tolist(),
            "notes": dict(self.notes),
        }
