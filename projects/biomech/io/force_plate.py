# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force-plate GRF / COP / free-moment extraction from C3D analog data.

For the S001 split-belt instrumented treadmill each belt is a C3D TYPE-2
(force+moment) plate with 6 channels (Fx, Fy, Fz [N]; Mx, My, Mz [N*mm]).

Frames & conventions (validated against the S001 capture)
---------------------------------------------------------
- Plates are axis-aligned with the lab frame: ``FORCE_PLATFORM:CORNERS`` are
  axis-aligned and share a constant z, so the plate frame differs from the lab
  frame only by a translation. The moment reference point equals the corner
  centroid (== ``FORCE_PLATFORM:ORIGIN`` for this vendor), expressed in world.
- Vertical force is negative during stance (force *onto* the plate). The GRF
  (force *onto the subject*) is the negation, +Z up during stance.
- COP is computed on the plate surface plane and returned in the **world**
  frame; it is ``NaN`` when the vertical load is below ``fz_threshold`` so that
  swing samples do not produce garbage COP values.

COP derivation (moment ``M`` measured about world point ``O``)::

    M = (COP - O) x F,  with  COP.z = z_surface
    COP.x = O.x + [(z_surface - O.z) * Fx - My] / Fz
    COP.y = O.y + [ Mx + (z_surface - O.z) * Fy] / Fz
    free_moment_z = Mz - [(COP.x - O.x) * Fy - (COP.y - O.y) * Fx]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..frames import MM_TO_M, NMM_TO_NM, CoordinateFrame
from .c3d import C3DFile


@dataclass
class ForcePlate:
    """Per-plate ground-reaction signals, all in SI world-frame quantities."""

    index: int  # 0-based plate index
    label: str
    plate_type: int
    rate: float  # Hz (analog rate)
    channels: List[int]  # 1-based analog channel indices used

    # Geometry (world frame, meters)
    corners_world: np.ndarray  # [4, 3]
    center_world: np.ndarray  # [3] corner centroid == moment reference
    origin_param_world: np.ndarray  # [3] FORCE_PLATFORM:ORIGIN as stored (m)
    surface_z: float
    x_sign: int  # +1 if plate centered at +x, -1 if -x (belt-side hint)

    # Signals [n_samples, 3] / [n_samples]
    force_measured: np.ndarray  # force ON plate, N (as recorded)
    moment_measured: np.ndarray  # moment about center_world, N*m
    grf: np.ndarray  # force ON subject = -force_measured, N (+Z up in stance)
    cop_world: np.ndarray  # [n, 3] m, NaN below fz_threshold
    free_moment_z: np.ndarray  # [n] N*m about COP, NaN below fz_threshold

    fz_threshold: float = 20.0
    frame: CoordinateFrame = CoordinateFrame.WORLD
    warnings: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.force_measured.shape[0]

    @property
    def vertical_load(self) -> np.ndarray:
        """Upward GRF magnitude on the subject (N), clamped at 0."""

        return np.clip(self.grf[:, 2], a_min=0.0, a_max=None)

    def summary(self) -> Dict[str, Any]:
        vz = self.vertical_load
        stance = vz > self.fz_threshold
        return {
            "index": self.index,
            "label": self.label,
            "type": self.plate_type,
            "x_sign": self.x_sign,
            "center_world_m": self.center_world.tolist(),
            "surface_z_m": self.surface_z,
            "peak_vertical_grf_N": float(np.nanmax(vz)) if vz.size else 0.0,
            "stance_fraction": float(stance.mean()) if stance.size else 0.0,
            "mean_stance_vertical_grf_N": (
                float(vz[stance].mean()) if stance.any() else 0.0
            ),
        }


def _reshape_corners(raw: Any) -> np.ndarray:
    """Return ``[n_plates, 4, 3]`` corners in meters.

    C3D stores ``FORCE_PLATFORM:CORNERS`` with dims ``[3, 4, n_plates]`` in
    column-major order.
    """

    arr = np.asarray(raw, dtype=np.float64).ravel()
    n_plates = arr.size // 12
    # Fortran order matches C3D's [3, 4, n_plates] layout.
    corners = arr.reshape(3, 4, n_plates, order="F")
    corners = np.transpose(corners, (2, 1, 0))  # [n_plates, 4, 3]
    return corners * MM_TO_M


def _reshape_origin(raw: Any, n_plates: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64).ravel()
    origin = arr.reshape(3, n_plates, order="F").T  # [n_plates, 3]
    return origin * MM_TO_M


def _reshape_channels(raw: Any, n_plates: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.int64).ravel()
    per_plate = arr.size // n_plates
    return arr.reshape(per_plate, n_plates, order="F").T  # [n_plates, per_plate]


def compute_force_plates(
    c3d: C3DFile, fz_threshold: float = 20.0
) -> List[ForcePlate]:
    """Extract per-plate GRF/COP/free-moment signals from a parsed C3D file."""

    used = int(c3d.param("FORCE_PLATFORM", "USED", 0) or 0)
    if used <= 0:
        return []

    types = c3d.param("FORCE_PLATFORM", "TYPE", [2] * used)
    if not isinstance(types, list):
        types = [types]
    corners = _reshape_corners(c3d.param("FORCE_PLATFORM", "CORNERS"))
    origin = _reshape_origin(c3d.param("FORCE_PLATFORM", "ORIGIN"), used)
    channels = _reshape_channels(c3d.param("FORCE_PLATFORM", "CHANNEL"), used)

    plates: List[ForcePlate] = []
    for p in range(used):
        plate_channels = [int(ch) for ch in channels[p]]
        plate_type = int(types[p]) if p < len(types) else 2
        warnings: List[str] = []

        if plate_type not in (2, 4):
            warnings.append(
                f"plate {p}: TYPE {plate_type} not validated; treating as "
                f"force+moment (TYPE 2)."
            )
        if len(plate_channels) < 6:
            raise ValueError(
                f"force plate {p} has {len(plate_channels)} channels; need >=6"
            )

        fx_i, fy_i, fz_i, mx_i, my_i, mz_i = [ci - 1 for ci in plate_channels[:6]]
        force = np.column_stack(
            [c3d.analog[:, fx_i], c3d.analog[:, fy_i], c3d.analog[:, fz_i]]
        )
        # Moments recorded in N*mm -> N*m.
        moment = np.column_stack(
            [c3d.analog[:, mx_i], c3d.analog[:, my_i], c3d.analog[:, mz_i]]
        ) * NMM_TO_NM

        plate_corners = corners[p]
        center = plate_corners.mean(axis=0)
        surface_z = float(plate_corners[:, 2].mean())

        # Axis-alignment check: corners should share a constant z and form an
        # axis-aligned rectangle. If not, the pure-translation COP math is
        # invalid and we flag it for a downstream (rotated-plate) handler.
        if float(np.ptp(plate_corners[:, 2])) > 1e-3:
            warnings.append(
                f"plate {p}: corners not coplanar in z "
                f"(ptp={np.ptp(plate_corners[:, 2]):.4f} m); COP may be off."
            )

        origin_world = origin[p]
        # Moment reference point in world: use corner centroid, cross-checked
        # against ORIGIN. For this vendor they coincide.
        ref = center.copy()
        if not np.allclose(origin_world[:2], center[:2], atol=5e-3):
            warnings.append(
                f"plate {p}: ORIGIN {origin_world.tolist()} != corner centroid "
                f"{center.tolist()}; using ORIGIN as moment reference."
            )
            ref = origin_world.copy()

        fz = force[:, 2]
        loaded = np.abs(fz) > fz_threshold
        dz = surface_z - ref[2]

        cop = np.full_like(force, np.nan)
        free_mz = np.full(force.shape[0], np.nan)
        safe = loaded & (np.abs(fz) > 1e-6)
        fz_safe = fz[safe]
        cop_x = ref[0] + (dz * force[safe, 0] - moment[safe, 1]) / fz_safe
        cop_y = ref[1] + (moment[safe, 0] + dz * force[safe, 1]) / fz_safe
        cop[safe, 0] = cop_x
        cop[safe, 1] = cop_y
        cop[safe, 2] = surface_z
        free_mz[safe] = moment[safe, 2] - (
            (cop_x - ref[0]) * force[safe, 1] - (cop_y - ref[1]) * force[safe, 0]
        )

        label = str(
            _plate_description(c3d, plate_channels[0]) or f"ForcePlate{p + 1}"
        )

        plates.append(
            ForcePlate(
                index=p,
                label=label,
                plate_type=plate_type,
                rate=c3d.analog_rate,
                channels=plate_channels,
                corners_world=plate_corners,
                center_world=center,
                origin_param_world=origin_world,
                surface_z=surface_z,
                x_sign=int(np.sign(center[0])) or 1,
                force_measured=force,
                moment_measured=moment,
                grf=-force,
                cop_world=cop,
                free_moment_z=free_mz,
                fz_threshold=fz_threshold,
                warnings=warnings,
            )
        )

    return plates


def _plate_description(c3d: C3DFile, one_based_channel: int) -> Optional[str]:
    desc = c3d.param("ANALOG", "DESCRIPTIONS")
    if isinstance(desc, list) and 0 < one_based_channel <= len(desc):
        return desc[one_based_channel - 1]
    return None
