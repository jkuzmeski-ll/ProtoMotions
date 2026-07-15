# SPDX-License-Identifier: MIT
#
# Stance segmentation + flat-foot ground registration for the contact pipeline.
#
# Why this exists: predicting instantaneous GRF from *prescribed* (reconstructed)
# kinematics with a stiff distributed spring is ill-conditioned — the reconstructed
# foot vertical position carries centimeter-scale, time-varying error (heel-strike vs
# midstance can differ by cm), so a single floating ground plane over a stride cannot
# make the per-frame penetration physical. Two robustness tools help:
#
#   1. Register the ground plane from *flat-foot* frames only (foot planted, sole nearly
#      horizontal, low vertical speed, high measured load) — the frames where the
#      prescribed-kinematics contact model is actually trustworthy.
#   2. Segment measured GRF into contiguous stance phases so calibration can use an
#      aggregate (stance-mean / impulse) objective that averages out unbiased per-frame
#      kinematic noise instead of chasing every noisy instantaneous sample.
#
# World Z-up; foot pose is world position + xyzw quaternion (COMMON), matching
# ``biomech.contact.elastic_foundation`` / ``biomech.export.motion``.

"""Stance segmentation + flat-foot ground registration."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from biomech.contact.elastic_foundation import FootSole, _quat_rotate_np


# ---------------------------------------------------------------------------
# Sole kinematics helpers
# ---------------------------------------------------------------------------


def sole_world_points(sole: FootSole, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """World coordinates of every sole patch over a trajectory: ``(F, N, 3)``."""
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    F, N = pos.shape[0], sole.n
    pl = np.broadcast_to(sole.points, (F, N, 3))
    return pos[:, None, :] + _quat_rotate_np(quat[:, None, :], pl)


def sole_world_z(sole: FootSole, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """World z of every sole patch over a foot trajectory: ``(F, N)``."""
    return sole_world_points(sole, pos, quat)[:, :, 2]


def sole_world_normal_z(sole: FootSole, quat: np.ndarray) -> np.ndarray:
    """World z-component of the (area-weighted) mean sole normal per frame: ``(F,)``.

    The plantar normal points *down* in the body frame (``~ -z``), so when the foot is
    flat on the ground this is close to ``-1``. ``|value| -> 1`` means horizontal;
    ``-> 0`` means the sole is vertical (edge contact).
    """
    quat = np.asarray(quat, dtype=np.float64)
    F = quat.shape[0]
    w = sole.areas / (np.sum(sole.areas) + 1e-12)
    mean_n = (sole.normals * w[:, None]).sum(axis=0)  # (3,) body-frame mean normal
    mean_n = np.broadcast_to(mean_n, (F, 3))
    world_n = _quat_rotate_np(quat, mean_n)  # (F, 3)
    return world_n[:, 2]


# ---------------------------------------------------------------------------
# Stance segmentation
# ---------------------------------------------------------------------------


def stance_mask(fz: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of loaded frames (measured vertical GRF above ``threshold``)."""
    return np.asarray(fz, dtype=np.float64) > threshold


def segment_contacts(
    fz: np.ndarray, threshold: float, min_len: int = 1
) -> List[Tuple[int, int]]:
    """Contiguous half-open stance intervals ``[(start, end), ...]`` where ``fz>threshold``.

    Intervals shorter than ``min_len`` frames are dropped (debounce brief spikes).
    """
    mask = stance_mask(fz, threshold)
    segs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segs.append((start, len(mask)))
    return segs


# ---------------------------------------------------------------------------
# Flat-foot detection + robust ground registration
# ---------------------------------------------------------------------------


def flat_foot_mask(
    sole: FootSole,
    pos: np.ndarray,
    quat: np.ndarray,
    linvel: Optional[np.ndarray] = None,
    fz: Optional[np.ndarray] = None,
    fz_frac: float = 0.5,
    fz_threshold: float = 20.0,
    tilt_cos: float = 0.94,  # |normal_z| >= this  (~20 deg of horizontal)
    speed_tol: float = 0.15,  # m/s max foot vertical speed to count as planted
) -> np.ndarray:
    """Frames where the foot is planted flat (trustworthy for ground registration).

    A frame qualifies when the sole is near-horizontal (``|world normal_z| >= tilt_cos``),
    the foot vertical speed is low (``|linvel_z| <= speed_tol``, if ``linvel`` given), and
    the measured load is high (``fz >= max(fz_frac*peak, fz_threshold)``, if ``fz`` given).
    """
    quat = np.asarray(quat, dtype=np.float64)
    ok = np.abs(sole_world_normal_z(sole, quat)) >= tilt_cos
    if linvel is not None:
        ok &= np.abs(np.asarray(linvel, dtype=np.float64)[:, 2]) <= speed_tol
    if fz is not None:
        fz = np.asarray(fz, dtype=np.float64)
        peak = float(np.nanmax(fz)) if fz.size else 0.0
        ok &= fz >= max(fz_frac * peak, fz_threshold)
    return ok


def register_ground_flatfoot(
    sole: FootSole,
    pos: np.ndarray,
    quat: np.ndarray,
    flat_mask: np.ndarray,
    penetration: float = 0.005,
    fallback: Optional[np.ndarray] = None,
) -> float:
    """Ground plane z from flat-foot frames: median lowest sole z + ``penetration``.

    Flat-foot frames are where the reconstructed foot is genuinely planted, so their
    lowest sole point marks the true belt surface far more reliably than a percentile
    over the whole (rolling) stride. Falls back to ``fallback`` frames (else all frames)
    when no flat-foot frame is found.
    """
    zsole = sole_world_z(sole, pos, quat)  # (F, N)
    per_frame_min = np.min(zsole, axis=1)  # (F,)
    flat_mask = np.asarray(flat_mask, dtype=bool)
    if flat_mask.any():
        sel = per_frame_min[flat_mask]
    elif fallback is not None and np.asarray(fallback, dtype=bool).any():
        sel = per_frame_min[np.asarray(fallback, dtype=bool)]
    else:
        sel = per_frame_min
    return float(np.median(sel)) + penetration
