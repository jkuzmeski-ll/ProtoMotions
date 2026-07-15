# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-phase Butterworth low-pass filtering for kinematics + kinetics preprocessing.

Standard biomechanics preprocessing: a 4th-order Butterworth low-pass applied with
**zero phase lag** (forward-backward, ``scipy.signal.filtfilt``) so marker trajectories
and ground-reaction forces are not time-shifted relative to each other or to gait events.

Why a *matched* cutoff on both streams: feeding kinematics (segment accelerations) and
kinetics (GRF) of different bandwidth into inverse dynamics produces artifactual joint
moments, most visibly a spike at heel strike. Filtering both at the same cutoff avoids
this (Kristianslund, Krosshaug & van den Bogert, 2012, *J. Biomech.* 45:666-671;
Bisseling & Hof, 2006). A 4th-order Butterworth through ``filtfilt`` is effectively an
8th-order zero-lag response -- the conventional recipe (Winter, *Biomechanics and Motor
Control of Human Movement*).

Marker data has occlusion **gaps** (NaN). Butterworth filtering cannot cross a NaN, so
:func:`filter_markers` filters each contiguous finite run per marker/axis independently
and leaves the gaps as NaN (no data is fabricated across gaps -- gap filling is a
separate concern, see ``fitting/marker_fixer.py``). Force/moment analog channels are
dense, so they are filtered directly.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def design_butter_lowpass(
    cutoff_hz: float, fs_hz: float, order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(b, a)`` for a Butterworth low-pass at ``cutoff_hz`` given ``fs_hz``.

    Raises ``ValueError`` if the cutoff is not strictly below the Nyquist frequency.
    """
    from scipy.signal import butter

    nyq = 0.5 * float(fs_hz)
    if not (0.0 < cutoff_hz < nyq):
        raise ValueError(
            f"cutoff {cutoff_hz} Hz must be in (0, Nyquist={nyq} Hz) for fs={fs_hz} Hz"
        )
    wn = float(cutoff_hz) / nyq
    res = cast(Any, butter(order, wn, btype="low"))
    b = np.asarray(res[0], dtype=np.float64)
    a = np.asarray(res[1], dtype=np.float64)
    return b, a


def _filtfilt_run(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Zero-phase filter a single 1-D finite run, padding safely for short runs."""
    from scipy.signal import filtfilt

    n = x.size
    ntaps = max(len(a), len(b))
    if n <= ntaps:
        # Too short to filter meaningfully; leave as-is rather than fabricate edges.
        return x
    padlen = min(3 * ntaps, n - 1)
    return filtfilt(b, a, x, padlen=padlen)


def lowpass_nan_1d(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Zero-phase low-pass a 1-D signal, filtering each contiguous finite run.

    NaN gaps are preserved; each run of finite samples between gaps is filtered on its
    own so the filter never crosses (and never fills) a gap.
    """
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    finite = np.isfinite(x)
    if finite.all():
        return _filtfilt_run(x, b, a)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return out
    # split the finite indices into contiguous runs
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, splits):
        if run.size:
            out[run] = _filtfilt_run(x[run], b, a)
    return out


def lowpass_dense(
    x: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4, axis: int = 0
) -> np.ndarray:
    """Zero-phase Butterworth low-pass a dense (NaN-free) array along ``axis``."""
    from scipy.signal import filtfilt

    b, a = design_butter_lowpass(cutoff_hz, fs_hz, order)
    return filtfilt(b, a, np.asarray(x, dtype=np.float64), axis=axis)


def filter_markers(
    markers: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 4
) -> np.ndarray:
    """Zero-phase low-pass a ``(F, M, 3)`` marker array, preserving NaN gaps.

    Filters each marker/axis time series independently over its contiguous finite runs.
    """
    markers = np.asarray(markers, dtype=np.float64)
    if markers.ndim != 3 or markers.shape[2] != 3:
        raise ValueError(f"markers must be (F, M, 3), got {markers.shape}")
    b, a = design_butter_lowpass(cutoff_hz, fs_hz, order)
    out = markers.copy()
    F, M, D = markers.shape
    for m in range(M):
        for c in range(D):
            out[:, m, c] = lowpass_nan_1d(markers[:, m, c], b, a)
    return out
