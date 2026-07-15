# SPDX-License-Identifier: MIT

"""Tests for robust marker cleanup (biomech.fitting.marker_fixer, MarkerFixer port).

Synthetic rigid-body marker cloud: 4 markers rigidly attached to one body, moved through
a smooth trajectory. We inject a single-frame spike, a short dropout gap, and a gross
rigid-body outlier, then check the fixer blanks the bad samples and fills the short gaps
back to (near) the ground truth.
"""

from __future__ import annotations

import numpy as np

from biomech.fitting.marker_fixer import MarkerFixConfig, fix_markers


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _clean_cloud(F=40):
    # 4 markers on one rigid body (a small tetrahedron), moving smoothly.
    base = np.array(
        [
            [0.05, 0.0, 0.0],
            [-0.05, 0.03, 0.0],
            [0.0, -0.05, 0.02],
            [0.0, 0.0, 0.06],
        ]
    )
    obs = np.zeros((F, 4, 3))
    for t in range(F):
        R = _rot_z(0.02 * t)
        trans = np.array([0.01 * t, 0.005 * t, 0.2 + 0.001 * t])
        obs[t] = (base @ R.T) + trans
    return obs


def test_fills_short_gap():
    obs = _clean_cloud()
    truth = obs.copy()
    obs[10:13, 1] = np.nan  # 3-frame dropout on marker 1
    cleaned, rep = fix_markers(obs, config=MarkerFixConfig())
    assert rep.n_filled >= 3
    assert np.isfinite(cleaned[10:13, 1]).all()
    assert np.max(np.abs(cleaned[10:13, 1] - truth[10:13, 1])) < 2e-3


def test_blanks_velocity_spike():
    obs = _clean_cloud()
    obs[20, 0] += np.array([0.5, 0.0, 0.0])  # 50 cm ghost jump on marker 0
    cleaned, rep = fix_markers(obs, config=MarkerFixConfig())
    assert rep.n_spikes_rejected + rep.n_rigid_rejected > 0
    # the ghost value must not survive (either NaN or filled back near truth)
    truth = _clean_cloud()
    assert (not np.isfinite(cleaned[20, 0]).all()) or (
        np.linalg.norm(cleaned[20, 0] - truth[20, 0]) < 5e-2
    )


def test_rigid_outlier_rejected():
    obs = _clean_cloud()
    truth = obs.copy()
    # marker 2 jumps 20 cm off the rigid body at frame 15 but is "smoothly" placed so the
    # velocity gate alone would be borderline; the rigid pairwise-distance check catches it.
    obs[15, 2] += np.array([0.0, 0.2, 0.0])
    body = np.zeros(4, dtype=int)  # all 4 on the same body
    cleaned, rep = fix_markers(obs, body_of_marker=body, config=MarkerFixConfig())
    assert rep.n_rigid_rejected + rep.n_spikes_rejected > 0
    # after fixing, marker 2 at frame 15 is either blanked or interpolated back near truth
    assert (not np.isfinite(cleaned[15, 2]).all()) or (
        np.linalg.norm(cleaned[15, 2] - truth[15, 2]) < 2e-2
    )


def test_clean_data_is_untouched():
    obs = _clean_cloud()
    body = np.zeros(4, dtype=int)
    cleaned, rep = fix_markers(obs.copy(), body_of_marker=body, config=MarkerFixConfig())
    assert rep.n_rigid_rejected == 0
    assert rep.n_spikes_rejected == 0
    assert rep.n_filled == 0
    assert np.allclose(cleaned, obs)


def test_input_not_mutated():
    obs = _clean_cloud()
    obs[5, 0] += np.array([0.4, 0.0, 0.0])
    snapshot = obs.copy()
    fix_markers(obs, config=MarkerFixConfig())
    assert np.array_equal(obs, snapshot, equal_nan=True)
