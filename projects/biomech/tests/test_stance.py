# SPDX-License-Identifier: MIT

"""Tests for stance segmentation + flat-foot ground registration (biomech.contact.stance).

Validates the contact-pipeline robustness helpers on synthetic data:

- ``segment_contacts`` finds contiguous loaded intervals and debounces short spikes,
- ``sole_world_normal_z`` reports how horizontal the sole is,
- ``flat_foot_mask`` selects genuinely planted (flat, slow, loaded) frames,
- ``register_ground_flatfoot`` places the ground from flat-foot frames only and is
  robust to a foot that rolls / lifts (unlike a percentile over the whole stride).

No pytest dependency: run ``python projects/biomech/run_tests.py``.
"""

from __future__ import annotations

import numpy as np

from biomech.contact.elastic_foundation import sample_flat_sole
from biomech.contact.stance import (
    flat_foot_mask,
    register_ground_flatfoot,
    segment_contacts,
    sole_world_normal_z,
    sole_world_z,
    stance_mask,
)

_QID = np.array([0.0, 0.0, 0.0, 1.0])


def _quat_about_y(angle):
    # xyzw quaternion for a rotation of `angle` about the world/body y axis
    return np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])


def test_stance_mask_and_segments():
    fz = np.array([0.0, 5.0, 30.0, 40.0, 10.0, 0.0, 50.0, 60.0, 0.0])
    m = stance_mask(fz, threshold=20.0)
    assert m.tolist() == [False, False, True, True, False, False, True, True, False]
    segs = segment_contacts(fz, threshold=20.0)
    assert segs == [(2, 4), (6, 8)]


def test_segment_debounces_short_spikes():
    fz = np.array([0.0, 100.0, 0.0, 50.0, 50.0, 50.0, 0.0])
    # min_len=2 drops the single-frame spike at index 1 but keeps the 3-frame stance
    segs = segment_contacts(fz, threshold=20.0, min_len=2)
    assert segs == [(3, 6)]


def test_segment_trailing_stance_closes():
    fz = np.array([0.0, 30.0, 30.0])
    assert segment_contacts(fz, threshold=20.0) == [(1, 3)]


def test_sole_world_normal_z_flat_vs_tilted():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)  # normal points -z (down) in body frame
    # flat foot: world normal z ~ -1
    nz_flat = sole_world_normal_z(sole, _QID[None, :])
    assert abs(nz_flat[0] + 1.0) < 1e-6
    # rotate 90 deg about y: sole normal becomes horizontal -> nz ~ 0
    nz_tilt = sole_world_normal_z(sole, _quat_about_y(np.pi / 2)[None, :])
    assert abs(nz_tilt[0]) < 1e-6


def test_flat_foot_mask_selects_planted_frames():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    F = 6
    quat = np.tile(_QID, (F, 1))
    # frame 3 is tilted (heel strike), frame 4 has high vertical speed (lifting)
    quat[3] = _quat_about_y(0.6)
    linvel = np.zeros((F, 3))
    linvel[4, 2] = 0.5  # fast lift
    fz = np.array([0.0, 600.0, 600.0, 600.0, 600.0, 5.0])  # frame 5 unloaded
    mask = flat_foot_mask(sole, np.zeros((F, 3)), quat, linvel=linvel, fz=fz,
                          fz_frac=0.5, fz_threshold=20.0)
    # planted, flat, slow, loaded: frames 1 and 2 only
    assert mask.tolist() == [False, True, True, False, False, False]


def test_register_ground_flatfoot_ignores_roll_and_lift():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    F = 8
    quat = np.tile(_QID, (F, 1))
    pos = np.zeros((F, 3))
    # planted flat-foot frames sit near z=0; rolling/lifting frames sit much higher
    pos[:, 2] = [0.0, 0.0, 0.0, 0.06, 0.10, 0.15, 0.20, 0.28]
    linvel = np.zeros((F, 3))
    linvel[3:, 2] = 0.6  # lifting during the later frames
    fz = np.array([600.0, 600.0, 600.0, 400.0, 200.0, 50.0, 0.0, 0.0])
    flat = flat_foot_mask(sole, pos, quat, linvel=linvel, fz=fz)
    gz = register_ground_flatfoot(sole, pos, quat, flat, penetration=0.005)
    # ground registered from the planted frames (~z=0), not dragged up by the lift
    assert abs(gz - 0.005) < 1e-6


def test_register_ground_flatfoot_fallback():
    sole = sample_flat_sole(0.2, 0.1, 6, 4)
    F = 4
    quat = np.tile(_QID, (F, 1))
    pos = np.zeros((F, 3))
    pos[:, 2] = [0.02, 0.03, 0.02, 0.03]
    no_flat = np.zeros(F, dtype=bool)
    fallback = np.ones(F, dtype=bool)
    gz = register_ground_flatfoot(sole, pos, quat, no_flat, penetration=0.0,
                                  fallback=fallback)
    # falls back to the median lowest sole z over all frames
    zmin = sole_world_z(sole, pos, quat).min(axis=1)
    assert abs(gz - float(np.median(zmin))) < 1e-9
