# SPDX-License-Identifier: MIT

"""Tests for the fit priors (biomech.fitting.priors: MarkerOffsetPrior + Anthropometrics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biomech.fitting.marker_fitter import MarkerFitConfig
from biomech.fitting.priors import Anthropometrics, MarkerOffsetPrior
from biomech.osim import parse_osim
from biomech.tests import SUBJECT_MP, require

_ROOT = Path(__file__).resolve().parents[1]
_OSIM = _ROOT / "models" / "rajagopal_data" / "Rajagopal2015.osim"

_spec_cache = None


def _spec():
    global _spec_cache
    if _spec_cache is None:
        _spec_cache = parse_osim(str(_OSIM))
    return _spec_cache


def test_marker_offset_prior_weights():
    prior = MarkerOffsetPrior(base_weight=1.0, anatomical_factor=25.0)
    anat = np.array([True, False, True, False])
    w = prior.per_marker_weights(anat)
    assert np.allclose(w, [25.0, 1.0, 25.0, 1.0])


def test_marker_offset_prior_anatomical_flags():
    spec = _spec()
    flags = MarkerOffsetPrior.anatomical_flags(spec)
    assert flags.shape == (len(spec.markers),)
    assert flags.dtype == bool


def test_anthropometrics_scale_prior_shapes():
    spec = _spec()
    mp = require(SUBJECT_MP)
    anthro = Anthropometrics(mp_path=mp)
    G = len(spec.scale_groups)
    target, weights, diag = anthro.scale_prior(spec)
    assert target.shape == (3 * G,)
    assert weights.shape == (3 * G,)
    assert np.all(weights >= 0.0)
    assert np.all(target > 0.0)
    # at least the lower-body segment-length axes should be constrained
    assert np.count_nonzero(weights) >= 3
    assert "subject_upper_leg_length_m" in diag


def test_anthropometrics_apply_to_config():
    spec = _spec()
    mp = require(SUBJECT_MP)
    cfg = MarkerFitConfig()
    assert cfg.scale_prior_target is None
    diag = Anthropometrics(mp_path=mp).apply_to_config(spec, cfg)
    G = len(spec.scale_groups)
    assert cfg.scale_prior_target is not None
    assert cfg.scale_prior_target.shape == (3 * G,)
    assert cfg.scale_prior_weights.shape == (3 * G,)
    assert isinstance(diag, dict) and diag
